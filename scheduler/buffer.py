"""
ActiveTokenBuffer — scheduler/buffer.py

Data structure centrale du scheduler dynamique.
Maintient la liste des groupes de tokens actifs à chaque couche,
et les cu_seqlens nécessaires pour FlashAttention varlen.

Principe :
  - Un buffer global (batch, seq, d_model) est pré-alloué une fois.
    Jamais réalloué pendant le forward pass.
  - active_group_idx : indices des groupes encore actifs dans le buffer.
    Un groupe = warp_size tokens consécutifs dans la dimension batch.
    Un groupe est stoppé seulement si TOUS ses tokens ont halt_prob > 0.5.
  - cu_seqlens : offsets cumulatifs pour FlashAttention varlen.
    Recalculé en O(n_active) après chaque update.

Relation avec FlashAttention varlen :
  flash_attn_varlen_qkvpacked_func attend :
    qkv        : (total_active_tokens, 3, n_heads, head_dim)
    cu_seqlens : (n_active_samples + 1,)  — [0, seq_len, 2*seq_len, ...]
    max_seqlen : seq_len
  get_active_qkv() retourne exactement ce format.
"""

import torch
from torch import Tensor


class ActiveTokenBuffer:
    """
    Buffer de tokens actifs pour le scheduler warp-aligned.

    Args:
        batch_size : nombre de samples dans le batch
        seq_len    : longueur de séquence (fixe — c'est la longueur, pas la profondeur)
        d_model    : dimension du modèle
        T_max      : profondeur maximale (pour le suivi des stop times)
        warp_size  : granularité de la partition (32 par défaut — taille d'un warp GPU)
        device     : device CUDA
        dtype      : dtype du modèle
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        d_model: int,
        T_max: int,
        warp_size: int = 32,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.batch_size  = batch_size
        self.seq_len     = seq_len
        self.d_model     = d_model
        self.T_max       = T_max
        self.warp_size   = warp_size
        self.device      = torch.device(device)
        self.dtype       = dtype

        # Nombre de groupes (arrondi inférieur — les tokens résiduels sont ignorés)
        self.n_groups    = batch_size // warp_size

        # Buffer global — pré-alloué une fois, jamais réalloué
        # Shape : (batch_size, seq_len, d_model)
        self.states      = torch.zeros(batch_size, seq_len, d_model, device=self.device, dtype=dtype)

        # Indices des groupes actifs — initialement tous actifs
        # Shape variable : (n_active_groups,)
        self.active_group_idx = torch.arange(self.n_groups, device=self.device)

        # Stop time par groupe — pour le calcul de L_halt
        # T_i = step auquel le groupe s'est arrêté (ou T_max si jamais arrêté)
        self.group_stop_times = torch.full(
            (self.n_groups,), T_max, dtype=torch.long, device=self.device
        )

        # cu_seqlens pour FlashAttention varlen
        # Shape : (n_active_groups + 1,)
        # = [0, seq_len, 2*seq_len, ..., n_active_groups * seq_len]
        self._update_cu_seqlens()

        # Step courant (pour stop times)
        self.current_step = 0

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def load(self, x: Tensor) -> None:
        """
        Charge les états initiaux depuis les embeddings.
        x : (batch_size, seq_len, d_model)
        À appeler une fois avant la boucle des couches.
        """
        assert x.shape == (self.batch_size, self.seq_len, self.d_model), \
            f"Shape attendu ({self.batch_size}, {self.seq_len}, {self.d_model}), reçu {x.shape}"
        self.states.copy_(x)
        self.active_group_idx = torch.arange(self.n_groups, device=self.device)
        self.group_stop_times.fill_(self.T_max)
        self.current_step = 0
        self._update_cu_seqlens()

    # ------------------------------------------------------------------
    # Accès aux états actifs
    # ------------------------------------------------------------------

    @property
    def n_active(self) -> int:
        """Nombre de groupes actifs."""
        return len(self.active_group_idx)

    @property
    def n_active_tokens(self) -> int:
        """Nombre de tokens actifs (pour FlashAttention)."""
        return self.n_active * self.warp_size

    def get_active_states(self) -> Tensor:
        """
        Retourne les états des tokens actifs.
        Shape : (n_active_groups * warp_size, seq_len, d_model)

        Note : retourne une copie contiguë (nécessaire pour FlashAttention).
        Les indices de groupes → indices de tokens dans le buffer global.
        """
        # Convertir les indices de groupes en indices de tokens
        token_idx = self._group_idx_to_token_idx(self.active_group_idx)
        return self.states[token_idx].contiguous()

    def write_active_states(self, h_new: Tensor) -> None:
        """
        Écrit les états mis à jour des tokens actifs dans le buffer global.
        h_new : (n_active_tokens, seq_len, d_model)
        """
        token_idx = self._group_idx_to_token_idx(self.active_group_idx)
        self.states[token_idx] = h_new

    def get_all_states(self) -> Tensor:
        """
        Retourne tous les états (actifs + stoppés).
        Shape : (batch_size, seq_len, d_model)
        Pour assembler la sortie finale après la boucle.
        """
        return self.states[:self.n_groups * self.warp_size]

    # ------------------------------------------------------------------
    # Update : partition actifs / stoppés
    # ------------------------------------------------------------------

    def update(self, halt_probs: Tensor, step: int) -> Tensor:
        """
        Partitionne les groupes actifs entre ceux qui continuent et ceux qui s'arrêtent.
        Met à jour active_group_idx et group_stop_times.

        halt_probs : (n_active_tokens,) ou (n_active_groups * warp_size,)
                     probabilités d'arrêt par token, calculées par la halt head

        Règle warp-aligned : un groupe s'arrête ssi TOUS ses tokens ont halt_prob > 0.5.
        Règle min — pas vote majoritaire.

        Retourne : stopped_mask (n_active_groups,) bool — True = groupe stoppé
        """
        self.current_step = step

        n_active = self.n_active
        assert halt_probs.shape[0] == n_active * self.warp_size, \
            f"halt_probs shape {halt_probs.shape} incompatible avec {n_active} groupes de {self.warp_size}"

        # Reshape en (n_active_groups, warp_size)
        probs_grouped = halt_probs.view(n_active, self.warp_size)

        # Règle min : groupe stoppé ssi min > 0.5
        group_stopped = probs_grouped.min(dim=1).values > 0.5  # (n_active_groups,)

        # Enregistrer les stop times pour les groupes qui s'arrêtent maintenant
        stopping_global_idx = self.active_group_idx[group_stopped]
        self.group_stop_times[stopping_global_idx] = step

        # Mettre à jour la liste des groupes actifs
        self.active_group_idx = self.active_group_idx[~group_stopped]

        # Recalculer cu_seqlens
        self._update_cu_seqlens()

        return group_stopped

    # ------------------------------------------------------------------
    # Défragmentation
    # ------------------------------------------------------------------

    def defrag_if_needed(self, step: int, freq: int = 4) -> bool:
        """
        Défragmente les slots actifs pour maintenir la coalescence mémoire.
        Copie les états actifs dans des slots contigus.

        À appeler seulement si nsight montre l1tex__t_sector_hit_rate < 50%.
        Sur GH200 HBM3e, peut ne pas être nécessaire — mesurer avant d'activer.

        Retourne True si défragmentation effectuée.
        """
        if step % freq != 0:
            return False
        if self.n_active == 0:
            return False

        # Récupérer les états actifs
        active_states = self.get_active_states()  # (n_active_tokens, seq, d_model)

        # Les réécrire dans les premiers slots du buffer
        n_active_tokens = self.n_active * self.warp_size
        self.states[:n_active_tokens] = active_states

        # Mettre à jour les indices : 0, 1, 2, ..., n_active-1
        self.active_group_idx = torch.arange(self.n_active, device=self.device)
        self._update_cu_seqlens()

        return True

    # ------------------------------------------------------------------
    # cu_seqlens pour FlashAttention varlen
    # ------------------------------------------------------------------

    def _update_cu_seqlens(self) -> None:
        """
        Recalcule cu_seqlens après chaque update.
        cu_seqlens[i] = i * seq_len (tous les samples ont la même longueur de séquence)
        Shape : (n_active + 1,)
        """
        n = self.n_active
        self.cu_seqlens = torch.arange(
            0, (n + 1) * self.seq_len, self.seq_len,
            dtype=torch.int32,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _group_idx_to_token_idx(self, group_idx: Tensor) -> Tensor:
        """
        Convertit des indices de groupes en indices de tokens dans le buffer global.
        groupe g → tokens [g * warp_size, (g+1) * warp_size)
        """
        # group_idx : (n,) → token_idx : (n * warp_size,)
        base = group_idx.unsqueeze(1) * self.warp_size  # (n, 1)
        offsets = torch.arange(self.warp_size, device=self.device).unsqueeze(0)  # (1, warp_size)
        return (base + offsets).reshape(-1)  # (n * warp_size,)

    def get_stop_times(self) -> Tensor:
        """
        Retourne les stop times par token (pas par groupe) pour le calcul de L_halt.
        Shape : (batch_size,)
        T_i = step d'arrêt du groupe du token i.
        """
        # Étendre les stop times de groupe vers les tokens
        group_times = self.group_stop_times  # (n_groups,)
        return (
            group_times.unsqueeze(1)
                       .expand(-1, self.warp_size)
                       .reshape(self.n_groups * self.warp_size)
        )

    def summary(self) -> dict:
        """Stats courantes du buffer — pour le monitoring d'entraînement."""
        return {
            "step":           self.current_step,
            "n_active":       self.n_active,
            "n_stopped":      self.n_groups - self.n_active,
            "active_ratio":   round(self.n_active / self.n_groups, 3) if self.n_groups > 0 else 0,
            "mean_stop_time": round(self.group_stop_times.float().mean().item(), 2),
        }
