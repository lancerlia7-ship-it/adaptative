"""
WarpAlignedScheduler — scheduler/scheduler.py

Opération non-différentiable qui partitionne les tokens entre actifs et stoppés.
Le gradient la contourne via le pattern straight-through.

Règle de partition (warp-aligned) :
  Un groupe de warp_size tokens est stoppé ssi TOUS ses tokens ont halt_prob > 0.5.
  Règle min — pas vote majoritaire.
  Raison : le GPU alloue de toute façon 32 threads par warp. Si un seul token
  d'un groupe continue, tout le groupe doit continuer pour éviter la divergence.

Gradient straight-through corrigé :
  - Le gradient est annulé pour TOUS les tokens d'un groupe stoppé,
    y compris ceux qui voulaient continuer (halt_prob <= 0.5 mais groupe stoppé
    parce que ses voisins ont tous halt_prob > 0.5 — cas rare mais possible).
  - Pas de gradient vers halt_probs : la halt head est entraînée uniquement
    via λ_t * L_halt + λ_w * L_warp dans la loss, pas via le scheduler.

Signal contradictoire à éviter :
  Sans cette correction, un token qui voulait s'arrêter à t=3 mais dont le groupe
  a continué jusqu'à t=7 reçoit T_i = 3 dans la loss — mais le scheduler l'a
  quand même forcé à continuer. La halt head reçoit un signal incohérent.
  La correction : T_i de chaque token = T_i de son groupe (voir model/loss.py).
"""

import torch
import torch.nn as nn
from torch import Tensor

from scheduler.buffer import ActiveTokenBuffer


class WarpAlignedScheduler(torch.autograd.Function):
    """
    Opération de partition warp-aligned.
    Utilisée comme fonction custom dans le forward pass du Universal Transformer.

    Usage dans la boucle :
        h_new, token_stopped = WarpAlignedScheduler.apply(h_new, halt_probs, warp_size)
        buf.update(token_stopped, step=t)
    """

    @staticmethod
    def forward(
        ctx,
        h: Tensor,
        halt_probs: Tensor,
        warp_size: int = 32,
    ) -> tuple[Tensor, Tensor]:
        """
        h          : (batch_size, seq_len, d_model) — états courants
        halt_probs : (batch_size,) — probabilité d'arrêt par token (sortie de HaltHead)
        warp_size  : granularité de la partition

        Retourne :
          h           : (batch_size, seq_len, d_model) — inchangé (straight-through)
          token_stopped : (batch_size,) bool — True = token dans un groupe stoppé
        """
        n = halt_probs.shape[0]
        n_warps = n // warp_size
        n_aligned = n_warps * warp_size

        # Reshape en groupes
        probs_grouped = halt_probs[:n_aligned].view(n_warps, warp_size)

        # Règle min : groupe stoppé ssi tous > 0.5
        group_stopped = probs_grouped.min(dim=1).values > 0.5  # (n_warps,)

        # Expansion au niveau token
        token_stopped_aligned = (
            group_stopped.unsqueeze(1)
                         .expand(-1, warp_size)
                         .reshape(n_aligned)
        )

        # Tokens résiduels (si batch_size % warp_size != 0) → toujours actifs
        if n > n_aligned:
            remainder = torch.zeros(n - n_aligned, dtype=torch.bool, device=h.device)
            token_stopped = torch.cat([token_stopped_aligned, remainder])
        else:
            token_stopped = token_stopped_aligned

        ctx.save_for_backward(token_stopped)
        ctx.warp_size = warp_size

        return h, token_stopped

    @staticmethod
    def backward(ctx, grad_h: Tensor, _grad_stopped) -> tuple[Tensor, None, None]:
        """
        Straight-through : le gradient passe directement à travers la partition.
        Annulé pour les tokens dans des groupes stoppés.

        grad_h : gradient de la loss par rapport à h
        _grad_stopped : ignoré (token_stopped n'est pas différentiable)

        Retourne les gradients pour chaque argument de forward :
          (grad_h, grad_halt_probs, grad_warp_size)
          grad_halt_probs = None : la halt head est entraînée via la loss directement
        """
        token_stopped, = ctx.saved_tensors
        grad_h = grad_h.clone()

        # Annuler le gradient pour les tokens stoppés
        # token_stopped : (batch_size,) → broadcaster sur (batch_size, seq_len, d_model)
        stopped_mask = token_stopped.view(-1, 1, 1).expand_as(grad_h)
        grad_h[stopped_mask] = 0.0

        return grad_h, None, None


class SchedulerStep(nn.Module):
    """
    Wrapper nn.Module autour de WarpAlignedScheduler pour une utilisation
    plus naturelle dans le forward pass du Universal Transformer.

    Gère aussi la mise à jour du buffer.
    """

    def __init__(self, warp_size: int = 32):
        super().__init__()
        self.warp_size = warp_size

    def forward(
        self,
        h: Tensor,
        halt_probs: Tensor,
        buf: ActiveTokenBuffer,
        step: int,
    ) -> tuple[Tensor, Tensor, int]:
        """
        h          : (batch_size, seq_len, d_model)
        halt_probs : (batch_size,)
        buf        : buffer à mettre à jour
        step       : étape courante (pour les stop times)

        Retourne :
          h              : états (inchangés, gradient straight-through)
          token_stopped  : masque des tokens stoppés
          n_newly_stopped : nombre de groupes stoppés à cette étape
        """
        h, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, self.warp_size)

        # Mettre à jour le buffer avec les halt_probs
        group_stopped = buf.update(halt_probs, step)
        n_newly_stopped = group_stopped.sum().item()

        return h, token_stopped, n_newly_stopped


def compute_group_stop_times(
    halt_probs_history: list[Tensor],
    warp_size: int,
    T_max: int,
) -> Tensor:
    """
    Calcule les stop times par groupe à partir de l'historique des halt_probs.
    Utile pour déboguer et pour vérifier que les stop times dans le buffer sont corrects.

    halt_probs_history : liste de tenseurs (batch_size,), un par step
    Retourne : (n_groups,) — step d'arrêt par groupe (T_max si jamais arrêté)
    """
    batch_size = halt_probs_history[0].shape[0]
    n_groups   = batch_size // warp_size
    stop_times = torch.full((n_groups,), T_max, dtype=torch.long)

    for t, probs in enumerate(halt_probs_history):
        probs_grouped = probs[:n_groups * warp_size].view(n_groups, warp_size)
        group_stopped = probs_grouped.min(dim=1).values > 0.5

        # Premier step où le groupe s'arrête
        newly_stopped = group_stopped & (stop_times == T_max)
        stop_times[newly_stopped] = t

    return stop_times
