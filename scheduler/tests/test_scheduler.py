"""
Tests de correction — WarpAlignedScheduler
scheduler/tests/test_scheduler.py

Ce qu'on teste :
  1. forward() : partition correcte avec la règle min
  2. forward() : token_stopped cohérent avec halt_probs
  3. backward() : gradient annulé pour les tokens stoppés
  4. backward() : gradient intact pour les tokens actifs
  5. backward() : pas de gradient vers halt_probs (None)
  6. Tokens résiduels (batch_size % warp_size != 0)
  7. compute_group_stop_times() : calcul correct des stop times

Run : pytest scheduler/tests/test_scheduler.py -v
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scheduler.scheduler import WarpAlignedScheduler, SchedulerStep, compute_group_stop_times


WARP_SIZE = 32
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32


# ---------------------------------------------------------------------------
# 1. forward() — partition correcte
# ---------------------------------------------------------------------------

class TestForward:
    def test_all_stop(self):
        batch = 64
        h = torch.randn(batch, 8, 32, device=DEVICE, dtype=DTYPE)
        halt_probs = torch.ones(batch, device=DEVICE) * 0.9
        _, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        assert token_stopped.all()

    def test_none_stop(self):
        batch = 64
        h = torch.randn(batch, 8, 32, device=DEVICE, dtype=DTYPE)
        halt_probs = torch.zeros(batch, device=DEVICE) + 0.1
        _, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        assert not token_stopped.any()

    def test_rule_min_one_low_keeps_group(self):
        """Un seul token < 0.5 dans un groupe → tout le groupe continue."""
        batch = 64
        h = torch.randn(batch, 8, 32, device=DEVICE, dtype=DTYPE)
        halt_probs = torch.ones(batch, device=DEVICE) * 0.9
        halt_probs[0] = 0.1  # Un seul token du premier groupe à 0.1
        _, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        # Le premier groupe (tokens 0..31) doit être entièrement actif
        assert not token_stopped[:WARP_SIZE].any()
        # Le deuxième groupe (tokens 32..63) doit être entièrement stoppé
        assert token_stopped[WARP_SIZE:].all()

    def test_token_stopped_aligned_to_warp(self):
        """token_stopped doit être uniforme par groupe (tous True ou tous False)."""
        batch = 128
        h = torch.randn(batch, 8, 32, device=DEVICE, dtype=DTYPE)
        halt_probs = torch.rand(batch, device=DEVICE)
        _, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)

        n_warps = batch // WARP_SIZE
        grouped = token_stopped[:n_warps * WARP_SIZE].view(n_warps, WARP_SIZE)
        # Chaque ligne doit être uniforme (tous True ou tous False)
        for i in range(n_warps):
            row = grouped[i]
            assert row.all() or not row.any(), \
                f"Groupe {i} n'est pas uniforme : {row}"

    def test_h_unchanged(self):
        """forward() ne doit pas modifier h."""
        batch = 64
        h = torch.randn(batch, 8, 32, device=DEVICE, dtype=DTYPE)
        h_original = h.clone()
        halt_probs = torch.ones(batch, device=DEVICE) * 0.9
        h_out, _ = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        assert torch.allclose(h_out, h_original)

    def test_output_shape(self):
        batch, seq, d = 128, 16, 64
        h = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE)
        halt_probs = torch.rand(batch, device=DEVICE)
        h_out, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        assert h_out.shape == (batch, seq, d)
        assert token_stopped.shape == (batch,)


# ---------------------------------------------------------------------------
# 2. Tokens résiduels
# ---------------------------------------------------------------------------

class TestResidualTokens:
    def test_residual_tokens_always_active(self):
        """
        Si batch_size % warp_size != 0, les tokens résiduels doivent être actifs.
        """
        batch = 70  # 70 % 32 = 6 tokens résiduels
        h = torch.randn(batch, 8, 32, device=DEVICE, dtype=DTYPE)
        halt_probs = torch.ones(batch, device=DEVICE) * 0.9
        _, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)

        n_aligned = (batch // WARP_SIZE) * WARP_SIZE
        # Les tokens alignés (0..63) doivent être stoppés
        assert token_stopped[:n_aligned].all()
        # Les tokens résiduels (64..69) doivent être actifs
        assert not token_stopped[n_aligned:].any()


# ---------------------------------------------------------------------------
# 3-5. backward() — gradient straight-through
# ---------------------------------------------------------------------------

class TestBackward:
    def test_gradient_zero_for_stopped_tokens(self):
        """
        Le gradient doit être annulé pour les tokens dans des groupes stoppés.
        """
        batch, seq, d = 64, 8, 32
        h = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE, requires_grad=True)
        halt_probs = torch.ones(batch, device=DEVICE) * 0.9  # tout s'arrête

        h_out, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)

        # Loss quelconque sur h_out
        loss = h_out.sum()
        loss.backward()

        # Tous les tokens sont stoppés → gradient doit être zéro partout
        assert h.grad is not None
        assert torch.allclose(h.grad, torch.zeros_like(h.grad))

    def test_gradient_intact_for_active_tokens(self):
        """
        Le gradient doit être intact pour les tokens dans des groupes actifs.
        """
        batch, seq, d = 64, 8, 32
        h = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE, requires_grad=True)
        halt_probs = torch.zeros(batch, device=DEVICE) + 0.1  # rien ne s'arrête

        h_out, token_stopped = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        loss = h_out.sum()
        loss.backward()

        # Aucun token stoppé → gradient doit être ones (d/d(sum) = 1)
        assert h.grad is not None
        assert torch.allclose(h.grad, torch.ones_like(h.grad))

    def test_gradient_mixed(self):
        """
        Premier groupe stoppé, deuxième actif.
        Gradient = 0 pour le premier groupe, 1 pour le second.
        """
        batch, seq, d = 64, 8, 32
        h = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE, requires_grad=True)
        halt_probs = torch.zeros(batch, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9  # Premier groupe stoppé

        h_out, _ = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        loss = h_out.sum()
        loss.backward()

        # Premier groupe : gradient = 0
        assert torch.allclose(h.grad[:WARP_SIZE], torch.zeros(WARP_SIZE, seq, d, device=DEVICE))
        # Deuxième groupe : gradient = 1
        assert torch.allclose(h.grad[WARP_SIZE:], torch.ones(WARP_SIZE, seq, d, device=DEVICE))

    def test_no_gradient_to_halt_probs(self):
        """
        halt_probs ne doit pas recevoir de gradient depuis le scheduler.
        La halt head est entraînée uniquement via L_halt dans la loss.
        """
        batch, seq, d = 64, 8, 32
        h = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE, requires_grad=True)
        halt_probs = torch.rand(batch, device=DEVICE, requires_grad=True)

        h_out, _ = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        loss = h_out.sum()
        loss.backward()

        # halt_probs.grad doit être None (pas de gradient vers halt_probs)
        assert halt_probs.grad is None

    def test_gradient_warp_aligned_not_token_aligned(self):
        """
        Vérification que le masquage se fait au niveau du groupe, pas du token.
        Si le groupe est stoppé, TOUS ses tokens ont gradient = 0,
        même ceux qui avaient halt_prob <= 0.5.
        """
        batch, seq, d = 64, 8, 32
        h = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE, requires_grad=True)
        halt_probs = torch.ones(batch, device=DEVICE) * 0.9
        halt_probs[0] = 0.1  # Un token à 0.1 dans le premier groupe
        # Le premier groupe a min(halt_probs[:32]) = 0.1 → groupe actif

        h_out, _ = WarpAlignedScheduler.apply(h, halt_probs, WARP_SIZE)
        loss = h_out.sum()
        loss.backward()

        # Premier groupe : actif (à cause du token 0) → gradient = 1
        assert torch.allclose(h.grad[:WARP_SIZE], torch.ones(WARP_SIZE, seq, d, device=DEVICE))
        # Deuxième groupe : stoppé → gradient = 0
        assert torch.allclose(h.grad[WARP_SIZE:], torch.zeros(WARP_SIZE, seq, d, device=DEVICE))


# ---------------------------------------------------------------------------
# 6. compute_group_stop_times()
# ---------------------------------------------------------------------------

class TestComputeGroupStopTimes:
    def test_all_stop_at_step_0(self):
        batch = 64
        T_max = 8
        # À chaque step, tous les halt_probs > 0.5
        history = [torch.ones(batch, device=DEVICE) * 0.9 for _ in range(T_max)]
        stop_times = compute_group_stop_times(history, WARP_SIZE, T_max)
        # Tous les groupes s'arrêtent au step 0
        assert (stop_times == 0).all()

    def test_none_stop(self):
        batch = 64
        T_max = 8
        history = [torch.zeros(batch, device=DEVICE) + 0.1 for _ in range(T_max)]
        stop_times = compute_group_stop_times(history, WARP_SIZE, T_max)
        # Aucun groupe ne s'arrête → stop_time = T_max
        assert (stop_times == T_max).all()

    def test_first_group_stops_at_step_2(self):
        batch = 64
        T_max = 8
        n_warps = batch // WARP_SIZE

        history = []
        for t in range(T_max):
            probs = torch.zeros(batch, device=DEVICE)
            if t >= 2:
                probs[:WARP_SIZE] = 0.9  # Premier groupe s'arrête à partir de t=2
            history.append(probs)

        stop_times = compute_group_stop_times(history, WARP_SIZE, T_max)
        assert stop_times[0].item() == 2
        for i in range(1, n_warps):
            assert stop_times[i].item() == T_max


# ---------------------------------------------------------------------------
# 7. SchedulerStep (nn.Module wrapper)
# ---------------------------------------------------------------------------

class TestSchedulerStep:
    def test_forward_returns_correct_shapes(self):
        from scheduler.buffer import ActiveTokenBuffer

        batch, seq, d = 64, 8, 32
        T_max = 8
        buf = ActiveTokenBuffer(
            batch_size=batch, seq_len=seq, d_model=d,
            T_max=T_max, warp_size=WARP_SIZE, device=DEVICE, dtype=DTYPE
        )
        x = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE)
        buf.load(x)

        sched = SchedulerStep(warp_size=WARP_SIZE)
        h = torch.randn(batch, seq, d, device=DEVICE, dtype=DTYPE, requires_grad=True)
        halt_probs = torch.rand(batch, device=DEVICE)

        h_out, token_stopped, n_stopped = sched(h, halt_probs, buf, step=1)

        assert h_out.shape == (batch, seq, d)
        assert token_stopped.shape == (batch,)
        assert isinstance(n_stopped, (int, float))
