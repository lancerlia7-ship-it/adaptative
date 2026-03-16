"""
Tests de correction — ActiveTokenBuffer
scheduler/tests/test_buffer.py

Ce qu'on teste :
  1. Initialisation correcte (dimensions, cu_seqlens)
  2. load() charge les états sans copie parasite
  3. update() partitionne correctement avec la règle min
  4. get_active_states() retourne les bons tokens
  5. write_active_states() écrit au bon endroit dans le buffer global
  6. get_stop_times() retourne les T_i au niveau token
  7. cu_seqlens est correct après chaque update
  8. defrag_if_needed() maintient la cohérence

Run : pytest scheduler/tests/test_buffer.py -v
"""

import pytest
import torch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scheduler.buffer import ActiveTokenBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH     = 128
SEQ       = 16
D_MODEL   = 64
T_MAX     = 8
WARP_SIZE = 32
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE     = torch.float32  # float32 pour les tests (comparaisons exactes)


@pytest.fixture
def buf():
    return ActiveTokenBuffer(
        batch_size=BATCH,
        seq_len=SEQ,
        d_model=D_MODEL,
        T_max=T_MAX,
        warp_size=WARP_SIZE,
        device=DEVICE,
        dtype=DTYPE,
    )


@pytest.fixture
def x():
    return torch.randn(BATCH, SEQ, D_MODEL, device=DEVICE, dtype=DTYPE)


# ---------------------------------------------------------------------------
# 1. Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_states_shape(self, buf):
        assert buf.states.shape == (BATCH, SEQ, D_MODEL)

    def test_all_groups_active_initially(self, buf):
        assert buf.n_active == BATCH // WARP_SIZE

    def test_stop_times_initialized_to_T_max(self, buf):
        assert (buf.group_stop_times == T_MAX).all()

    def test_cu_seqlens_shape(self, buf):
        n_groups = BATCH // WARP_SIZE
        assert buf.cu_seqlens.shape == (n_groups + 1,)

    def test_cu_seqlens_values(self, buf):
        n_groups = BATCH // WARP_SIZE
        expected = torch.arange(0, (n_groups + 1) * SEQ, SEQ, dtype=torch.int32, device=DEVICE)
        assert torch.equal(buf.cu_seqlens, expected)

    def test_n_groups(self, buf):
        assert buf.n_groups == BATCH // WARP_SIZE


# ---------------------------------------------------------------------------
# 2. load()
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_copies_states(self, buf, x):
        buf.load(x)
        assert torch.allclose(buf.states[:BATCH], x)

    def test_load_resets_active_idx(self, buf, x):
        # Simuler un update partiel, puis reload
        halt_probs = torch.ones(buf.n_active_tokens, device=DEVICE) * 0.9
        buf.update(halt_probs, step=1)
        assert buf.n_active < BATCH // WARP_SIZE

        buf.load(x)
        assert buf.n_active == BATCH // WARP_SIZE

    def test_load_resets_stop_times(self, buf, x):
        halt_probs = torch.ones(buf.n_active_tokens, device=DEVICE) * 0.9
        buf.update(halt_probs, step=2)
        buf.load(x)
        assert (buf.group_stop_times == T_MAX).all()

    def test_load_wrong_shape_raises(self, buf):
        x_bad = torch.randn(BATCH + 1, SEQ, D_MODEL, device=DEVICE)
        with pytest.raises(AssertionError):
            buf.load(x_bad)


# ---------------------------------------------------------------------------
# 3. update() — règle min warp-aligned
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_all_stop_when_all_probs_high(self, buf, x):
        buf.load(x)
        # Tous les halt_probs > 0.5 → tous les groupes s'arrêtent
        halt_probs = torch.ones(buf.n_active_tokens, device=DEVICE) * 0.9
        buf.update(halt_probs, step=1)
        assert buf.n_active == 0

    def test_none_stop_when_all_probs_low(self, buf, x):
        buf.load(x)
        # Tous les halt_probs < 0.5 → aucun groupe ne s'arrête
        halt_probs = torch.zeros(buf.n_active_tokens, device=DEVICE) * 0.1
        buf.update(halt_probs, step=1)
        assert buf.n_active == BATCH // WARP_SIZE

    def test_rule_min_one_low_keeps_group(self, buf, x):
        """
        Règle min : si un seul token d'un groupe a halt_prob <= 0.5,
        le groupe entier continue.
        """
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.ones(n_active * WARP_SIZE, device=DEVICE) * 0.9
        # Mettre un seul token du premier groupe à 0.1
        halt_probs[0] = 0.1
        buf.update(halt_probs, step=1)
        # Le premier groupe doit rester actif
        assert buf.n_active == n_active  # aucun groupe stoppé

    def test_rule_min_all_high_stops_group(self, buf, x):
        """
        Tous les tokens d'un groupe > 0.5 → groupe stoppé.
        """
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        # Premier groupe : tous à 0.9
        halt_probs[:WARP_SIZE] = 0.9
        buf.update(halt_probs, step=1)
        assert buf.n_active == n_active - 1

    def test_stop_times_recorded_correctly(self, buf, x):
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9  # Premier groupe s'arrête au step 3
        buf.update(halt_probs, step=3)
        # Le stop time du groupe 0 (global idx 0) doit être 3
        assert buf.group_stop_times[0].item() == 3

    def test_stop_times_not_overwritten(self, buf, x):
        """Un groupe qui s'est arrêté à t=2 ne doit pas avoir son stop time modifié à t=5."""
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9
        buf.update(halt_probs, step=2)

        # Deuxième update — le premier groupe n'est plus actif, son stop time reste 2
        if buf.n_active > 0:
            halt_probs2 = torch.zeros(buf.n_active_tokens, device=DEVICE)
            buf.update(halt_probs2, step=5)

        assert buf.group_stop_times[0].item() == 2


# ---------------------------------------------------------------------------
# 4. get_active_states()
# ---------------------------------------------------------------------------

class TestGetActiveStates:
    def test_shape_all_active(self, buf, x):
        buf.load(x)
        h = buf.get_active_states()
        assert h.shape == (BATCH, SEQ, D_MODEL)

    def test_shape_after_partial_stop(self, buf, x):
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9  # Un groupe stoppé
        buf.update(halt_probs, step=1)

        h = buf.get_active_states()
        expected_tokens = (n_active - 1) * WARP_SIZE
        assert h.shape == (expected_tokens, SEQ, D_MODEL)

    def test_values_match_buffer(self, buf, x):
        buf.load(x)
        h = buf.get_active_states()
        # Tous les groupes actifs → doit correspondre aux états initiaux
        assert torch.allclose(h, x[:buf.n_active * WARP_SIZE])

    def test_is_contiguous(self, buf, x):
        buf.load(x)
        h = buf.get_active_states()
        assert h.is_contiguous()


# ---------------------------------------------------------------------------
# 5. write_active_states()
# ---------------------------------------------------------------------------

class TestWriteActiveStates:
    def test_write_updates_buffer(self, buf, x):
        buf.load(x)
        h_new = torch.ones(buf.n_active_tokens, SEQ, D_MODEL, device=DEVICE, dtype=DTYPE)
        buf.write_active_states(h_new)
        h_read = buf.get_active_states()
        assert torch.allclose(h_read, h_new)

    def test_write_only_updates_active(self, buf, x):
        buf.load(x)
        n_active = buf.n_active

        # Stopper le premier groupe
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9
        buf.update(halt_probs, step=1)

        # Sauvegarder l'état du groupe stoppé
        stopped_state_before = buf.states[:WARP_SIZE].clone()

        # Écrire de nouveaux états pour les groupes actifs
        h_new = torch.full(
            (buf.n_active_tokens, SEQ, D_MODEL), 99.0, device=DEVICE, dtype=DTYPE
        )
        buf.write_active_states(h_new)

        # Le groupe stoppé ne doit pas avoir changé
        assert torch.allclose(buf.states[:WARP_SIZE], stopped_state_before)


# ---------------------------------------------------------------------------
# 6. get_stop_times()
# ---------------------------------------------------------------------------

class TestGetStopTimes:
    def test_shape(self, buf, x):
        buf.load(x)
        times = buf.get_stop_times()
        assert times.shape == (buf.n_groups * WARP_SIZE,)

    def test_all_T_max_initially(self, buf, x):
        buf.load(x)
        times = buf.get_stop_times()
        assert (times == T_MAX).all()

    def test_group_stop_propagated_to_tokens(self, buf, x):
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9
        buf.update(halt_probs, step=3)

        times = buf.get_stop_times()
        # Les WARP_SIZE premiers tokens (groupe 0) ont stop_time = 3
        assert (times[:WARP_SIZE] == 3).all()
        # Les autres ont stop_time = T_MAX
        assert (times[WARP_SIZE:] == T_MAX).all()


# ---------------------------------------------------------------------------
# 7. cu_seqlens
# ---------------------------------------------------------------------------

class TestCuSeqlens:
    def test_cu_seqlens_after_update(self, buf, x):
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9  # Premier groupe stoppé
        buf.update(halt_probs, step=1)

        n_remaining = n_active - 1
        assert buf.cu_seqlens.shape == (n_remaining + 1,)
        expected = torch.arange(0, (n_remaining + 1) * SEQ, SEQ, dtype=torch.int32, device=DEVICE)
        assert torch.equal(buf.cu_seqlens, expected)

    def test_cu_seqlens_empty(self, buf, x):
        buf.load(x)
        halt_probs = torch.ones(buf.n_active_tokens, device=DEVICE) * 0.9
        buf.update(halt_probs, step=1)
        assert buf.n_active == 0
        assert torch.equal(buf.cu_seqlens, torch.tensor([0], dtype=torch.int32, device=DEVICE))


# ---------------------------------------------------------------------------
# 8. defrag_if_needed()
# ---------------------------------------------------------------------------

class TestDefrag:
    def test_defrag_maintains_state_values(self, buf, x):
        buf.load(x)
        n_active = buf.n_active

        # Stopper le premier groupe
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9
        buf.update(halt_probs, step=1)

        # États actifs avant défrag
        states_before = buf.get_active_states().clone()

        buf.defrag_if_needed(step=4, freq=4)

        # États actifs après défrag — doivent être identiques
        states_after = buf.get_active_states()
        assert torch.allclose(states_before, states_after)

    def test_defrag_makes_indices_contiguous(self, buf, x):
        buf.load(x)
        n_active = buf.n_active
        halt_probs = torch.zeros(n_active * WARP_SIZE, device=DEVICE)
        halt_probs[:WARP_SIZE] = 0.9
        buf.update(halt_probs, step=1)

        buf.defrag_if_needed(step=4, freq=4)

        # Après défrag, les indices doivent être 0, 1, 2, ...
        expected = torch.arange(buf.n_active, device=DEVICE)
        assert torch.equal(buf.active_group_idx, expected)

    def test_defrag_skipped_when_freq_not_reached(self, buf, x):
        buf.load(x)
        result = buf.defrag_if_needed(step=3, freq=4)
        assert result is False

    def test_defrag_called_at_freq(self, buf, x):
        buf.load(x)
        result = buf.defrag_if_needed(step=4, freq=4)
        assert result is True


# ---------------------------------------------------------------------------
# 9. summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_summary_keys(self, buf, x):
        buf.load(x)
        s = buf.summary()
        for key in ["step", "n_active", "n_stopped", "active_ratio", "mean_stop_time"]:
            assert key in s

    def test_summary_active_ratio_initially_one(self, buf, x):
        buf.load(x)
        assert buf.summary()["active_ratio"] == 1.0
