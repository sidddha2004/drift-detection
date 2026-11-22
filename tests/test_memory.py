"""A6 deployable memory: candidate scoring, undo / restore / reject decisions and the cache bound."""
import numpy as np
import torch

from src.fl.adapter import AdaptedLSTM
from src.fl.engine import DEVICE, TrainConfig, make_model, predict
from src.fl.streaming import Policy, _cache, _Client, _memory_check

N_CLS, W_LEN = 4, 16


def _state(seed=0):
    torch.manual_seed(seed)
    m = AdaptedLSTM(make_model(N_CLS, 9, TrainConfig()).eval()).eval()
    st = _Client(1, None, None, [], m, None, None)
    st.init_state = m.adapter_state()
    st.err_hist = [0.0] * 20
    x = torch.randn(60, W_LEN, 9, device=DEVICE)
    return st, x


def _perturbed(st, scale):
    s = {k: v.clone() for k, v in st.init_state.items()}
    s["in_w"] = s["in_w"] * scale
    return s


def _labels_of(st, state, x):
    st.model.load_adapter_state(state)
    return torch.as_tensor(predict(st.model, x), device=DEVICE)


def _same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_undo_is_chosen_when_the_original_model_fits_again():
    st, x = _state()
    y = _labels_of(st, st.init_state, x)                 # data on which the original model is perfect
    bad = _perturbed(st, -3.0)
    st.model.load_adapter_state(bad); st.adapted, st.episode_left = True, 5
    assert _memory_check(st, x, y, k=2, tol=0.05)
    assert _same(st.model.adapter_state(), st.init_state) and st.episode_left == 0 and not st.adapted
    assert len(st.library) == 1 and _same(st.library[0], bad)            # the abandoned state is cached


def test_cached_state_is_restored_on_recurrence_and_leaves_the_cache():
    st, x = _state()
    good = _perturbed(st, -3.0)
    y = _labels_of(st, good, x)
    st.library = [good]
    st.model.load_adapter_state(_perturbed(st, 0.2)); st.adapted = True
    assert _memory_check(st, x, y, k=2, tol=0.05)
    assert _same(st.model.adapter_state(), good) and st.adapted
    assert all(not _same(e, good) for e in st.library) and len(st.library) == 1


def test_no_switch_when_no_candidate_is_good_enough():
    st, x = _state()
    y = (_labels_of(st, st.init_state, x) + 1) % N_CLS      # every candidate is wrong everywhere
    cur = _perturbed(st, 0.5)
    st.model.load_adapter_state(cur); st.adapted, st.episode_left = True, 3
    assert not _memory_check(st, x, y, k=2, tol=0.05)
    assert _same(st.model.adapter_state(), cur) and st.episode_left == 3 and st.library == []


def test_k0_ignores_the_cache():
    st, x = _state()
    good = _perturbed(st, -3.0)
    y = _labels_of(st, good, x)
    st.library = [good]
    st.model.load_adapter_state(_perturbed(st, 0.2))
    if _memory_check(st, x, y, k=0, tol=0.05):
        assert not _same(st.model.adapter_state(), good)


def test_cache_keeps_the_most_recent_k():
    st, _ = _state()
    for i in range(5):
        _cache(st, {"i": torch.tensor(i)}, 2)
    assert [int(e["i"]) for e in st.library] == [3, 4]
    _cache(st, {"i": torch.tensor(9)}, 0)
    assert st.library == []


def test_memory_policy_runs_end_to_end():
    from tests.test_streaming import _client, _events
    from src.fl.streaming import Response, run_stream
    clients = {c: _client(c, c) for c in (1, 2, 3)}
    torch.manual_seed(0)
    models = {c: make_model(6, 9, TrainConfig()).eval() for c in clients}
    pol = Policy("m", "detector", Response("a", "align", False, rounds=2), detector=(0.0, 1.0), memory=1)
    res = run_stream(clients, models, _events(), 30, 5, 6, pol, TrainConfig())
    assert "memory_hits" in res and res["up_bytes"].sum() == 0
    assert all(n <= 1 for n in res["library_sizes"].values())
