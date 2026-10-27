"""Structural tests of the streaming engine on tiny synthetic clients (untrained models: only mechanics are checked)."""
import copy

import numpy as np
import pytest
import torch

from src.data import ClientData
from src.drift.schedule import DriftEvent
from src.eval.streaming import stream_metrics
from src.fl.adapter import AdaptedLSTM
from src.fl.engine import DEVICE, TrainConfig, make_model, state_bytes
from src.fl.streaming import Policy, Response, run_stream

N_CLS, W_LEN, ROUNDS = 6, 16, 30
TC = TrainConfig()


def _client(cid, seed):
    r = np.random.default_rng(seed)
    mean = r.normal(0, 2, 9).astype(np.float32)
    std = r.uniform(1, 3, 9).astype(np.float32)

    def split(per_class):
        y = np.repeat(np.arange(N_CLS), per_class).astype(np.int64)
        x = (r.normal(0, 1, (len(y), W_LEN, 9)) + y[:, None, None] * 0.5).astype(np.float32)
        return x, y

    x = {},
    xs, ys = {}, {}
    for s, k in (("train", 20), ("val", 32), ("test", 8)):
        xs[s], ys[s] = split(k)
    return ClientData(client_id=cid, x=xs, y=ys, mean=mean, std=std)


@pytest.fixture(scope="module")
def world():
    clients = {c: _client(c, c) for c in (1, 2, 3)}
    torch.manual_seed(0)
    models = {c: make_model(N_CLS, 9, TC).eval() for c in clients}
    return clients, models


def _events(drifted=(1,), start=10, transform="offset", mode="virtual"):
    ev = {1: [], 2: [], 3: []}
    for c in drifted:
        ev[c] = [DriftEvent(c, 0, "sudden", mode, transform, tuple(range(9)) if mode == "virtual" else (),
                            1.0 if mode == "virtual" else 2.0, start, remap=((0, 1),) if mode == "real" else ())]
    return ev


def _run(world, policy, events=None):
    clients, models = world
    return run_stream(clients, models, events or _events(), ROUNDS, 5, N_CLS, policy, TC)


def test_frozen_policy_without_drift_gives_constant_scores_and_no_bytes(world):
    res = _run(world, Policy("frozen", "none"), _events(drifted=()))
    np.testing.assert_allclose(res["f1"], res["f1"][:, :1].repeat(ROUNDS, 1))
    assert res["up_bytes"].sum() == 0 and res["alarms"] == []


def test_oracle_trigger_fires_only_for_drifted_clients_at_onset_plus_delay(world):
    res = _run(world, Policy("o", "oracle", Response("a", "align", False, rounds=2), oracle_delay=3))
    assert res["alarms"] == [(1, 13)]


def test_unaffected_clients_keep_their_model_exactly(world):
    clients, models = world
    res = _run(world, Policy("o", "oracle", Response("full_local", "full", False, rounds=2, epochs=1), oracle_delay=2))
    # clients 2 and 3 never trigger: their scores equal the frozen scores
    frozen = _run(world, Policy("frozen", "none"))
    np.testing.assert_allclose(res["f1"][1:], frozen["f1"][1:])
    assert not np.allclose(res["f1"][0, 12:], frozen["f1"][0, 12:])      # the drifted client did change


def test_local_responses_cost_zero_bytes(world):
    for resp in (Response("l", "full", False, rounds=2, epochs=1), Response("a", "align", False, rounds=2),
                 Response("al", "adapter", False, rounds=2, epochs=1)):
        res = _run(world, Policy("o", "oracle", resp))
        assert res["up_bytes"].sum() == 0 and res["down_bytes"].sum() == 0


def test_federated_full_bytes_equal_participants_times_model_size(world):
    clients, models = world
    res = _run(world, Policy("o", "oracle", Response("f", "full", True, rounds=2, epochs=1), oracle_delay=2),
               _events(drifted=(1, 2)))
    full = state_bytes(models[1])
    assert res["up_bytes"][12] == 2 * full and res["up_bytes"][13] == 2 * full and res["up_bytes"].sum() == 4 * full
    np.testing.assert_array_equal(res["up_bytes"], res["down_bytes"])


def test_federated_adapter_bytes_are_a_small_fraction_of_the_model(world):
    clients, models = world
    res = _run(world, Policy("o", "oracle", Response("f", "adapter", True, rounds=2, epochs=1), oracle_delay=2),
               _events(drifted=(1, 2)))
    ad = AdaptedLSTM(models[1]).adapter_bytes()
    assert res["up_bytes"][12] == 2 * ad
    assert ad < 0.05 * state_bytes(models[1])
    assert res["adapter_bytes"] == ad


def test_periodic_trigger_touches_every_client_at_the_period(world):
    clients, models = world
    res = _run(world, Policy("p", "periodic", Response("f", "full", True, rounds=1, epochs=1), period=10), _events(drifted=()))
    nz = np.flatnonzero(res["up_bytes"])
    assert list(nz) == [10, 20] and res["up_bytes"][10] == 3 * state_bytes(models[1])


def test_run_is_deterministic_for_a_seed(world):
    pol = Policy("o", "oracle", Response("al", "adapter", False, rounds=2, epochs=1))
    a, b = _run(world, pol), _run(world, pol)
    np.testing.assert_allclose(a["f1"], b["f1"])


def test_stream_metrics_recovery_and_post_change_definition():
    n = 40
    f1 = np.full((2, n), 0.9)
    f1[0, 20:30] = 0.3                      # client 0 drifts at round 20, recovers at round 30
    res = {"policy": "x", "f1": f1, "alpha": np.vstack([np.r_[np.zeros(20), np.ones(20)], np.zeros(n)]), "clients": [1, 2],
           "onsets": {1: [20], 2: []}, "up_bytes": np.zeros(n), "down_bytes": np.zeros(n), "alarms": [], "seconds": 1.0}
    m = stream_metrics(res)
    assert m["post_change_f1"] == pytest.approx((0.3 * 10 + 0.9 * 10) / 20)
    assert m["pre_drift_f1"] == pytest.approx(0.9) and m["recovery_rounds"] == 10
    assert m["max_drop"] == pytest.approx(0.6) and m["clean_clients_post_f1"] == pytest.approx(0.9)


# ---- typed-response machinery ---------------------------------------------------------------------------------------
def _recurring(c=1, start=6, period=12, duration=6):
    return {1: [DriftEvent(c, 0, "recurring", "virtual", "offset", tuple(range(9)), 1.5, start, period=period, duration=duration)], 2: [], 3: []}


def _typed(info):
    if info["event"] == "offset":
        return Response("rev", "revert", False)
    return Response("al", "align", False, rounds=2)


def test_revert_returns_the_input_map_to_identity_and_caches_the_fit(world):
    pol = Policy("t", "oracle", respond_by_type=_typed, oracle_delay=2, oracle_offsets=True)
    res = _run(world, pol, _recurring())
    W, b = res["in_maps"][1]
    np.testing.assert_array_equal(W, np.eye(9))
    np.testing.assert_array_equal(b, np.zeros(9))
    assert res["library_sizes"][1] >= 1
    assert (1, 6 + 2) in res["alarms"] and (1, 12 + 2) in res["alarms"]       # onset and offset events


def test_restore_reloads_the_cached_map_without_a_new_fit(world):
    def typed(info):
        if info["event"] == "offset":
            return Response("rev", "revert", False)
        return Response("res" if info["occurrence"] else "al", "restore" if info["occurrence"] else "align", False, rounds=2)
    pol = Policy("t", "oracle", respond_by_type=typed, oracle_delay=2, oracle_offsets=True)
    res = _run(world, pol, _recurring(start=4, period=12, duration=4))        # onsets 4, 16; offsets 8, 20
    W, b = res["in_maps"][1]
    # last event is an offset at round 22 -> identity again; the restore at round 18 happened in between
    np.testing.assert_array_equal(W, np.eye(9))
    assert res["library_sizes"][1] == 1                                        # the restore did not create a new entry


def test_imus_restriction_keeps_other_sensors_at_identity(world):
    clients, models = world
    pol = Policy("t", "oracle", Response("al", "align", False, rounds=2, imus=(1,)), oracle_delay=2)
    res = _run(world, pol, _events(drifted=(1,)))
    W, b = res["in_maps"][1]
    keep = np.zeros(9, dtype=bool); keep[3:6] = True
    np.testing.assert_array_equal(W[np.ix_(~keep, ~keep)], np.eye(6))
    assert np.all(b[~keep] == 0)


def test_tracking_mode_runs_for_the_whole_episode(world):
    pol = Policy("t", "oracle", Response("al", "align", False, rounds=10, track=True, buffer_rounds=6), oracle_delay=2)
    res = _run(world, pol, _events(drifted=(1,)))
    assert res["adapt_rounds"] == 10 and res["up_bytes"].sum() == 0


def test_adwin_online_detector_fires_on_an_error_increase_only_and_resets():
    from src.fl.streaming import _ADWINOnline
    d = _ADWINOnline(0.3)
    rng = np.random.default_rng(0)
    low = [d.update(float(rng.random() < 0.03)) for _ in range(80)]
    high = [d.update(float(rng.random() < 0.5)) for _ in range(80)]
    assert not any(low) and any(high)
    d.reset()
    assert not any(d.update(0.0) for _ in range(20))


def test_policy_can_use_adwin_as_trigger(world):
    pol = Policy("e4", "detector", Response("f", "full", True, rounds=1, epochs=1), detector=(0.3, 0.0), detector_kind="adwin")
    res = _run(world, pol, _events(drifted=()))
    assert res["f1"].shape == (3, ROUNDS)
