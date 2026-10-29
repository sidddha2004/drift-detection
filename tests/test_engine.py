import numpy as np
import pytest
import torch

from src.data import ClientData
from src.eval import aggregate_clients, classification_metrics
from src.fl.baselines import run_fedavg, run_local_only
from src.fl.engine import TrainConfig, fedavg, make_model, state_bytes


def _client(cid, rng, n=40, w=32, c=9, k=3):
    def split(m):
        y = rng.integers(0, k, m)
        x = (rng.standard_normal((m, w, c)) + y[:, None, None]).astype(np.float32)
        return x, y.astype(np.int64)
    d = {s: split(n) for s in ("train", "val", "test")}
    return ClientData(cid, {s: d[s][0] for s in d}, {s: d[s][1] for s in d},
                      np.zeros(c, np.float32), np.ones(c, np.float32))


@pytest.fixture
def tiny():
    rng = np.random.default_rng(0)
    return {1: _client(1, rng), 2: _client(2, rng)}


TC = TrainConfig(hidden_size=8, batch_size=16)


def test_fedavg_state_average_is_weighted():
    a = {"w": torch.tensor([0.0, 0.0])}
    b = {"w": torch.tensor([4.0, 8.0])}
    out = fedavg([a, b], [1, 3])
    torch.testing.assert_close(out["w"], torch.tensor([3.0, 6.0]))


def test_state_bytes_matches_parameter_count():
    assert state_bytes(make_model(18, 9, TrainConfig())) == 53_650 * 4
    assert state_bytes(make_model(12, 9, TrainConfig())) == 53_260 * 4   # the 12-class model used in Phase 1


def test_same_seed_gives_identical_results(tiny):
    a = run_fedavg(tiny, 3, 9, TC, rounds=2, local_epochs=1, seed=1)
    b = run_fedavg(tiny, 3, 9, TC, rounds=2, local_epochs=1, seed=1)
    assert a["val_curve"] == b["val_curve"]
    assert a["test"] == b["test"]


def test_different_seed_changes_results(tiny):
    a = run_local_only(tiny, 3, 9, TC, epochs=2, seed=1)
    b = run_local_only(tiny, 3, 9, TC, epochs=2, seed=2)
    assert a["val_curves"] != b["val_curves"]


def test_fedavg_communication_accounting(tiny):
    r = run_fedavg(tiny, 3, 9, TC, rounds=2, local_epochs=1, seed=1)
    p = state_bytes(make_model(3, 9, TC))
    assert r["uplink_bytes_per_round"] == p * 2
    assert r["comm_bytes_total"] == 2 * p * 2 * 2


def test_metrics_macro_f1_ignores_classes_absent_from_truth_and_pred():
    y = np.array([0, 0, 1, 1]); p = np.array([0, 0, 1, 1])
    m = classification_metrics(y, p, n_classes=5)
    assert m["macro_f1"] == 1.0 and m["accuracy"] == 1.0


def test_aggregate_clients_reports_worst():
    agg = aggregate_clients({1: {"macro_f1": 0.9, "accuracy": 0.9}, 2: {"macro_f1": 0.5, "accuracy": 0.6}})
    assert agg["worst_macro_f1"] == 0.5 and agg["mean_macro_f1"] == pytest.approx(0.7)
