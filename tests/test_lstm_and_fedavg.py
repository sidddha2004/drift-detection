import numpy as np
import torch

from src.drift_detection import ActivityLSTM, LSTMConfig
from src.federated import AggregationConfig, FedAvgAggregator


def test_forward_shape():
    cfg = LSTMConfig(output_dim=12)
    out = ActivityLSTM(cfg)(torch.randn(4, 256, 9))
    assert out.shape == (4, 12)


def test_weights_roundtrip():
    torch.manual_seed(0)
    a, b = ActivityLSTM(), ActivityLSTM()
    b.set_weights(a.get_weights())
    x = torch.randn(2, 256, 9)
    a.eval(); b.eval()
    torch.testing.assert_close(a(x), b(x))


def test_fedavg_is_sample_weighted():
    agg = FedAvgAggregator(AggregationConfig(min_clients=2))
    agg.receive_update(0, {"w": np.array([0.0, 0.0])}, n_samples=100)
    agg.receive_update(1, {"w": np.array([4.0, 8.0])}, n_samples=300)
    np.testing.assert_allclose(agg.aggregate()["w"], [3.0, 6.0])


def test_fedavg_withholds_below_min_clients():
    agg = FedAvgAggregator(AggregationConfig(min_clients=3))
    agg.receive_update(0, {"w": np.zeros(2)}, 10)
    assert agg.aggregate() is None
