"""A6 M2: clipping, noise scale, privacy accounting, and the engine's DP upload path."""
import math

import numpy as np
import torch

from src.fl.dp import clip_and_noise, epsilon


def _pair(norm):
    start = {"w": torch.zeros(4, 25), "steps": torch.tensor(3)}
    v = torch.randn(4, 25)
    return {"w": v / v.norm() * norm, "steps": torch.tensor(3)}, start


def test_update_above_the_bound_is_clipped_exactly_and_ints_untouched():
    s, start = _pair(5.0)
    out, n = clip_and_noise(s, start, clip=1.0, z=0.0, gen=torch.Generator().manual_seed(0))
    assert abs(n - 5.0) < 1e-5 and abs(float(out["w"].norm()) - 1.0) < 1e-5 and int(out["steps"]) == 3


def test_update_below_the_bound_is_unchanged_and_clip0_only_logs():
    s, start = _pair(0.5)
    out, _ = clip_and_noise(s, start, clip=1.0, z=0.0, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(out["w"], s["w"])
    s, start = _pair(9.0)
    out, n = clip_and_noise(s, start, clip=0.0, z=2.0, gen=torch.Generator().manual_seed(0))
    assert torch.allclose(out["w"], s["w"]) and abs(n - 9.0) < 1e-4


def test_noise_has_std_z_times_clip():
    start = {"w": torch.zeros(200, 200)}
    out, _ = clip_and_noise({"w": torch.zeros(200, 200)}, start, clip=0.5, z=2.0, gen=torch.Generator().manual_seed(1))
    assert abs(float(out["w"].std()) - 1.0) < 0.02


def test_epsilon_limits_and_monotonicity():
    assert epsilon(0, 1.0) == 0.0 and math.isinf(epsilon(3, 0.0))
    assert epsilon(5, 2.0) > epsilon(5, 4.0) and epsilon(10, 2.0) > epsilon(5, 2.0)


def test_epsilon_matches_the_classical_gaussian_mechanism_scale():
    # classical bound: sigma = sqrt(2 ln(1.25/delta)) / eps; eps = 1, delta = 1e-5 -> sigma ~ 4.84 (sensitivity 1)
    sigma = math.sqrt(2 * math.log(1.25 / 1e-5))
    assert abs(epsilon(1, sigma, sensitivity_factor=1.0) - 1.0) < 0.1


def test_engine_dp_logging_without_noise_reproduces_the_plain_federated_run():
    from tests.test_streaming import _client, _events
    from src.fl.engine import TrainConfig, make_model
    from src.fl.streaming import Policy, Response, run_stream
    clients = {c: _client(c, c) for c in (1, 2, 3)}
    torch.manual_seed(0)
    models = {c: make_model(6, 9, TrainConfig()).eval() for c in clients}
    ev = _events(drifted=(1, 2))
    base = Response("ad", "adapter", True, rounds=2, epochs=1)
    plain = run_stream(clients, models, ev, 20, 5, 6, Policy("p", "oracle", base), TrainConfig())
    logged = run_stream(clients, models, ev, 20, 5, 6, Policy("p", "oracle", Response(**{**base.__dict__, "dp_log": True})), TrainConfig())
    np.testing.assert_allclose(plain["f1"], logged["f1"], atol=1e-6)
    assert len(logged["update_norms"]) == 4 and sum(logged["n_uploads"].values()) == 4
    noisy = run_stream(clients, models, ev, 20, 5, 6,
                       Policy("p", "oracle", Response(**{**base.__dict__, "dp_clip": 0.01, "dp_noise": 4.0})), TrainConfig())
    assert noisy["up_bytes"].sum() == plain["up_bytes"].sum()
