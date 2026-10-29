import copy

import numpy as np
import pytest
import torch

from src.fl.adapter import AdaptedLSTM
from src.fl.engine import DEVICE, TrainConfig, make_model

TC = TrainConfig()


def _base():
    torch.manual_seed(0)
    return make_model(12, 9, TC).eval()


def _data(n=48, shift=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 32, 9, generator=g).to(DEVICE) + shift
    return x


def test_initial_adapter_reproduces_the_base_model_exactly():
    base = _base()
    ad = AdaptedLSTM(base).eval()
    x = _data()
    with torch.no_grad():
        np.testing.assert_allclose(base(x).cpu().numpy(), ad(x).cpu().numpy(), atol=1e-5)


def test_parameter_counts_and_bytes():
    ad = AdaptedLSTM(_base(), bottleneck=8)
    assert ad.adapter_params() == (81 + 9) + (64 * 8 + 8) + (8 * 64 + 64) + (64 * 12 + 12) == 1_966
    assert ad.adapter_bytes() == 1_966 * 4
    full = sum(p.numel() for p in _base().parameters())
    assert full == 53_260 and ad.adapter_bytes() < 0.05 * full * 4          # < 5% of the full-model payload


def test_backbone_is_frozen_and_never_changes():
    ad = AdaptedLSTM(_base())
    before = {k: v.clone() for k, v in ad.base.state_dict().items()}
    ad.train()
    opt = torch.optim.Adam([p for p in ad.parameters() if p.requires_grad], lr=1e-2)
    x, y = _data(), torch.randint(0, 12, (48,)).to(DEVICE)
    for _ in range(5):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(ad(x), y).backward()
        opt.step()
    for k, v in ad.base.state_dict().items():
        assert torch.equal(v, before[k]), k
    assert not any(p.requires_grad for p in ad.base.parameters())


def test_set_trainable_selects_parts():
    ad = AdaptedLSTM(_base())
    ad.set_trainable(["input"])
    assert ad.trainable_params() == 90
    ad.set_trainable(["head"])
    assert ad.trainable_params() == 64 * 12 + 12
    ad.set_trainable(["input", "feature", "head"])
    assert ad.trainable_params() == ad.adapter_params()


def _trained_base(n_classes=4, seed=0):
    """A base model trained on a separable synthetic task: class k has its own signal energy (std) on every channel,
    so a constant input offset does not carry class information (it is pure virtual drift)."""
    g = torch.Generator().manual_seed(seed)
    y = torch.randint(0, n_classes, (256,), generator=g)
    x = torch.randn(256, 32, 9, generator=g) * (0.4 + 0.6 * y.float())[:, None, None]
    x, y = x.to(DEVICE), y.to(DEVICE)
    torch.manual_seed(seed)
    m = make_model(n_classes, 9, TC)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(120):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(m(x), y).backward()
        opt.step()
    return m.eval(), x, y


def test_input_adapter_recovers_accuracy_after_a_sensor_offset():
    """Virtual drift = constant bias on the inputs; a trained input recalibration should absorb it."""
    base, x, y = _trained_base()
    xs = x + 1.0
    with torch.no_grad():
        acc_clean = (base(x).argmax(1) == y).float().mean().item()
        acc_drift = (base(xs).argmax(1) == y).float().mean().item()
    assert acc_clean > 0.95 and acc_drift < 0.6
    ad = AdaptedLSTM(base)
    ad.set_trainable(["input"])
    ad.train()
    opt = torch.optim.Adam([p for p in ad.parameters() if p.requires_grad], lr=2e-2)
    for _ in range(100):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(ad(xs), y).backward()
        opt.step()
    assert (ad(xs).argmax(1) == y).float().mean().item() > 0.9


def test_head_adapter_recovers_accuracy_after_a_label_swap():
    """Real drift = P(y|x) changes: only the head (and feature adapter) can fix it, the input adapter cannot."""
    base, x, y = _trained_base()
    swap = torch.tensor([1, 0, 3, 2], device=DEVICE)
    ys = swap[y]
    res = {}
    for parts in (["input"], ["head"]):
        ad = AdaptedLSTM(base)
        ad.set_trainable(parts)
        ad.train()
        opt = torch.optim.Adam([p for p in ad.parameters() if p.requires_grad], lr=2e-2)
        for _ in range(100):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(ad(x), ys).backward()
            opt.step()
        res[parts[0]] = (ad(x).argmax(1) == ys).float().mean().item()
    assert res["head"] > 0.9 and res["input"] < 0.9


def test_adapter_state_roundtrip_and_averaging_keys():
    a, b = AdaptedLSTM(_base()), AdaptedLSTM(_base())
    with torch.no_grad():
        a.in_b += 1.0
    b.load_adapter_state(a.adapter_state())
    assert torch.allclose(b.in_b, a.in_b)
    assert set(a.adapter_state()) == set(a.adapter_names())
    assert all(not k.startswith("base.") for k in a.adapter_state())
