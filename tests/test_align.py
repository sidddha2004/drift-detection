"""The closed-form input recalibration must undo known virtual drift from a few labelled windows of a few classes."""
import numpy as np
import pytest

from src.drift.schedule import DriftEvent
from src.drift.transforms import apply_virtual
from src.fl.align import fit_input_alignment, reference_means

C, W_LEN = 9, 64


def _client(seed=0, n_classes=6, per_class=40):
    r = np.random.default_rng(seed)
    mean = r.normal(0, 2, C).astype(np.float32)
    std = r.uniform(1.5, 4, C).astype(np.float32)
    # class k: its own gravity-like mean vector per IMU (raw units) + noise
    centers = r.normal(0, 4, (n_classes, C))
    y = np.repeat(np.arange(n_classes), per_class)
    raw = centers[y][:, None, :] + r.normal(0, 1.5, (len(y), W_LEN, C))
    x = ((raw - mean) / std).astype(np.float32)
    return x, y, mean, std


def _ev(transform, mag, channels=tuple(range(C))):
    return DriftEvent(101, 0, "sudden", "virtual", transform, channels, mag, 0)


def _err(x_true, x_fixed):
    return float(np.abs(x_true - x_fixed).mean())


@pytest.mark.parametrize("transform,mag", [("offset", 1.0), ("offset", 2.0), ("gain", 1.3), ("gain", 1.6), ("rotation", 20.0), ("rotation", 30.0)])
def test_alignment_recovers_the_clean_signal(transform, mag):
    x, y, mean, std = _client()
    ref = reference_means(x, y, mean, std)
    xd = apply_virtual(x, mean, std, _ev(transform, mag), 1.0)
    sel = np.isin(y, [0, 2, 3, 5]) & (np.arange(len(y)) % 4 == 0)              # 4 classes, 10 windows each
    W, b = fit_input_alignment(xd[sel], y[sel], ref, mean, std, ridge=0.05)
    before, after = _err(x, xd), _err(x, xd @ W + b)
    assert after < 0.35 * before, (transform, mag, before, after)


def test_identity_without_drift():
    x, y, mean, std = _client()
    ref = reference_means(x, y, mean, std)
    W, b = fit_input_alignment(x[::3], y[::3], ref, mean, std, ridge=0.05)
    assert np.abs(W - np.eye(C)).max() < 0.05 and np.abs(b).max() < 0.05


def test_single_sensor_drift_leaves_other_sensors_untouched():
    x, y, mean, std = _client()
    ref = reference_means(x, y, mean, std)
    xd = apply_virtual(x, mean, std, _ev("offset", 2.0, channels=(3, 4, 5)), 1.0)
    W, b = fit_input_alignment(xd[::3], y[::3], ref, mean, std, ridge=0.05)
    fixed = xd @ W + b
    np.testing.assert_allclose(fixed[..., [0, 1, 2, 6, 7, 8]], x[..., [0, 1, 2, 6, 7, 8]], atol=0.15)
    assert _err(x[..., 3:6], fixed[..., 3:6]) < 0.35 * _err(x[..., 3:6], xd[..., 3:6])


def test_no_evidence_returns_identity_and_one_class_recovers_a_pure_offset():
    x, y, mean, std = _client()
    ref = reference_means(x, y, mean, std)
    xd = apply_virtual(x, mean, std, _ev("offset", 2.0), 1.0)
    W, b = fit_input_alignment(xd[:0], y[:0], ref, mean, std)                           # empty buffer
    np.testing.assert_array_equal(W, np.eye(C))
    np.testing.assert_array_equal(b, np.zeros(C))
    one = y == 1
    W, b = fit_input_alignment(xd[one][:10], y[one][:10], ref, mean, std, ridge=0.05)     # a single class is enough for a bias
    assert _err(x, xd @ W + b) < 0.35 * _err(x, xd)


def test_real_drift_leaves_the_inputs_alone():
    """Relabelled windows with unchanged inputs must not move the input layer much when classes are seen under swapped labels."""
    x, y, mean, std = _client()
    ref = reference_means(x, y, mean, std)
    y_swapped = np.where(y == 0, 1, np.where(y == 1, 0, y))
    sel = np.isin(y, [0, 1, 2, 3])
    W, b = fit_input_alignment(x[sel], y_swapped[sel], ref, mean, std, ridge=1.0)
    # with a strong ridge the (wrong) pull towards swapped class means stays small relative to the clean signal scale
    assert _err(x, x @ W + b) < 1.0
