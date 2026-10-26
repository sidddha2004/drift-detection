import numpy as np
import pytest

from src.data_pipeline import DriftConfig, DriftInjector, DriftType


def _inject(x, **kw):
    return DriftInjector(DriftConfig(**kw)).inject_drift(x)


def test_sudden_offset_applies_only_after_injection_point(stream):
    x, _ = stream
    out = _inject(x, drift_type=DriftType.SUDDEN, magnitude=1.5, injection_point=3000)
    np.testing.assert_array_equal(out[:3000], x[:3000])
    np.testing.assert_allclose(out[3000:], x[3000:] + 1.5, rtol=1e-6)


def test_injection_does_not_mutate_input(stream):
    x, _ = stream
    before = x.copy()
    _inject(x, drift_type=DriftType.SUDDEN, injection_point=10)
    np.testing.assert_array_equal(x, before)


def test_feature_level_touches_one_axis_only(stream):
    x, _ = stream
    out = _inject(x, drift_type=DriftType.FEATURE_LEVEL, magnitude=2.0,
                  injection_point=1000, target_axis=3)
    changed = np.where(~np.isclose(out, x).all(axis=0))[0]
    assert list(changed) == [3]


def test_gradual_ramp_is_monotone(stream):
    x, _ = stream
    out = _inject(x, drift_type=DriftType.GRADUAL, magnitude=1.0,
                  injection_point=1000, ramp_samples=2000)
    offset = (out - x)[:, 0]
    assert offset[999] == 0 and offset[-1] == pytest.approx(1.0, abs=1e-3)
    assert np.all(np.diff(offset[1000:3000]) >= -1e-6)


def test_deterministic_given_seed(stream):
    x, _ = stream
    a = _inject(x, drift_type=DriftType.RECURRING, seed=5, injection_point=500)
    b = _inject(x, drift_type=DriftType.RECURRING, seed=5, injection_point=500)
    np.testing.assert_array_equal(a, b)


def test_drift_mask_marks_post_injection_region(stream):
    x, _ = stream
    inj = DriftInjector(DriftConfig(drift_type=DriftType.SUDDEN, injection_point=3000))
    mask = inj.get_drift_mask(len(x))
    assert not mask[:3000].any() and mask[3000:].all()


@pytest.mark.xfail(strict=True, reason="Found in Phase 0: `injection_point or config.injection_point` "
                                       "treats an explicit 0 as 'unset' (drift from the first sample)")
def test_drift_mask_accepts_explicit_zero_injection_point():
    inj = DriftInjector(DriftConfig(drift_type=DriftType.SUDDEN, injection_point=3000))
    assert inj.get_drift_mask(100, injection_point=0).all()


def test_current_injector_is_label_preserving_virtual_drift_only(stream):
    """Documents audit A4: injectors take no labels, so they cannot produce real drift."""
    import inspect
    assert "labels" not in inspect.signature(DriftInjector.inject_drift).parameters
