"""Drift-type features and rule classifier on synthetic streams with a known signature."""
import numpy as np

from src.detect.detectors import make_calibration
from src.detect.drift_type import FEATURE_NAMES, RuleClassifier, run_episodes, series

N, PER_ROUND = 240, 3


def _cal(seed=0, n=150):
    r = np.random.default_rng(seed)
    pred = np.repeat(np.arange(5), n // 5)
    return make_calibration({"ent": r.gamma(2, 0.05, n).astype(np.float32), "pred": pred.astype(np.int16),
                             "err": (r.random(n) < 0.03).astype(np.float32), "feat": r.normal(0, 1, (n, 27)).astype(np.float32)})


def _stream(alpha, channels=range(27), shift=3.0, seed=1):
    """alpha(i) in [0,1] per window; the listed feature columns move by shift*alpha."""
    r = np.random.default_rng(seed)
    a = np.asarray(alpha, dtype=float)
    feat = r.normal(0, 1, (N, 27))
    feat[:, list(channels)] += shift * a[:, None]
    pred = np.repeat(np.arange(5), N // 5)
    return {"feat": feat.astype(np.float32), "ent": (r.gamma(2, 0.05, N) + 0.1 * a).astype(np.float32),
            "pred": pred.astype(np.int16), "err": (r.random(N) < 0.03 + 0.2 * a).astype(np.float32)}


def _alpha_sudden(t):
    return (np.arange(N) >= t).astype(float)


def _alpha_gradual(t, ramp=45):
    return np.clip((np.arange(N) - t) / ramp, 0, 1)


def _imu_cols(m):
    return [s * 9 + 3 * m + c for s in range(3) for c in range(3)]


def _type(alpha, alarm, channels=range(27), cal=None):
    cal = cal or _cal()
    S = _stream(alpha, channels)
    eps = run_episodes(series(S, cal), PER_ROUND, [alarm])
    assert eps, "episode was dropped"
    return RuleClassifier().predict_one(eps[0][1])


def test_feature_vector_has_the_declared_names():
    S = _stream(_alpha_sudden(100))
    (_, f), = run_episodes(series(S, _cal()), PER_ROUND, [110])
    assert len(f) == len(FEATURE_NAMES) and np.isfinite(f).all()


def test_sudden_shift_across_all_sensors_is_typed_sudden():
    assert _type(_alpha_sudden(100), alarm=106) == "sudden"


def test_slow_ramp_is_typed_gradual():
    assert _type(_alpha_gradual(100), alarm=112) == "gradual"


def test_single_sensor_shift_is_typed_feature_level():
    assert _type(_alpha_sudden(100), alarm=106, channels=_imu_cols(1)) == "feature_level"


def test_returned_and_repeated_signature_is_typed_recurring():
    cal = _cal()
    on = (np.arange(N) >= 60) & (np.arange(N) < 110) | (np.arange(N) >= 150)      # A, back to baseline, A again
    S = _stream(on.astype(float))
    eps = run_episodes(series(S, cal), PER_ROUND, [66, 156])
    rc = RuleClassifier()
    assert [rc.predict_one(f) for _, f in eps] == ["sudden", "recurring"]         # first occurrence is indistinguishable


def test_a_different_second_drift_is_not_recurring():
    cal = _cal()
    a1 = ((np.arange(N) >= 60) & (np.arange(N) < 110)).astype(float)
    S = _stream(a1)
    r = np.random.default_rng(5)
    other = (np.arange(N) >= 150)
    S["feat"][other] += (r.normal(0, 3, 27)).astype(np.float32)                   # unrelated direction
    eps = run_episodes(series(S, cal), PER_ROUND, [66, 156])
    assert RuleClassifier().predict_one(eps[1][1]) != "recurring"


def test_alarm_too_close_to_the_end_is_dropped():
    S = _stream(_alpha_sudden(100))
    assert run_episodes(series(S, _cal()), PER_ROUND, [N - 5]) == []


def test_features_do_not_peek_beyond_the_lookahead():
    cal = _cal()
    S = _stream(_alpha_sudden(100))
    f1 = run_episodes(series(S, cal), PER_ROUND, [106])[0][1]
    S2 = {k: v.copy() for k, v in S.items()}
    S2["feat"][106 + 5 * PER_ROUND + 2:] += 50                                    # change only the far future
    S2["err"][106 + 5 * PER_ROUND + 2:] = 1
    f2 = run_episodes(series(S2, cal), PER_ROUND, [106])[0][1]
    np.testing.assert_allclose(f1, f2, atol=1e-9)
