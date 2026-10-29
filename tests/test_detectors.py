"""Detector and matching tests on synthetic signals (no dataset, no GPU)."""
import numpy as np
import pytest

from src.detect.detectors import (ADWINErr, MMDDet, PageHinkley, PHEnt, PHErr, TwoLevel, make_calibration, mmd2_unbiased,
                                  run_detector)
from src.detect.evaluate import aggregate, match_stream

N, ONSET = 160, 80


def _cal(n=120, seed=0):
    r = np.random.default_rng(seed)
    pred = np.repeat(np.arange(6), n // 6)
    return {"ent": r.gamma(2, 0.05, n).astype(np.float32) + 0.1 * (pred % 3), "pred": pred.astype(np.int16),
            "err": (r.random(n) < 0.05).astype(np.float32), "feat": r.normal(0, 1, (n, 27)).astype(np.float32)}


def _stream(drift: str, seed=0, n=N, onset=ONSET):
    """drift: none | err (error jumps) | ent (entropy up) | feat (inputs shift)."""
    r = np.random.default_rng(seed)
    pred = np.repeat(np.arange(6), int(np.ceil(n / 6)))[:n]
    post = np.arange(n) >= onset
    err = (r.random(n) < np.where(post & (drift == "err"), 0.5, 0.05)).astype(np.float32)
    ent = r.gamma(2, 0.05, n) + 0.1 * (pred % 3) + (0.8 * post if drift == "ent" else 0)
    feat = r.normal(0, 1, (n, 27)) + (2.0 * post[:, None] if drift == "feat" else 0)
    return {"err": err, "ent": ent.astype(np.float32), "pred": pred.astype(np.int16), "feat": feat.astype(np.float32),
            "alpha": post.astype(np.float32) if drift != "none" else np.zeros(n, np.float32),
            "rnd": (np.arange(n) // 2).astype(np.int32)}


def test_page_hinkley_fires_on_increase_not_on_stationary():
    r = np.random.default_rng(0)
    ph = PageHinkley(0.05, 5.0, warm=20)
    assert not any(ph.update(float(v)) for v in r.normal(0, 0.1, 300))
    ph.reset()
    fired = [ph.update(float(v)) for v in np.r_[r.normal(0, 0.1, 40), r.normal(2.0, 0.1, 40)]]
    assert any(fired) and np.argmax(fired) >= 40                   # never before the change


def test_page_hinkley_ignores_decrease():
    ph = PageHinkley(0.05, 5.0, warm=20)
    assert not any(ph.update(v) for v in np.r_[np.ones(30), np.zeros(100)])


def test_supervised_detectors_fire_after_error_jump():
    S = _stream("err")
    for det in (ADWINErr(0.3), PHErr(0.02, 3.0)):
        alarms, labels = run_detector(det, S)
        assert alarms and alarms[0] >= ONSET and labels == N        # all labels consumed


def test_supervised_detectors_quiet_without_drift():
    S = _stream("none")
    assert run_detector(PHErr(0.02, 8.0), S)[0] == []


def test_entropy_detector_is_label_free_and_fires():
    cal = make_calibration(_cal())
    alarms, labels = run_detector(PHEnt(cal, 0.3, 10.0), _stream("ent"))
    assert alarms and alarms[0] >= ONSET and labels == 0


def test_mmd_detects_covariate_shift_and_not_the_clean_stream():
    cal = make_calibration(_cal())
    alarms, labels = run_detector(MMDDet(cal, q=0.99, persist=2), _stream("feat"))
    assert alarms and ONSET <= alarms[0] <= ONSET + 60 and labels == 0
    assert run_detector(MMDDet(cal, q=0.999, persist=3), _stream("none"))[0] == []


def test_mmd_statistic_is_near_zero_for_equal_and_large_for_shifted_samples():
    r = np.random.default_rng(1)
    a, b, c = r.normal(0, 1, (60, 5)), r.normal(0, 1, (60, 5)), r.normal(3, 1, (60, 5))
    assert abs(mmd2_unbiased(a, b, 10.0)) < 0.05 < mmd2_unbiased(a, c, 10.0)


def test_two_level_confirms_a_real_error_increase_and_counts_labels():
    cal = make_calibration(_cal())
    S = _stream("err")
    det = TwoLevel(cal, ent=(0.3, 10.0), mmd=(0.99, 2), sentinel=(4, 2.0), alpha_b=0.05, L=16)
    alarms, labels = run_detector(det, S)
    assert alarms and alarms[0] >= ONSET
    assert 0 < labels < N                                           # cheaper than labelling every window


def test_two_level_dismisses_a_benign_covariate_shift():
    """Inputs move but the model stays correct: stage A may trigger, stage B must refuse to confirm."""
    cal = make_calibration(_cal())
    S = _stream("feat")                                             # errors stay at the base rate
    det = TwoLevel(cal, ent=(0.3, 10.0), mmd=(0.99, 2), sentinel=None, alpha_b=0.01, L=16)
    assert run_detector(det, S)[0] == []
    assert det.n_susp_total >= 1                                    # it was suspected, then dismissed


def test_without_sentinel_label_free_stage_cannot_see_real_drift():
    cal = make_calibration(_cal())
    det = TwoLevel(cal, ent=(0.3, 10.0), mmd=(0.99, 2), sentinel=None, alpha_b=0.05, L=16)
    alarms, labels = run_detector(det, _stream("err"))
    assert alarms == [] and labels == 0


def test_cooldown_suppresses_repeated_alarms():
    S = _stream("err")
    a1, _ = run_detector(PHErr(0.0, 1.0, warm=10), S, cooldown=1)
    a2, _ = run_detector(PHErr(0.0, 1.0, warm=10), S, cooldown=60)
    assert len(a2) < len(a1) and all(y - x >= 60 for x, y in zip(a2, a2[1:]))


# ---- matching rules --------------------------------------------------------------------------------------------
def _S(alpha_from=80):
    n = 160
    return {"rnd": (np.arange(n) // 2).astype(np.int32), "alpha": (np.arange(n) >= alpha_from).astype(np.float32)}


def test_match_counts_tp_delay_miss_false_and_redundant():
    S = _S()                                                         # onset round 40 = window 80
    m = match_stream([10, 85, 120], S, onsets=[40])
    assert m["tp"] == 1 and m["delays_w"] == [5] and m["delays_r"] == [2]
    assert m["false_alarms"] == 1 and m["redundant"] == 1 and m["miss"] == 0
    m = match_stream([], S, onsets=[40])
    assert m["miss"] == 1 and m["tp"] == 0


def test_alarm_outside_horizon_is_a_miss_not_a_hit():
    m = match_stream([150], _S(), onsets=[40], horizon=10)           # round 75 >= 40 + 10
    assert m["tp"] == 0 and m["miss"] == 1 and m["redundant"] == 1


def test_aggregate_objective_charges_misses_the_full_horizon():
    parts = [match_stream([85], _S(), [40]), match_stream([], _S(), [40])]
    a = aggregate(parts, horizon=20)
    assert a["recall"] == 0.5 and a["objective_delay_rounds"] == pytest.approx((2 + 20) / 2)
    assert a["fa_per_1000"] == 0
