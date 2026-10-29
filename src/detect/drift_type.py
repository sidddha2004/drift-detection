"""4-way drift-type features and classifiers (sudden / gradual / recurring / feature_level), Phase 3.

Typing happens AFTER an alarm, with a fixed look-ahead of `LOOKAHEAD_ROUNDS` rounds (the typing delay), from three
evidence series that a client can compute locally:
  d_feat  how far the window statistics are from the class-matched calibration mean   (label-free; total + per IMU)
  ent_z   entropy z-scored per predicted class                                         (label-free)
  err     error bit                                                                    (labels of the confirmation phase)
Recurrence is recognised from a per-client MEMORY of earlier episode signatures: a drift is `recurring` when its
signature matches a stored one and the stream had returned to baseline in between. The first onset of a recurring
drift cannot be told from a sudden one at alarm time (reported separately as `recurring_first`).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .detectors import Calibration

TYPES = ("sudden", "gradual", "recurring", "feature_level")
PRE, PRE_LONG, LOOKAHEAD_ROUNDS = 30, 60, 5
FEATURE_NAMES = ["dF_pre", "dF_post", "dF_delta", "dF_ramp", "dF_slope", "ent_delta", "ent_ramp", "err_delta", "err_ramp",
                 "imu_share_max", "imu_ratio", "imu_entropy", "mem_cos", "mem_returned", "mem_recur", "n_prior",
                 "rise_dF", "rise_ent", "rise_err", "sr_dF", "sr_ent", "sr_err"]


def series(S: Dict[str, np.ndarray], cal: Calibration) -> Dict[str, np.ndarray]:
    """Evidence series of a stream, expressed as excess over the clean calibration level."""
    z = (S["feat"].astype(np.float64) - cal.f_mu) / cal.f_sd
    mu_k = {int(k): cal.feat[cal.pred == k].mean(0) for k in np.unique(cal.pred)}
    mu_all = cal.feat.mean(0)
    ref = np.stack([mu_k.get(int(k), mu_all) for k in S["pred"]])
    dev = np.abs(z - ref)                                                     # (n, 27)
    cal_dev = np.abs(cal.feat - np.stack([mu_k[int(k)] for k in cal.pred]))   # cal windows vs their class mean
    imu_idx = [[s * 9 + 3 * m + c for s in range(3) for c in range(3)] for m in range(3)]
    d_imu = np.stack([dev[:, ix].mean(1) for ix in imu_idx], 1) - np.array([cal_dev[:, ix].mean() for ix in imu_idx])
    k = S["pred"].astype(int)
    ent_z = np.array([(e - cal.ent_mu.get(c, cal.ent_mu_all)) / cal.ent_sd.get(c, cal.ent_sd_all) for e, c in zip(S["ent"], k)])
    return {"d_feat": dev.mean(1) - cal_dev.mean(), "d_imu": d_imu, "ent_z": ent_z, "err": S["err"].astype(np.float64) - cal.err_rate,
            "dev": dev - cal_dev.mean(0)}


def _slope(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    x = np.arange(len(y)) - (len(y) - 1) / 2
    return float((x * (y - y.mean())).sum() / (x ** 2).sum())


def _ramp(y: np.ndarray) -> float:
    """Late third minus early third of the post-alarm series: ~0 for a step, > 0 while still rising."""
    k = max(1, len(y) // 3)
    return float(y[-k:].mean() - y[:k].mean())


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) < k:
        return y
    c = np.convolve(y, np.ones(k) / k, mode="same")
    return c


def rise_and_fit(y: np.ndarray, per_round: int, smooth: int = 5) -> Tuple[float, float]:
    """Retrospective shape of a change inside a segment (history before the alarm + look-ahead).

    rise  : rounds between 10% and 90% of the total level change (0 if there is no clear change)
    sr    : SSE(best ramp) / SSE(best step), < 1 means a ramp explains the data better than a step
    """
    n = len(y)
    if n < 20:
        return 0.0, 1.0
    ys = _smooth(y, smooth)
    b, f = ys[:8].mean(), ys[-8:].mean()
    amp = f - b
    noise = y[:max(8, n // 4)].std() + 1e-9
    rise = 0.0
    if abs(amp) > 1.0 * noise / np.sqrt(8):
        lo = np.flatnonzero((ys - b) / amp >= 0.1)
        hi = np.flatnonzero((ys - b) / amp >= 0.9)
        if len(lo) and len(hi) and hi[0] >= lo[0]:
            rise = float(hi[0] - lo[0]) / per_round
    # SSE of a step vs a ramp (coarse grid over change points)
    cs = np.arange(3, n - 3, 2)
    csum, csq = np.cumsum(y), np.cumsum(y ** 2)
    sse_step = np.inf
    for c in cs:
        m1, m2 = csum[c - 1] / c, (csum[-1] - csum[c - 1]) / (n - c)
        sse_step = min(sse_step, (csq[c - 1] - c * m1 ** 2) + ((csq[-1] - csq[c - 1]) - (n - c) * m2 ** 2))
    sse_ramp = np.inf
    x = np.arange(n)
    for c1 in cs[::2]:
        for c2 in cs[::2]:
            if c2 - c1 < 6:
                continue
            ramp = np.clip((x - c1) / (c2 - c1), 0, 1)
            A = np.stack([np.ones(n), ramp], 1)
            coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
            sse_ramp = min(sse_ramp, float(((A @ coef - y) ** 2).sum()))
    return rise, float(sse_ramp / (sse_step + 1e-9)) if np.isfinite(sse_step) and np.isfinite(sse_ramp) else 1.0


@dataclass
class Memory:
    sigs: List[np.ndarray] = field(default_factory=list)
    levels: List[float] = field(default_factory=list)


def episode_features(ser: Dict[str, np.ndarray], t0: int, per_round: int, mem: Memory) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Features and signature of the episode that raised an alarm at window t0 (None if the stream ends too early)."""
    F = LOOKAHEAD_ROUNDS * per_round
    n = len(ser["d_feat"])
    if t0 + F >= n or t0 < 10:
        return None
    pre = slice(max(0, t0 - PRE), max(1, t0 - 5))
    post = slice(t0 - 3, t0 + F + 1)
    dF, ent, err, dev = ser["d_feat"], ser["ent_z"], ser["err"], ser["dev"]
    pre_dF, post_dF = dF[pre].mean(), dF[post].mean()
    d_imu = ser["d_imu"][post].mean(0) - ser["d_imu"][pre].mean(0)
    pos = np.clip(d_imu, 0, None)
    tot = pos.sum() + 1e-9
    share = pos / tot
    imu_entropy = float(-(share[share > 0] * np.log(share[share > 0])).sum())
    shift = np.concatenate([dev[post].mean(0) - dev[pre].mean(0), [ent[post].mean() - ent[pre].mean(),
                                                                    5 * (err[post].mean() - err[pre].mean())]])
    norm = float(np.linalg.norm(shift)) + 1e-9
    sig = shift / norm
    cos, returned = 0.0, 0.0
    if mem.sigs:
        cos = float(max(float(sig @ s) for s in mem.sigs))
        pre_sig = np.concatenate([dev[pre].mean(0) - 0.0, [ent[pre].mean(), 5 * err[pre].mean()]])
        returned = float(1.0 - min(1.0, np.linalg.norm(pre_sig) / (max(mem.levels) + 1e-9)))
    seg = slice(max(0, t0 - PRE_LONG), t0 + F + 1)
    r_d, s_d = rise_and_fit(dF[seg], per_round)
    r_e, s_e = rise_and_fit(ent[seg], per_round)
    r_r, s_r = rise_and_fit(_smooth(err[seg], 9), per_round, smooth=1)
    feats = np.array([pre_dF, post_dF, post_dF - pre_dF, _ramp(dF[post]), _slope(dF[t0 - 10:t0 + F + 1]),
                      ent[post].mean() - ent[pre].mean(), _ramp(ent[post]),
                      err[post].mean() - err[pre].mean(), _ramp(err[post]),
                      share.max(), float(pos.max() / (np.median(pos) + 1e-3)), imu_entropy,
                      cos, returned, cos * returned, float(len(mem.sigs)), r_d, r_e, r_r, s_d, s_e, s_r])
    return feats, np.append(sig, norm)


def run_episodes(ser, per_round, alarms: List[int]) -> List[Tuple[int, np.ndarray]]:
    """Features of all alarms of one client stream in order, with the memory growing after every episode."""
    mem, out = Memory(), []
    for a in alarms:
        r = episode_features(ser, a, per_round, mem)
        if r is None:
            continue
        feats, sig = r
        out.append((a, feats))
        mem.sigs.append(sig[:-1])
        mem.levels.append(float(sig[-1]))
    return out


class RuleClassifier:
    """Hand-set rules with a-priori thresholds (fixed before any result was seen)."""
    SHARE, RAMP, RECUR, RETURNED = 0.6, 0.5, 0.8, 0.5
    SHIFT_FLOOR = 0.05                          # below this total shift there is no covariate evidence for sensor asymmetry

    def predict_one(self, f: np.ndarray) -> str:
        v = dict(zip(FEATURE_NAMES, f))
        if v["mem_cos"] >= self.RECUR and v["mem_returned"] >= self.RETURNED:
            return "recurring"
        if v["dF_delta"] > self.SHIFT_FLOOR and v["imu_share_max"] >= self.SHARE:
            return "feature_level"
        if v["dF_ramp"] > self.RAMP * max(v["dF_delta"], 1e-6) or v["err_ramp"] > 0.3:
            return "gradual"
        return "sudden"

    def predict(self, X: np.ndarray) -> List[str]:
        return [self.predict_one(x) for x in X]
