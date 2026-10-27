"""Streaming drift detectors over the per-window signals of a frozen model (Phase 3).

All detectors are causal: `step(i)` may only look at signals with index <= i. `run_detector` adds a cooldown after
every alarm (the detector state is reset) and returns the alarm window indices plus the number of labels consumed.

D1 ADWIN on the error bit                       (needs labels for every window)
D2 Page-Hinkley on the error bit                (needs labels for every window)
D3 Page-Hinkley on entropy, z-scored by the predicted class with calibration statistics   (label-free)
D4 MMD between recent window statistics and a class-matched calibration reference          (label-free)
D5 two-level: stage A (D3 or D4 or sparse-label sentinel) raises a suspicion, stage B asks for labels on the next
   L windows and confirms only if the error rate is significantly above the calibrated baseline.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binom


# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Calibration:
    """Reference statistics from the clean, labelled calibration pool (pre-deployment data)."""
    ent_mu: Dict[int, float]
    ent_sd: Dict[int, float]
    ent_mu_all: float
    ent_sd_all: float
    err_rate: float                      # overall error rate on the calibration pool
    feat: np.ndarray                     # (n, F) standardised window statistics
    pred: np.ndarray                     # (n,) predicted class
    f_mu: np.ndarray
    f_sd: np.ndarray
    bw2: float                           # RBF bandwidth^2 (median heuristic on the reference)
    null: Optional[np.ndarray] = None    # cached null distribution of the MMD statistic for (B,) buffers


def make_calibration(cal: Dict[str, np.ndarray], min_class: int = 5) -> Calibration:
    ent, pred = cal["ent"].astype(np.float64), cal["pred"]
    mu_a, sd_a = float(ent.mean()), float(ent.std() + 1e-3)
    mu, sd = {}, {}
    for k in np.unique(pred):
        e = ent[pred == k]
        if len(e) >= min_class:
            mu[int(k)], sd[int(k)] = float(e.mean()), float(max(e.std(), 0.05 * sd_a, 1e-3))
    f = cal["feat"].astype(np.float64)
    f_mu, f_sd = f.mean(0), f.std(0) + 1e-6
    z = (f - f_mu) / f_sd
    sub = z[np.random.default_rng(0).permutation(len(z))[:200]]
    d2 = ((sub[:, None] - sub[None]) ** 2).sum(-1)
    return Calibration(mu, sd, mu_a, sd_a, float(cal["err"].mean()), z, pred, f_mu, f_sd,
                       float(np.median(d2[np.triu_indices(len(sub), 1)]) + 1e-9))


def _rbf(a, b, bw2):
    return np.exp(-((a[:, None] - b[None]) ** 2).sum(-1) / bw2)


def mmd2_unbiased(a: np.ndarray, b: np.ndarray, bw2: float) -> float:
    n, m = len(a), len(b)
    kaa, kbb, kab = _rbf(a, a, bw2), _rbf(b, b, bw2), _rbf(a, b, bw2)
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())


def class_matched_reference(cal: Calibration, pred_buf: np.ndarray, rng: np.random.Generator, exclude=None) -> np.ndarray:
    """For every buffer window draw a calibration window with the same predicted class (fallback: any)."""
    ok = np.ones(len(cal.pred), dtype=bool)
    if exclude is not None:
        ok[exclude] = False
    allowed = np.flatnonzero(ok)
    out = np.empty((len(pred_buf), cal.feat.shape[1]))
    for k in np.unique(pred_buf):
        sel = np.flatnonzero(pred_buf == k)
        pool = allowed[cal.pred[allowed] == k]
        if len(pool) == 0:
            pool = allowed
        out[sel] = cal.feat[rng.choice(pool, size=len(sel))]
    return out


def mmd_null(cal: Calibration, B: int, draws: int = 120, seed: int = 0) -> np.ndarray:
    """Null of the class-matched MMD^2 for buffers of B consecutive CLEAN calibration windows."""
    rng = np.random.default_rng(seed)
    n = len(cal.pred)
    out = []
    for _ in range(draws):
        s = int(rng.integers(0, n - B + 1))
        idx = np.arange(s, s + B)
        ref = class_matched_reference(cal, cal.pred[idx], rng, exclude=idx)
        out.append(mmd2_unbiased(cal.feat[idx], ref, cal.bw2))
    return np.sort(np.asarray(out))


# ---------------------------------------------------------------------------------------------------------------------
class PageHinkley:
    """One-sided (increase) Page-Hinkley with a reference mean frozen after `warm` samples."""

    def __init__(self, delta: float, lam: float, warm: int = 25):
        self.delta, self.lam, self.warm = delta, lam, warm
        self.reset()

    def reset(self):
        self.n, self.sum, self.ref, self.m, self.mmin = 0, 0.0, None, 0.0, 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        if self.ref is None:
            self.sum += x
            if self.n >= self.warm:
                self.ref = self.sum / self.n
            return False
        self.m += x - self.ref - self.delta
        self.mmin = min(self.mmin, self.m)
        return self.m - self.mmin > self.lam


class Detector:
    labels_per_window = 0.0

    def reset(self):
        pass

    def step(self, i: int, S: Dict[str, np.ndarray]) -> bool:
        raise NotImplementedError

    def labels_used(self, n_windows: int) -> float:
        return self.labels_per_window * n_windows


class ADWINErr(Detector):
    labels_per_window = 1.0

    def __init__(self, delta: float):
        self.delta = delta
        self.reset()

    def reset(self):
        from river.drift import ADWIN
        self.a = ADWIN(delta=self.delta)

    def step(self, i, S):
        before = self.a.estimation
        self.a.update(float(S["err"][i]))
        return bool(self.a.drift_detected and self.a.estimation > before)


class PHErr(Detector):
    labels_per_window = 1.0

    def __init__(self, delta: float, lam: float, warm: int = 25):
        self.ph = PageHinkley(delta, lam, warm)

    def reset(self):
        self.ph.reset()

    def step(self, i, S):
        return self.ph.update(float(S["err"][i]))


class PHEnt(Detector):
    def __init__(self, cal: Calibration, delta: float, lam: float, warm: int = 25):
        self.cal, self.ph = cal, PageHinkley(delta, lam, warm)

    def reset(self):
        self.ph.reset()

    def z(self, i, S):
        k = int(S["pred"][i])
        mu, sd = self.cal.ent_mu.get(k, self.cal.ent_mu_all), self.cal.ent_sd.get(k, self.cal.ent_sd_all)
        return (float(S["ent"][i]) - mu) / sd

    def step(self, i, S):
        return self.ph.update(self.z(i, S))


class MMDDet(Detector):
    def __init__(self, cal: Calibration, q: float, persist: int, B: int = 32, step: int = 4):
        self.cal, self.B, self.every, self.persist = cal, B, step, persist
        if cal.null is None or len(cal.null) == 0 or getattr(cal, "_null_B", None) != B:
            cal.null = mmd_null(cal, B)
            cal._null_B = B
        self.thr = float(np.quantile(cal.null, q))
        self.rng = np.random.default_rng(1)
        self.reset()

    def reset(self):
        self.start, self.hits = None, 0

    def step(self, i, S):
        if self.start is None:
            self.start = i
        n = i - self.start + 1
        if n < self.B or (n - self.B) % self.every:
            return False
        idx = np.arange(i - self.B + 1, i + 1)
        buf = (S["feat"][idx] - self.cal.f_mu) / self.cal.f_sd
        ref = class_matched_reference(self.cal, S["pred"][idx], self.rng)
        stat = mmd2_unbiased(buf, ref, self.cal.bw2)
        self.hits = self.hits + 1 if stat > self.thr else 0
        return self.hits >= self.persist


class TwoLevel(Detector):
    """Stage A: PH on entropy OR class-matched MMD OR a sparse-label sentinel. Stage B: confirm with labels."""

    def __init__(self, cal: Calibration, ent: Tuple[float, float], mmd: Tuple[float, int], sentinel: Optional[Tuple[int, float]],
                 alpha_b: float, L: int = 16):
        self.cal, self.alpha_b, self.L = cal, alpha_b, L
        self.d3 = PHEnt(cal, ent[0], ent[1])
        self.d4 = MMDDet(cal, mmd[0], mmd[1])
        self.sent = sentinel                         # (period, PH lambda) or None
        self.sph = PageHinkley(0.02, sentinel[1], warm=10) if sentinel else None
        self.n_sent_total = self.n_susp_total = 0
        self.reset()

    def reset(self):
        self.d3.reset(); self.d4.reset()
        if self.sph:
            self.sph.reset()
        self.pending = None

    def labels_used(self, n_windows):
        return self.n_sent_total + self.L * self.n_susp_total

    def step(self, i, S):
        if self.pending is not None:                          # stage B: labels requested for windows pending..i
            if i - self.pending + 1 < self.L:
                return False
            errs = int(S["err"][self.pending:i + 1].sum())
            self.pending = None
            p0 = max(self.cal.err_rate, 0.03)
            return bool(binom.sf(errs - 1, self.L, p0) < self.alpha_b)
        trig = self.d3.step(i, S)
        trig = self.d4.step(i, S) or trig
        if self.sent and i % self.sent[0] == 0:
            self.n_sent_total += 1
            trig = self.sph.update(float(S["err"][i])) or trig
        if trig:
            self.pending = i + 1
            self.n_susp_total += 1
        return False


def run_detector(det: Detector, S: Dict[str, np.ndarray], cooldown: int = 30) -> Tuple[List[int], float]:
    """Alarm window indices of one client stream and the labels consumed."""
    n = len(S["err"])
    alarms, last = [], -10 ** 9
    det.reset()
    for i in range(n):
        if det.step(i, S):
            if i - last >= cooldown:
                alarms.append(i)
            last = i
            det.reset()
    return alarms, det.labels_used(n)
