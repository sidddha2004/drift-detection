"""Apply a DriftEvent at strength alpha to normalised windows / labels.

Virtual transforms act in RAW sensor units (g): windows are de-normalised with the client's frozen train statistics,
transformed, and re-normalised with the SAME statistics, so a bias of 1 g is 1 g for every client and the detector
sees exactly what a deployed (frozen-normaliser) model would see. Real drift only relabels.
"""
from typing import Tuple

import numpy as np

from .schedule import DriftEvent


def _rotate_x(raw: np.ndarray, imu: int, deg: float) -> None:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    y, z = raw[..., 3 * imu + 1].copy(), raw[..., 3 * imu + 2].copy()
    raw[..., 3 * imu + 1] = c * y - s * z
    raw[..., 3 * imu + 2] = s * y + c * z


def apply_virtual(x: np.ndarray, mean: np.ndarray, std: np.ndarray, ev: DriftEvent, alpha: float) -> np.ndarray:
    """x: (n, W, C) normalised float32. Returns a new array; alpha == 0 returns an exact copy."""
    if alpha <= 0 or ev.mode != "virtual":
        return x.copy()
    raw = x.astype(np.float64) * std + mean
    ch = list(ev.channels)
    if ev.transform == "offset":
        raw[..., ch] += ev.magnitude * alpha
    elif ev.transform == "gain":
        raw[..., ch] *= 1.0 + (ev.magnitude - 1.0) * alpha
    elif ev.transform == "rotation":
        for imu in sorted({c // 3 for c in ch}):
            _rotate_x(raw, imu, ev.magnitude * alpha)
    else:
        raise ValueError(ev.transform)
    return ((raw - mean) / std).astype(np.float32)


def apply_real(y: np.ndarray, ev: DriftEvent, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Swap labels of the event's class pairs for each window with probability alpha."""
    out = y.copy()
    if alpha <= 0 or ev.mode != "real":
        return out
    hit = np.ones(len(y), dtype=bool) if alpha >= 1 else rng.random(len(y)) < alpha
    for a, b in ev.remap:
        out[hit & (y == a)] = b
        out[hit & (y == b)] = a
    return out


def apply_events(x, y, mean, std, events, alphas, rng) -> Tuple[np.ndarray, np.ndarray]:
    """Compose all events of a client at one round (`alphas[i]` belongs to events[i])."""
    for ev, a in zip(events, alphas):
        x = apply_virtual(x, mean, std, ev, a)
        y = apply_real(y, ev, a, rng)
    return x, y
