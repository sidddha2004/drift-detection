"""PAMAP2 loading with activity *bouts* as the unit of segmentation.

A bout is a maximal run of one non-transient activity on a gap-free timeline.
Windows are only ever cut inside a bout, so a window can never straddle an
activity boundary and removing activity 0 never glues unrelated segments
together (audit A7).

Column layout (0-indexed) of a PAMAP2 .dat row, per IMU block
(hand 3-19, chest 20-36, ankle 37-53):
    +0 temperature | +1..+3 acc (+-16g) | +4..+6 acc (+-6g) | +7..+9 gyro | +10..+12 mag | +13..+16 orient
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

TIMESTAMP_COL, ACTIVITY_COL = 0, 1

# Correct +-16g accelerometer columns: hand, chest, ankle (x, y, z each).
ACC16_COLS = [4, 5, 6, 21, 22, 23, 38, 39, 40]
# What the pre-Phase-1 code used by mistake: temperature + only two acc axes (audit A14).
LEGACY_COLS = [3, 4, 5, 20, 21, 22, 37, 38, 39]
TEMP_COLS = [3, 20, 37]

PROTOCOL_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]
ACTIVITY_NAMES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running", 6: "cycling",
    7: "nordic walking", 9: "watching TV", 10: "computer work", 11: "car driving",
    12: "ascending stairs", 13: "descending stairs", 16: "vacuum cleaning",
    17: "ironing", 18: "folding laundry", 19: "house cleaning", 20: "playing soccer",
    24: "rope jumping",
}

MAX_GAP_S = 0.05          # a timestamp jump larger than this splits a bout
MAX_NAN_RUN = 50          # bouts with a missing run longer than this (0.5 s) are dropped; shorter runs are interpolated


@dataclass
class Bout:
    subject: int
    activity: int
    data: np.ndarray          # (n, n_channels) float32, NaN interpolated inside the bout
    start_row: int            # row offset in the raw file (provenance)
    nan_frac: float


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def load_bouts(path, subject: int, cols: Sequence[int] = ACC16_COLS,
               activities: Sequence[int] = PROTOCOL_ACTIVITIES,
               stats: dict = None) -> List[Bout]:
    """Return the usable bouts of one subject file. `stats` (optional dict) collects audit counters."""
    stats = stats if stats is not None else {}
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN"])
    ts = df[TIMESTAMP_COL].to_numpy()
    act = df[ACTIVITY_COL].to_numpy().astype(int)
    x = df[list(cols)].to_numpy(dtype=np.float32)

    # run boundaries: label change OR timestamp gap
    gap = np.r_[False, np.diff(ts) > MAX_GAP_S]
    change = np.r_[True, act[1:] != act[:-1]] | gap
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(act)]

    stats.update(rows=len(df), transient_rows=int((act == 0).sum()), n_gaps=int(gap.sum()),
                 dropped_nan=0, dropped_activity=0)
    bouts: List[Bout] = []
    for s, e in zip(starts, ends):
        a = int(act[s])
        if a == 0:
            continue
        if a not in activities:
            stats["dropped_activity"] += e - s
            continue
        seg = x[s:e]
        nan_rows = np.isnan(seg).any(axis=1)
        nan_frac = float(nan_rows.mean())
        if _longest_run(nan_rows) > MAX_NAN_RUN:
            stats["dropped_nan"] += e - s
            continue
        if nan_frac > 0:
            seg = (pd.DataFrame(seg).interpolate(limit_direction="both").to_numpy(dtype=np.float32))
        bouts.append(Bout(subject, a, seg, int(s), nan_frac))
    return bouts
