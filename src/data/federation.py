"""Build per-client (= per-subject) train/val/test sets without leakage.

Protocol
--------
* Windows are cut inside bouts only (see pamap2.py): a window has exactly one label.
* Inside every bout windows are split 60/15/25 (train/val/test) with a purge of ceil(W/S)-1 windows
  between segments of different splits, so no two windows from different splits share a sample.
  Every activity a subject performed therefore appears in all three splits.
  Two modes: "blocked" (default, the stationary control: contiguous blocks of `block_windows` windows
  are assigned to splits at random, so train/val/test cover the same time span) and "temporal"
  (train = first part of each bout, test = last part; measures natural within-activity drift).
* Normalisation statistics are fit on the client's TRAIN samples only, then frozen and applied
  to val/test (audit A6).
"""
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .pamap2 import ACC16_COLS, PROTOCOL_ACTIVITIES, Bout, load_bouts

SPLITS = ("train", "val", "test")


@dataclass
class DataConfig:
    root: str = "PAMAP2_Dataset/Protocol"
    subjects: Sequence[int] = tuple(range(101, 109))   # 109 has a single 64 s rope-jumping bout: excluded (D9)
    channels: str = "acc16"            # acc16 | legacy | temp   (legacy/temp only for the A14 check)
    window: int = 256
    stride: int = 128
    train_frac: float = 0.60
    val_frac: float = 0.15
    min_windows_per_bout: int = 10
    split_mode: str = "blocked"        # blocked | temporal
    block_windows: int = 16            # blocked mode: windows per contiguous block
    split_seed: int = 0                # fixes the block->split assignment (independent of the model seed)
    activities: Sequence[int] = tuple(PROTOCOL_ACTIVITIES)
    trim_samples: int = 0              # samples cut from BOTH ends of every bout before windowing (D13; 0 = frozen v1 protocol)


@dataclass
class ClientData:
    client_id: int
    x: Dict[str, np.ndarray]           # split -> (n, W, C) float32, normalised
    y: Dict[str, np.ndarray]           # split -> (n,) int64 (global label index)
    mean: np.ndarray                   # (C,) train statistics
    std: np.ndarray
    # provenance (for leak tests and the audit): split -> (n, 3) [bout_id, start_sample, activity_id]
    prov: Dict[str, np.ndarray] = field(default_factory=dict)
    audit: Dict[str, float] = field(default_factory=dict)

    def n(self, split: str) -> int:
        return len(self.y[split])


def _channel_cols(name: str) -> List[int]:
    from .pamap2 import LEGACY_COLS, TEMP_COLS
    return {"acc16": ACC16_COLS, "legacy": LEGACY_COLS, "temp": TEMP_COLS}[name]


def window_starts(n_samples: int, window: int, stride: int) -> np.ndarray:
    return np.arange(0, n_samples - window + 1, stride)


def split_bound(n: int, cfg: DataConfig):
    """Return (train_slice, val_slice, test_slice) over window indices with purges."""
    purge = math.ceil(cfg.window / cfg.stride) - 1
    n_tr = int(n * cfg.train_frac)
    n_va = int(n * cfg.val_frac)
    tr = (0, n_tr)
    va = (n_tr + purge, n_tr + purge + n_va)
    te = (va[1] + purge, n)
    return tr, va, te


def assign_splits(n: int, cfg: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """Per-window split code (0 train, 1 val, 2 test, -1 purged) for one bout of n windows."""
    purge = math.ceil(cfg.window / cfg.stride) - 1
    code = np.full(n, -1, dtype=np.int8)
    n_chunks = math.ceil(n / cfg.block_windows)
    if cfg.split_mode == "temporal" or n_chunks < 3:
        for k, (lo, hi) in enumerate(split_bound(n, cfg)):
            code[lo:hi] = k
        return code
    n_va = max(1, round(cfg.val_frac * n_chunks))
    n_te = max(1, round((1 - cfg.train_frac - cfg.val_frac) * n_chunks))
    chunk_split = np.array([1] * n_va + [2] * n_te + [0] * (n_chunks - n_va - n_te))
    chunk_split = rng.permutation(chunk_split)
    for k in range(n_chunks):
        lo, hi = k * cfg.block_windows, min(n, (k + 1) * cfg.block_windows)
        code[lo:hi] = chunk_split[k]
        if k > 0 and chunk_split[k] != chunk_split[k - 1]:
            code[lo:lo + purge] = -1
    return code


def build_federation(cfg: DataConfig, project_root: Optional[Path] = None):
    """Returns (clients: {client_id: ClientData}, label_map: {activity_id: index}, audit: dict)."""
    root = Path(project_root or Path(__file__).resolve().parents[2]) / cfg.root
    cols = _channel_cols(cfg.channels)
    label_map = {a: i for i, a in enumerate(sorted(cfg.activities))}
    clients: Dict[int, ClientData] = {}
    audit: Dict[int, dict] = {}

    for subj in cfg.subjects:
        path = root / f"subject{subj}.dat"
        if not path.exists():
            continue
        raw_stats: dict = {}
        bouts = load_bouts(path, subj, cols, cfg.activities, raw_stats)
        buckets = {s: {"x": [], "y": [], "p": []} for s in SPLITS}
        train_samples: List[np.ndarray] = []
        dropped_short = 0

        for bid, b in enumerate(bouts):
            t = cfg.trim_samples
            data = b.data[t:len(b.data) - t] if t else b.data
            starts = window_starts(len(data), cfg.window, cfg.stride)
            if len(starts) < cfg.min_windows_per_bout:
                dropped_short += 1
                continue
            code = assign_splits(len(starts), cfg, np.random.default_rng((cfg.split_seed, subj, bid)))
            for k, split in enumerate(SPLITS):
                st = starts[code == k]
                if len(st) == 0:
                    continue
                w = np.stack([data[s:s + cfg.window] for s in st])
                buckets[split]["x"].append(w)
                buckets[split]["y"].append(np.full(len(st), label_map[b.activity], dtype=np.int64))
                buckets[split]["p"].append(np.stack(
                    [np.full(len(st), bid), st + t, np.full(len(st), b.activity)], axis=1))
                if split == "train":
                    # union of sample ranges, so overlapping windows are not double counted
                    mask = np.zeros(len(data), dtype=bool)
                    for s0 in st:
                        mask[s0:s0 + cfg.window] = True
                    train_samples.append(data[mask])

        if not train_samples:
            continue
        flat = np.concatenate(train_samples)
        mean = flat.mean(axis=0).astype(np.float32)
        std = (flat.std(axis=0) + 1e-6).astype(np.float32)

        x = {s: ((np.concatenate(buckets[s]["x"]) - mean) / std).astype(np.float32) for s in SPLITS}
        y = {s: np.concatenate(buckets[s]["y"]) for s in SPLITS}
        prov = {s: np.concatenate(buckets[s]["p"]) for s in SPLITS}
        clients[subj] = ClientData(
            client_id=subj, x=x, y=y, mean=mean, std=std, prov=prov,
            audit=dict(raw_stats, n_bouts=len(bouts), dropped_short_bouts=dropped_short,
                       **{f"n_{s}": int(len(y[s])) for s in SPLITS}))
        audit[subj] = clients[subj].audit
    return clients, label_map, audit
