"""Per-client drifting stream built on a ClientData (src/data) and the events of the manifest.

Pools (all come from the leak-free Phase 1 split, v2 protocol):
  train : clean pre-drift training data (never drifted: the model is fit before the stream starts)
  val   : split by contiguous blocks of `block` windows (alternating per class) into
          cal     clean, never drifted: pre-deployment calibration/reference data for detectors, and
          arrival  the ARRIVAL stream: blocks come in random order, windows inside a block stay in time order, so the
                  stream has realistic activity bouts. `per_round` windows arrive per round; drift is applied per the
                  schedule. This is what detectors monitor and what adaptation may use (subject to the label-supervision
                  model chosen in Phase 3/4).
  test  : held-out EVALUATION pool. At every round the whole pool is transformed with that round's drift strength,
          so post-change macro-F1 is measured on unseen windows with low variance.
"""
from typing import Dict, List, Tuple

import numpy as np

from src.data import ClientData

from .schedule import DriftEvent, events_from_manifest
from .transforms import apply_events

_POOL_ID = {"val": 1, "test": 2}


def split_val_blocks(y_val: np.ndarray, block: int, rng: np.random.Generator):
    """Contiguous runs of equal label are cut into blocks of <= `block` windows; blocks of each class alternate
    between calibration and arrival (a class with a single block goes to arrival). Returns (cal_idx, arrival_blocks)."""
    blocks, start = [], 0
    for i in range(1, len(y_val) + 1):
        if i == len(y_val) or y_val[i] != y_val[start]:
            for b in range(start, i, block):
                blocks.append((int(y_val[start]), np.arange(b, min(i, b + block))))
            start = i
    cal, arr, seen = [], [], {}
    flip = {int(k): int(rng.integers(0, 2)) for k in np.unique(y_val)}
    for k, idx in blocks:
        seen[k] = seen.get(k, 0) + 1
        n_k = sum(1 for kk, _ in blocks if kk == k)
        (arr if n_k == 1 or (seen[k] + flip[k]) % 2 == 0 else cal).append(idx)
    cal_idx = np.concatenate(cal) if cal else np.array([], dtype=int)
    return cal_idx, arr


class ClientStream:
    def __init__(self, client: ClientData, events: List[DriftEvent], n_rounds: int, seed: int, per_round: int = 0,
                 block: int = 16, passes: int = 1):
        self.c, self.events, self.n_rounds, self.seed = client, events, n_rounds, seed
        rng = np.random.default_rng((seed, 31, client.client_id))
        self.cal_idx, blocks = split_val_blocks(client.y["val"], block, rng)
        # `passes` > 1 (benchmark v2): the arrival pool is replayed in a new random block order for every pass
        self.order = np.concatenate([blocks[i] for _ in range(passes) for i in rng.permutation(len(blocks))])
        self.per_round = per_round or max(1, len(self.order) // n_rounds)      # floor: every arrival window is used at most once
        self._alpha = [e.alpha(n_rounds) for e in events]

    def alphas(self, r: int) -> List[float]:
        return [float(a[r]) for a in self._alpha]

    def drift_active(self, r: int) -> bool:
        return any(a > 0 for a in self.alphas(r))

    def _transform(self, x, y, r, pool):
        rng = np.random.default_rng((self.seed, 41, self.c.client_id, _POOL_ID[pool], r))
        return apply_events(x, y, self.c.mean, self.c.std, self.events, self.alphas(r), rng)

    def train_pool(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.c.x["train"], self.c.y["train"]

    def cal_pool(self) -> Tuple[np.ndarray, np.ndarray]:
        """Clean labelled calibration windows (never drifted)."""
        return self.c.x["val"][self.cal_idx], self.c.y["val"][self.cal_idx]

    def arrival_batch(self, r: int) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.order)
        idx = self.order[(np.arange(self.per_round) + r * self.per_round) % n]
        return self._transform(self.c.x["val"][idx], self.c.y["val"][idx], r, "val")

    def eval_batch(self, r: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._transform(self.c.x["test"], self.c.y["test"], r, "test")


def build_streams(clients: Dict[int, ClientData], manifest: Dict, per_round: int = 0) -> Dict[int, ClientStream]:
    ev = events_from_manifest(manifest)
    return {cid: ClientStream(c, ev[cid], manifest["n_rounds"], manifest["seed"], per_round, passes=manifest.get("passes", 1))
            for cid, c in clients.items()}
