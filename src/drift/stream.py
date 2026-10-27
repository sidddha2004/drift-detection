"""Per-client drifting stream built on a ClientData (src/data) and the events of the manifest.

Pools (all come from the leak-free Phase 1 split, v2 protocol):
  train : clean pre-drift training data (never drifted: the model is fit before the stream starts)
  val   : ARRIVAL stream. Windows arrive `per_round` at a time in randomised-block order; drift is applied
          according to the schedule. This is what detectors monitor and what adaptation may use (subject to the
          label-supervision model chosen in Phase 3/4).
  test  : held-out EVALUATION pool. At every round the whole pool is transformed with that round's drift strength,
          so post-change macro-F1 is measured on unseen windows with low variance.
"""
from typing import Dict, List, Tuple

import numpy as np

from src.data import ClientData

from .schedule import DriftEvent, events_from_manifest
from .transforms import apply_events

_POOL_ID = {"val": 1, "test": 2}


class ClientStream:
    def __init__(self, client: ClientData, events: List[DriftEvent], n_rounds: int, seed: int, per_round: int = 0):
        self.c, self.events, self.n_rounds, self.seed = client, events, n_rounds, seed
        n_val = client.n("val")
        self.per_round = per_round or max(1, int(np.ceil(n_val / n_rounds)))
        self.order = np.random.default_rng((seed, 31, client.client_id)).permutation(n_val)
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

    def arrival_batch(self, r: int) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.order)
        idx = self.order[(np.arange(self.per_round) + r * self.per_round) % n]
        return self._transform(self.c.x["val"][idx], self.c.y["val"][idx], r, "val")

    def eval_batch(self, r: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._transform(self.c.x["test"], self.c.y["test"], r, "test")


def build_streams(clients: Dict[int, ClientData], manifest: Dict, per_round: int = 0) -> Dict[int, ClientStream]:
    ev = events_from_manifest(manifest)
    return {cid: ClientStream(c, ev[cid], manifest["n_rounds"], manifest["seed"], per_round) for cid, c in clients.items()}
