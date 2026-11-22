"""A6: on-device memory budget per system (computed from the real model, adapter and data sizes; no training, no streaming run).

    DRIFT_BENCH=v2 python experiments/phase7_memory_budget.py   -> results/phase7/memory_budget.md / .json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, build_federation  # noqa: E402
from src.drift import ClientStream  # noqa: E402
from src.fl.adapter import AdaptedLSTM  # noqa: E402
from src.fl.engine import TrainConfig, make_model, state_bytes  # noqa: E402
from src.utils.io import load_config  # noqa: E402

PH_STATE = 8 * 8          # PageHinkley: delta, lambda, warm, n, sum, ref, m, mmin (8 numbers, float64)
KB = 1024


def main():
    cfg = load_config(ROOT / "configs/phase1_v2_trimmed.yaml")
    clients, lm, _ = build_federation(DataConfig(**cfg["data"]))
    tc = TrainConfig(**cfg["train"])
    model = make_model(len(lm), 9, tc)
    full = state_bytes(model)
    ad = AdaptedLSTM(model)
    adapter = ad.adapter_bytes()
    in_map = (ad.in_w.numel() + ad.in_b.numel()) * 4
    win = cfg["data"]["window"] * 9 * 4                                  # one FP32 window
    per_round = int(np.median([ClientStream(c, [], 120, 0, passes=2).per_round for c in clients.values()]))
    n_train = [len(c.y["train"]) for c in clients.values()]
    replay = int(np.median(n_train)) * win
    buf = {"align (lookback 8 + episode 8 rounds)": 16, "head_long (45 rounds)": 45, "federated retraining (15 rounds)": 15}
    rows = [("model (FP32, deployed)", full), ("adapter (input map + bottleneck + head)", adapter), ("input map only (align fit)", in_map),
            ("Page-Hinkley detector state", PH_STATE)]
    rows += [(f"labelled arrival buffer: {k}", r * per_round * win) for k, r in buf.items()]
    rows += [("replay pool (client train windows, median client)", replay)]
    rows += [(f"memory cache, K = {k} adapter states", k * adapter) for k in (1, 2, 4)]
    systems = {
        "E3b frozen": full + PH_STATE * 0,
        "E5 / E7a (PH + full fed retraining)": full + PH_STATE + 15 * per_round * win + replay + full,     # + episode-start copy
        "PH + federated adapter": full + PH_STATE + adapter + 15 * per_round * win + replay,
        "E8 (typed: align / head_long)": full + PH_STATE + adapter + 45 * per_round * win + replay,
    }
    systems.update({f"E8 + memory K = {k}": systems["E8 (typed: align / head_long)"] + k * adapter for k in (0, 1, 2, 4)})
    out = ROOT / "results/phase7"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# On-device memory budget (A6)", "",
          f"FP32. One window = 256 x 9 x 4 B = {win / KB:.1f} kB; about {per_round} labelled windows arrive per round; median client "
          f"train pool = {int(np.median(n_train))} windows (range {min(n_train)}-{max(n_train)}).", "",
          "| component | bytes | kB |", "|---|---|---|"] + [f"| {n} | {b:,} | {b / KB:.1f} |" for n, b in rows]
    md += ["", "| system (peak, per client) | kB |", "|---|---|"] + [f"| {n} | {b / KB:.0f} |" for n, b in systems.items()]
    md += ["", "The replay pool dominates every adapting system; the memory cache of E8+memory adds 7.7 kB per entry, under 2% of the total even at K = 4. "
           "The simulator keeps the whole arrival history; a device needs only the buffer listed above."]
    (out / "memory_budget.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "memory_budget.json").write_text(json.dumps({"components": dict(rows), "systems": systems, "per_round": per_round,
                                                        "n_train": n_train}, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
