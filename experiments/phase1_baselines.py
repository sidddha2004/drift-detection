"""Phase 1: stationary baselines E1 (local-only), E2 (centralised), E3 (FedAvg) over all seeds.

    python experiments/phase1_baselines.py --config configs/phase1_baselines.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, build_federation  # noqa: E402
from src.fl.baselines import run_centralized, run_fedavg, run_local_only  # noqa: E402
from src.fl.engine import TrainConfig  # noqa: E402
from src.utils.io import load_config, run_metadata, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/phase1_baselines.yaml"))
    ap.add_argument("--split-mode", choices=["blocked", "temporal"], default=None)
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply epochs/rounds (budget sensitivity)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.split_mode:
        cfg["data"]["split_mode"] = args.split_mode
    mode = cfg["data"]["split_mode"]
    tag = f"{mode}" + (f"_x{args.budget_scale:g}" if args.budget_scale != 1 else "")
    out = args.out or str(ROOT / f"results/phase1/baselines_{tag}.json")
    dc = DataConfig(**cfg["data"])
    tc = TrainConfig(**cfg["train"])
    clients, label_map, _ = build_federation(dc)
    n_classes, n_ch = len(label_map), 9
    b = {k: int(v * args.budget_scale) if k != "fedavg_local_epochs" else v for k, v in cfg["budget"].items()}
    cfg["budget"] = b
    runs = []
    for seed in args.seeds or cfg["seeds"]:
        for fn, kw in [
            (run_local_only, dict(epochs=b["total_epochs"])),
            (run_centralized, dict(epochs=b["total_epochs"])),
            (run_fedavg, dict(rounds=b["fedavg_rounds"], local_epochs=b["fedavg_local_epochs"])),
        ]:
            r = fn(clients, n_classes, n_ch, tc, seed=seed, **kw)
            runs.append(r)
            t = r["test"]
            print(f"seed {seed:<5} {r['method']:<16} macroF1 {t['mean_macro_f1']:.4f}  worst {t['worst_macro_f1']:.4f}"
                  f"  acc {t['mean_accuracy']:.4f}  [{r['train_seconds']:.0f}s]", flush=True)
    save_json({"meta": run_metadata(), "config": cfg, "label_map": label_map,
               "clients": sorted(clients), "runs": runs}, out)
    print("saved", out)


if __name__ == "__main__":
    main()
