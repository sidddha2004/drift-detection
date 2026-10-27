"""Phase 1c: FedAvg + local fine-tuning (E3b), the personalisation baseline the proposed system must beat.

Runs FedAvg at the locked budget with the same seeds as Phase 1, then fine-tunes the selected global model per
client (<= 20 epochs, best val macro-F1). The FedAvg part must reproduce the frozen E3 numbers exactly.
    python experiments/phase1c_fedavg_ft.py --split-mode blocked
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, build_federation  # noqa: E402
from src.fl.baselines import run_fedavg  # noqa: E402
from src.fl.engine import TrainConfig  # noqa: E402
from src.utils.io import load_config, run_metadata, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-mode", choices=["blocked", "temporal"], default="blocked")
    ap.add_argument("--finetune-epochs", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(ROOT / "configs/phase1_baselines.yaml")
    cfg["data"]["split_mode"] = args.split_mode
    clients, lm, _ = build_federation(DataConfig(**cfg["data"]))
    tc, b = TrainConfig(**cfg["train"]), cfg["budget"]
    runs = []
    for seed in args.seeds or cfg["seeds"]:
        r = run_fedavg(clients, len(lm), 9, tc, b["fedavg_rounds"], b["fedavg_local_epochs"], seed,
                       finetune_epochs=args.finetune_epochs)
        ft = r.pop("ft")
        runs += [r, ft]
        print(f"seed {seed:<5} E3 {r['test']['mean_macro_f1']:.4f} (worst {r['test']['worst_macro_f1']:.4f})   "
              f"E3b {ft['test']['mean_macro_f1']:.4f} (worst {ft['test']['worst_macro_f1']:.4f})  "
              f"ft epochs {sorted(ft['finetune_best_epoch'].values())}", flush=True)
    save_json({"meta": run_metadata(), "config": cfg, "finetune_epochs": args.finetune_epochs,
               "label_map": lm, "clients": sorted(clients), "runs": runs},
              ROOT / f"results/phase1/fedavg_ft_{args.split_mode}.json")


if __name__ == "__main__":
    main()
