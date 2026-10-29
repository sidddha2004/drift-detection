"""Audit A14: quantify the effect of the wrong channel indices used before Phase 1.

Trains E1 (local-only, same split/budget) on three channel sets:
  acc16  - the correct 9 +-16g accelerometer axes
  legacy - what the old code used: temperature + 2 acc axes per IMU
  temp   - IMU temperature sensors only (3 channels; carries no motion information)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, build_federation  # noqa: E402
from src.fl.baselines import run_local_only  # noqa: E402
from src.fl.engine import TrainConfig  # noqa: E402
from src.utils.io import load_config, run_metadata, save_json  # noqa: E402


def main():
    cfg = load_config(ROOT / "configs/phase1_baselines.yaml")
    tc = TrainConfig(**cfg["train"])
    rows = []
    for ch in ["acc16", "legacy", "temp"]:
        dc = DataConfig(**dict(cfg["data"], channels=ch))
        clients, lm, _ = build_federation(dc)
        n_ch = next(iter(clients.values())).x["train"].shape[-1]
        for seed in cfg["seeds"][:3]:
            r = run_local_only(clients, len(lm), n_ch, tc, epochs=30, seed=seed)
            rows.append({"channels": ch, "seed": seed, **r["test"]})
            print(f"{ch:<7} seed {seed:<5} macroF1 {r['test']['mean_macro_f1']:.4f}  acc {r['test']['mean_accuracy']:.4f}",
                  flush=True)
    save_json({"meta": run_metadata(), "rows": rows}, ROOT / "results/phase1/channel_check.json")


if __name__ == "__main__":
    main()
