"""Phase 1: data audit tables (windows per client/class/split, dropped data) -> results/phase1/data_audit.{json,md}."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ACTIVITY_NAMES, DataConfig, build_federation  # noqa: E402
from src.utils.io import load_config, save_json  # noqa: E402


def main():
    cfg = load_config(ROOT / "configs/phase1_baselines.yaml")
    dc = DataConfig(**cfg["data"])
    clients, lm, audit = build_federation(dc)
    inv = {v: k for k, v in lm.items()}
    classes = sorted(inv)
    lines = ["# Phase 1 data audit", "",
             f"Window {dc.window} / stride {dc.stride}, split {dc.train_frac:.0%}/{dc.val_frac:.0%}/"
             f"{1 - dc.train_frac - dc.val_frac:.0%} (minus purge), channels `{dc.channels}`.", "",
             "## Windows per client and split", "",
             "| client | bouts | train | val | test | classes | dropped rows (long NaN) | dropped short bouts |",
             "|---|---|---|---|---|---|---|---|"]
    for cid, c in clients.items():
        a = audit[cid]
        lines.append(f"| {cid} | {a['n_bouts']} | {a['n_train']} | {a['n_val']} | {a['n_test']} | "
                     f"{len(set(c.y['train'].tolist()))} | {a['dropped_nan']} | {a['dropped_short_bouts']} |")
    tot = {s: sum(c.n(s) for c in clients.values()) for s in ("train", "val", "test")}
    lines.append(f"| **total** | | {tot['train']} | {tot['val']} | {tot['test']} | | | |")
    lines += ["", "## Train windows per client and class", "",
              "| client | " + " | ".join(ACTIVITY_NAMES[inv[k]] for k in classes) + " |",
              "|---|" + "---|" * len(classes)]
    for cid, c in clients.items():
        cnt = np.bincount(c.y["train"], minlength=len(classes))
        lines.append(f"| {cid} | " + " | ".join(str(int(v)) for v in cnt) + " |")
    glob = sum(np.bincount(c.y["train"], minlength=len(classes)) for c in clients.values())
    lines.append("| **all** | " + " | ".join(f"**{int(v)}**" for v in glob) + " |")
    lines += ["", f"Imbalance ratio (max/min class, all clients): {glob.max() / max(glob.min(), 1):.1f}"]
    out = ROOT / "results/phase1"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")
    save_json({"audit": audit, "totals": tot, "class_train_counts": glob}, out / "data_audit.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
