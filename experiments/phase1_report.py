"""Summarise Phase 1 results -> results/phase1/summary.md (tables used verbatim in the paper draft)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.stats import bootstrap_ci, holm, paired_wilcoxon  # noqa: E402

NAMES = {"E1_local_only": "E1 Local-only", "E2_centralized": "E2 Centralised (single global model)", "E3_fedavg": "E3 FedAvg"}


def ms(v):
    return f"{np.mean(v):.4f} ± {np.std(v):.4f}"


def val_at_selected(r):
    if r["method"] == "E1_local_only":
        return float(np.mean([max(c) for c in r["val_curves"].values()]))
    return float(r["val_curve"][r.get("best_round", r.get("best_epoch"))])


def section(res, heading, desc):
    runs = res["runs"]
    by = defaultdict(list)
    for r in runs:
        by[r["method"]].append(r)
    seeds = sorted({r["seed"] for r in runs})
    L = [f"# {heading}", "", desc, "",
         f"Run `{res['meta']['git_commit']}` on {res['meta']['device']} | seeds {seeds} | "
         f"{len(res['clients'])} clients (subjects {res['clients']}) | split `{res['config']['data']['split_mode']}` | test split only.", "",
         "## Headline (mean over seeds ± std; CI = 95% bootstrap over seeds)", "",
         "| method | test macro-F1 | 95% CI | val macro-F1 (selected ckpt) | worst-client macro-F1 | accuracy | client std |",
         "|---|---|---|---|---|---|---|"]
    for m, rs in by.items():
        f1 = [r["test"]["mean_macro_f1"] for r in rs]
        lo, hi = bootstrap_ci(f1)
        L.append(f"| {NAMES[m]} | {ms(f1)} | [{lo:.4f}, {hi:.4f}] | {ms([val_at_selected(r) for r in rs])} | "
                 f"{ms([r['test']['worst_macro_f1'] for r in rs])} | {ms([r['test']['mean_accuracy'] for r in rs])} | "
                 f"{ms([r['test']['std_macro_f1'] for r in rs])} |")

    L += ["", "## Paired tests across seeds (macro-F1, Wilcoxon, Holm-adjusted)", ""]
    pairs = [("E3_fedavg", "E1_local_only"), ("E2_centralized", "E3_fedavg"), ("E2_centralized", "E1_local_only")]
    raw = {}
    for a, b in pairs:
        fa = [r["test"]["mean_macro_f1"] for r in sorted(by[a], key=lambda r: r["seed"])]
        fb = [r["test"]["mean_macro_f1"] for r in sorted(by[b], key=lambda r: r["seed"])]
        raw[f"{a} vs {b}"] = (paired_wilcoxon(fa, fb), np.mean(fa) - np.mean(fb))
    adj = holm({k: v[0] for k, v in raw.items()})
    L += ["| comparison | Δ macro-F1 | p (raw) | p (Holm) |", "|---|---|---|---|"]
    for k, (p, d) in raw.items():
        L.append(f"| {k} | {d:+.4f} | {p:.4f} | {adj[k]:.4f} |")
    L.append(f"\nWith n={len(seeds)} seeds the smallest attainable two-sided Wilcoxon p is "
             f"{2 / 2 ** len(seeds):.4f}; Holm over 3 tests multiplies it. Treat these tests as indicative only.")

    L += ["", "## Per-client macro-F1 (mean over seeds)", ""]
    cids = res["clients"]
    L += ["| method | " + " | ".join(str(c) for c in cids) + " |", "|---|" + "---|" * len(cids)]
    for m, rs in by.items():
        L.append(f"| {NAMES[m]} | " + " | ".join(
            f"{np.mean([r['test_per_client'][str(c)]['macro_f1'] for r in rs]):.3f}" for c in cids) + " |")

    fa = by["E3_fedavg"]
    L += ["", "## Cost", "",
          "| method | train time / run (s) | comm. total (MB) | comm. to best round (MB) |", "|---|---|---|---|"]
    for m, rs in by.items():
        t = np.mean([r["train_seconds"] for r in rs])
        tot = "0" if m == "E1_local_only" else ("n/a (data pooled)" if m == "E2_centralized"
                                                else f"{np.mean([r['comm_bytes_total'] for r in rs]) / 1e6:.1f}")
        best = f"{np.mean([r['comm_bytes_to_best'] for r in rs]) / 1e6:.1f}" if m == "E3_fedavg" else "-"
        L.append(f"| {NAMES[m]} | {t:.0f} | {tot} | {best} |")
    L.append(f"\nFedAvg full-model payload: {fa[0]['uplink_bytes_per_round'] // len(cids):,} B per client per direction "
             f"(53,260 FP32 parameters, 12-class head).")
    L.append(f"FedAvg selected round (best val): {sorted(r['best_round'] for r in fa)} of "
             f"{res['config']['budget']['fedavg_rounds']}.")

    return L, by


def main():
    out = []
    res_b = json.loads((ROOT / "results/phase1/baselines_blocked.json").read_text())
    Lb, by_b = section(res_b, "Phase 1 results", "## S0 stationary control: blocked split (train/val/test interleaved in time)")
    out += Lb
    tp = ROOT / "results/phase1/baselines_temporal.json"
    if tp.exists():
        res_t = json.loads(tp.read_text())
        Lt, by_t = section(res_t, "Natural-drift probe", "Temporal split: train = first part of each activity bout, test = last part. No injected drift: any loss vs the blocked control is natural within-activity temporal shift.")
        out += [""] + [l.replace("# ", "## ", 1) if l.startswith("# ") else l for l in Lt]
        out += ["", "### Natural-shift gap (blocked − temporal, test macro-F1)", "", "| method | blocked | temporal | gap |", "|---|---|---|---|"]
        for m in by_b:
            a = np.mean([r["test"]["mean_macro_f1"] for r in by_b[m]]); b = np.mean([r["test"]["mean_macro_f1"] for r in by_t[m]])
            out.append(f"| {NAMES[m]} | {a:.4f} | {b:.4f} | {a - b:+.4f} |")
    L = out
    cc = ROOT / "results/phase1/channel_check.json"
    if cc.exists():
        rows = json.loads(cc.read_text())["rows"]
        L += ["", "## Audit A14: effect of the wrong channel indices (E1, 30 epochs, 3 seeds)", "",
              "| channels | macro-F1 | accuracy |", "|---|---|---|"]
        for ch, d in [("acc16", "correct ±16g accelerometer (9)"), ("legacy", "old code: temperature + 2 acc axes per IMU (9)"),
                      ("temp", "temperature sensors only (3)")]:
            rr = [r for r in rows if r["channels"] == ch]
            L.append(f"| {d} | {ms([r['mean_macro_f1'] for r in rr])} | {ms([r['mean_accuracy'] for r in rr])} |")
    (ROOT / "results/phase1/summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
