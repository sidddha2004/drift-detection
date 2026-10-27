"""Phase 1d: compare protocol v1 (frozen, untrimmed) with v2 (5 s edge trim). Rule fixed in advance (decisions.md, D13):
a ranking change = for a pair of methods the sign of the mean paired difference differs between protocols AND the
95% bootstrap CI (over seeds) excludes 0 in at least one protocol. Ties that merely reshuffle are not changes.
Output: results/phase1d/comparison.md (+ json)
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.stats import bootstrap_ci  # noqa: E402

NAMES = {"E1_local_only": "E1 local-only", "E2_centralized": "E2 centralised", "E3_fedavg": "E3 FedAvg",
         "E3b_fedavg_ft": "E3b FedAvg+FT"}


def load(base, mode):
    """-> {method: {seed: (macro_f1, worst_client_f1)}}"""
    out = {}
    for name in (f"baselines_{mode}.json", f"fedavg_ft_{mode}.json"):
        p = ROOT / base / name
        if not p.exists():
            return None
        for r in json.loads(p.read_text())["runs"]:
            if r["method"] in NAMES:
                out.setdefault(r["method"], {})[r["seed"]] = (r["test"]["mean_macro_f1"], r["test"]["worst_macro_f1"])
    return out


def main():
    L, res = ["# Phase 1d: v1 (frozen, untrimmed) vs v2 (5 s edge trim)", ""], {}
    for mode in ("blocked", "temporal"):
        v1, v2 = load("results/phase1", mode), load("results/phase1d", mode)
        if v1 is None or v2 is None:
            L += [f"## {mode}: v2 results not available yet", ""]
            continue
        seeds = sorted(set(v1["E1_local_only"]) & set(v2["E1_local_only"]))
        L += [f"## {mode} split ({len(seeds)} seeds)", "", "| method | v1 macro-F1 | v2 macro-F1 | shift | v1 worst-client | v2 worst-client |",
              "|---|---|---|---|---|---|"]
        for m in NAMES:
            a = np.array([v1[m][s][0] for s in seeds]); b = np.array([v2[m][s][0] for s in seeds])
            wa = np.mean([v1[m][s][1] for s in seeds]); wb = np.mean([v2[m][s][1] for s in seeds])
            L.append(f"| {NAMES[m]} | {a.mean():.4f} ± {a.std():.4f} | {b.mean():.4f} ± {b.std():.4f} | {b.mean() - a.mean():+.4f} | {wa:.3f} | {wb:.3f} |")
        L += ["", "| pair (A − B) | v1 diff [95% CI] | v2 diff [95% CI] | ranking change? |", "|---|---|---|---|"]
        changes = []
        for A, B in itertools.combinations(NAMES, 2):
            row = []
            for v in (v1, v2):
                d = np.array([v[A][s][0] - v[B][s][0] for s in seeds])
                lo, hi = bootstrap_ci(d)
                row.append((d.mean(), lo, hi))
            sig = any(lo > 0 or hi < 0 for _, lo, hi in row)
            flip = np.sign(row[0][0]) != np.sign(row[1][0])
            ch = bool(sig and flip)
            if ch:
                changes.append((A, B))
            L.append(f"| {NAMES[A]} − {NAMES[B]} | {row[0][0]:+.4f} [{row[0][1]:+.4f}, {row[0][2]:+.4f}] | "
                     f"{row[1][0]:+.4f} [{row[1][1]:+.4f}, {row[1][2]:+.4f}] | {'**YES**' if ch else 'no'} |")
        order = lambda v: " > ".join(NAMES[m] for m in sorted(NAMES, key=lambda m: -np.mean([v[m][s][0] for s in seeds])))
        L += ["", f"Order by mean, v1: {order(v1)}", f"Order by mean, v2: {order(v2)}", "",
              f"**Ranking changes under the pre-set rule: {len(changes)}** {changes}", ""]
        res[mode] = {"changes": changes}
    out = ROOT / "results/phase1d"
    (out / "comparison.md").write_text("\n".join(L), encoding="utf-8")
    (out / "comparison.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
