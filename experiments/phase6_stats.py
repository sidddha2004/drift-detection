"""Phase 6 confirmatory statistics (analysis_plan.md Section 5 + Amendment A5): results/phase4_v2/conf_*.json -> results/phase6/confirmatory.md/json.

Unit = seed (post-change macro-F1 averaged over the scenario's variants). Paired Wilcoxon, 10,000-resample bootstrap CI of the mean
paired difference, Holm-Bonferroni over H1, H2, H3, H5 x S1-S3 (12 tests). H4 = non-inferiority (margin 1.0 point).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SRC = ROOT / "results/phase4_v2"
OUT = ROOT / "results/phase6"
E8 = "typed_learned"
FAMILY = {"H1": "frozen_E3b_personal", "H2": "periodic_E3c_full_fed", "H3": "ph_fedprox_E7a", "H5": "ph_head_long"}   # comparators fixed in A5
SCEN = ["S1", "S2", "S3"]
rng = np.random.default_rng(0)


def load():
    per = defaultdict(lambda: defaultdict(list))          # (policy, scenario) -> seed -> [variant values]
    extra = defaultdict(lambda: defaultdict(list))
    for f in sorted(SRC.glob("conf_*.json")):
        for r in json.loads(f.read_text())["runs"]:
            m = r["metrics"]
            per[(r["policy"], r["scenario"])][r["seed"]].append(m["post_change_f1"])
            extra[(r["policy"], r["scenario"])]["up"].append(m["uplink_bytes"])
            extra[(r["policy"], r["scenario"])]["worst"].append(m.get("worst_client_post_f1", np.nan))
    return per, extra


def seedvec(per, pol, scn, seeds):
    return np.array([np.mean(per[(pol, scn)][s]) for s in seeds])


def boot_ci(d, n=10000):
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def test(a, b):
    d = a - b
    try:
        p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = 1.0
    lo, hi = boot_ci(d)
    return d.mean(), lo, hi, p


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def main():
    per, extra = load()
    seeds = sorted({s for k in per for s in per[k]})
    n_var = {scn: len(per[(E8, scn)][seeds[0]]) for scn in SCEN}
    L = ["# Phase 6 confirmatory results (10 evaluation seeds)", "",
         f"Seeds: {seeds}. Unit = seed; post-change macro-F1 on drifted clients averaged over the scenario's variants "
         f"({n_var}). E8 = `{E8}`. Wilcoxon two-sided, bootstrap 10,000, Holm over the 12 family tests.", ""]
    rows, res = [], []
    for h, comp in FAMILY.items():
        for scn in SCEN:
            a, b = seedvec(per, E8, scn, seeds), seedvec(per, comp, scn, seeds)
            d, lo, hi, p = test(a, b)
            rows.append((h, scn, comp, a.mean(), b.mean(), d, lo, hi, p))
    adj = holm(np.array([r[-1] for r in rows]))
    L += ["## Confirmatory family", "", "| H | scenario | comparator | E8 | comparator F1 | diff | 95% CI | p | Holm p | significant |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r, pa in zip(rows, adj):
        sig = pa < 0.05 and (r[6] > 0 or r[7] < 0)
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:+.3f} | [{r[6]:+.3f}, {r[7]:+.3f}] | {r[8]:.4f} | {pa:.4f} | {'yes' if sig else 'no'} |")
        res.append(dict(h=r[0], scenario=r[1], comparator=r[2], diff=r[5], ci=[r[6], r[7]], p=r[8], p_holm=pa, significant=bool(sig)))
    # H4 non-inferiority: full retraining (E5) minus federated adapter must have CI upper bound < 0.01
    L += ["", "## H4 non-inferiority: ph_full_fed_E5 minus ph_adapter_fed (margin +0.010) and uplink", "",
          "| scenario | E5 F1 | adapter F1 | E5 - adapter | 95% CI | non-inferior | uplink E5 (kB) | uplink adapter (kB) | reduction |", "|---|---|---|---|---|---|---|---|---|"]
    h4 = []
    for scn in SCEN:
        a, b = seedvec(per, "ph_full_fed_E5", scn, seeds), seedvec(per, "ph_adapter_fed", scn, seeds)
        d = a - b; lo, hi = boot_ci(d)
        ue, ua = np.mean(extra[("ph_full_fed_E5", scn)]["up"]) / 1e3, np.mean(extra[("ph_adapter_fed", scn)]["up"]) / 1e3
        L.append(f"| {scn} | {a.mean():.3f} | {b.mean():.3f} | {d.mean():+.3f} | [{lo:+.3f}, {hi:+.3f}] | {'yes' if hi < 0.01 else 'no'} | {ue:.0f} | {ua:.0f} | {1 - ua / ue:.0%} |")
        h4.append(dict(scenario=scn, diff=d.mean(), ci=[lo, hi], noninferior=bool(hi < 0.01), uplink_full=ue, uplink_adapter=ua))
    # descriptive table, all systems x scenarios incl. S4
    pols = sorted({p for p, _ in per})
    L += ["", "## All systems, post-change macro-F1 (mean over seeds ± std over seeds), uplink per run", "",
          "| system | " + " | ".join(f"{s} F1" for s in SCEN + ["S4"]) + " | uplink S1 (kB) |", "|---|" + "---|" * 5]
    for p in pols:
        cells = []
        for s in SCEN + ["S4"]:
            v = seedvec(per, p, s, seeds)
            cells.append(f"{v.mean():.3f} ± {v.std():.3f}")
        L.append(f"| {p} | " + " | ".join(cells) + f" | {np.mean(extra[(p, 'S1')]['up']) / 1e3:.0f} |")
    # typing ablation: 4-way vs 2-way oracle, deployable vs oracle mode
    L += ["", "## Typing ablations (descriptive)", "", "| comparison | scenario | diff | 95% CI | p (uncorrected) |", "|---|---|---|---|---|"]
    for name, a_, b_ in (("oracle 4-way - oracle 2-way", "typed_oracle", "mode_typed_oracle"),
                         ("deployable E8 - E8 with true mode", E8, "ph_mode_oracle")):
        for scn in SCEN + ["S4"]:
            d, lo, hi, p = test(seedvec(per, a_, scn, seeds), seedvec(per, b_, scn, seeds))
            L.append(f"| {name} | {scn} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.4f} |")
    (OUT / "confirmatory.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "confirmatory.json").write_text(json.dumps({"family": res, "h4": h4}, default=float, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
