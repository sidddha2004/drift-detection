"""A6 M2 (descriptive): post-change F1 vs (epsilon, delta=1e-5) vs uplink for the two federated responders under DP.

    python experiments/phase7_dp_tradeoff.py   -> results/phase7/dp_tradeoff.md / .json, docs/paper/figs/dp_tradeoff.{pdf,png}

z = 0 rows come from the confirmatory runs (results/phase4_v2/conf_*.json, same 10 evaluation seeds); z > 0 from dp_eval_*.json.
Epsilon is per drifted client over the whole stream (its number of uploads), then averaged over drifted clients and seeds.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fl.dp import epsilon  # noqa: E402

RES = ROOT / "results/phase4_v2"
SYSTEMS = {"ph_adapter_fed": "PH-FedAdapter", "ph_full_fed_E5": "PH-FedAvg"}
REFS = {"typed_learned": "TLA (no upload)", "frozen_E3b_personal": "Frozen-Personal"}
ZS = [0.0, 0.5, 1.0, 2.0, 4.0]
SCEN = ["S1", "S2", "S3"]


def _runs(pattern):
    out = []
    for f in sorted(RES.glob(pattern)):
        out += json.load(open(f))["runs"]
    return out


def split_policy(p):
    for s in SYSTEMS:
        if p == s:
            return s, 0.0
        if p.startswith(s + "_dp_z"):
            return s, float(p.split("_dp_z")[1])
    return None, None


def collect():
    conf = [r for r in _runs("conf_*.json") if r["scenario"] in SCEN]
    dp = _runs("dp_eval_*.json")
    cell = defaultdict(lambda: defaultdict(list))        # (system, z) -> scenario -> list of per-run dicts
    for r in conf + dp:
        s, z = split_policy(r["policy"])
        if s is None:
            continue
        drifted = [str(c) for c in r["drifted"]]
        nup = [int(r["n_uploads"].get(c, 0)) for c in drifted] if r.get("n_uploads") else None
        cell[(s, z)][r["scenario"]].append({
            "seed": r["seed"], "variant": r["variant"], "f1": r["metrics"]["post_change_f1"],
            "clean": r["metrics"]["clean_clients_post_f1"], "up": r["metrics"]["uplink_bytes"],
            "eps": float(np.mean([epsilon(n, z) for n in nup])) if (nup and z > 0) else (np.inf if z == 0 else np.nan),
            "uploads": float(np.mean(nup)) if nup else np.nan})
    refs = defaultdict(lambda: defaultdict(list))
    for r in conf:
        if r["policy"] in REFS:
            refs[r["policy"]][r["scenario"]].append(r["metrics"]["post_change_f1"])
    return cell, refs


def summarise(cell, refs):
    rows = []
    for s in SYSTEMS:
        for z in ZS:
            for sc in SCEN + ["all"]:
                xs = [x for k in SCEN for x in cell[(s, z)][k]] if sc == "all" else cell[(s, z)][sc]
                if not xs:
                    continue
                # seed-level means first (each seed weighs equally), then mean and sd across seeds
                by_seed = defaultdict(list)
                for x in xs:
                    by_seed[x["seed"]].append(x["f1"])
                seed_f1 = np.array([np.mean(v) for v in by_seed.values()])
                rows.append({"system": s, "z": z, "scenario": sc, "n_runs": len(xs), "n_seeds": len(by_seed),
                             "f1": float(seed_f1.mean()), "f1_sd": float(seed_f1.std(ddof=1)) if len(seed_f1) > 1 else 0.0,
                             "clean_f1": float(np.mean([x["clean"] for x in xs])),
                             "uplink_kB": float(np.mean([x["up"] for x in xs]) / 1024),
                             "uploads": float(np.mean(u)) if (u := [x["uploads"] for x in xs if np.isfinite(x["uploads"])]) else float("nan"),
                             "eps": float(np.nanmean([x["eps"] for x in xs])) if z > 0 else float("inf")})
    ref_rows = {p: {sc: float(np.mean(v)) for sc, v in d.items()} | {"all": float(np.mean([x for v in d.values() for x in v]))}
                for p, d in refs.items()}
    return rows, ref_rows


def figure(rows, ref_rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    colors = {"ph_adapter_fed": "#1f77b4", "ph_full_fed_E5": "#d62728"}
    for ax, sc in zip(axes, SCEN):
        for s, lab in SYSTEMS.items():
            pts = sorted([r for r in rows if r["system"] == s and r["scenario"] == sc], key=lambda r: -r["z"])
            xs = [r["eps"] if np.isfinite(r["eps"]) else 300 for r in pts]
            ys = [r["f1"] for r in pts]
            es = [r["f1_sd"] for r in pts]
            ax.errorbar(xs, ys, yerr=es, marker="o", color=colors[s], label=lab, capsize=2, lw=1.4)
        for p, ls in zip(REFS, ["--", ":"]):
            ax.axhline(ref_rows[p][sc], color="k", ls=ls, lw=1, label=REFS[p])
        ax.set_xscale("log")
        ax.set_xticks([3, 10, 30, 100, 300])
        ax.set_xticklabels(["3", "10", "30", "100", "no DP"])
        ax.set_title(sc)
        ax.set_xlabel(r"$\varepsilon$ per drifted client ($\delta=10^{-5}$)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("post-change macro-F1 (drifted clients)")
    axes[0].legend(fontsize=7, loc="center left")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "dp_tradeoff.pdf")
    fig.savefig(out / "dp_tradeoff.png", dpi=150)
    plt.close(fig)


def main():
    cell, refs = collect()
    rows, ref_rows = summarise(cell, refs)
    out = ROOT / "results/phase7"
    (out / "dp_tradeoff.json").write_text(json.dumps({"rows": rows, "refs": ref_rows}, indent=1, default=str), encoding="utf-8")
    md = ["# DP trade-off (A6 M2, descriptive; 10 evaluation seeds; S1-S3)", "",
          "F1 = post-change macro-F1 of drifted clients, seed-level mean (sd across seeds). eps per drifted client, delta = 1e-5.", "",
          "| system | z | scenario | F1 (sd) | clean F1 | eps | uploads / drifted client | uplink kB |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        eps = "inf" if not np.isfinite(r["eps"]) else f"{r['eps']:.1f}"
        md.append(f"| {r['system']} | {r['z']:g} | {r['scenario']} | {r['f1']:.3f} ({r['f1_sd']:.3f}) | {r['clean_f1']:.3f} | {eps} | "
                  + (f"{r['uploads']:.1f}" if np.isfinite(r["uploads"]) else "-") + f" | {r['uplink_kB']:.0f} |")
    md += ["", "| reference (no upload, eps = 0) | " + " | ".join(SCEN + ["all"]) + " |", "|---|" + "---|" * 4]
    md += [f"| {REFS[p]} | " + " | ".join(f"{ref_rows[p][sc]:.3f}" for sc in SCEN + ["all"]) + " |" for p in REFS]
    (out / "dp_tradeoff.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    figure(rows, ref_rows, ROOT / "docs/paper/figs")
    print("\n".join(md))


if __name__ == "__main__":
    main()
