"""Compact column-width figures for the 6-page IJCNN version (supervisor feedback, 2026-11-24).

Cached results only (no training). Text is 8-9 pt at the printed size (figures are made at 3.5 in, the IEEE column width).
    python experiments/paper_figs/ijcnn_figs.py   -> docs/paper_ieee/figs/*.pdf (+ .png)
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.fl.dp import epsilon  # noqa: E402

RES = ROOT / "results/phase4_v2"
OUT = ROOT / "docs/paper_ieee/figs"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "pdf.fonttype": 42})
W = 3.5

NAME = {"frozen_E3_global": "Global (no adaptation)", "frozen_E3b_personal": "Personalized (no adaptation)",
        "periodic_E3c_full_fed": "Periodic full retraining", "adwin_full_fed_E4": "ADWIN + full retraining",
        "ph_full_fed_E5": "PH + full retraining", "ph_fedprox_E7a": "PH + FedProx retraining",
        "ph_full_local": "PH + local full retraining", "ph_adapter_fed": "PH + federated adapter",
        "ph_align": "Input alignment (fixed)", "ph_head_long": "Head retraining (fixed)", "typed_learned": "TLA (typed)",
        "typed_memory_K0_t02": "TLA + memory", "ph_mode_oracle": "TLA, true type", "mode_typed_oracle": "Oracle 2-way",
        "typed_oracle": "Oracle 4-way"}
COL = {"frozen_E3b_personal": "#6b7280", "periodic_E3c_full_fed": "#d97706", "ph_full_fed_E5": "#b91c1c", "ph_fedprox_E7a": "#9333ea",
       "ph_adapter_fed": "#2563eb", "ph_head_long": "#0891b2", "ph_align": "#65a30d", "typed_learned": "#16a34a",
       "typed_memory_K0_t02": "#15803d", "typed_oracle": "#111827"}


def load(pattern):
    runs = []
    for f in sorted(RES.glob(pattern)):
        runs += json.load(open(f))["runs"]
    return runs


def seed_mean(runs, pol, metric, scen, variant=None):
    by = defaultdict(list)
    for r in runs:
        if r["policy"] == pol and r["scenario"] in scen and (variant is None or r["variant"] == variant):
            v = r["metrics"][metric]
            if v == v:
                by[r["seed"]].append(v)
    vals = np.array([np.mean(v) for v in by.values()])
    return (float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0) if len(vals) else (np.nan, np.nan)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def fig_cost(conf):
    pols = ["frozen_E3b_personal", "periodic_E3c_full_fed", "ph_full_fed_E5", "ph_fedprox_E7a", "ph_adapter_fed", "ph_head_long", "typed_learned"]
    off = {"frozen_E3b_personal": (5, 1), "ph_head_long": (5, -3), "typed_learned": (5, -3), "ph_adapter_fed": (-5, 2),
           "ph_full_fed_E5": (5, -13), "ph_fedprox_E7a": (0, 6), "periodic_E3c_full_fed": (5, -2)}
    fig, ax = plt.subplots(figsize=(W, 2.35))
    for p in pols:
        f, _ = seed_mean(conf, p, "post_change_f1", ("S1", "S2", "S3"))
        u = np.mean([r["metrics"]["uplink_bytes"] / 1e3 for r in conf if r["policy"] == p and r["scenario"] in ("S1", "S2", "S3")])
        ax.scatter(u + 1, f, color=COL[p], s=28, zorder=3)
        dx, dy = off[p]
        text = NAME[p].replace(" full retraining", "\nfull retraining") if p in ("ph_full_fed_E5", "periodic_E3c_full_fed") else NAME[p]
        ax.annotate(text, (u + 1, f), xytext=(dx, dy), textcoords="offset points", fontsize=7,
                    ha="center" if dx == 0 else ("right" if dx < 0 else "left"))
    ax.set_xscale("log"); ax.set_xlim(0.5, 3e6); ax.set_ylim(0.675, 0.752)
    ax.set_xlabel("uplink per run (kB, log scale; local systems at 1)"); ax.set_ylabel("post-change macro-F1")
    ax.grid(alpha=0.3)
    save(fig, "cost_vs_uplink")


def fig_mistype(conf):
    pols = ["frozen_E3b_personal", "ph_align", "ph_head_long"]
    groups = [("S1", "real_partial"), ("S2", "real_partial"), ("S3", "real_partial")]
    fig, ax = plt.subplots(figsize=(W, 1.9))
    w = 0.26
    for i, p in enumerate(pols):
        vals = [seed_mean(conf, p, "post_change_f1", (s,), v) for s, v in groups]
        ax.bar(np.arange(3) + (i - 1) * w, [m for m, _ in vals], w, yerr=[e for _, e in vals], capsize=2, color=COL[p], label=NAME[p])
    ax.set_xticks(range(3)); ax.set_xticklabels(["S1 real", "S2 real", "S3 real"]); ax.set_ylim(0.3, 0.95)
    ax.set_ylabel("post-change macro-F1"); ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=6.2, columnspacing=0.8, handlelength=1.2); ax.grid(axis="y", alpha=0.3)
    save(fig, "mistype_cost")


def fig_recurring(conf, mem):
    pols = ["frozen_E3b_personal", "ph_full_fed_E5", "ph_align", "ph_head_long", "typed_learned"] + (["typed_memory_K0_t02"] if mem else [])
    runs = conf + mem
    fig, ax = plt.subplots(figsize=(W, 1.9))
    y = np.arange(len(pols))[::-1]
    on = [seed_mean(runs, p, "post_change_f1", ("S3",)) for p in pols]
    off = [seed_mean(runs, p, "off_phase_f1", ("S3",)) for p in pols]
    ax.barh(y + 0.2, [m for m, _ in off], 0.38, xerr=[e for _, e in off], capsize=2, color="#2563eb", label="between episodes (drift off)")
    ax.barh(y - 0.2, [m for m, _ in on], 0.38, xerr=[e for _, e in on], capsize=2, color="#d97706", label="post-change (whole S3)")
    ax.set_yticks(y); ax.set_yticklabels([NAME[p] for p in pols]); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("macro-F1 of drifted clients, S3 recurring")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5)
    ax.grid(axis="x", alpha=0.3)
    save(fig, "recurring")


def fig_dp(conf, dp):
    pols = {"ph_adapter_fed": "PH + federated adapter", "ph_full_fed_E5": "PH + full retraining"}
    fig, ax = plt.subplots(figsize=(W, 2.0))
    for p, lab in pols.items():
        pts = []
        for z in (0.5, 1.0, 2.0, 4.0):
            name = f"{p}_dp_z{z:g}"
            rs = [r for r in dp if r["policy"] == name]
            eps = np.mean([np.mean([epsilon(int(r["n_uploads"].get(str(c), 0)), z) for c in r["drifted"]]) for r in rs])
            pts.append((eps, seed_mean(dp, name, "post_change_f1", ("S1", "S2", "S3"))[0]))
        ax.plot([a for a, _ in pts], [b for _, b in pts], "o-", color=COL[p], label=lab + " (noisy)", ms=4)
        ax.plot([300], [seed_mean(conf, p, "post_change_f1", ("S1", "S2", "S3"))[0]], "s", color=COL[p], ms=5, mfc="white")
    for p, ls in (("typed_learned", "--"), ("frozen_E3b_personal", ":")):
        ax.axhline(seed_mean(conf, p, "post_change_f1", ("S1", "S2", "S3"))[0], color="k", ls=ls, lw=1, label=NAME[p] + ", no upload")
    ax.set_xscale("log"); ax.set_xticks([3, 10, 30, 100, 300]); ax.set_xticklabels(["3", "10", "30", "100", "no DP"])
    ax.set_xlabel(r"$\varepsilon$ per drifted client ($\delta=10^{-5}$)"); ax.set_ylabel("post-change macro-F1")
    ax.legend(loc="center", bbox_to_anchor=(0.45, 0.55), frameon=False, fontsize=6.3); ax.grid(alpha=0.3)
    save(fig, "dp_cliff")


def main():
    conf = load("conf_*.json")
    mem = load("mem_eval_*.json")
    fig_cost(conf)
    fig_mistype(conf)
    fig_recurring(conf, mem)
    fig_dp(conf, load("dp_eval_*.json"))
    print("figures ->", OUT, "(memory runs:", len(mem), ")")


if __name__ == "__main__":
    main()
