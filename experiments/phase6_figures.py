"""Paper figures from the confirmatory runs (results/phase4_v2/conf_*.json) -> docs/paper/figs/*.pdf (+png)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "results/phase4_v2"
OUT = ROOT / "docs/paper/figs"
OUT.mkdir(parents=True, exist_ok=True)

LABEL = {"frozen_E3b_personal": "E3b (frozen personalised)", "periodic_E3c_full_fed": "E3c (periodic full retrain)",
         "ph_full_fed_E5": "E5 (PH + full retrain, federated)", "ph_fedprox_E7a": "E7a (PH + FedProx)",
         "ph_head_long": "PH + head retrain (fixed)", "typed_learned": "E8 (typed, deployable)", "typed_oracle": "oracle 4-way typing",
         "ph_adapter_fed": "PH + adapter (federated)"}
COL = {"frozen_E3b_personal": "#6b7280", "periodic_E3c_full_fed": "#d97706", "ph_full_fed_E5": "#b91c1c", "ph_fedprox_E7a": "#9333ea",
       "ph_head_long": "#0891b2", "typed_learned": "#16a34a", "typed_oracle": "#111827", "ph_adapter_fed": "#2563eb"}


def load():
    runs = []
    for f in sorted(SRC.glob("conf_*.json")):
        runs += json.loads(f.read_text())["runs"]
    return runs


def curve(runs, pol, scn, var):
    cs = []
    for r in runs:
        if r["policy"] == pol and r["scenario"] == scn and r["variant"] == var:
            f1 = np.asarray(r["f1"]); idx = [r["clients"].index(c) for c in r["drifted"]]
            cs.append(f1[idx].mean(0))
    return np.array(cs)


def smooth(x, k=5):
    return np.convolve(np.pad(x, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def fig_curves(runs):
    panels = [("S1", "virtual_rotation_partial", "S1 sudden virtual drift (rotation)"), ("S1", "real_partial", "S1 sudden real drift (label remap)"),
              ("S3", "virtual_rotation_partial", "S3 recurring virtual drift (rotation)")]
    pols = ["frozen_E3b_personal", "ph_full_fed_E5", "ph_head_long", "typed_learned", "typed_oracle"]
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.9), sharey=True)
    for a, (s, v, t) in zip(ax, panels):
        for p in pols:
            c = curve(runs, p, s, v)
            if len(c):
                m = smooth(c.mean(0)); a.plot(m, label=LABEL[p], color=COL[p], lw=1.4, ls="--" if p == "typed_oracle" else "-")
        a.axvline(30, color="#9ca3af", ls=":", lw=1)
        a.set_title(t, fontsize=8); a.set_xlabel("round")
    ax[0].set_ylabel("macro-F1 (drifted clients)")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6.5, loc="lower center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT / "recovery_curves.pdf"); fig.savefig(OUT / "recovery_curves.png", dpi=150); plt.close(fig)


def fig_pareto(runs):
    by = defaultdict(list); up = defaultdict(list)
    for r in runs:
        if r["scenario"] in ("S1", "S2", "S3"):
            by[(r["policy"], r["seed"])].append(r["metrics"]["post_change_f1"]); up[r["policy"]].append(r["metrics"]["uplink_bytes"] / 1e3)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for p in LABEL:
        f = np.mean([np.mean(v) for (q, _), v in by.items() if q == p]); u = np.mean(up[p])
        ax.scatter(u + 1, f, color=COL[p], s=36, zorder=3)
        ax.annotate(LABEL[p], (u + 1, f), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("uplink per run (kB, +1, log scale)"); ax.set_ylabel("post-change macro-F1 (S1-S3)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "pareto.pdf"); fig.savefig(OUT / "pareto.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    runs = load()
    fig_curves(runs); fig_pareto(runs)
    print("saved", sorted(p.name for p in OUT.iterdir()))
