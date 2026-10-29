"""Paper figures and summary stats built ONLY from cached results (no training, no streaming runs; runs in seconds).

Inputs : results/phase4_v2/conf_*.json (confirmatory streaming runs, 10 evaluation seeds)
         results/phase6/confirmatory.json (registered tests)
         results/phase3_v2/detector_report.json, mode_typing.json (detector validation, mode classifier)
Output : docs/paper/figs/<name>.pdf + .png, docs/paper/figs/summary_stats.md
Run    : python experiments/paper_figs/run_all.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "results/phase4_v2"
CONF = ROOT / "results/phase6/confirmatory.json"
DET = ROOT / "results/phase3_v2/detector_report.json"
MODE = ROOT / "results/phase3_v2/mode_typing.json"
OUT = ROOT / "docs/paper/figs"

E8 = "typed_learned"
SCEN = ["S1", "S2", "S3"]
NAME = {"frozen_E3_global": "E3 frozen global", "frozen_E3b_personal": "E3b frozen pers.", "periodic_E3c_full_fed": "E3c periodic full",
        "adwin_full_fed_E4": "E4 ADWIN+full", "ph_full_fed_E5": "E5 PH+full", "ph_fedprox_E7a": "E7a PH+FedProx",
        "ph_full_local": "PH+full local", "ph_adapter_fed": "PH+adapter fed", "ph_align": "PH+align (fixed)",
        "ph_head_long": "PH+head (fixed)", "typed_learned": "E8 typed (deploy.)", "ph_mode_oracle": "E8 + true mode",
        "mode_typed_oracle": "oracle 2-way", "typed_oracle": "oracle 4-way"}
ORDER = list(NAME)
COMPARATOR = {"H1": "frozen_E3b_personal", "H2": "periodic_E3c_full_fed", "H3": "ph_fedprox_E7a", "H5": "ph_head_long"}


# ---------------------------------------------------------------- loading / aggregation (pure, tested)
def load_runs(src=RUNS):
    runs = []
    for f in sorted(Path(src).glob("conf_*.json")):
        runs += json.loads(f.read_text())["runs"]
    if not runs:
        raise FileNotFoundError(f"no conf_*.json in {src}")
    return runs


def per_seed(runs, metric="post_change_f1", scenarios=SCEN):
    """(policy, scenario) -> {seed: mean over the scenario's variants}; NaN values are dropped."""
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        v = r["metrics"].get(metric)
        if r["scenario"] in scenarios and v is not None and np.isfinite(v):
            acc[(r["policy"], r["scenario"])][r["seed"]].append(v)
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in acc.items()}


def onset_round(alpha_row):
    nz = np.flatnonzero(np.asarray(alpha_row) > 0)
    return int(nz[0]) if len(nz) else None


def alarm_delays(runs, policy=E8):
    """Rounds from a drifted client's onset to its first alarm at/after onset; None = missed. Alarms are [client, round]."""
    out = []
    for r in runs:
        if r["policy"] != policy:
            continue
        for c in r["drifted"]:
            on = onset_round(r["alpha"][r["clients"].index(c)])
            if on is None:
                continue
            after = [rd for cl, rd in r["alarms"] if cl == c and rd >= on]
            out.append(min(after) - on if after else None)
    return out


# ---------------------------------------------------------------- figures
def _save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=150); plt.close(fig)
    return name


def fig_forest(conf, out=OUT):
    fam = conf["family"]
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for i, t in enumerate(fam[::-1]):
        col = ("#16a34a" if t["diff"] > 0 else "#b91c1c") if t["significant"] else "#6b7280"
        ax.errorbar(t["diff"], i, xerr=[[t["diff"] - t["ci"][0]], [t["ci"][1] - t["diff"]]], fmt="o", color=col, ms=4, capsize=2)
    ax.set_yticks(range(len(fam))); ax.set_yticklabels([f"{t['h']} {t['scenario']}" for t in fam[::-1]], fontsize=7)
    ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("E8 minus comparator (post-change macro-F1), 95% CI", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5); ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
    ax.set_title("Confirmatory tests (colour = Holm-significant)", fontsize=8); ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); return _save(fig, "forest_confirmatory", out)


def fig_paired_seeds(ps, out=OUT):
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8), sharey=True)
    for ax, (h, comp) in zip(axes, COMPARATOR.items()):
        for j, s in enumerate(SCEN):
            a, b = ps.get((E8, s), {}), ps.get((comp, s), {})
            seeds = sorted(set(a) & set(b)); d = [a[k] - b[k] for k in seeds]
            ax.scatter(np.full(len(d), j) + np.linspace(-0.12, 0.12, len(d)), d, s=12, color="#2563eb", alpha=0.8)
            if d: ax.hlines(np.mean(d), j - 0.25, j + 0.25, color="k", lw=1.5)
        ax.axhline(0, color="#9ca3af", lw=0.8); ax.set_xticks(range(3)); ax.set_xticklabels(SCEN)
        ax.set_title(f"{h}: E8 - {NAME[comp]}", fontsize=7.5)
    axes[0].set_ylabel("paired difference per seed")
    fig.tight_layout(); return _save(fig, "paired_seeds", out)


def fig_heatmap(runs, out=OUT):
    cells = defaultdict(list)
    for r in runs:
        cells[(r["policy"], f"{r['scenario']} {r['variant'].replace('_partial', '')}")].append(r["metrics"]["post_change_f1"])
    cols = sorted({c for _, c in cells}); rows = [p for p in ORDER if any((p, c) in cells for c in cols)]
    m = np.array([[np.mean(cells[(p, c)]) if (p, c) in cells else np.nan for c in cols] for p in rows])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(m, cmap="viridis", aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="w" if m[i, j] < np.nanmean(m) else "k")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=6)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAME[p] for p in rows], fontsize=6.5)
    fig.colorbar(im, ax=ax, label="post-change macro-F1"); fig.tight_layout(); return _save(fig, "heatmap_policy_variant", out)


def _bars(ax, ps_metric, pols, scen, label):
    vals = [[ps_metric[(p, s)][k] for k in ps_metric[(p, s)]] if (p, s) in ps_metric else [np.nan] for p in pols for s in [scen]]
    ax.barh(range(len(pols)), [np.mean(v) for v in vals], xerr=[np.std(v) for v in vals], color="#2563eb", alpha=0.8, capsize=2)
    ax.set_yticks(range(len(pols))); ax.set_yticklabels([NAME[p] for p in pols], fontsize=6.5); ax.set_xlabel(label, fontsize=7)


def fig_s3_offphase(runs, out=OUT):
    ps = per_seed(runs, "off_phase_f1", ["S3"]); pols = [p for p in ORDER if (p, "S3") in ps]
    fig, ax = plt.subplots(figsize=(4.6, 3.4)); _bars(ax, ps, pols, "S3", "S3 off-phase macro-F1 (between drift episodes)")
    ax.set_title("Recurring drift: accuracy after each episode ends", fontsize=8); fig.tight_layout(); return _save(fig, "s3_offphase", out)


def fig_collateral(runs, out=OUT):
    clean = per_seed(runs, "clean_clients_post_f1"); worst = per_seed(runs, "worst_client_post_f1")
    pols = [p for p in ORDER if (p, "S1") in clean]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    _bars(ax[0], clean, pols, "S1", "S1 F1 of NON-drifted clients"); _bars(ax[1], worst, pols, "S1", "S1 worst drifted client F1")
    fig.tight_layout(); return _save(fig, "collateral_worst_client", out)


def fig_alarm_delay(runs, out=OUT):
    d = alarm_delays(runs); hit = [x for x in d if x is not None]
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.hist(hit, bins=np.arange(0, 92, 4), color="#0891b2")
    ax.set_xlabel("rounds from onset to first alarm (E8, PH detector)"); ax.set_ylabel("drifted clients")
    ax.set_title(f"detected {len(hit)}/{len(d)} ({len(hit) / max(len(d), 1):.0%}), median delay {np.median(hit) if hit else float('nan'):.0f} rounds", fontsize=8)
    fig.tight_layout(); return _save(fig, "alarm_delay", out)


def fig_detectors(det, out=OUT):
    v, pl = det["validation"], det["placebo"]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for i, (k, m) in enumerate(v.items()):
        lf = m["labels_per_1000"] == 0
        ax.scatter(m["fa_per_1000"], m["recall"], s=20 + m["labels_per_1000"] / 8, color="#b91c1c" if lf else "#2563eb", zorder=3)
        ax.annotate(f"{k.split('_')[0]} (placebo {pl[k]:.2f})", (m["fa_per_1000"], m["recall"]), fontsize=5.5,
                    xytext=(6, -9 if k.startswith("D2c") else 4), textcoords="offset points")
    ax.axvline(1.0, color="k", ls=":", lw=0.8)
    ax.text(1.02, 0.97, "FA bound", fontsize=6, transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlabel("false alarms per 1000 clean windows"); ax.set_ylabel("recall within 20 rounds")
    ax.set_title("Detectors (validation; red = label-free, size = labels used)", fontsize=7.5); ax.grid(alpha=0.3)
    fig.tight_layout(); return _save(fig, "detectors_tradeoff", out)


def fig_mode(mode, out=OUT):
    names = list(mode); x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(5.2, 3))
    for i, (k, c) in enumerate([("acc", "#111827"), ("recall_virtual", "#2563eb"), ("recall_real", "#d97706")]):
        ax.bar(x + (i - 1) * w, [mode[n][k] for n in names], w, label=k, color=c)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=6.5); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=6.5); ax.set_title("Virtual vs real mode classification (leave-one-seed-out)", fontsize=8)
    fig.tight_layout(); return _save(fig, "mode_classifier", out)


def fig_typing_ladder(runs, out=OUT):
    pols = ["frozen_E3b_personal", "ph_head_long", "typed_learned", "ph_mode_oracle", "mode_typed_oracle", "typed_oracle"]
    scen = ["S1", "S2", "S3", "S4"]; full = per_seed(runs, scenarios=scen)
    fig, ax = plt.subplots(figsize=(6, 3)); w = 0.8 / len(pols)
    for i, p in enumerate(pols):
        ax.bar(np.arange(4) + i * w, [np.mean(list(full.get((p, s), {np.nan: np.nan}).values())) for s in scen], w, label=NAME[p])
    ax.set_xticks(np.arange(4) + 0.4 - w / 2); ax.set_xticklabels(scen); ax.set_ylim(0.5, 0.9); ax.set_ylabel("post-change macro-F1")
    ax.legend(fontsize=6, ncol=2); ax.set_title("What typing is worth: deployable -> true mode -> oracle 2-way -> 4-way", fontsize=8)
    fig.tight_layout(); return _save(fig, "typing_ladder", out)


def fig_cost(runs, out=OUT):
    drop, rec, up = per_seed(runs, "max_drop"), per_seed(runs, "recovery_rounds"), per_seed(runs, "uplink_bytes")
    pols = [p for p in ORDER if (p, "S1") in drop]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    _bars(ax[0], drop, pols, "S1", "S1 max drop (F1)"); _bars(ax[1], rec, pols, "S1", "S1 recovery rounds")
    up_kb = {k: {s: v / 1e3 for s, v in d.items()} for k, d in up.items()}; _bars(ax[2], up_kb, pols, "S1", "S1 uplink per run (kB)")
    ax[2].set_xscale("symlog"); fig.tight_layout(); return _save(fig, "cost_recovery", out)


# ---------------------------------------------------------------- stats table
def summary_stats(runs, conf):
    ps = per_seed(runs, scenarios=["S1", "S2", "S3", "S4"])
    lines = ["# Summary stats from cached results (no reruns)", "", "Post-change macro-F1, mean +- std over seeds.", "",
             "| system | S1 | S2 | S3 | S4 |", "|---|---|---|---|---|"]
    for p in ORDER:
        row = [f"{np.mean(list(ps[(p, s)].values())):.3f} +- {np.std(list(ps[(p, s)].values())):.3f}" if (p, s) in ps else "-" for s in ["S1", "S2", "S3", "S4"]]
        lines.append(f"| {NAME[p]} | " + " | ".join(row) + " |")
    d = alarm_delays(runs); hit = [x for x in d if x is not None]
    lines += ["", f"E8 alarms: {len(hit)}/{len(d)} drifted clients detected, median delay {np.median(hit):.1f} rounds." if hit else "E8 alarms: none.",
              "", "Confirmatory family (from results/phase6/confirmatory.json):", ""]
    lines += [f"- {t['h']} {t['scenario']}: {t['diff']:+.3f} [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}], Holm p {t['p_holm']:.3f}" for t in conf["family"]]
    return "\n".join(lines) + "\n"
