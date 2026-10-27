"""Phase 4 report: tables and recovery curves from results/phase4/*.json -> results/phase4/summary.md + curves_*.png.

All numbers are tuning-side seeds (1001-1003): a design look, NOT the confirmatory evaluation (Phase 6, 10 evaluation seeds).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results/phase4"
FILES = ["s1_virtual_first_look", "s1_real_first_look", "typed_virtual", "typed_real", "mode_typed", "learned_s1", "learned_s234", "learned_conf80"]


def load():
    runs = []
    for f in FILES:
        p = OUT / f"{f}.json"
        if p.exists():
            runs += [dict(r, file=f) for r in json.loads(p.read_text())["runs"]]
    return runs


def ms(v):
    return f"{np.mean(v):.3f} ± {np.std(v):.3f}"


def table(runs, key, L):
    by = defaultdict(list)
    for r in runs:
        by[(r["scenario"], r["variant"])].append(r)
    for (scn, var), rs in sorted(by.items()):
        pol = defaultdict(list)
        for r in rs:
            pol[r["policy"]].append(r["metrics"])
        L += [f"### {scn} {var} (severity {rs[0]['severity']}, seeds {sorted({r['seed'] for r in rs})})", "",
              "| policy | post-change F1 | pre-drift F1 | max drop | recovery (rounds) | off-phase F1 | clean-clients F1 | uplink (kB) | alarms |",
              "|---|---|---|---|---|---|---|---|---|"]
        for p, ms_ in pol.items():
            g = lambda k: [m[k] for m in ms_]
            L.append(f"| {p} | {ms(g('post_change_f1'))} | {np.mean(g('pre_drift_f1')):.3f} | {np.mean(g('max_drop')):.3f} | "
                     f"{np.mean(g('recovery_rounds')):.1f} | {np.nanmean(g('off_phase_f1')) if not np.all(np.isnan(g('off_phase_f1'))) else float('nan'):.3f} | "
                     f"{np.nanmean(g('clean_clients_post_f1')):.3f} | {np.mean(g('uplink_bytes')) / 1e3:.0f} | {np.mean(g('alarms')):.1f} |")
        L.append("")


def curves(runs, name, variants, policies):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = [(s, v) for s, v in variants if any(r["scenario"] == s and r["variant"] == v for r in runs)]
    if not sel:
        return
    fig, ax = plt.subplots(1, len(sel), figsize=(4.6 * len(sel), 3.6), squeeze=False)
    for j, (s, v) in enumerate(sel):
        for p in policies:
            curve = []
            for r in runs:
                if r["scenario"] == s and r["variant"] == v and r["policy"] == p:
                    f1 = np.asarray(r["f1"])
                    idx = [r["clients"].index(c) for c in r["drifted"]]
                    curve.append(f1[idx].mean(0))
            if curve:
                ax[0, j].plot(np.mean(curve, 0), label=p, lw=1.4)
        ax[0, j].axvline(20, ls=":", c="gray")
        ax[0, j].set_title(f"{s} {v}", fontsize=9)
        ax[0, j].set_xlabel("round")
        ax[0, j].set_ylabel("macro-F1 (drifted clients)")
    ax[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / f"curves_{name}.png", dpi=130)
    plt.close(fig)


def main():
    runs = load()
    L = ["# Phase 4 summary: streaming FL, adaptation responses and drift typing (design look)", "",
         "Tuning-side seeds only (1001-1003), `med` severity, 8 clients, 60 rounds, deployed model = E3b personalised model, "
         "immediate labels. Means ± std over seeds; with 3 seeds nothing here is a confirmatory test.", ""]
    if not runs:
        L.append("No result files yet.")
    else:
        table(runs, "post_change_f1", L)
        # typed vs fixed
        L += ["## Typed (oracle) vs fixed responses, post-change macro-F1 per variant", "",
              "| variant | typed_oracle | best fixed (name) | typed - best fixed | frozen E3b |", "|---|---|---|---|---|"]
        by = defaultdict(lambda: defaultdict(list))
        for r in runs:
            by[(r["scenario"], r["variant"])][r["policy"]].append(r["metrics"]["post_change_f1"])
        diffs = []
        for (s, v), pol in sorted(by.items()):
            if "typed_oracle" not in pol:
                continue
            fixed = {p: np.mean(x) for p, x in pol.items() if p.startswith("fixed_")}
            if not fixed:
                continue
            bf = max(fixed, key=fixed.get)
            d = np.mean(pol["typed_oracle"]) - fixed[bf]
            diffs.append(d)
            L.append(f"| {s} {v} | {np.mean(pol['typed_oracle']):.3f} | {fixed[bf]:.3f} ({bf}) | {d:+.3f} | {np.mean(pol['frozen_E3b_personal']):.3f} |")
        if diffs:
            L += ["", f"Mean (typed - best fixed) over {len(diffs)} variants: {np.mean(diffs):+.3f}. "
                      "The best fixed policy is picked per variant, i.e. the comparison favours the fixed side.", ""]
    # deployable (detector-triggered) systems side by side
    cols = ["frozen_E3b_personal", "ph_full_fed_E5", "ph_align", "ph_head_long", "typed_learned", "typed_learned_conf80", "ph_mode_oracle"]
    dep = defaultdict(lambda: defaultdict(list))
    for r in runs:
        dep[(r["scenario"], r["variant"])][r["policy"]].append(r["metrics"]["post_change_f1"])
    L += ["## Deployable (detector-triggered) systems, post-change macro-F1 (3 tuning seeds, immediate labels)", "",
          "| variant | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    tot = defaultdict(list)
    for k, v in sorted(dep.items()):
        if not any(c in v for c in cols[1:]):
            continue
        cells = []
        for c in cols:
            if c in v:
                cells.append(f"{np.mean(v[c]):.3f}")
                tot[c].append((k, np.mean(v[c])))
            else:
                cells.append("-")
        L.append(f"| {k[0]} {k[1]} | " + " | ".join(cells) + " |")
    common = set.intersection(*[{k for k, _ in tot[c]} for c in cols if tot[c]]) if tot else set()
    if common:
        L.append("| **mean (variants common to all)** | " + " | ".join(
            f"**{np.mean([m for k, m in tot[c] if k in common]):.3f}**" if tot[c] else "-" for c in cols) + " |")
        L += ["", f"{len(common)} variants common to all columns. typed_* use a mode classifier (virtual / real) after a 5-round "
                  "typing delay; ph_mode_oracle uses the true mode; conf80 applies alignment only if the classifier is >= 80% sure the drift is virtual.", ""]
    # what does the 4-way type add on top of the 2-way (virtual / real) mode?  (oracle typing, S2-S4)
    by2 = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by2[(r["scenario"], r["variant"])][r["policy"]].append(r["metrics"]["post_change_f1"])
    four = {k: pol for k, pol in by2.items() if "typed_oracle" in pol and "mode_typed_oracle" in pol}
    if four:
        L += ["## Value of the 4-way type beyond the 2-way mode (both with oracle labels), post-change F1", "",
              "| variant | 4-way typed_oracle | 2-way mode_typed_oracle | difference |", "|---|---|---|---|"]
        dd = []
        for (sc, v), pol in sorted(four.items()):
            a, b = np.mean(pol["typed_oracle"]), np.mean(pol["mode_typed_oracle"])
            dd.append(a - b)
            L.append(f"| {sc} {v} | {a:.3f} | {b:.3f} | {a - b:+.3f} |")
        L += ["", f"Mean 4-way minus 2-way over {len(dd)} variants: {np.mean(dd):+.3f}.", ""]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    curves(runs, "s1_virtual", [("S1", "virtual_offset_partial"), ("S1", "virtual_gain_partial"), ("S1", "virtual_rotation_partial")],
           ["frozen_E3b_personal", "periodic_E3c_full_fed", "oracle_full_fed", "oracle_adapter_fed", "oracle_align", "ph_full_fed_E5", "ph_align"])
    curves(runs, "s1_real", [("S1", "real_partial")], ["frozen_E3b_personal", "oracle_full_fed", "oracle_head_long", "oracle_adapter_long", "ph_head_long"])
    print("\n".join(L))


if __name__ == "__main__":
    main()
