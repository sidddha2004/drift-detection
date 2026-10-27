"""Phase 3, step 2: tune the detectors on seeds 1001-1003 under the false-alarm bound, validate on seeds 1004-1010.

Objective (Amendment A3): minimise mean detection delay in rounds (a miss costs the full horizon) subject to
<= 1 false alarm per 1,000 clean windows, on S0 streams + all variants at `med` severity of S1-S4 (fixed timing).
Output: results/phase3/detector_report.{md,json}
"""
import itertools
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detect.detectors import (ADWINErr, MMDDet, PHEnt, PHErr, TwoLevel, make_calibration,  # noqa: E402
                                  run_detector)
from src.detect.evaluate import aggregate, match_stream  # noqa: E402
from src.utils.io import run_metadata  # noqa: E402

TUNE_SEEDS, VAL_SEEDS = [1001, 1002, 1003], list(range(1004, 1011))
FA_BOUND = 1.0
CACHE = ROOT / "results/raw/phase3"

GRIDS = {
    "D1_adwin_err": [dict(delta=d) for d in (0.002, 0.02, 0.1, 0.3, 0.6)],
    "D2_ph_err": [dict(delta=d, lam=l) for d in (0.02, 0.1) for l in (2, 4, 8, 16)],
    "D3_ph_entropy": [dict(delta=d, lam=l) for d in (0.1, 0.5) for l in (5, 10, 20, 40)],
    "D4_mmd": [dict(q=q, persist=m) for q in (0.95, 0.99, 0.999) for m in (1, 2, 3)],
    "D5_two_level": [dict(ent_lam=e, q=q, sent_lam=s, alpha_b=a) for e in (10, 20) for q in (0.99, 0.999) for s in (2, 4) for a in (0.05, 0.01)],
    "D5ns_two_level_no_sentinel": [dict(ent_lam=e, q=q, alpha_b=a) for e in (10, 20) for q in (0.99, 0.999) for a in (0.05, 0.01)],
}


def build(name, cfg, cal):
    if name == "D1_adwin_err":
        return ADWINErr(cfg["delta"])
    if name == "D2_ph_err":
        return PHErr(cfg["delta"], cfg["lam"])
    if name == "D3_ph_entropy":
        return PHEnt(cal, cfg["delta"], cfg["lam"])
    if name == "D4_mmd":
        return MMDDet(cal, cfg["q"], cfg["persist"])
    sent = (5, cfg["sent_lam"]) if name == "D5_two_level" else None
    return TwoLevel(cal, ent=(0.3, cfg["ent_lam"]), mmd=(cfg["q"], 2), sentinel=sent, alpha_b=cfg["alpha_b"])


def main():
    t0 = time.time()
    sev_tune = {"med"}
    data = {}
    for s in TUNE_SEEDS + VAL_SEEDS:
        d = pickle.load(open(CACHE / f"signals_seed{s}.pkl", "rb"))
        cal = {cid: make_calibration(c) for cid, c in d["cal"].items()}
        data[s] = (cal, d)
    print(f"loaded [{time.time() - t0:.0f}s]", flush=True)

    def run(name, cfg, seeds, severities):
        parts, labs, wins = defaultdict(list), 0.0, 0
        for s in seeds:
            cal, d = data[s]
            for (scn, v, lvl, cid), S in d["streams"].items():
                if scn == "S0":
                    onsets, mode, typ = [], "none", None
                else:
                    if lvl not in severities:
                        continue
                    m = d["manifests"][(scn, v, lvl)]
                    ev = [e for e in m["events"] if e["client"] == cid]
                    onsets = [c["round"] for e in ev for c in e["change_points"] if c["kind"] == "onset"]
                    mode, typ = (ev[0]["mode"], ev[0]["type4"]) if ev else ("none", None)
                alarms, lab = run_detector(build(name, cfg, cal[cid]), S)
                labs += lab
                wins += len(S["err"])
                parts[(scn, mode, typ, lvl)].append(match_stream(alarms, S, onsets))
        return parts, labs / wins * 1000.0

    def flat(parts, keep=lambda k: True):
        return [p for k, v in parts.items() if keep(k) for p in v]

    tuned, rows = {}, []
    for name, grid in GRIDS.items():
        best = None
        for cfg in grid:
            parts, lab = run(name, cfg, TUNE_SEEDS, sev_tune)
            a = aggregate(flat(parts))
            feasible = a["fa_per_1000"] <= FA_BOUND
            key = (0 if feasible else 1, a["objective_delay_rounds"] if feasible else a["fa_per_1000"])
            if best is None or key < best[0]:
                best = (key, cfg, a, lab, feasible)
        tuned[name] = {"cfg": best[1], "tune": best[2], "labels_per_1000": best[3], "feasible": best[4]}
        print(f"{name}: tuned {best[1]} feasible={best[4]} delay={best[2]['objective_delay_rounds']:.2f} "
              f"recall={best[2]['recall']:.2f} fa/1000={best[2]['fa_per_1000']:.2f} [{time.time() - t0:.0f}s]", flush=True)

    report = {"tuned": tuned, "validation": {}, "sweep": {}}
    L = ["# Phase 3: detector tuning and validation", "",
         f"Tuned on seeds {TUNE_SEEDS} (S0 + all S1-S4 variants, `med` severity, fixed timing) under the bound "
         f"<= {FA_BOUND} false alarm per 1,000 clean windows; validated on seeds {VAL_SEEDS}. "
         "Horizon 20 rounds; a miss costs the full horizon in the delay objective.", "",
         "## Tuned configurations", "", "| detector | config | feasible on tuning | tuning delay (rounds) | tuning recall | tuning FA/1000 |", "|---|---|---|---|---|---|"]
    for n, t in tuned.items():
        L.append(f"| {n} | `{t['cfg']}` | {t['feasible']} | {t['tune']['objective_delay_rounds']:.2f} | {t['tune']['recall']:.2f} | {t['tune']['fa_per_1000']:.2f} |")

    L += ["", "## Validation (unseen seeds), `med` severity, all drifted variants", "",
          "| detector | recall | mean delay (rounds, detected only) | objective delay | FA / 1000 clean windows | redundant alarms | labels / 1000 windows |",
          "|---|---|---|---|---|---|---|"]
    val_parts = {}
    for name, t in tuned.items():
        parts, lab = run(name, t["cfg"], VAL_SEEDS, {"med"})
        val_parts[name] = parts
        a = aggregate(flat(parts))
        report["validation"][name] = {**a, "labels_per_1000": lab}
        L.append(f"| {name} | {a['recall']:.2f} | {a['delay_rounds']:.1f} | {a['objective_delay_rounds']:.1f} | "
                 f"{a['fa_per_1000']:.2f} ({a['false_alarms']}/{a['clean_windows']}) | {a['redundant']} | {lab:.0f} |")

    for title, keyfn, groups in [("by drift mode (recall / delay rounds)", lambda k: k[1], ["virtual", "real"]),
                                 ("by ground-truth type (recall / delay rounds)", lambda k: k[2], ["sudden", "gradual", "recurring", "feature_level"])]:
        L += ["", f"## Validation {title}", "", "| detector | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
        for name in tuned:
            cells = []
            for g in groups:
                a = aggregate(flat(val_parts[name], lambda k, g=g: keyfn(k) == g))
                cells.append(f"{a['recall']:.2f} / {a['delay_rounds']:.1f}")
            L.append(f"| {name} | " + " | ".join(cells) + " |")

    L += ["", "## Validation severity sweep (recall / FA per 1000 on the same streams)", "", "| detector | low | med | high |", "|---|---|---|---|"]
    for name, t in tuned.items():
        cells = []
        parts, _ = run(name, t["cfg"], VAL_SEEDS, {"low", "med", "high"})
        for lvl in ("low", "med", "high"):
            a = aggregate(flat(parts, lambda k, lvl=lvl: k[3] == lvl))
            cells.append(f"{a['recall']:.2f} / {a['fa_per_1000']:.2f}")
        report["sweep"][name] = cells
        L.append(f"| {name} | " + " | ".join(cells) + " |")

    out = ROOT / "results/phase3"
    (out / "detector_report.md").write_text("\n".join(L), encoding="utf-8")
    (out / "detector_report.json").write_text(json.dumps({"meta": run_metadata(), **report}, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
