"""Phase 3, step 3: drift-type classification (sudden / gradual / recurring / feature_level).

Episodes are built from the cached signals of seeds 1001-1010 (tuning side only, Amendment A3). Two alarm sources:
  oracle    alarm = true onset + 2 rounds        (upper bound: typing quality independent of detection quality)
  detected  alarms of the tuned D2 (Page-Hinkley on error) detector (matched to an onset within the 20-round horizon)
Classifiers: majority baseline, a-priori rule classifier, logistic regression and gradient boosting (learned), all scored
with leave-one-seed-out. recurring_first episodes (first onset of a recurring drift) are not identifiable at alarm
time and are reported separately; they are excluded from training and from the 4-way score.
Output: results/phase3/typing_report.{md,json}
"""
import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.phase3_detectors import build  # noqa: E402
from src.utils.bench import raw_dir, res_dir, scn_dir  # noqa: E402
from src.detect.detectors import make_calibration, run_detector  # noqa: E402
import src.detect.drift_type as DT  # noqa: E402
from src.detect.drift_type import FEATURE_NAMES, TYPES, RuleClassifier, run_episodes, series  # noqa: E402
from src.detect.evaluate import HORIZON_ROUNDS  # noqa: E402
from src.utils.io import run_metadata  # noqa: E402

SEEDS = list(range(1001, 1011))
CACHE = raw_dir(ROOT, "phase3")
ORACLE_DELAY_ROUNDS = 2
# `n_prior` (number of earlier alarms) is a schedule artefact with oracle alarms (every second alarm is a re-occurrence), so
# the LEARNED models never see it; recurrence must come from the signature-memory features.
LEARN_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n != "n_prior"]
DETECTOR = "D2_ph_err"          # best validated detector of the detector report (full labels)


def collect(source: str, tuned: dict):
    rows = []
    for seed in SEEDS:
        d = pickle.load(open(CACHE / f"signals_seed{seed}.pkl", "rb"))
        cal = {c: make_calibration(v) for c, v in d["cal"].items()}
        for (scn, v, lvl, cid), S in d["streams"].items():
            if scn == "S0":
                continue
            ev = [e for e in d["manifests"][(scn, v, lvl)]["events"] if e["client"] == cid]
            if not ev:
                continue
            e = ev[0]
            onsets = [c["round"] for c in e["change_points"] if c["kind"] == "onset"]
            rnd = S["rnd"]
            if source == "oracle":
                alarms = [int(np.argmax(rnd >= r0 + ORACLE_DELAY_ROUNDS)) for r0 in onsets]
            else:
                alarms, _ = run_detector(build(DETECTOR, tuned[DETECTOR]["cfg"], cal[cid]), S)
            ser = series(S, cal[cid])
            for a, f in run_episodes(ser, int(S["per_round"]), alarms):
                j = next((k for k, r0 in enumerate(onsets) if r0 <= rnd[a] < r0 + HORIZON_ROUNDS), None)
                if j is None:
                    continue                                   # false alarm: no ground-truth type
                label = e["type4"]
                if label == "recurring" and j == 0:
                    label = "recurring_first"
                rows.append(dict(seed=seed, scn=scn, variant=v, severity=lvl, mode=e["mode"], client=cid, label=label, x=f))
    return rows


def score(y, p, labels=TYPES):
    cm = confusion_matrix(y, p, labels=list(labels))
    rec = {l: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else float("nan")) for i, l in enumerate(labels)}
    return {"acc": float(np.mean(np.asarray(y) == np.asarray(p))), "macro_f1": float(f1_score(y, p, labels=list(labels), average="macro", zero_division=0)),
            "recall": rec, "cm": cm.tolist(), "n": int(len(y))}


def evaluate(rows):
    ident = [r for r in rows if r["label"] in TYPES]
    first = [r for r in rows if r["label"] == "recurring_first"]
    X, y = np.stack([r["x"] for r in ident]), np.array([r["label"] for r in ident])
    Xr = X                                                       # rule classifier sees all columns (it ignores n_prior)
    X = X[:, LEARN_COLS]
    seeds = np.array([r["seed"] for r in ident])
    out = {"n_identifiable": len(ident), "n_recurring_first": len(first), "class_counts": dict(Counter(y.tolist()))}
    preds = {"majority": np.empty(len(y), dtype=object), "rule": np.array(RuleClassifier().predict(Xr)),
             "logreg": np.empty(len(y), dtype=object), "gboost": np.empty(len(y), dtype=object)}
    first_pred = {"rule": RuleClassifier().predict(np.stack([r["x"] for r in first])) if first else []}
    first_X = np.stack([r["x"] for r in first])[:, LEARN_COLS] if first else None
    fp = {"logreg": [], "gboost": []}
    for s in SEEDS:
        te, tr = seeds == s, seeds != s
        if not te.any():
            continue
        preds["majority"][te] = Counter(y[tr].tolist()).most_common(1)[0][0]
        sc = StandardScaler().fit(X[tr])
        for name, mk in (("logreg", lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
                         ("gboost", lambda: GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0))):
            m = mk().fit(sc.transform(X[tr]), y[tr])
            preds[name][te] = m.predict(sc.transform(X[te]))
            if first_X is not None:
                fs = np.array([r["seed"] for r in first]) == s
                if fs.any():
                    fp[name] += list(m.predict(sc.transform(first_X[fs])))
    out["models"] = {n: score(y, p) for n, p in preds.items()}
    out["by_severity"] = {n: {lvl: float(np.mean(p[np.array([r["severity"] == lvl for r in ident])] == y[np.array([r["severity"] == lvl for r in ident])]))
                              for lvl in ("low", "med", "high")} for n, p in preds.items()}
    out["by_mode"] = {n: {md: float(np.mean(p[np.array([r["mode"] == md for r in ident])] == y[np.array([r["mode"] == md for r in ident])]))
                          for md in ("virtual", "real")} for n, p in preds.items()}
    out["recurring_first_predictions"] = {"rule": dict(Counter(first_pred["rule"])), **{n: dict(Counter(v)) for n, v in fp.items()}}
    # which features matter (full-data gradient boosting; descriptive only)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(StandardScaler().fit_transform(X), y)
    out["feature_importance"] = dict(sorted(zip([FEATURE_NAMES[i] for i in LEARN_COLS], map(float, gb.feature_importances_)), key=lambda t: -t[1]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookahead", type=int, default=5, help="rounds of evidence after the alarm")
    args = ap.parse_args()
    DT.LOOKAHEAD_ROUNDS = args.lookahead
    rep_path = res_dir(ROOT, "phase3") / "detector_report.json"
    sources = {"oracle": None}
    if rep_path.exists():
        sources["detected"] = json.loads(rep_path.read_text())["tuned"]
    report, L = {}, ["# Phase 3: drift-type classification", "",
                     f"Seeds {SEEDS[0]}-{SEEDS[-1]} (tuning side), leave-one-seed-out. Look-ahead {args.lookahead} rounds after the alarm; learned models do not see `n_prior`. "
                     "Identifiable 4-way score excludes the first onset of recurring drifts.", ""]
    for src, tuned in sources.items():
        rows = collect(src, tuned)
        r = evaluate(rows)
        report[src] = r
        L += [f"## Alarm source: {src}", "", f"{r['n_identifiable']} identifiable episodes {r['class_counts']}; {r['n_recurring_first']} recurring_first episodes.", "",
              "| classifier | accuracy | macro-F1 | " + " | ".join(f"recall {t}" for t in TYPES) + " |", "|---|---|---|" + "---|" * len(TYPES)]
        for n, m in r["models"].items():
            L.append(f"| {n} | {m['acc']:.3f} | {m['macro_f1']:.3f} | " + " | ".join(f"{m['recall'][t]:.2f}" for t in TYPES) + " |")
        L += ["", "Accuracy by severity (low / med / high) and by mode (virtual / real):", "", "| classifier | low | med | high | virtual | real |", "|---|---|---|---|---|---|"]
        for n in r["models"]:
            s, md = r["by_severity"][n], r["by_mode"][n]
            L.append(f"| {n} | {s['low']:.2f} | {s['med']:.2f} | {s['high']:.2f} | {md['virtual']:.2f} | {md['real']:.2f} |")
        L += ["", "Confusion matrix of gboost (rows = truth, columns = predicted, order " + ", ".join(TYPES) + "):", ""]
        L += ["    " + str(row) for row in r["models"]["gboost"]["cm"]]
        L += ["", f"recurring_first (not identifiable) predicted as: {r['recurring_first_predictions']}", "",
              "Feature importance (gboost, descriptive): " + ", ".join(f"{k} {v:.2f}" for k, v in list(r["feature_importance"].items())[:6]), ""]
    out = res_dir(ROOT, "phase3")
    (out / f"typing_report_la{args.lookahead}.md").write_text("\n".join(L), encoding="utf-8")
    (out / f"typing_report_la{args.lookahead}.json").write_text(json.dumps({"meta": run_metadata(), **report}, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
