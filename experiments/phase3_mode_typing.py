"""Phase 3/4 bridge: can the drift MODE (virtual = inputs changed, real = labels changed) be identified from an alarm?

Same episodes and features as phase3_typing (detected alarms of D2, leave-one-seed-out, tuning-side seeds), label = mode.
Majority baseline, a one-feature rule on the label-free covariate evidence, logistic regression and gradient boosting.
Output: results/phase3/mode_typing.md
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.phase3_typing import LEARN_COLS, SEEDS, collect  # noqa: E402
from src.utils.bench import raw_dir, res_dir, scn_dir  # noqa: E402
from src.detect.drift_type import FEATURE_NAMES  # noqa: E402

LABEL_FREE = [FEATURE_NAMES.index(n) for n in ("dF_pre", "dF_post", "dF_delta", "dF_ramp", "dF_slope", "ent_delta", "ent_ramp",
                                               "imu_share_max", "imu_ratio", "imu_entropy", "rise_dF", "rise_ent", "sr_dF", "sr_ent")]


def main():
    tuned = json.loads((res_dir(ROOT, "phase3") / "detector_report.json").read_text())["tuned"]
    rows = collect("detected", tuned)
    X = np.stack([r["x"] for r in rows])
    y = np.array([r["mode"] for r in rows])
    seeds = np.array([r["seed"] for r in rows])
    sev = np.array([r["severity"] for r in rows])
    res = {}
    major = np.empty(len(y), dtype=object)
    rule = np.where(X[:, FEATURE_NAMES.index("dF_delta")] > 0.05, "virtual", "real")
    preds = {"majority": major, "rule dF_delta>0.05": rule}
    for name, cols in (("logreg label-free", LABEL_FREE), ("gboost label-free", LABEL_FREE), ("gboost all features", LEARN_COLS)):
        preds[name] = np.empty(len(y), dtype=object)
    for s in SEEDS:
        te, tr = seeds == s, seeds != s
        if not te.any():
            continue
        major[te] = max(set(y[tr]), key=list(y[tr]).count)
        for name, cols in (("logreg label-free", LABEL_FREE), ("gboost label-free", LABEL_FREE), ("gboost all features", LEARN_COLS)):
            sc = StandardScaler().fit(X[tr][:, cols])
            m = (LogisticRegression(max_iter=2000, class_weight="balanced") if name.startswith("logreg")
                 else GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0))
            m.fit(sc.transform(X[tr][:, cols]), y[tr])
            preds[name][te] = m.predict(sc.transform(X[te][:, cols]))
    L = ["# Mode (virtual vs real) identification from detected alarms", "",
         f"{len(y)} episodes ({int((y == 'virtual').sum())} virtual, {int((y == 'real').sum())} real), leave-one-seed-out over seeds {SEEDS[0]}-{SEEDS[-1]}.", "",
         "| classifier | accuracy | recall virtual | recall real | acc low | acc med | acc high |", "|---|---|---|---|---|---|---|"]
    for n, p in preds.items():
        acc = float(np.mean(p == y))
        rv, rr = float(np.mean(p[y == "virtual"] == "virtual")), float(np.mean(p[y == "real"] == "real"))
        by = [float(np.mean(p[sev == s] == y[sev == s])) for s in ("low", "med", "high")]
        L.append(f"| {n} | {acc:.3f} | {rv:.2f} | {rr:.2f} | {by[0]:.2f} | {by[1]:.2f} | {by[2]:.2f} |")
        res[n] = {"acc": acc, "recall_virtual": rv, "recall_real": rr}
    (res_dir(ROOT, "phase3") / "mode_typing.md").write_text("\n".join(L), encoding="utf-8")
    (res_dir(ROOT, "phase3") / "mode_typing.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
