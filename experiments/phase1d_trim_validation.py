"""Phase 1d: validate the bout-edge trimming rule (D13) BEFORE any model is trained.

Rule under test: cut TRIM_S = 5 s (500 samples = ~2 windows) from both ends of every bout.
Rationale (a priori, not performance based): PAMAP2 labels at activity transitions are only approximately aligned
with the movement, so edge windows may be partly another activity; they are outside the steady-state target.
Checks (no training): (1) coverage kept per client x class, (2) edge windows are atypical of their class,
(3) trimming does not change the covariate content of what remains (MMD v1 vs v2).
Output: results/phase1/trim_validation.{json,md}
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.phase1b_natural_drift import mmd2, window_features  # noqa: E402
from src.data import DataConfig, build_federation  # noqa: E402
from src.utils.io import load_config, run_metadata, save_json  # noqa: E402

TRIM_S, FS = 5, 100
MIN_PER_SPLIT = {"train": 20, "val": 5, "test": 5}   # coverage floor stated before looking at counts


def main():
    cfg = load_config(ROOT / "configs/phase1_baselines.yaml")
    d1 = DataConfig(**cfg["data"])
    d2 = DataConfig(**dict(cfg["data"], trim_samples=TRIM_S * FS))
    c1, lm, _ = build_federation(d1)
    c2, _, _ = build_federation(d2)
    n_cls = len(lm)
    res = {"trim_s": TRIM_S, "per_client": {}, "lost_pairs": [], "below_floor": []}
    edge_d, mid_d, mmds = [], [], []
    L = ["# Phase 1d: validation of the 5 s bout-edge trim", "",
         "| client | windows v1 | windows v2 | kept | classes v1 | classes v2 |", "|---|---|---|---|---|---|"]
    for s in c1:
        a, b = c1[s], c2[s]
        n1 = sum(a.n(k) for k in ("train", "val", "test"))
        n2 = sum(b.n(k) for k in ("train", "val", "test"))
        cl1 = set(np.concatenate([a.y[k] for k in a.y]).tolist())
        cl2 = set(np.concatenate([b.y[k] for k in b.y]).tolist())
        res["per_client"][s] = {"n_v1": n1, "n_v2": n2, "classes_v1": len(cl1), "classes_v2": len(cl2)}
        res["lost_pairs"] += [(s, int(k)) for k in cl1 - cl2]
        for k in cl2:
            for sp, floor in MIN_PER_SPLIT.items():
                n = int((b.y[sp] == k).sum())
                if n < floor:
                    res["below_floor"].append((s, int(k), sp, n))
        L.append(f"| {s} | {n1} | {n2} | {n2 / n1:.2f} | {len(cl1)} | {len(cl2)} |")
        # (2) edge windows vs interior windows in v1: distance to class train centroid
        fa = window_features(np.concatenate([a.x[k] for k in a.x]))
        ya = np.concatenate([a.y[k] for k in a.y])
        pa = np.concatenate([a.prov[k] for k in a.prov])
        # window end position inside bout needs bout length: use window start only for the leading edge, and
        # classify "edge" as starting within the first 5 s; trailing edge via max start per bout
        last = {int(bd): pa[pa[:, 0] == bd][:, 1].max() for bd in np.unique(pa[:, 0])}
        is_edge = np.array([st < TRIM_S * FS or st > last[int(bd)] - TRIM_S * FS + cfg["data"]["window"] // 1 - cfg["data"]["window"]
                            for bd, st, _ in pa])
        ftr = window_features(a.x["train"]); ytr = a.y["train"]
        for k in np.unique(ya):
            ref = ftr[ytr == k]
            sel = ya == k
            if len(ref) < 20 or is_edge[sel].sum() < 5 or (~is_edge[sel]).sum() < 5:
                continue
            mu, sd = ref.mean(0), ref.std(0) + 1e-6
            dist = np.sqrt((((fa[sel] - mu) / sd) ** 2).mean(1))
            edge_d.append(float(np.median(dist[is_edge[sel]]))); mid_d.append(float(np.median(dist[~is_edge[sel]])))
        # (3) covariate content preserved: MMD between v1 and v2 windows of the same class
        fb = window_features(np.concatenate([b.x[k] for k in b.x])); yb = np.concatenate([b.y[k] for k in b.y])
        rng = np.random.default_rng(0)
        for k in cl1 & cl2:
            u, v = fa[ya == k], fb[yb == k]
            if len(u) < 20 or len(v) < 20:
                continue
            sd = u.std(0) + 1e-6
            u = u[rng.permutation(len(u))[:300]]; v = v[rng.permutation(len(v))[:300]]
            mmds.append(mmd2(u / sd, v / sd))
    res["edge_vs_interior_median_distance"] = {"edge": float(np.mean(edge_d)), "interior": float(np.mean(mid_d)),
                                               "frac_pairs_edge_farther": float(np.mean(np.array(edge_d) > np.array(mid_d))),
                                               "n_pairs": len(edge_d)}
    res["mmd2_v1_vs_v2"] = {"mean": float(np.mean(mmds)), "max": float(np.max(mmds)), "n_pairs": len(mmds)}
    e = res["edge_vs_interior_median_distance"]
    L += ["", f"Coverage floor (fixed before looking): train>={MIN_PER_SPLIT['train']}, val>={MIN_PER_SPLIT['val']}, "
              f"test>={MIN_PER_SPLIT['test']} windows per client x class.", "",
          f"- (client, class) pairs lost entirely by trimming: **{len(res['lost_pairs'])}** {res['lost_pairs']}",
          f"- pairs below the floor after trimming: **{len(res['below_floor'])}** {res['below_floor']}", "",
          "## Are edge windows atypical of their class? (median distance to class train centroid, train-std units)", "",
          f"edge {e['edge']:.3f} vs interior {e['interior']:.3f}; edge farther in {e['frac_pairs_edge_farther']:.0%} of "
          f"{e['n_pairs']} client x class pairs. (Supporting evidence only; the rule is justified by label alignment, "
          f"not by this.)", "",
          "## Is the covariate content of the retained windows unchanged? (RBF-MMD^2, v1 vs v2, per client x class)", "",
          f"mean {res['mmd2_v1_vs_v2']['mean']:.4f}, max {res['mmd2_v1_vs_v2']['max']:.4f} "
          f"(reference: train-vs-test MMD^2 under the blocked split in Phase 1b, see natural_drift.md)."]
    out = ROOT / "results/phase1"
    (out / "trim_validation.md").write_text("\n".join(L), encoding="utf-8")
    save_json({"meta": run_metadata(), **res}, out / "trim_validation.json")
    print("\n".join(L))


if __name__ == "__main__":
    main()
