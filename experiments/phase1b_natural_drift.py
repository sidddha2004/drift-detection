"""Phase 1b: what is the natural temporal shift inside PAMAP2 activity bouts?

The temporal split loses 6-8 macro-F1 points vs the blocked split with no injected drift. Candidate causes:
  (a) input-distribution shift (covariate): feature MMD and distance-to-train-centroid vs position
  (b) label-conditional shift / intrinsically harder late windows: blocked-model error at late positions
  (c) class-frequency change: TV distance of label histograms
  (d) acquisition effects (sensor warm-up): per-bout temperature rise and its rank correlation
Output: results/phase1/natural_drift.{json,md,png}
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import ACC16_COLS, DataConfig, build_federation, load_bouts  # noqa: E402
from src.data.pamap2 import TEMP_COLS  # noqa: E402
from src.fl.baselines import run_local_only  # noqa: E402
from src.fl.engine import TrainConfig  # noqa: E402
from src.utils.io import load_config, run_metadata, save_json  # noqa: E402

SEEDS = [42, 123, 999]
EPOCHS = 60                      # local-only saturates by ~60 epochs (Phase 1 budget sensitivity)
POS_BINS = [0.0, 0.25, 0.5, 0.75, 1.0]


def window_features(x):
    """(n, W, C) -> (n, 3C): per-channel mean, std, std of first difference."""
    return np.concatenate([x.mean(1), x.std(1), np.diff(x, axis=1).std(1)], axis=1)


def mmd2(a, b):
    """Unbiased RBF-MMD^2, median-heuristic bandwidth."""
    z = np.vstack([a, b])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    med = np.median(d2[np.triu_indices(len(z), 1)])
    k = np.exp(-d2 / (med + 1e-12))
    n, m = len(a), len(b)
    kaa, kbb, kab = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1))
                 - 2 * kab.mean())


def bout_meta(root, cfg, subj):
    """Per-bout length and temperature rise, indexed like the provenance bout ids."""
    path = root / cfg.root / f"subject{subj}.dat"
    bouts = load_bouts(path, subj, ACC16_COLS, cfg.activities)
    df = pd.read_csv(path, sep=r"\s+", header=None, na_values=["NaN"], usecols=[0] + TEMP_COLS)
    temp = df[TEMP_COLS].to_numpy()
    out = []
    for b in bouts:
        t = temp[b.start_row: b.start_row + len(b.data)]
        k = max(1, len(t) // 10)
        out.append({"len": len(b.data), "dtemp": float(np.nanmean(t[-k:]) - np.nanmean(t[:k]))})
    return out


def rel_pos(prov, meta, W):
    L = np.array([meta[int(b)]["len"] for b in prov[:, 0]], dtype=float)
    return (prov[:, 1] + W / 2) / L


def main():
    cfg = load_config(ROOT / "configs/phase1_baselines.yaml")
    W = cfg["data"]["window"]
    fed, meta, res = {}, {}, {}
    for mode in ("blocked", "temporal"):
        dc = DataConfig(**dict(cfg["data"], split_mode=mode))
        fed[mode], lm, _ = build_federation(dc)
    n_classes = len(lm)
    for subj in fed["blocked"]:
        meta[subj] = bout_meta(ROOT, DataConfig(**cfg["data"]), subj)

    # (a) input shift
    mmd = {m: [] for m in fed}
    dist_by_pos = defaultdict(list)
    for mode, clients in fed.items():
        for subj, c in clients.items():
            ftr, fte = window_features(c.x["train"]), window_features(c.x["test"])
            for k in np.unique(c.y["train"]):
                a, b = ftr[c.y["train"] == k], fte[c.y["test"] == k]
                if len(a) < 20 or len(b) < 10:
                    continue
                sd = a.std(0) + 1e-6
                mmd[mode].append(mmd2(a / sd, b / sd))
            if mode == "temporal":
                for split in ("train", "val", "test"):
                    f, y = window_features(c.x[split]), c.y[split]
                    pos = rel_pos(c.prov[split], meta[subj], W)
                    for k in np.unique(c.y["train"]):
                        a = ftr[c.y["train"] == k]
                        if len(a) < 20:
                            continue
                        mu, sd = a.mean(0), a.std(0) + 1e-6
                        sel = y == k
                        d = np.sqrt((((f[sel] - mu) / sd) ** 2).mean(1))
                        for p, dd in zip(pos[sel], d):
                            dist_by_pos[min(int(p * 10), 9)].append(dd)
    res["mmd2_train_vs_test"] = {m: {"mean": float(np.mean(v)), "n_pairs": len(v)} for m, v in mmd.items()}
    res["centroid_distance_by_position_decile"] = {str(k): float(np.median(v)) for k, v in sorted(dist_by_pos.items())}

    # (c) class frequency
    tv = {}
    for mode, clients in fed.items():
        vals = []
        for c in clients.values():
            p = np.bincount(c.y["train"], minlength=n_classes) / c.n("train")
            q = np.bincount(c.y["test"], minlength=n_classes) / c.n("test")
            vals.append(0.5 * np.abs(p - q).sum())
        tv[mode] = float(np.mean(vals))
    res["class_freq_tv_train_vs_test"] = tv

    # (b) error by position
    err = {m: defaultdict(list) for m in fed}
    per_bout_err = defaultdict(list)
    for mode, clients in fed.items():
        for seed in SEEDS:
            r = run_local_only(clients, n_classes, 9, TrainConfig(**cfg["train"]), EPOCHS, seed, keep_predictions=True)
            print(f"{mode} seed {seed}: macroF1 {r['test']['mean_macro_f1']:.4f}", flush=True)
            for subj, c in clients.items():
                wrong = (np.asarray(r["test_pred"][subj]) != c.y["test"]).astype(float)
                pos = rel_pos(c.prov["test"], meta[subj], W)
                for lo, hi in zip(POS_BINS[:-1], POS_BINS[1:]):
                    sel = (pos >= lo) & (pos < hi + (1e-9 if hi == 1 else 0))
                    if sel.any():
                        err[mode][f"{lo:.2f}-{hi:.2f}"].append(float(wrong[sel].mean()))
                if mode == "temporal":
                    for bid in np.unique(c.prov["test"][:, 0]):
                        per_bout_err[(subj, int(bid))].append(float(wrong[c.prov["test"][:, 0] == bid].mean()))
    res["error_rate_by_position"] = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in err.items()}

    # (d) warm-up
    dt, e = [], []
    for (subj, bid), v in per_bout_err.items():
        dt.append(meta[subj][bid]["dtemp"])
        e.append(np.mean(v))
    rho, p = spearmanr(dt, e)
    res["warmup"] = {"mean_bout_temp_rise_C": float(np.mean(dt)), "spearman_dtemp_vs_temporal_error": float(rho),
                     "p_value": float(p), "n_bouts": len(dt)}

    # report
    labels = [f"{a:.2f}-{b:.2f}" for a, b in zip(POS_BINS[:-1], POS_BINS[1:])]
    L = ["# Phase 1b: natural temporal shift in PAMAP2", "",
         f"Local-only LSTM, {EPOCHS} epochs, seeds {SEEDS}, 8 clients. The temporal split loses ~6-8 macro-F1 vs blocked; why?", "",
         "## (a) Input-distribution shift", "",
         "Unbiased RBF-MMD^2 between train and test window features (per client x activity; mean over pairs):", "",
         "| split | MMD^2 | pairs |", "|---|---|---|"]
    for m in mmd:
        L.append(f"| {m} | {res['mmd2_train_vs_test'][m]['mean']:.4f} | {len(mmd[m])} |")
    dec = res["centroid_distance_by_position_decile"]
    L += ["", "Median distance of a window's features to its class's train centroid (train-std units) by position in bout "
              "(temporal split; train covers 0-0.6, val ~0.6-0.75, test 0.75-1):", "",
          "| position decile | " + " | ".join(str(k) for k in range(10)) + " |", "|---|" + "---|" * 10,
          "| median distance | " + " | ".join(f"{dec.get(str(k), float('nan')):.2f}" for k in range(10)) + " |",
          "", "## (b) Are late windows intrinsically harder? Test error rate by position in the bout", "",
          "| split | " + " | ".join(labels) + " |", "|---|" + "---|" * 4]
    for m in err:
        L.append(f"| {m} | " + " | ".join(f"{res['error_rate_by_position'][m].get(l, float('nan')):.3f}" for l in labels) + " |")
    L += ["", "If the blocked model is also worse late in the bout, late windows are intrinsically harder (e.g. label noise near "
              "activity ends). If it is flat while the temporal model is bad, the loss is extrapolation to shifted input.", "",
          "## (c) Class frequencies", "",
          f"TV distance of train vs test label histograms: blocked {tv['blocked']:.4f}, temporal {tv['temporal']:.4f}.", "",
          "## (d) Sensor warm-up", "",
          f"Mean temperature rise over a bout: {res['warmup']['mean_bout_temp_rise_C']:.2f} C. "
          f"Spearman(bout temperature rise, temporal-split error) = {rho:.3f} (p={p:.3g}, n={len(dt)} bouts)."]
    out = ROOT / "results/phase1"
    (out / "natural_drift.md").write_text("\n".join(L), encoding="utf-8")
    save_json({"meta": run_metadata(), **res}, out / "natural_drift.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ks = sorted(int(k) for k in dec)
    ax[0].plot([k / 10 + 0.05 for k in ks], [dec[str(k)] for k in ks], "o-")
    ax[0].axvline(0.6, ls=":", c="gray")
    ax[0].set_xlabel("relative position in bout")
    ax[0].set_ylabel("median distance to train centroid")
    ax[0].set_title("Input shift vs time (temporal split)")
    xs = [(a + b) / 2 for a, b in zip(POS_BINS[:-1], POS_BINS[1:])]
    for m in err:
        ax[1].plot(xs, [res["error_rate_by_position"][m].get(l, np.nan) for l in labels], "o-", label=m)
    ax[1].set_xlabel("relative position in bout")
    ax[1].set_ylabel("test error rate")
    ax[1].legend()
    ax[1].set_title("Error by position")
    fig.tight_layout()
    fig.savefig(out / "natural_drift.png", dpi=150)
    print("\n".join(L))


if __name__ == "__main__":
    main()
