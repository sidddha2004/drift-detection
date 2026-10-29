"""Phase 3, step 1: train the frozen local models and cache the per-window signals of every scenario.

For each seed: 8 local LSTMs (v2 protocol), then signals of every client for S0 and for every (scenario, variant,
severity) in S1-S4 (fixed timing). Unaffected clients reuse their clean (S0) signals. Cache: results/raw/phase3/
(git-ignored). Only tuning-side seeds are allowed here (Amendment A3).
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.bench import raw_dir, res_dir, scn_dir  # noqa: E402
from src.data import DataConfig, build_federation  # noqa: E402
from src.detect.signals import client_signals, train_local_models  # noqa: E402
from src.drift import build_schedule, events_from_manifest, load_yaml  # noqa: E402
from src.fl.engine import TrainConfig  # noqa: E402
from src.utils.io import load_config  # noqa: E402

SCENARIOS = ["s1_sudden", "s2_gradual", "s3_recurring", "s4_feature"]
TUNING_SIDE = set(range(1001, 1011))
EPOCHS = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1001, 1011)))
    ap.add_argument("--severities", nargs="+", default=["low", "med", "high"])
    args = ap.parse_args()
    assert set(args.seeds) <= TUNING_SIDE, "evaluation seeds must not be used in Phase 3 (Amendment A3)"
    cfg = load_config(ROOT / "configs/phase1_v2_trimmed.yaml")
    clients, lm, _ = build_federation(DataConfig(**cfg["data"]))
    cids = sorted(clients)
    classes = {c: sorted(set(clients[c].y["train"].tolist())) for c in cids}
    sev = load_yaml(scn_dir(ROOT) / "severity.yaml")
    tc = TrainConfig(**cfg["train"])
    out_dir = raw_dir(ROOT, "phase3")
    out_dir.mkdir(parents=True, exist_ok=True)
    s0 = load_yaml(scn_dir(ROOT) / "s0_none.yaml")
    n_rounds = s0["n_rounds"]
    for seed in args.seeds:
        path = out_dir / f"signals_seed{seed}.pkl"
        if path.exists():
            print("skip", seed, flush=True)
            continue
        t0 = time.time()
        models = train_local_models(clients, len(lm), seed, EPOCHS, tc, out_dir)
        print(f"seed {seed}: models ready [{time.time() - t0:.0f}s]", flush=True)
        clean, cal = {}, {}
        for cid in cids:
            clean[cid], cal[cid] = client_signals(models[cid], clients[cid], [], n_rounds, seed, passes=s0.get("passes", 1))
        streams = {("S0", None, None, cid): clean[cid] for cid in cids}
        manifests = {}
        for name in SCENARIOS:
            scn = load_yaml(scn_dir(ROOT) / f"{name}.yaml")
            for v in scn["variants"]:
                for lvl in args.severities:
                    m = build_schedule(scn, sev, v, lvl, seed, cids, classes)
                    manifests[(scn["scenario"], v, lvl)] = m
                    ev = events_from_manifest(m)
                    for cid in cids:
                        if ev[cid]:
                            streams[(scn["scenario"], v, lvl, cid)] = client_signals(models[cid], clients[cid], ev[cid], n_rounds, seed, passes=m.get("passes", 1))[0]
                        else:
                            streams[(scn["scenario"], v, lvl, cid)] = clean[cid]
        with open(path, "wb") as f:
            pickle.dump({"seed": seed, "streams": streams, "cal": cal, "manifests": manifests}, f)
        print(f"seed {seed}: {len(streams)} streams saved [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
