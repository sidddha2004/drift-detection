"""Phase 4: streaming federated runs. Policies x scenario variants x seeds -> results/phase4/<tag>.json (+ per-round curves).

Only tuning-side seeds are allowed (1001-1010); evaluation seeds are reserved for Phase 6 (Amendment A3 carried forward).
Usage: python experiments/phase4_streaming.py --tag s1_first_look --variants S1:virtual_offset_partial ... --policies ...
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.bench import raw_dir, res_dir, scn_dir  # noqa: E402
from src.data import DataConfig, build_federation  # noqa: E402
from src.drift import ClientStream, build_schedule, events_from_manifest, load_yaml  # noqa: E402
from src.eval.streaming import stream_metrics  # noqa: E402
from src.fl.engine import DEVICE, TrainConfig, make_model  # noqa: E402
from src.fl.streaming import Policy, Response, pretrain_federation, run_stream  # noqa: E402
from src.utils.io import load_config, run_metadata  # noqa: E402
from src.detect.detectors import make_calibration  # noqa: E402
from src.detect.drift_type import FEATURE_NAMES, Memory, episode_features, series  # noqa: E402
from src.detect.signals import cal_signals  # noqa: E402

TUNING_SIDE = set(range(1001, 1011))
SCN_FILE = {"S1": "s1_sudden", "S2": "s2_gradual", "S3": "s3_recurring", "S4": "s4_feature"}
CACHE = ROOT / "results/raw/phase4"

R_FULL_LOCAL = Response("full_local", "full", federated=False)
R_FULL_FED = Response("full_fed", "full", federated=True)
R_AD_LOCAL = Response("adapter_local", "adapter", federated=False)
R_AD_FED = Response("adapter_fed", "adapter", federated=True)
R_IN_LOCAL = Response("input_local", "input", federated=False)
R_IN_FED = Response("input_fed", "input", federated=True)
R_FULL_FED_1R = Response("full_fed_1round", "full", federated=True, rounds=1)

R_ALIGN_ONESHOT = Response("align_oneshot", "align", False, rounds=8)
R_ALIGN_TRACK = Response("align_track", "align", False, rounds=30, track=True, buffer_rounds=12)
R_HEAD_LONG = Response("head_long", "head", False, rounds=40, buffer_rounds=45, epochs=8, replay_unseen=True)
R_REVERT = Response("revert", "revert", False)


def typed_oracle(info):
    """Oracle-typed response (ground-truth drift type from the manifest): the upper bound of what typing can buy."""
    if info.get("event") == "offset":
        return R_REVERT
    if info["mode"] == "real":
        return R_HEAD_LONG
    t = info["type4"]
    if t == "gradual":
        return R_ALIGN_TRACK
    if t == "recurring" and info.get("occurrence", 0) > 0:
        return Response("restore", "restore", False)
    if t == "feature_level":
        return Response("align_imus", "align", False, rounds=8, imus=info["imus"])
    return R_ALIGN_ONESHOT


def mode_typed_oracle(info):
    """2-way oracle typing only (virtual vs real): no tracking, no revert/restore, no sensor restriction."""
    return R_HEAD_LONG if info["mode"] == "real" else R_ALIGN_ONESHOT


POLICIES = {
    "frozen_E3_global": Policy("frozen_E3_global", "none", start_model="global"),
    "frozen_E3b_personal": Policy("frozen_E3b_personal", "none"),
    "periodic_E3c_full_fed": Policy("periodic_E3c_full_fed", "periodic", R_FULL_FED_1R, period=10),
    "oracle_full_local": Policy("oracle_full_local", "oracle", R_FULL_LOCAL),
    "oracle_full_fed": Policy("oracle_full_fed", "oracle", R_FULL_FED),
    "oracle_adapter_local": Policy("oracle_adapter_local", "oracle", R_AD_LOCAL),
    "oracle_adapter_fed": Policy("oracle_adapter_fed", "oracle", R_AD_FED),
    "oracle_input_local": Policy("oracle_input_local", "oracle", R_IN_LOCAL),
    "oracle_input_fed": Policy("oracle_input_fed", "oracle", R_IN_FED),
    "oracle_input_local_nr": Policy("oracle_input_local_nr", "oracle", Response("input_local_nr", "input", False, replay=0.0)),
    "oracle_input_local_nr_long": Policy("oracle_input_local_nr_long", "oracle", Response("input_local_nr_long", "input", False, replay=0.0, buffer_rounds=40, epochs=30, rounds=5)),
    "oracle_adapter_local_nr": Policy("oracle_adapter_local_nr", "oracle", Response("adapter_local_nr", "adapter", False, replay=0.0)),
    "oracle_align": Policy("oracle_align", "oracle", Response("align", "align", False, rounds=8)),
    "ph_align": Policy("ph_align", "detector", Response("align", "align", False, rounds=8)),
    "oracle_head_unseen_replay": Policy("oracle_head_unseen_replay", "oracle", Response("head_ur", "head", False, replay_unseen=True)),
    "oracle_adapter_unseen_replay": Policy("oracle_adapter_unseen_replay", "oracle", Response("adapter_ur", "adapter", False, replay_unseen=True)),
    "oracle_full_unseen_replay": Policy("oracle_full_unseen_replay", "oracle", Response("full_ur", "full", False, replay_unseen=True)),
    "oracle_head_long": Policy("oracle_head_long", "oracle", Response("head_long", "head", False, rounds=40, buffer_rounds=45, epochs=8, replay_unseen=True)),
    "oracle_adapter_long": Policy("oracle_adapter_long", "oracle", Response("adapter_long", "adapter", False, rounds=40, buffer_rounds=45, epochs=8, replay_unseen=True)),
    "oracle_full_long": Policy("oracle_full_long", "oracle", Response("full_long", "full", False, rounds=40, buffer_rounds=45, epochs=8, replay_unseen=True)),
    "ph_head_long": Policy("ph_head_long", "detector", Response("head_long", "head", False, rounds=40, buffer_rounds=45, epochs=8, replay_unseen=True)),
    "typed_oracle": Policy("typed_oracle", "oracle", respond_by_type=typed_oracle, oracle_offsets=True),
    "mode_typed_oracle": Policy("mode_typed_oracle", "oracle", respond_by_type=mode_typed_oracle, oracle_offsets=False),
    "fixed_align_oneshot": Policy("fixed_align_oneshot", "oracle", R_ALIGN_ONESHOT, oracle_offsets=False),
    "fixed_align_track": Policy("fixed_align_track", "oracle", R_ALIGN_TRACK),
    "fixed_head_long": Policy("fixed_head_long", "oracle", R_HEAD_LONG),
    "fixed_full_fed": Policy("fixed_full_fed", "oracle", R_FULL_FED),
    "fixed_adapter_fed": Policy("fixed_adapter_fed", "oracle", R_AD_FED),
    "typed_learned": Policy("typed_learned", "detector", respond_by_type=mode_typed_oracle),
    "typed_learned_conf80": Policy("typed_learned_conf80", "detector", respond_by_type=mode_typed_oracle),
    "ph_mode_oracle": Policy("ph_mode_oracle", "detector", respond_by_type=mode_typed_oracle),
    "adwin_full_fed_E4": Policy("adwin_full_fed_E4", "detector", R_FULL_FED, detector_kind="adwin"),
    "ph_full_fed_E5": Policy("ph_full_fed_E5", "detector", R_FULL_FED),
    "ph_fedprox_mu001": Policy("ph_fedprox_mu001", "detector", Response("full_fed_prox", "full", federated=True, prox=0.01)),
    "ph_fedprox_mu1": Policy("ph_fedprox_mu1", "detector", Response("full_fed_prox", "full", federated=True, prox=1.0)),
    "ph_fedprox_E7a": Policy("ph_fedprox_E7a", "detector", Response("full_fed_prox", "full", federated=True, prox=1.0)),
    "ph_full_local": Policy("ph_full_local", "detector", R_FULL_LOCAL),
    "ph_adapter_local": Policy("ph_adapter_local", "detector", R_AD_LOCAL),
    "ph_adapter_fed": Policy("ph_adapter_fed", "detector", R_AD_FED),
}


LABEL_FREE = [FEATURE_NAMES.index(n) for n in ("dF_pre", "dF_post", "dF_delta", "dF_ramp", "dF_slope", "ent_delta", "ent_ramp",
                                               "imu_share_max", "imu_ratio", "imu_entropy", "rise_dF", "rise_ent", "sr_dF", "sr_ent")]


def train_mode_classifier():
    """Virtual-vs-real classifier on Phase 3 episodes of the seeds 1004-1010 (disjoint from the 1001-1003 stream seeds)."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from experiments.phase3_typing import collect
    tuned = json.loads((res_dir(ROOT, "phase3") / "detector_report.json").read_text())["tuned"]
    import experiments.phase3_typing as T3
    old = T3.SEEDS
    T3.SEEDS = list(range(1004, 1011))
    rows = collect("detected", tuned)
    T3.SEEDS = old
    X = np.stack([r["x"] for r in rows])[:, LABEL_FREE]
    y = np.array([r["mode"] for r in rows])
    sc = StandardScaler().fit(X)
    clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0).fit(sc.transform(X), y)
    print(f"mode classifier trained on {len(y)} episodes ({(y == 'virtual').mean():.2f} virtual)", flush=True)
    return clf, sc


def make_typer(clf, sc, models, clients, seed, events=None, p_virtual=0.5):
    """Deployable typer: features of the episode (label-free subset) -> predicted mode. With `events` it returns the TRUE mode
    instead (oracle mode, to price the classifier), defaulting to 'virtual' for clients without drift."""
    cal = {c: make_calibration(cal_signals(models[c], ClientStream(clients[c], [], 60, seed))) for c in clients}
    mem = {c: Memory() for c in clients}

    def typer(cid, rec):
        ser = series(rec, cal[cid])
        out = episode_features(ser, rec["alarm_window"], rec["per_round"], mem[cid])
        if out is None:
            return None
        feats, sig = out
        mem[cid].sigs.append(sig[:-1]); mem[cid].levels.append(float(sig[-1]))
        if events is not None:
            return {"mode": events[cid][0].mode if events[cid] else "virtual", "type4": None, "channels": ()}
        proba = clf.predict_proba(sc.transform(feats[LABEL_FREE][None]))[0]
        pv = float(proba[list(clf.classes_).index("virtual")])
        # asymmetric costs: aligning a REAL drift hurts a lot, head adaptation on a virtual drift is nearly harmless,
        # so "virtual" must be at least `p_virtual` confident
        return {"mode": "virtual" if pv >= p_virtual else "real", "type4": None, "channels": ()}
    return typer


def apply_tuned(pol):
    """Detector settings come from the tuned detector report of the active benchmark (Phase 3), not from constants."""
    path = res_dir(ROOT, "phase3") / "detector_report.json"
    if pol.trigger != "detector" or not path.exists():
        return pol
    tuned = json.loads(path.read_text())["tuned"]
    if pol.detector_kind == "adwin":
        return dataclasses.replace(pol, detector=(tuned["D1_adwin_err"]["cfg"]["delta"], 0.0))
    c = tuned["D2_ph_err"]["cfg"]
    return dataclasses.replace(pol, detector=(c["delta"], c["lam"]))


def pretrained(seed, clients, cal, n_classes, tc, epochs_rounds=(60, 2, 20)):
    path = CACHE / f"pretrained_seed{seed}.pt"
    if path.exists():
        blob = torch.load(path, map_location=DEVICE)
        glob = make_model(n_classes, 9, tc)
        glob.load_state_dict(blob["global"])
        personal = {}
        for c, st in blob["personal"].items():
            m = make_model(n_classes, 9, tc)
            m.load_state_dict(st)
            personal[int(c)] = m.eval()
        return glob.eval(), personal
    glob, personal = pretrain_federation(clients, cal, n_classes, tc, seed, *epochs_rounds)
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save({"global": glob.state_dict(), "personal": {c: m.state_dict() for c, m in personal.items()}}, path)
    return glob, personal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1001, 1002, 1003])
    ap.add_argument("--variants", nargs="+", required=True, help="S1:virtual_offset_partial ...")
    ap.add_argument("--policies", nargs="+", default=list(POLICIES))
    ap.add_argument("--severity", default="med")
    ap.add_argument("--timing", default=None)
    ap.add_argument("--confirmatory", action="store_true", help="Phase 6: the 10 evaluation seeds (analysis_plan A5)")
    args = ap.parse_args()
    assert args.confirmatory or set(args.seeds) <= TUNING_SIDE, "evaluation seeds are reserved for Phase 6 (--confirmatory)"
    assert not args.confirmatory or not set(args.seeds) & TUNING_SIDE, "--confirmatory takes evaluation seeds only"
    cfg = load_config(ROOT / "configs/phase1_v2_trimmed.yaml")
    clients, lm, _ = build_federation(DataConfig(**cfg["data"]))
    cids = sorted(clients)
    classes = {c: sorted(set(clients[c].y["train"].tolist())) for c in cids}
    sev = load_yaml(scn_dir(ROOT) / "severity.yaml")
    tc = TrainConfig(**cfg["train"])
    out = res_dir(ROOT, "phase4")
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    clf = sc = None
    if any(p in ("typed_learned", "typed_learned_conf80") for p in args.policies):
        clf, sc = train_mode_classifier()
    for seed in args.seeds:
        t0 = time.time()
        cal = {c: ClientStream(clients[c], [], 60, seed).cal_pool() for c in cids}
        glob, personal = pretrained(seed, clients, cal, len(lm), tc)
        print(f"seed {seed}: pretrained [{time.time() - t0:.0f}s]", flush=True)
        for sv in args.variants:
            scn_key, variant = sv.split(":")
            scn = load_yaml(scn_dir(ROOT) / f"{SCN_FILE[scn_key]}.yaml")
            m = build_schedule(scn, sev, variant, args.severity, seed, cids, classes, args.timing)
            ev = events_from_manifest(m)
            for pname in args.policies:
                pol = apply_tuned(POLICIES[pname])
                start = {c: (glob if pol.start_model == "global" else personal[c]) for c in cids}
                if pname == "typed_learned":
                    pol = dataclasses.replace(pol, typer=make_typer(clf, sc, start, clients, seed))
                elif pname == "typed_learned_conf80":
                    pol = dataclasses.replace(pol, typer=make_typer(clf, sc, start, clients, seed, p_virtual=0.8))
                elif pname == "ph_mode_oracle":
                    pol = dataclasses.replace(pol, typer=make_typer(None, None, start, clients, seed, events=ev))
                res = run_stream(clients, start, ev, m["n_rounds"], seed, len(lm), pol, tc, passes=m.get("passes", 1))
                met = stream_metrics(res)
                runs.append({"seed": seed, "scenario": scn_key, "variant": variant, "severity": args.severity, "policy": pname,
                             "metrics": met, "f1": res["f1"].tolist(), "alpha": res["alpha"].tolist(), "alarms": res["alarms"],
                             "up_bytes": res["up_bytes"].tolist(), "down_bytes": res["down_bytes"].tolist(),
                             "drifted": m["drifted_clients"], "clients": res["clients"]})
                print(f"  {scn_key}:{variant:<26} {pname:<24} post {met['post_change_f1']:.3f} pre {met['pre_drift_f1']:.3f} "
                      f"rec {met['recovery_rounds']:.1f} clean {met['clean_clients_post_f1']:.3f} up {met['uplink_bytes'] / 1e3:.0f}kB "
                      f"alarms {met['alarms']} [{res['seconds']:.0f}s]", flush=True)
    (out / f"{args.tag}.json").write_text(json.dumps({"meta": run_metadata(), "args": vars(args), "runs": runs}, default=float), encoding="utf-8")
    print("saved", out / f"{args.tag}.json")


if __name__ == "__main__":
    main()
