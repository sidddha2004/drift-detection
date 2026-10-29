"""Metrics of a streaming run (see run_stream): primary endpoint and the secondary adaptation / cost metrics.

Primary endpoint (analysis plan, section 2): post-change macro-F1 = mean over the rounds from a client's ground-truth
onset to the end of the stream of its macro-F1 on the held-out test pool, averaged over the drifted clients.
"""
from typing import Dict

import numpy as np


def stream_metrics(res: Dict, recover_frac: float = 0.95) -> Dict:
    f1, alpha, cids = res["f1"], res["alpha"], res["clients"]
    n_rounds = f1.shape[1]
    drifted = [k for k, c in enumerate(cids) if res["onsets"][c]]
    first = min((res["onsets"][cids[k]][0] for k in drifted), default=n_rounds)
    out = {"policy": res["policy"], "n_drifted": len(drifted)}
    post, pre, rec, drop, retention = [], [], [], [], []
    for k in drifted:
        o = res["onsets"][cids[k]][0]
        p0 = float(f1[k, :o].mean())
        post.append(float(f1[k, o:].mean()))
        pre.append(p0)
        drop.append(p0 - float(f1[k, o:].min()))
        thr = recover_frac * p0
        low = o + int(np.argmin(f1[k, o:]))
        back = np.flatnonzero(f1[k, low:] >= thr)
        rec.append(float(back[0]) + (low - o) if len(back) else float(n_rounds - o))
        off = (alpha[k] == 0) & (np.arange(n_rounds) >= o)
        if off.any():
            retention.append(float(f1[k, off].mean()))
    out.update(post_change_f1=float(np.mean(post)) if post else float("nan"),
               pre_drift_f1=float(np.mean(pre)) if pre else float("nan"),
               max_drop=float(np.mean(drop)) if drop else float("nan"),
               recovery_rounds=float(np.mean(rec)) if rec else float("nan"),
               off_phase_f1=float(np.mean(retention)) if retention else float("nan"))
    out["all_clients_post_f1"] = float(f1[:, first:].mean())
    out["worst_client_post_f1"] = float(f1[:, first:].mean(1).min())
    out["clean_clients_post_f1"] = float(np.mean([f1[k, first:].mean() for k in range(len(cids)) if k not in drifted])) if len(drifted) < len(cids) else float("nan")
    out["uplink_bytes"] = float(res["up_bytes"].sum())
    out["downlink_bytes"] = float(res["down_bytes"].sum())
    out["alarms"] = len(res["alarms"])
    out["seconds"] = res["seconds"]
    return out
