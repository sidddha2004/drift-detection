"""Detection metrics against the manifest (matching rules fixed in Amendment A3).

ground truth  = manifest ONSET change points of the client's events
TP            = first alarm whose round lies in [onset, onset + horizon)
delay         = alarm round - onset round (and in windows); a miss counts as `horizon` for the tuning objective
false alarm   = alarm raised while drift strength alpha == 0
redundant     = alarm while drift is active that is not the first detection of an event (reported separately)
"""
from typing import Dict, List

import numpy as np

HORIZON_ROUNDS = 20


def match_stream(alarms: List[int], S: Dict[str, np.ndarray], onsets: List[int], horizon: int = HORIZON_ROUNDS) -> Dict:
    rnd, alpha = S["rnd"], S["alpha"]
    used, delays_r, delays_w, miss = set(), [], [], 0
    for r0 in onsets:
        w0 = int(np.argmax(rnd >= r0))
        hit = [a for a in alarms if r0 <= rnd[a] < r0 + horizon]
        if hit:
            a = hit[0]
            used.add(a)
            delays_r.append(int(rnd[a] - r0))
            delays_w.append(int(a - w0))
        else:
            miss += 1
    fa = sum(1 for a in alarms if alpha[a] == 0)
    red = sum(1 for a in alarms if alpha[a] > 0 and a not in used)
    return {"n_events": len(onsets), "tp": len(delays_r), "miss": miss, "delays_r": delays_r, "delays_w": delays_w,
            "false_alarms": fa, "redundant": red, "clean_windows": int((alpha == 0).sum()), "windows": int(len(alpha))}


def aggregate(parts: List[Dict], horizon: int = HORIZON_ROUNDS) -> Dict:
    ev = sum(p["n_events"] for p in parts)
    tp = sum(p["tp"] for p in parts)
    dr = [d for p in parts for d in p["delays_r"]]
    dw = [d for p in parts for d in p["delays_w"]]
    clean = sum(p["clean_windows"] for p in parts)
    fa = sum(p["false_alarms"] for p in parts)
    miss = sum(p["miss"] for p in parts)
    return {"events": ev, "tp": tp, "recall": tp / ev if ev else float("nan"),
            "delay_rounds": float(np.mean(dr)) if dr else float("nan"),
            "delay_windows": float(np.mean(dw)) if dw else float("nan"),
            "objective_delay_rounds": float((sum(dr) + miss * horizon) / ev) if ev else float("nan"),
            "false_alarms": fa, "clean_windows": clean, "fa_per_1000": 1000.0 * fa / max(clean, 1),
            "redundant": sum(p["redundant"] for p in parts)}
