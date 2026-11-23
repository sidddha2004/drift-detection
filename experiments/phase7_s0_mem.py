"""Supervisor feedback C8/C9 (descriptive): S0 no-drift control and the A6-selected memory configuration on S3 (10 evaluation seeds).

    python experiments/phase7_s0_mem.py   -> results/phase7/s0_mem.md / .json
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/phase4_v2"


def load(pattern):
    out = []
    for f in sorted(RES.glob(pattern)):
        out += json.load(open(f))["runs"]
    return out


def boot_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(d, (n, len(d))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def s0(runs):
    rows = {}
    for p in sorted({r["policy"] for r in runs}):
        rs = [r for r in runs if r["policy"] == p]
        windows = []
        alarms = [len(r["alarms"]) for r in rs]
        f1 = defaultdict(list)
        for r in rs:
            f1[r["seed"]].append(float(np.mean(r["f1"])))
        rows[p] = {"seeds": len(f1), "alarms_per_stream": float(np.mean(alarms)),
                   "alarms_per_client_stream": float(np.mean(alarms)) / len(rs[0]["clients"]),
                   "f1_whole_stream": float(np.mean([np.mean(v) for v in f1.values()])),
                   "f1_by_seed": {s: float(np.mean(v)) for s, v in f1.items()},
                   "uplink_kB": float(np.mean([sum(r["up_bytes"]) for r in rs]) / 1e3)}
    base = rows.get("frozen_E3b_personal")
    if base:
        for p, row in rows.items():
            d = np.array([row["f1_by_seed"][s] - base["f1_by_seed"][s] for s in row["f1_by_seed"] if s in base["f1_by_seed"]])
            row["diff_vs_personalized"] = float(d.mean())
            row["diff_ci"] = boot_ci(d) if np.any(d) else (0.0, 0.0)
    return rows


def mem(mruns, conf):
    out = {}
    e8 = defaultdict(lambda: defaultdict(list))
    for r in conf:
        if r["policy"] == "typed_learned" and r["scenario"] == "S3":
            e8[r["seed"]]["post"].append(r["metrics"]["post_change_f1"]); e8[r["seed"]]["off"].append(r["metrics"]["off_phase_f1"])
    for p in sorted({r["policy"] for r in mruns}):
        by = defaultdict(lambda: defaultdict(list)); hits = 0
        for r in mruns:
            if r["policy"] == p:
                by[r["seed"]]["post"].append(r["metrics"]["post_change_f1"]); by[r["seed"]]["off"].append(r["metrics"]["off_phase_f1"])
                hits += len(r["memory_hits"])
        seeds = sorted(set(by) & set(e8))
        res = {"seeds": len(seeds), "memory_switches": hits}
        for k in ("post", "off"):
            a = np.array([np.mean(by[s][k]) for s in seeds]); b = np.array([np.mean(e8[s][k]) for s in seeds])
            d = a - b
            res[k] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "tla": float(b.mean()), "diff": float(d.mean()),
                      "ci": boot_ci(d) if np.any(d) else (0.0, 0.0),
                      "p": float(wilcoxon(d).pvalue) if np.any(d) else 1.0}
        out[p] = res
    return out


def main():
    s0r, memr = s0(load("s0_eval_*.json")), mem(load("mem_eval_*.json"), load("conf_*.json"))
    (ROOT / "results/phase7/s0_mem.json").write_text(json.dumps({"s0": s0r, "memory": memr}, indent=1, default=str), encoding="utf-8")
    md = ["# S0 control and memory on S3 (descriptive, 10 evaluation seeds)", "", "## S0 (no drift)", "",
          "| system | seeds | alarms / stream (8 clients) | F1 whole stream | diff vs Personalized [95% CI] | uplink kB |", "|---|---|---|---|---|---|"]
    for p, r in s0r.items():
        md.append(f"| {p} | {r['seeds']} | {r['alarms_per_stream']:.2f} | {r['f1_whole_stream']:.3f} | "
                  f"{r.get('diff_vs_personalized', 0):+.4f} [{r.get('diff_ci', (0, 0))[0]:+.4f}, {r.get('diff_ci', (0, 0))[1]:+.4f}] | {r['uplink_kB']:.0f} |")
    md += ["", "## Memory on S3 (paired with TLA = typed_learned from the confirmatory runs)", "",
           "| system | seeds | switches | post F1 (TLA) | diff [CI] p | off-phase F1 (TLA) | diff [CI] p |", "|---|---|---|---|---|---|---|"]
    for p, r in memr.items():
        a, b = r["post"], r["off"]
        md.append(f"| {p} | {r['seeds']} | {r['memory_switches']} | {a['mean']:.3f} ({a['tla']:.3f}) | {a['diff']:+.4f} [{a['ci'][0]:+.4f}, {a['ci'][1]:+.4f}] p={a['p']:.3f} | "
                  f"{b['mean']:.3f} ({b['tla']:.3f}) | {b['diff']:+.4f} [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}] p={b['p']:.3f} |")
    (ROOT / "results/phase7/s0_mem.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
