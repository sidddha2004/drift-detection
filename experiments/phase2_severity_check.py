"""Phase 2: do the severity levels bite, and monotonically?

Train one frozen local LSTM per client on the clean v2 train split (same recipe as E1, tuning seed 1001), then apply every
transform at alpha = 1 to the TEST pool of all clients and measure macro-F1. Expected: low > med > high and a
clear drop for the high level; the control (no drift) equals the clean score. Output: results/phase2/severity_check.{md,json}
"""
import copy
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import DataConfig, build_federation  # noqa: E402
from src.drift import load_yaml  # noqa: E402
from src.drift.schedule import DriftEvent  # noqa: E402
from src.drift.transforms import apply_real, apply_virtual  # noqa: E402
from src.fl.engine import TrainConfig, evaluate, make_model, to_tensors, train_epochs  # noqa: E402
from src.utils.io import load_config, run_metadata, save_json  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402

import torch  # noqa: E402

SEED, EPOCHS = 1001, 60          # tuning seed (analysis plan): design calibration never uses evaluation seeds
ALL, HAND, CHEST, ANKLE = tuple(range(9)), (0, 1, 2), (3, 4, 5), (6, 7, 8)
CASES = [("offset", "all sensors", ALL), ("offset", "hand only", HAND), ("gain", "all sensors", ALL), ("gain", "ankle only", ANKLE),
         ("rotation", "all sensors", ALL), ("rotation", "chest only", CHEST), ("label_remap", "real drift", ())]


def main():
    cfg = load_config(ROOT / "configs/phase1_v2_trimmed.yaml")
    clients, lm, _ = build_federation(DataConfig(**cfg["data"]))
    n_cls, tc = len(lm), TrainConfig(**cfg["train"])
    sev = load_yaml(ROOT / "configs/scenarios/severity.yaml")
    seed_everything(SEED)
    models, T = {}, {}
    for cid, c in clients.items():
        T[cid] = {s: to_tensors(c.x[s], c.y[s]) for s in ("train", "val", "test")}
        gen = torch.Generator().manual_seed(SEED * 1000 + cid)
        m = make_model(n_cls, 9, tc)
        opt = torch.optim.Adam(m.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        best, state = -1, None
        for ep in range(EPOCHS):
            train_epochs(m, *T[cid]["train"], 1, tc, gen, opt)
            f1 = evaluate(m, *T[cid]["val"], n_cls)["macro_f1"]
            if f1 > best:
                best, state = f1, copy.deepcopy(m.state_dict())
        m.load_state_dict(state)
        models[cid] = m
        print("trained", cid, f"val {best:.3f}", flush=True)

    def score(make_event):
        f1s = []
        for cid, c in clients.items():
            ev = make_event(cid, c)
            x, y = c.x["test"], c.y["test"]
            if ev is not None:
                x = apply_virtual(x, c.mean, c.std, ev, 1.0)
                y = apply_real(y, ev, 1.0, np.random.default_rng(0))
            f1s.append(evaluate(models[cid], *to_tensors(x, y), n_cls)["macro_f1"])
        return float(np.mean(f1s)), float(np.min(f1s))

    res = {"clean": score(lambda cid, c: None)}
    L = [f"# Phase 2: severity check (frozen local model per client, test pool, alpha = 1, seed {SEED}, 8 clients)", "",
         f"Clean (no drift) macro-F1: mean {res['clean'][0]:.3f}, worst client {res['clean'][1]:.3f}", "",
         "| transform | scope | low | med | high |", "|---|---|---|---|---|"]
    for t, scope, ch in CASES:
        row = []
        for lvl in ("low", "med", "high"):
            mag = float(sev[t][lvl])
            if t == "label_remap":
                def mk(cid, c, mag=mag):
                    cls = sorted(set(c.y["train"].tolist()))
                    rng = np.random.default_rng((SEED, cid))
                    k = min(int(mag), len(cls) - len(cls) % 2)
                    pick = [int(v) for v in rng.permutation(cls)[:k]]
                    return DriftEvent(cid, 0, "sudden", "real", t, (), mag, 0, remap=tuple((pick[i], pick[i + 1]) for i in range(0, k - 1, 2)))
            else:
                def mk(cid, c, mag=mag, ch=ch, t=t):
                    return DriftEvent(cid, 0, "sudden", "virtual", t, ch, mag, 0)
            row.append(score(mk))
            res[f"{t}|{scope}|{lvl}"] = row[-1]
        L.append(f"| {t} | {scope} | " + " | ".join(f"{m:.3f} (worst {w:.2f})" for m, w in row) + " |")
        res[f"mono|{t}|{scope}"] = bool(row[0][0] >= row[1][0] >= row[2][0])
    L += ["", "Monotone (low >= med >= high): " + ", ".join(f"{k.split('|', 1)[1]}={v}" for k, v in res.items() if k.startswith("mono"))]
    out = ROOT / "results/phase2"
    (out / "severity_check.md").write_text("\n".join(L), encoding="utf-8")
    save_json({"meta": run_metadata(), "results": res}, out / "severity_check.json")
    print("\n".join(L))


if __name__ == "__main__":
    main()
