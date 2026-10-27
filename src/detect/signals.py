"""Per-window monitoring signals of a frozen model on a client's drifting arrival stream.

signals (one row per arriving window, in arrival order):
  err    1 if the frozen model's prediction != (possibly drifted) label        [needs the label]
  ent    entropy of the softmax output (nats)                                   [label-free]
  feat   window statistics: mean, std, std of first difference per channel (27) [label-free, model-free]
  alpha  drift strength (max over events) of the round the window arrived in    [ground truth, evaluation only]
  rnd    round index of the window                                              [ground truth clock]
"""
import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.data import ClientData
from src.drift import ClientStream
from src.drift.schedule import DriftEvent
from src.fl.engine import DEVICE, TrainConfig, evaluate, make_model, to_tensors, train_epochs
from src.utils.seed import seed_everything


def window_stats(x: np.ndarray) -> np.ndarray:
    """(n, W, C) -> (n, 3C): per-channel mean, std and std of the first difference."""
    return np.concatenate([x.mean(1), x.std(1), np.diff(x, axis=1).std(1)], axis=1).astype(np.float32)


def train_local_models(clients: Dict[int, ClientData], n_classes: int, seed: int, epochs: int, tc: TrainConfig,
                       cache_dir: Path = None) -> Dict[int, torch.nn.Module]:
    """One frozen local LSTM per client (E1 recipe: best validation macro-F1 checkpoint)."""
    path = Path(cache_dir) / f"models_seed{seed}.pt" if cache_dir else None
    if path is not None and path.exists():
        states = torch.load(path, map_location=DEVICE)
        out = {}
        for cid, st in states.items():
            m = make_model(n_classes, 9, tc)
            m.load_state_dict(st)
            out[cid] = m.eval()
        return out
    seed_everything(seed)
    models, states = {}, {}
    for cid, c in clients.items():
        tr, va = to_tensors(c.x["train"], c.y["train"]), to_tensors(c.x["val"], c.y["val"])
        gen = torch.Generator().manual_seed(seed * 1000 + cid)
        m = make_model(n_classes, 9, tc)
        opt = torch.optim.Adam(m.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        best, best_state = -1.0, None
        for _ in range(epochs):
            train_epochs(m, *tr, 1, tc, gen, opt)
            f1 = evaluate(m, *va, n_classes)["macro_f1"]
            if f1 > best:
                best, best_state = f1, copy.deepcopy(m.state_dict())
        m.load_state_dict(best_state)
        models[cid], states[cid] = m.eval(), best_state
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(states, path)
    return models


@torch.no_grad()
def _model_signals(model, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    model.eval()
    logits = torch.cat([model(torch.from_numpy(x[i:i + 512]).to(DEVICE)) for i in range(0, len(x), 512)])
    p = torch.softmax(logits, 1)
    ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1).cpu().numpy()
    pred = logits.argmax(1).cpu().numpy()
    return {"err": (pred != y).astype(np.float32), "ent": ent.astype(np.float32), "pred": pred.astype(np.int16),
            "feat": window_stats(x)}


def cal_signals(model, stream: ClientStream) -> Dict[str, np.ndarray]:
    """Signals of the clean calibration pool (detector references)."""
    return _model_signals(model, *stream.cal_pool())


def client_signals(model, client: ClientData, events: List[DriftEvent], n_rounds: int, seed: int,
                   per_round: int = 0, passes: int = 1):
    """Returns (arrival signals, calibration signals) of one client for one drift schedule."""
    stream = ClientStream(client, events, n_rounds, seed, per_round, passes=passes)
    xs, ys, alpha, rnd = [], [], [], []
    for r in range(n_rounds):
        x, y = stream.arrival_batch(r)
        xs.append(x)
        ys.append(y)
        alpha += [max(stream.alphas(r), default=0.0)] * len(y)
        rnd += [r] * len(y)
    sig = _model_signals(model, np.concatenate(xs), np.concatenate(ys))
    sig.update(alpha=np.asarray(alpha, dtype=np.float32), rnd=np.asarray(rnd, dtype=np.int32),
               per_round=np.int32(stream.per_round))
    return sig, cal_signals(model, stream)
