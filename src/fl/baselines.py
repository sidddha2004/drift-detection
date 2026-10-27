"""Stationary baselines E1 (local-only), E2 (centralised), E3 (FedAvg).

Identical data, backbone, optimiser and epoch budget; checkpoint chosen by mean client
validation macro-F1; metrics reported on the held-out TEST split only.
"""
import copy
import time
from typing import Dict, List

import numpy as np
import torch

from src.data import ClientData
from src.eval import aggregate_clients
from src.fl.engine import (DEVICE, TrainConfig, evaluate, fedavg, make_model, predict, state_bytes,
                           to_tensors, train_epochs)
from src.utils.seed import seed_everything


def _tensors(clients: Dict[int, ClientData]):
    return {cid: {s: to_tensors(c.x[s], c.y[s]) for s in ("train", "val", "test")}
            for cid, c in clients.items()}


def _eval_all(model, T, split, n_classes) -> Dict[int, Dict]:
    return {cid: evaluate(model, *T[cid][split], n_classes) for cid in T}


def _finish(name, seed, test_by_client, extra) -> Dict:
    return {"method": name, "seed": seed,
            "test_per_client": test_by_client,
            "test": aggregate_clients(test_by_client), **extra}


def run_local_only(clients, n_classes, n_channels, tc: TrainConfig, epochs: int, seed: int,
                   keep_predictions: bool = False) -> Dict:
    seed_everything(seed)
    T = _tensors(clients)
    t0, test, curves, best_ep, preds = time.time(), {}, {}, {}, {}
    for cid in T:
        gen = torch.Generator().manual_seed(seed * 1000 + cid)
        model = make_model(n_classes, n_channels, tc)
        opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        best, best_state, curve = -1.0, None, []
        for ep in range(epochs):
            train_epochs(model, *T[cid]["train"], 1, tc, gen, opt)
            f1 = evaluate(model, *T[cid]["val"], n_classes)["macro_f1"]
            curve.append(f1)
            if f1 > best:
                best, best_state, best_ep[cid] = f1, copy.deepcopy(model.state_dict()), ep
        model.load_state_dict(best_state)
        test[cid] = evaluate(model, *T[cid]["test"], n_classes)
        curves[cid] = curve
        if keep_predictions:
            preds[cid] = predict(model, T[cid]["test"][0])
    extra = {"val_curves": curves, "best_epoch": best_ep, "train_seconds": time.time() - t0,
             "comm_bytes_total": 0}
    if keep_predictions:
        extra["test_pred"] = preds
    return _finish("E1_local_only", seed, test, extra)


def run_centralized(clients, n_classes, n_channels, tc: TrainConfig, epochs: int, seed: int) -> Dict:
    seed_everything(seed)
    T = _tensors(clients)
    x = torch.cat([T[c]["train"][0] for c in T]); y = torch.cat([T[c]["train"][1] for c in T])
    gen = torch.Generator().manual_seed(seed)
    model = make_model(n_classes, n_channels, tc)
    opt = torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    t0, best, best_state, best_ep, curve = time.time(), -1.0, None, 0, []
    for ep in range(epochs):
        train_epochs(model, x, y, 1, tc, gen, opt)
        f1 = aggregate_clients(_eval_all(model, T, "val", n_classes))["mean_macro_f1"]
        curve.append(f1)
        if f1 > best:
            best, best_state, best_ep = f1, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(best_state)
    return _finish("E2_centralized", seed, _eval_all(model, T, "test", n_classes),
                   {"val_curve": curve, "best_epoch": best_ep, "train_seconds": time.time() - t0,
                    "comm_bytes_total": None})


def finetune_clients(glob, T, n_classes, tc: TrainConfig, epochs: int, seed: int):
    """Personalise a global model: each client fine-tunes a copy on its own train data (fresh Adam), keeping the
    epoch (1..epochs) with the best client val macro-F1. Local compute only: no extra communication."""
    test, best_ep = {}, {}
    for cid in T:
        gen = torch.Generator().manual_seed(seed * 1000 + cid)
        m = make_model(n_classes, T[cid]["train"][0].shape[-1], tc)
        m.load_state_dict(glob.state_dict())
        opt = torch.optim.Adam(m.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        best, best_state = -1.0, None
        for ep in range(epochs):
            train_epochs(m, *T[cid]["train"], 1, tc, gen, opt)
            f1 = evaluate(m, *T[cid]["val"], n_classes)["macro_f1"]
            if f1 > best:
                best, best_state, best_ep[cid] = f1, copy.deepcopy(m.state_dict()), ep + 1
        m.load_state_dict(best_state)
        test[cid] = evaluate(m, *T[cid]["test"], n_classes)
    return test, best_ep


def run_fedavg(clients, n_classes, n_channels, tc: TrainConfig, rounds: int, local_epochs: int,
               seed: int, finetune_epochs: int = 0) -> Dict:
    seed_everything(seed)
    T = _tensors(clients)
    cids = list(T)
    gen = torch.Generator().manual_seed(seed)
    glob = make_model(n_classes, n_channels, tc)
    payload = state_bytes(glob)
    local = make_model(n_classes, n_channels, tc)
    t0, best, best_state, best_round, curve = time.time(), -1.0, None, 0, []
    for r in range(rounds):
        states, sizes = [], []
        for cid in cids:
            local.load_state_dict(glob.state_dict())
            train_epochs(local, *T[cid]["train"], local_epochs, tc, gen)   # fresh Adam each round
            states.append(copy.deepcopy(local.state_dict()))
            sizes.append(len(T[cid]["train"][1]))
        glob.load_state_dict(fedavg(states, sizes))
        f1 = aggregate_clients(_eval_all(glob, T, "val", n_classes))["mean_macro_f1"]
        curve.append(f1)
        if f1 > best:
            best, best_state, best_round = f1, copy.deepcopy(glob.state_dict()), r
    glob.load_state_dict(best_state)
    extra = {"val_curve": curve, "best_round": best_round, "train_seconds": time.time() - t0,
             "uplink_bytes_per_round": payload * len(cids),
             "downlink_bytes_per_round": payload * len(cids),
             "comm_bytes_total": 2 * payload * len(cids) * rounds,
             "comm_bytes_to_best": 2 * payload * len(cids) * (best_round + 1)}
    out = _finish("E3_fedavg", seed, _eval_all(glob, T, "test", n_classes), extra)
    if finetune_epochs:
        t1 = time.time()
        ft_test, ft_ep = finetune_clients(glob, T, n_classes, tc, finetune_epochs, seed)
        out["ft"] = _finish("E3b_fedavg_ft", seed, ft_test,
                            {**{k: v for k, v in extra.items() if k not in ("val_curve", "train_seconds")},
                             "finetune_best_epoch": ft_ep, "finetune_seconds": time.time() - t1,
                             "train_seconds": extra["train_seconds"] + time.time() - t1})
    return out
