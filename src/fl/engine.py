"""Minimal GPU training engine shared by all baselines (Phase 1).

Everything lives on one device as tensors; "clients" are just data shards, so 8 clients x many
rounds is limited by kernel launches, not by transfers. Phase 4 extends this with the
streaming/drift machinery.
"""
import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.data import ClientData
from src.drift_detection.lstm_model import ActivityLSTM, LSTMConfig
from src.eval import classification_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    dropout: float = 0.3
    hidden_size: int = 64
    num_layers: int = 2


def make_model(n_classes: int, n_channels: int, tc: TrainConfig) -> ActivityLSTM:
    cfg = LSTMConfig(input_size=n_channels, output_dim=n_classes, hidden_size=tc.hidden_size,
                     num_layers=tc.num_layers, dropout=tc.dropout)
    return ActivityLSTM(cfg).to(DEVICE)


def to_tensors(x: np.ndarray, y: np.ndarray):
    return (torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE))


def train_epochs(model, x: torch.Tensor, y: torch.Tensor, epochs: int, tc: TrainConfig,
                 gen: torch.Generator, opt=None) -> float:
    """Adam over mini-batches. Returns the mean loss of the last epoch."""
    opt = opt or torch.optim.Adam(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    model.train()
    n, last = len(y), 0.0
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen, device="cpu").to(DEVICE)
        tot = 0.0
        for i in range(0, n, tc.batch_size):
            idx = perm[i:i + tc.batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        last = tot / n
    return last


@torch.no_grad()
def predict(model, x: torch.Tensor, batch_size: int = 512) -> np.ndarray:
    model.eval()
    out = [model(x[i:i + batch_size]).argmax(1) for i in range(0, len(x), batch_size)]
    return torch.cat(out).cpu().numpy()


def evaluate(model, x: torch.Tensor, y: torch.Tensor, n_classes: int) -> Dict:
    return classification_metrics(y.cpu().numpy(), predict(model, x), n_classes)


def state_bytes(model) -> int:
    """FP32 payload of the full model (what a naive FL round transmits)."""
    return sum(p.numel() * p.element_size() for p in model.state_dict().values())


def fedavg(states: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out = copy.deepcopy(states[0])
    for k in out:
        out[k] = sum(float(wi) * s[k].float() for wi, s in zip(w, states)).to(states[0][k].dtype)
    return out
