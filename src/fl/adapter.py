"""Adapter-only adaptation of a frozen LSTM activity classifier (WP2).

AdaptedLSTM(base) = input recalibration -> frozen LSTM backbone -> bottleneck feature adapter -> head.

  input   x' = x W_in + b_in     W_in (C x C) initialised to the identity, b_in to 0. Can undo per-channel offsets and gains
                                 and 3-axis rotations of a sensor (virtual drift).                  C*C + C parameters
  feature h' = h + up(relu(down(h)))   residual bottleneck on the last hidden state, `up` zero-initialised
  head    a trainable copy of the classifier layer (needed when P(y|x) changes: real drift)

With the initial values the module reproduces the base model exactly. The backbone never receives an update, so only
`adapter_state()` has to be transmitted (a few KB instead of the whole model).
"""
import copy
from typing import Dict

import torch
import torch.nn as nn

PARTS = ("input", "feature", "head")


class AdaptedLSTM(nn.Module):
    def __init__(self, base: nn.Module, bottleneck: int = 8):
        super().__init__()
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.lstm.dropout = 0.0            # cuDNN needs train mode to backprop through the LSTM; no dropout => same maths
        c, h = self.base.lstm.input_size, self.base.lstm.hidden_size
        self.in_w = nn.Parameter(torch.eye(c, device=_dev(base)))
        self.in_b = nn.Parameter(torch.zeros(c, device=_dev(base)))
        self.down = nn.Linear(h, bottleneck).to(_dev(base))
        self.up = nn.Linear(bottleneck, h).to(_dev(base))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.head = copy.deepcopy(self.base.fc)
        for p in self.head.parameters():
            p.requires_grad_(True)
        self.base.fc = nn.Identity()
        self.base.lstm.flatten_parameters()
        self.set_trainable(PARTS)

    def set_trainable(self, parts) -> None:
        parts = set(parts)
        self.in_w.requires_grad_("input" in parts)
        self.in_b.requires_grad_("input" in parts)
        for m in (self.down, self.up):
            for p in m.parameters():
                p.requires_grad_("feature" in parts)
        for p in self.head.parameters():
            p.requires_grad_("head" in parts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.in_w + self.in_b
        _, (h_n, _) = self.base.lstm(x)
        h = h_n[-1]
        h = h + self.up(torch.relu(self.down(h)))
        return self.head(h)

    # ---- what is communicated -------------------------------------------------------------------------------------
    def adapter_names(self):
        return ["in_w", "in_b", "down.weight", "down.bias", "up.weight", "up.bias", "head.weight", "head.bias"]

    def adapter_state(self) -> Dict[str, torch.Tensor]:
        sd = self.state_dict()
        return {k: sd[k].detach().clone() for k in self.adapter_names()}

    def load_adapter_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.load_state_dict(state, strict=False)

    def adapter_params(self) -> int:
        return sum(v.numel() for v in self.adapter_state().values())

    def adapter_bytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.adapter_state().values())

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _dev(m: nn.Module) -> torch.device:
    return next(m.parameters()).device
