"""A6 M2: client-level Gaussian mechanism on uploaded updates and its (epsilon, delta) accounting.

Every upload releases  start + clip(update, C) + N(0, (z C)^2 I).  The privacy unit is the client: a neighbouring stream may
replace everything the client holds, so the L2 sensitivity of one release is 2C (the difference of two clipped vectors), and
each release is the Gaussian mechanism with effective noise multiplier z / 2. n releases compose in Renyi DP
(eps_RDP(alpha) = n alpha / (2 z_eff^2)), converted to (epsilon, delta). No subsampling amplification is used.
"""
import math
from typing import Dict, Tuple

import numpy as np
import torch

ALPHAS = np.concatenate([np.linspace(1.01, 10, 200), np.arange(11, 512)])


def clip_and_noise(state: Dict[str, torch.Tensor], start: Dict[str, torch.Tensor], clip: float, z: float,
                   gen: torch.Generator) -> Tuple[Dict[str, torch.Tensor], float]:
    """Return (released state, unclipped update norm). clip <= 0 leaves the update unclipped (norm logging only)."""
    keys = [k for k in state if state[k].is_floating_point()]
    upd = {k: state[k] - start[k] for k in keys}
    norm = float(torch.sqrt(sum((v.double() ** 2).sum() for v in upd.values())))
    scale = min(1.0, clip / max(norm, 1e-12)) if clip > 0 else 1.0
    out = dict(state)
    for k in keys:
        noise = torch.randn(upd[k].shape, generator=gen).to(upd[k].device) * (z * clip) if (z > 0 and clip > 0) else 0.0
        out[k] = start[k] + upd[k] * scale + noise
    return out, norm


def epsilon(n_releases: int, z: float, delta: float = 1e-5, sensitivity_factor: float = 2.0) -> float:
    """(epsilon, delta) of n releases of the Gaussian mechanism with noise multiplier z (sigma = z C, sensitivity factor x C)."""
    if n_releases == 0:
        return 0.0
    if z <= 0:
        return math.inf
    z_eff = z / sensitivity_factor
    rdp = n_releases * ALPHAS / (2 * z_eff ** 2)
    return float(np.min(rdp + math.log(1 / delta) / (ALPHAS - 1)))
