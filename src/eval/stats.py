"""Statistical protocol (plan §3.6): bootstrap CIs and paired Wilcoxon with Holm correction."""
from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon


def bootstrap_ci(values: Sequence[float], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.allclose(a, b):
        return 1.0
    return float(wilcoxon(a, b).pvalue)
