"""Shared pytest fixtures. Tests never depend on the PAMAP2 files unless marked."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAMAP2_SUBJECT_101 = ROOT / "PAMAP2_Dataset" / "Protocol" / "subject101.dat"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def stream(rng):
    """Synthetic 9-channel stream (6000 samples) with 3 contiguous label blocks."""
    x = rng.standard_normal((6000, 9)).astype(np.float32)
    y = np.repeat(np.array([1, 2, 3]), 2000).astype(np.int64)
    return x, y


requires_pamap2 = pytest.mark.skipif(
    not PAMAP2_SUBJECT_101.exists(), reason="PAMAP2 dataset not present"
)
