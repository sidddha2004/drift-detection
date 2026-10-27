"""Benchmark version switch: env var DRIFT_BENCH = v1 (default, 60-round streams) | v2 (Amendment A4, 120-round streams).

Selects the scenario directory and keeps the caches / result folders of the two benchmarks apart. Pretrained federation
models do not depend on the benchmark and are shared (results/raw/phase4).
"""
import os
from pathlib import Path

BENCH = os.environ.get("DRIFT_BENCH", "v1")


def scn_dir(root: Path) -> Path:
    return Path(root) / ("configs/scenarios" if BENCH == "v1" else f"configs/scenarios_{BENCH}")


def _sub(base: Path, phase: str) -> Path:
    return Path(base) / (phase if BENCH == "v1" else f"{phase}_{BENCH}")


def raw_dir(root: Path, phase: str) -> Path:
    return _sub(Path(root) / "results/raw", phase)


def res_dir(root: Path, phase: str) -> Path:
    return _sub(Path(root) / "results", phase)
