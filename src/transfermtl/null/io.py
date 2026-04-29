"""IO for random-partition null distributions (plan §2.13)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from transfermtl.utils.io import read_npy, write_npy

OUTPUTS_DIR = Path("outputs")


def null_path(dataset: str, statistic: str, M: int, base: Path | None = None) -> Path:
    root = base if base is not None else OUTPUTS_DIR
    return root / "analysis" / dataset / f"null_dist_{statistic}_M{M}.npy"


def save_null_distribution(
    dataset: str,
    statistic: str,
    M: int,
    arr: np.ndarray,
    base: Path | None = None,
) -> Path:
    return write_npy(null_path(dataset, statistic, M, base), np.asarray(arr, dtype=float))


def load_null_distribution(
    dataset: str,
    statistic: str,
    M: int,
    base: Path | None = None,
) -> np.ndarray:
    return read_npy(null_path(dataset, statistic, M, base))
