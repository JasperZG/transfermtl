"""Typed parquet/npy read/write helpers with optional schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_parquet(
    path: str | Path,
    df: pd.DataFrame,
    schema: Any | None = None,
) -> Path:
    p = ensure_parent(path)
    if schema is not None:
        df = schema.validate(df)
    df.to_parquet(p, index=False)
    return p


def read_parquet(
    path: str | Path,
    schema: Any | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if schema is not None:
        df = schema.validate(df)
    return df


def write_npy(path: str | Path, arr: np.ndarray) -> Path:
    p = ensure_parent(path)
    np.save(p, arr, allow_pickle=False)
    return p


def read_npy(path: str | Path) -> np.ndarray:
    arr: np.ndarray = np.load(path, allow_pickle=False)
    return arr
