"""Adapter from a SplitSchema parquet (+ A2's PyG cache) to PyG `Data` lists.

Two loading modes:

- `cache_dir`: read pre-featurized PyG `Data` from
  `outputs/cache/featurized/{smi_hash}.pt` (A2 produces these).
- `featurize_fn`: call a featurizer at load time. The synthetic fixture path
  (tests) goes through this.

Either path attaches:
  - `data.y`    — tensor of shape (1, n_tasks); NaN for missing labels
  - `data.smi`  — original SMILES string (used by the prediction writer)
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data


def smiles_hash(smi: str) -> str:
    """16-char sha256 prefix used by A2's featurization cache."""
    return hashlib.sha256(smi.encode()).hexdigest()[:16]


def cache_path_for(cache_dir: Path, smi: str) -> Path:
    return cache_dir / f"{smiles_hash(smi)}.pt"


def _attach_labels(data: Data, row: pd.Series, tasks: list[str]) -> Data:
    y_vals = []
    for t in tasks:
        v = row[t]
        y_vals.append(float("nan") if pd.isna(v) else float(v))
    data.y = torch.tensor([y_vals], dtype=torch.float32)
    return data


def load_pyg_dataset(
    split_df: pd.DataFrame,
    split: str,
    tasks: list[str],
    cache_dir: Path | None = None,
    featurize_fn: Callable[[str], Data] | None = None,
) -> list[Data]:
    """Return a list of `Data` for the requested split with `.y` and `.smi` set.

    For pairwise / all-task MTL, drop rows where all listed tasks are NaN — they
    contribute no gradient regardless of architecture (plan §2.5).
    """
    if cache_dir is None and featurize_fn is None:
        raise ValueError("Provide either cache_dir or featurize_fn")

    rows = split_df[split_df["split"] == split]
    out: list[Data] = []
    for _, row in rows.iterrows():
        smi = str(row["smiles"])

        # Drop rows where every requested task label is NaN.
        if all(pd.isna(row[t]) for t in tasks):
            continue

        if featurize_fn is not None:
            data = featurize_fn(smi)
        else:
            assert cache_dir is not None
            data = _load_cached(cache_dir, smi)

        data = _attach_labels(data, row, tasks)
        data.smi = smi
        out.append(data)
    return out


def _load_cached(cache_dir: Path, smi: str) -> Data:
    path = cache_path_for(cache_dir, smi)
    obj: Any = torch.load(path, weights_only=False)
    if not isinstance(obj, Data):
        raise TypeError(f"Cached file {path} is {type(obj).__name__}, expected Data")
    return obj
