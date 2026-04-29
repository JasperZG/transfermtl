"""Scaffold-stratified greedy split (plan §2.2 step 6).

The split is deterministic given the input dataframe + seed: scaffold groups
are sorted by size descending, then assigned to the bin furthest below its
target fraction (with ties broken by bin order: train > val > test).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "val", "test"]


def scaffold_stratified_split(
    df: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42,
    scaffold_col: str = "scaffold",
) -> pd.Series:
    """Return a pd.Series[SplitName] indexed like `df`.

    Algorithm:
      1. group by scaffold
      2. sort scaffold groups by size, descending (ties broken by sorted
         scaffold value, then by a seeded permutation for stable randomness)
      3. for each group, assign to the split whose current count is furthest
         below target_frac * total_n; ties -> train > val > test
    """
    if not np.isclose(train + val + test, 1.0):
        raise ValueError(f"train/val/test must sum to 1.0, got {train + val + test}")

    n = len(df)
    targets = {"train": int(round(train * n)), "val": int(round(val * n))}
    targets["test"] = n - targets["train"] - targets["val"]

    rng = np.random.default_rng(seed)

    # Reproducible scaffold ordering: size desc, then deterministic random tie-break.
    groups = df.groupby(scaffold_col, sort=False).indices
    items = [(scaff, list(idxs)) for scaff, idxs in groups.items()]
    permutation = rng.permutation(len(items))
    items = [items[i] for i in permutation]
    items.sort(key=lambda kv: -len(kv[1]))

    counts = {"train": 0, "val": 0, "test": 0}
    assignment = pd.Series(index=df.index, dtype=object)

    for _scaff, idxs in items:
        # Pick the split most under its target.
        deficits = {s: targets[s] - counts[s] for s in ("train", "val", "test")}
        best = max(("train", "val", "test"), key=lambda s: deficits[s])
        for i in idxs:
            assignment.iloc[i] = best
        counts[best] += len(idxs)

    return assignment.astype(str)
