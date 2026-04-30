"""Region-restricted performance metric (plan §2.8).

For classification: ROC-AUC over the test compounds whose ground-truth label
is non-NaN. Returns NaN if the regional test set fails the §2.11 size or
class-balance check (n_test < test_min, or single-class).

For regression: negative RMSE so a *higher* number is better, matching the
sign convention used by Δ_ij(r) downstream.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TaskType = Literal["clf", "reg"]


def regional_perf(
    predictions: pd.DataFrame,
    region_compounds: list[str] | set[str],
    task: str,
    task_type: TaskType = "clf",
    test_min_clf: int = 30,
    test_min_reg: int = 50,
    test_min_pos: int = 5,
    test_min_neg: int = 5,
) -> float:
    """Region-restricted ROC-AUC (clf) or negative RMSE (reg). NaN if invalid."""
    region_set = set(region_compounds)
    df = predictions[
        (predictions["task"] == task) & (predictions["smiles"].isin(region_set))
    ].copy()
    df = df.dropna(subset=["y_true", "y_pred"])
    if df.empty:
        return float("nan")

    n = len(df)
    if task_type == "clf":
        if n < test_min_clf:
            return float("nan")
        y_true = df["y_true"].to_numpy()
        n_pos = int((y_true == 1).sum())
        n_neg = int((y_true == 0).sum())
        if n_pos < test_min_pos or n_neg < test_min_neg:
            return float("nan")
        try:
            return float(roc_auc_score(y_true, df["y_pred"].to_numpy()))
        except ValueError:
            return float("nan")

    if n < test_min_reg:
        return float("nan")
    err = df["y_true"].to_numpy() - df["y_pred"].to_numpy()
    rmse = float(np.sqrt(np.mean(err**2)))
    return -rmse
