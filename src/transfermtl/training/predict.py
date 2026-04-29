"""Write per-test-compound predictions in PredictionSchema format.

One row per (smiles, task). For multi-head models, the same compound generates
n_tasks rows.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from transfermtl.models.multihead import MultiHeadModel
from transfermtl.utils.io import write_parquet
from transfermtl.utils.schemas import PredictionSchema


def _logits_to_prob(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


@torch.no_grad()
def predict_dataset(
    model: MultiHeadModel,
    dataset: Iterable[Data],
    seed: int,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> pd.DataFrame:
    """Return a PredictionSchema-valid DataFrame with one row per (smi, task)."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    loader = DataLoader(list(dataset), batch_size=batch_size, shuffle=False)
    rows: list[dict[str, object]] = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)  # dict[task] -> (B,)
        smi_list = list(getattr(batch, "smi", []))
        if not smi_list:
            raise ValueError("Batch is missing `smi`; load via training.data.load_pyg_dataset.")
        # batch.y is (B, n_tasks).
        y_true = batch.y.detach().cpu()
        for ti, task in enumerate(model.task_names):
            probs = _logits_to_prob(out[task]).detach().cpu().tolist()
            for bi, smi in enumerate(smi_list):
                yt = float(y_true[bi, ti].item())
                rows.append(
                    {
                        "smiles": smi,
                        "task": task,
                        "y_true": float("nan") if yt != yt else yt,
                        "y_pred": float(probs[bi]),
                        "seed": int(seed),
                    }
                )
    return pd.DataFrame(rows)


def save_predictions(
    path: str | Path,
    model: MultiHeadModel,
    dataset: Iterable[Data],
    seed: int,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
) -> Path:
    df = predict_dataset(model, dataset, seed=seed, batch_size=batch_size, device=device)
    return write_parquet(path, df, schema=PredictionSchema)
