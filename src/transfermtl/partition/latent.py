"""Latent embedding cluster partitioning (plan §2.6.2).

K-means on encoder representations from a single all-task MTL run (seed=0),
M ∈ {3, 5, 8, 10}, k-means++ init with 10 restarts.

Two-layer API:
- ``latent_partition_from_embeddings`` — pure compute, takes an [N, D]
  embedding matrix. Used by tests with synthetic embeddings; never touches
  disk.
- ``compute_latent_partition`` — IO wrapper that loads A3's all-task
  checkpoint and computes embeddings. **Stub policy**: until A3 lands a real
  checkpoint, callers can pass a custom ``embed_fn`` (e.g. a torch model
  closure) so this module is exercisable on the synthetic fixture and on
  ad-hoc fake checkpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from transfermtl.utils.io import read_parquet
from transfermtl.utils.schemas import PartitionSchema, SplitSchema

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def latent_partition_from_embeddings(
    smiles: Sequence[str],
    embeddings: np.ndarray,
    M: int = 5,
    seed: int = 0,
    n_init: int = 10,
) -> pd.DataFrame:
    """K-means on encoder embeddings.

    Parameters
    ----------
    smiles:
        Compound identifiers, length N.
    embeddings:
        Encoder representations, shape ``[N, D]``.
    M:
        Number of clusters.
    seed:
        Random seed for k-means++ initialization.
    n_init:
        Number of k-means restarts (default 10 per plan §2.6.2).

    Returns
    -------
    PartitionSchema-valid DataFrame with columns ``[smiles, region_id]``.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    smiles_arr = np.asarray(list(smiles), dtype=object)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    if len(smiles_arr) != embeddings.shape[0]:
        raise ValueError(
            f"smiles length {len(smiles_arr)} != embeddings rows {embeddings.shape[0]}"
        )

    n_clusters = min(M, embeddings.shape[0])
    model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=n_init,
        random_state=seed,
    )
    labels = model.fit_predict(embeddings).astype(int)

    out = pd.DataFrame(
        {
            "smiles": smiles_arr.astype(str),
            "region_id": labels,
        }
    )
    return PartitionSchema.validate(out)


def _embed_via_checkpoint(
    checkpoint_path: str | Path,
    smiles: Sequence[str],  # noqa: ARG001
) -> np.ndarray:
    """Load A3's all-task checkpoint and return encoder embeddings.

    Stub: A3 has not yet defined the checkpoint loader API or the encoder's
    inference path on bare SMILES. This helper raises a clear error pointing
    callers at ``embed_fn`` (the path-injection hook used by tests) until A3
    lands.
    """
    raise NotImplementedError(
        "A3 has not landed yet — pass an explicit embed_fn or embeddings to "
        f"compute_latent_partition. Tried to load: {checkpoint_path}"
    )


def compute_latent_partition(
    dataset: str,
    M: int = 5,
    seed: int = 0,
    *,
    checkpoint_path: str | Path | None = None,
    embed_fn: EmbedFn | None = None,
    embeddings: np.ndarray | None = None,
    splits_root: str | Path = "outputs/splits",
    checkpoints_root: str | Path = "outputs/checkpoints",
) -> pd.DataFrame:
    """Top-level entry: load split, compute embeddings, run k-means.

    Three ways to provide embeddings (in priority order):

    1. ``embeddings`` — precomputed matrix; smiles order must match
       ``df_split``.
    2. ``embed_fn`` — callable mapping a list of SMILES to an ``[N, D]``
       array. Used by tests to inject a fake encoder.
    3. ``checkpoint_path`` (or default A3 path) — load checkpoint, run real
       encoder. Currently raises NotImplementedError until A3 lands.
    """
    split_path = Path(splits_root) / dataset / "split.parquet"
    df_split = read_parquet(split_path, schema=SplitSchema)
    smiles_list = df_split["smiles"].astype(str).tolist()

    if embeddings is None:
        if embed_fn is not None:
            embeddings = np.asarray(embed_fn(smiles_list))
        else:
            ckpt = (
                Path(checkpoint_path)
                if checkpoint_path is not None
                else Path(checkpoints_root) / dataset / "all_task" / "seed0.pt"
            )
            embeddings = _embed_via_checkpoint(ckpt, smiles_list)

    return latent_partition_from_embeddings(smiles_list, embeddings, M=M, seed=seed)
