"""kNN neighborhood partitioning (plan §2.6.3).

Use scaffold cluster centroids — the means of compound embeddings whose
scaffolds belong to each scaffold cluster — as anchor points; assign each
compound to the region of the nearest centroid in encoder space.

Two-layer API mirrors latent.py: a pure-compute helper for tests plus an IO
wrapper that loads A2/A3 outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from transfermtl.partition.latent import EmbedFn
from transfermtl.partition.scaffold import compute_scaffold_partition
from transfermtl.utils.io import read_parquet
from transfermtl.utils.schemas import PartitionSchema, SplitSchema

ScaffoldPartitionFn = Callable[[], pd.DataFrame]


def knn_partition_from_embeddings(
    smiles: Sequence[str],
    embeddings: np.ndarray,
    scaffold_partition: pd.DataFrame,
) -> pd.DataFrame:
    """Assign each compound to the nearest scaffold-cluster centroid.

    Parameters
    ----------
    smiles:
        Compound identifiers, length N.
    embeddings:
        Encoder representations, shape ``[N, D]``.
    scaffold_partition:
        PartitionSchema-valid frame mapping each smiles → scaffold-cluster
        ``region_id``. Provides the anchor groupings for centroid computation.

    Returns
    -------
    PartitionSchema-valid DataFrame with columns ``[smiles, region_id]``.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {embeddings.shape}")
    smiles_arr = np.asarray(list(smiles), dtype=object).astype(str)
    if len(smiles_arr) != embeddings.shape[0]:
        raise ValueError(
            f"smiles length {len(smiles_arr)} != embeddings rows {embeddings.shape[0]}"
        )

    scaffold_partition = PartitionSchema.validate(scaffold_partition)
    sp_lookup = dict(
        zip(
            scaffold_partition["smiles"].astype(str),
            scaffold_partition["region_id"].astype(int),
            strict=True,
        )
    )
    missing = [s for s in smiles_arr if s not in sp_lookup]
    if missing:
        raise KeyError(
            f"scaffold_partition missing {len(missing)} smiles; first few: {missing[:5]}"
        )
    anchor_region = np.asarray([sp_lookup[s] for s in smiles_arr], dtype=int)

    unique_regions = np.array(sorted(np.unique(anchor_region).tolist()), dtype=int)
    centroids = np.stack([embeddings[anchor_region == r].mean(axis=0) for r in unique_regions])

    # Pairwise squared Euclidean: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x_norm = (embeddings**2).sum(axis=1, keepdims=True)
    c_norm = (centroids**2).sum(axis=1, keepdims=True).T
    cross = embeddings @ centroids.T
    sq_dist = x_norm + c_norm - 2.0 * cross
    nearest_idx = np.argmin(sq_dist, axis=1)
    region_id = unique_regions[nearest_idx].astype(int)

    out = pd.DataFrame(
        {
            "smiles": smiles_arr.astype(str),
            "region_id": region_id,
        }
    )
    return PartitionSchema.validate(out)


def compute_knn_partition(
    dataset: str,
    M: int = 5,
    *,
    embed_fn: EmbedFn | None = None,
    embeddings: np.ndarray | None = None,
    scaffold_partition: pd.DataFrame | None = None,
    splits_root: str | Path = "outputs/splits",
    fps_root: str | Path = "outputs/cache/scaffold_fps",
) -> pd.DataFrame:
    """Top-level entry: read split + scaffold partition, compute kNN partition.

    Either ``embeddings`` or ``embed_fn`` must be provided (real encoder
    inference is gated on A3). The scaffold partition is computed on the fly
    if not passed.
    """
    split_path = Path(splits_root) / dataset / "split.parquet"
    df_split = read_parquet(split_path, schema=SplitSchema)
    smiles_list = df_split["smiles"].astype(str).tolist()

    if scaffold_partition is None:
        scaffold_partition = compute_scaffold_partition(
            dataset, M=M, splits_root=splits_root, fps_root=fps_root
        )

    if embeddings is None:
        if embed_fn is None:
            raise ValueError(
                "compute_knn_partition needs an embed_fn or embeddings until A3 lands."
            )
        embeddings = np.asarray(embed_fn(smiles_list))

    return knn_partition_from_embeddings(smiles_list, embeddings, scaffold_partition)
