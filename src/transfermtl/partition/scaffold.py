"""Bemis-Murcko scaffold partitioning (plan §2.6.1).

Hierarchical agglomerative clustering on Tanimoto distance between scaffold
Morgan fingerprints (average linkage), followed by an undersized-region merge
step that keeps every region at ``>= 2 * n_min`` training compounds when
possible.

Two-layer API:
- ``scaffold_partition_from_fps`` — pure compute, takes a scaffold→FP dict.
  Used directly by tests; never touches disk.
- ``compute_scaffold_partition`` — IO wrapper that loads A2's split parquet
  and Morgan-FP cache, then delegates to the pure layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from transfermtl.utils.io import read_parquet
from transfermtl.utils.schemas import PartitionSchema, SplitSchema

DEFAULT_N_MIN = 50  # mirrors configs/_shared/preprocess.yaml::n_min


def _tanimoto_distance_matrix(fp_matrix: np.ndarray) -> np.ndarray:
    """Compute the pairwise Tanimoto distance matrix for binary fingerprints.

    Tanimoto similarity for binary vectors a, b:
        T(a, b) = |a ∩ b| / (|a| + |b| - |a ∩ b|)
    Distance is 1 - T. The diagonal is forced to 0.
    """
    if fp_matrix.ndim != 2:
        raise ValueError(f"fp_matrix must be 2D, got shape {fp_matrix.shape}")

    x = fp_matrix.astype(np.float64)
    intersect = x @ x.T
    sums = x.sum(axis=1)
    denom = sums[:, None] + sums[None, :] - intersect
    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(denom > 0, intersect / denom, 0.0)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    # Numerical safety: clip into [0, 1] (Tanimoto is bounded).
    return np.clip(dist, 0.0, 1.0)


def _hac_assign(distance_matrix: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run sklearn AgglomerativeClustering with precomputed distances, average linkage."""
    n = distance_matrix.shape[0]
    if n_clusters >= n:
        # Each scaffold its own cluster.
        return np.arange(n, dtype=np.int64)
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    return np.asarray(model.fit_predict(distance_matrix), dtype=np.int64)


def _scaffold_train_counts(
    df_split: pd.DataFrame,
    scaffold_to_cluster: dict[str, int],
) -> dict[int, int]:
    """Count train-split compounds in each cluster."""
    train = df_split[df_split["split"] == "train"]
    cluster_col = train["scaffold"].map(scaffold_to_cluster)
    counts = cluster_col.value_counts().to_dict()
    # Ensure every cluster is represented even if it has zero train compounds.
    for cid in set(scaffold_to_cluster.values()):
        counts.setdefault(int(cid), 0)
    return {int(k): int(v) for k, v in counts.items()}


def _cluster_centroids(
    fp_matrix: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    """Mean fingerprint per cluster (continuous-valued centroid)."""
    out: dict[int, np.ndarray] = {}
    for cid in np.unique(cluster_ids):
        mask = cluster_ids == cid
        out[int(cid)] = fp_matrix[mask].mean(axis=0)
    return out


def _merge_undersized(
    cluster_ids: np.ndarray,
    fp_matrix: np.ndarray,
    train_counts: dict[int, int],
    min_train: int,
) -> np.ndarray:
    """Iteratively merge undersized clusters into their nearest neighbor.

    Termination: every cluster has ``>= min_train`` train compounds OR only
    one cluster remains.
    """
    cluster_ids = cluster_ids.copy()
    counts = dict(train_counts)

    while True:
        active = sorted(counts.keys())
        if len(active) <= 1:
            break
        undersized = [cid for cid in active if counts[cid] < min_train]
        if not undersized:
            break

        # Pick the smallest undersized cluster; tie-break by id for determinism.
        target = min(undersized, key=lambda c: (counts[c], c))

        centroids = _cluster_centroids(fp_matrix, cluster_ids)
        target_centroid = centroids[target]

        # Find the nearest active *other* cluster by Euclidean centroid distance.
        best_other = None
        best_dist = np.inf
        for other in active:
            if other == target:
                continue
            d = float(np.linalg.norm(target_centroid - centroids[other]))
            if d < best_dist or (d == best_dist and (best_other is None or other < best_other)):
                best_dist = d
                best_other = other

        assert best_other is not None  # guaranteed by len(active) > 1
        cluster_ids[cluster_ids == target] = best_other
        counts[best_other] = counts[best_other] + counts.pop(target)

    return cluster_ids


def _renumber(cluster_ids: np.ndarray) -> np.ndarray:
    """Renumber cluster ids to a contiguous 0..K-1 range, sorted by first appearance."""
    seen: dict[int, int] = {}
    out = np.empty_like(cluster_ids)
    next_id = 0
    for i, c in enumerate(cluster_ids):
        c = int(c)
        if c not in seen:
            seen[c] = next_id
            next_id += 1
        out[i] = seen[c]
    return out


def scaffold_partition_from_fps(
    df_split: pd.DataFrame,
    scaffold_fps: Mapping[str, np.ndarray],
    M: int = 5,
    n_min: int = DEFAULT_N_MIN,
) -> pd.DataFrame:
    """HAC on scaffold fingerprints, then merge undersized regions.

    Parameters
    ----------
    df_split:
        SplitSchema-valid frame with at least ``smiles``, ``scaffold``,
        ``split`` columns.
    scaffold_fps:
        Mapping of scaffold string to binary fingerprint vector. Must cover
        every scaffold present in ``df_split``.
    M:
        Target number of clusters before merging.
    n_min:
        Per-task minimum training-set size; merge step keeps every region at
        ``>= 2 * n_min`` training compounds when possible.

    Returns
    -------
    PartitionSchema-valid DataFrame with columns ``[smiles, region_id]``.
    """
    if M < 1:
        raise ValueError(f"M must be >= 1, got {M}")
    df_split = SplitSchema.validate(df_split)

    scaffolds = sorted(df_split["scaffold"].unique())
    missing = [s for s in scaffolds if s not in scaffold_fps]
    if missing:
        raise KeyError(
            f"scaffold_fps missing fingerprints for {len(missing)} scaffolds; "
            f"first few: {missing[:5]}"
        )

    fp_matrix = np.stack([np.asarray(scaffold_fps[s]).astype(np.uint8) for s in scaffolds])
    distance = _tanimoto_distance_matrix(fp_matrix)
    cluster_ids = _hac_assign(distance, n_clusters=min(M, len(scaffolds)))

    scaffold_to_cluster = dict(zip(scaffolds, cluster_ids.tolist(), strict=True))
    train_counts = _scaffold_train_counts(df_split, scaffold_to_cluster)

    cluster_ids = _merge_undersized(
        cluster_ids,
        fp_matrix,
        train_counts,
        min_train=2 * n_min,
    )
    cluster_ids = _renumber(cluster_ids)
    scaffold_to_region = dict(zip(scaffolds, cluster_ids.tolist(), strict=True))

    region_id = df_split["scaffold"].map(scaffold_to_region).astype(int)
    out = pd.DataFrame(
        {
            "smiles": df_split["smiles"].astype(str).to_numpy(),
            "region_id": region_id.to_numpy(dtype=int),
        }
    )
    return PartitionSchema.validate(out)


def _load_scaffold_fps(path: str | Path) -> dict[str, np.ndarray]:
    """Load A2's scaffold fingerprint cache into a dict.

    A2 writes ``outputs/cache/scaffold_fps/{dataset}.parquet`` with columns
    ``[scaffold, fp]`` (or legacy ``fingerprint``); the value is a list/array
    of uint8 bits.
    """
    df = pd.read_parquet(path)
    if "scaffold" not in df.columns:
        raise ValueError(f"{path} must have a 'scaffold' column; got {list(df.columns)}")
    if "fp" in df.columns:
        fp_col = "fp"
    elif "fingerprint" in df.columns:
        fp_col = "fingerprint"
    else:
        raise ValueError(f"{path} must have a 'fp' or 'fingerprint' column; got {list(df.columns)}")
    return {
        str(s): np.asarray(fp, dtype=np.uint8)
        for s, fp in zip(df["scaffold"], df[fp_col], strict=True)
    }


def compute_scaffold_partition(
    dataset: str,
    M: int = 5,
    n_min: int = DEFAULT_N_MIN,
    splits_root: str | Path = "outputs/splits",
    fps_root: str | Path = "outputs/cache/scaffold_fps",
) -> pd.DataFrame:
    """Top-level entry: read A2's split + FP cache, return the partition.

    Used by ``scripts/compute_partitions.py``. Tests prefer
    ``scaffold_partition_from_fps`` directly.
    """
    split_path = Path(splits_root) / dataset / "split.parquet"
    fp_path = Path(fps_root) / f"{dataset}.parquet"
    df_split = read_parquet(split_path, schema=SplitSchema)
    fps = _load_scaffold_fps(fp_path)
    return scaffold_partition_from_fps(df_split, fps, M=M, n_min=n_min)
