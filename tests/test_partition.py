"""Unit tests for ``transfermtl.partition`` (Wave 2 / A5).

The synthetic fixture has 40 scaffolds (20 per region × 2 regions = 200 mols)
with deliberately separable structure. We provide synthetic Morgan FPs and
encoder embeddings that mirror the fixture's two-region ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score, pair_confusion_matrix

from transfermtl.partition.io import partition_path, write_partition
from transfermtl.partition.knn import knn_partition_from_embeddings
from transfermtl.partition.latent import latent_partition_from_embeddings
from transfermtl.partition.random_null import random_partition_from_sizes
from transfermtl.partition.scaffold import (
    _tanimoto_distance_matrix,
    scaffold_partition_from_fps,
)
from transfermtl.utils.schemas import PartitionSchema

N_BITS = 256  # Smaller than 2048 to keep tests fast; same Tanimoto math.


def _build_scaffold_fps(scaffolds: Iterable[str], seed: int = 0) -> dict[str, np.ndarray]:
    """Synthetic FPs separating region A vs B scaffolds.

    - Every scaffold gets ~10 random per-scaffold noise bits.
    - region-A scaffolds (``scaff_A_*``) all share bits [0, 30).
    - region-B scaffolds (``scaff_B_*``) all share bits [100, 130).
    Tanimoto between within-region scaffolds ≈ 0.7-0.9; cross-region ≈ 0.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for s in scaffolds:
        fp = np.zeros(N_BITS, dtype=np.uint8)
        noise = rng.choice(N_BITS, size=10, replace=False)
        fp[noise] = 1
        if s.startswith("scaff_A_"):
            fp[:30] = 1
        elif s.startswith("scaff_B_"):
            fp[100:130] = 1
        else:  # pragma: no cover - defensive
            fp[200:230] = 1
        out[s] = fp
    return out


def _build_embeddings(
    smiles: list[str], partition: pd.DataFrame, dim: int = 8, seed: int = 0
) -> np.ndarray:
    """Embeddings clustered by region: small Gaussian noise around region anchors."""
    rng = np.random.default_rng(seed)
    region_lookup = dict(
        zip(partition["smiles"].astype(str), partition["region_id"].astype(int), strict=True)
    )
    n_regions = int(partition["region_id"].max()) + 1
    anchors = np.eye(n_regions, dim, dtype=np.float64) * 5.0  # well-separated
    emb = np.empty((len(smiles), dim), dtype=np.float64)
    for i, s in enumerate(smiles):
        r = region_lookup[s]
        emb[i] = anchors[r] + rng.normal(0, 0.05, size=dim)
    return emb


def _pair_jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Pair-clustering Jaccard between two label vectors: TP / (TP + FP + FN)."""
    tn, fp, fn, tp = pair_confusion_matrix(a, b).ravel()
    denom = tp + fp + fn
    if denom == 0:
        return 1.0
    return float(tp / denom)


# --------------------------------------------------------------------------- #
# scaffold partition                                                           #
# --------------------------------------------------------------------------- #


def test_tanimoto_distance_matrix_basic() -> None:
    fp = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],  # identical to row 0
            [0, 0, 1, 1],  # disjoint from row 0
        ],
        dtype=np.uint8,
    )
    d = _tanimoto_distance_matrix(fp)
    assert d.shape == (3, 3)
    np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-12)
    np.testing.assert_allclose(d[0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(d[0, 2], 1.0, atol=1e-12)
    np.testing.assert_allclose(d, d.T, atol=1e-12)


def test_scaffold_deterministic(synthetic_split: pd.DataFrame) -> None:
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    a = scaffold_partition_from_fps(synthetic_split, fps, M=5, n_min=1)
    b = scaffold_partition_from_fps(synthetic_split, fps, M=5, n_min=1)
    pd.testing.assert_frame_equal(
        a.sort_values("smiles").reset_index(drop=True),
        b.sort_values("smiles").reset_index(drop=True),
    )


@pytest.mark.parametrize("n_min", [25, 50, 100])
def test_scaffold_min_size_enforced(synthetic_split: pd.DataFrame, n_min: int) -> None:
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    df = scaffold_partition_from_fps(synthetic_split, fps, M=5, n_min=n_min)

    train = synthetic_split[synthetic_split["split"] == "train"]
    train_with_region = train.merge(df, on="smiles", how="inner")
    counts = train_with_region["region_id"].value_counts()
    n_regions = df["region_id"].nunique()
    if n_regions > 1:
        # Every region must clear 2*n_min train compounds — the merge step's job.
        bad = {r: int(c) for r, c in counts.items() if c < 2 * n_min}
        assert not bad, f"n_min={n_min}: regions below 2*n_min after merge: {bad}"
    # Otherwise the function fell back to a single region; nothing more to check.


def test_scaffold_partition_schema_validates(synthetic_split: pd.DataFrame) -> None:
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    df = scaffold_partition_from_fps(synthetic_split, fps, M=5, n_min=1)
    PartitionSchema.validate(df)
    assert set(df.columns) == {"smiles", "region_id"}
    assert (df["region_id"] >= 0).all()


@pytest.mark.parametrize("M", [3, 5, 8, 10])
def test_scaffold_M_variants(synthetic_split: pd.DataFrame, M: int) -> None:
    """With merging disabled (n_min=0), HAC should return exactly M regions."""
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    df = scaffold_partition_from_fps(synthetic_split, fps, M=M, n_min=0)
    n_regions = df["region_id"].nunique()
    assert n_regions == M, f"M={M}: expected {M} regions, got {n_regions}"


def test_scaffold_recovers_two_regions(
    synthetic_split: pd.DataFrame, synthetic_partition: pd.DataFrame
) -> None:
    """At M=2 with no merging, HAC should recover the fixture's A vs B split."""
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    df = scaffold_partition_from_fps(synthetic_split, fps, M=2, n_min=1)
    truth = synthetic_partition.set_index("smiles")["region_id"].astype(int)
    pred = df.set_index("smiles")["region_id"].astype(int)
    pred = pred.reindex(truth.index)
    ari = adjusted_rand_score(truth.to_numpy(), pred.to_numpy())
    assert ari > 0.95, f"ARI={ari}"


# --------------------------------------------------------------------------- #
# latent partition                                                             #
# --------------------------------------------------------------------------- #


def test_latent_partition_clusters_synthetic_correctly(
    synthetic_split: pd.DataFrame, synthetic_partition: pd.DataFrame
) -> None:
    """Stand-in for A3's checkpoint: synthetic embeddings clustered by region."""
    smiles = synthetic_split["smiles"].astype(str).tolist()
    embeddings = _build_embeddings(smiles, synthetic_partition, dim=8, seed=0)
    df = latent_partition_from_embeddings(smiles, embeddings, M=2, seed=0)
    truth = synthetic_partition.set_index("smiles")["region_id"].astype(int)
    pred = df.set_index("smiles")["region_id"].astype(int).reindex(truth.index)
    ari = adjusted_rand_score(truth.to_numpy(), pred.to_numpy())
    assert ari > 0.8, f"ARI={ari}"


def test_latent_partition_schema_validates(
    synthetic_split: pd.DataFrame, synthetic_partition: pd.DataFrame
) -> None:
    smiles = synthetic_split["smiles"].astype(str).tolist()
    embeddings = _build_embeddings(smiles, synthetic_partition, dim=8, seed=0)
    df = latent_partition_from_embeddings(smiles, embeddings, M=3, seed=0)
    PartitionSchema.validate(df)


def test_latent_partition_deterministic(
    synthetic_split: pd.DataFrame, synthetic_partition: pd.DataFrame
) -> None:
    smiles = synthetic_split["smiles"].astype(str).tolist()
    embeddings = _build_embeddings(smiles, synthetic_partition, dim=8, seed=0)
    a = latent_partition_from_embeddings(smiles, embeddings, M=2, seed=42)
    b = latent_partition_from_embeddings(smiles, embeddings, M=2, seed=42)
    pd.testing.assert_frame_equal(a, b)


# --------------------------------------------------------------------------- #
# kNN partition                                                                #
# --------------------------------------------------------------------------- #


def test_knn_consistent_with_scaffold(synthetic_split: pd.DataFrame) -> None:
    """kNN with same M as scaffold has Jaccard overlap >= 0.5 on synthetic fixture."""
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    scaffold_part = scaffold_partition_from_fps(synthetic_split, fps, M=2, n_min=1)
    smiles = synthetic_split["smiles"].astype(str).tolist()
    emb = _build_embeddings(smiles, scaffold_part, dim=8, seed=0)
    knn_part = knn_partition_from_embeddings(smiles, emb, scaffold_part)

    a = scaffold_part.set_index("smiles")["region_id"].astype(int).loc[smiles].to_numpy()
    b = knn_part.set_index("smiles")["region_id"].astype(int).loc[smiles].to_numpy()
    j = _pair_jaccard(a, b)
    assert j >= 0.5, f"pair-Jaccard={j}"


def test_knn_partition_schema_validates(synthetic_split: pd.DataFrame) -> None:
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    scaffold_part = scaffold_partition_from_fps(synthetic_split, fps, M=2, n_min=1)
    smiles = synthetic_split["smiles"].astype(str).tolist()
    emb = _build_embeddings(smiles, scaffold_part, dim=8, seed=0)
    df = knn_partition_from_embeddings(smiles, emb, scaffold_part)
    PartitionSchema.validate(df)


# --------------------------------------------------------------------------- #
# random partitions                                                            #
# --------------------------------------------------------------------------- #


def test_random_partition_size_distribution_matches(synthetic_split: pd.DataFrame) -> None:
    fps = _build_scaffold_fps(sorted(synthetic_split["scaffold"].unique()))
    scaffold_part = scaffold_partition_from_fps(synthetic_split, fps, M=2, n_min=1)
    region_sizes = scaffold_part["region_id"].astype(int).value_counts().sort_index().tolist()
    smiles = synthetic_split["smiles"].astype(str).tolist()

    rand_part = random_partition_from_sizes(smiles, region_sizes, seed=0)
    rand_sizes = rand_part["region_id"].astype(int).value_counts().sort_index().tolist()
    assert rand_sizes == region_sizes


def test_random_partition_seed_determinism(synthetic_split: pd.DataFrame) -> None:
    smiles = synthetic_split["smiles"].astype(str).tolist()
    sizes = [60, 80, 60]
    a = random_partition_from_sizes(smiles, sizes, seed=7)
    b = random_partition_from_sizes(smiles, sizes, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_random_partition_seeds_differ(synthetic_split: pd.DataFrame) -> None:
    smiles = synthetic_split["smiles"].astype(str).tolist()
    sizes = [60, 80, 60]
    a = random_partition_from_sizes(smiles, sizes, seed=0)
    b = random_partition_from_sizes(smiles, sizes, seed=1)
    # Same row order; differing labels.
    assert not a["region_id"].equals(b["region_id"])


def test_random_partition_schema_validates(synthetic_split: pd.DataFrame) -> None:
    smiles = synthetic_split["smiles"].astype(str).tolist()
    df = random_partition_from_sizes(smiles, [100, 100], seed=0)
    PartitionSchema.validate(df)


# --------------------------------------------------------------------------- #
# IO                                                                           #
# --------------------------------------------------------------------------- #


def test_write_partition_roundtrip(tmp_path, synthetic_partition: pd.DataFrame) -> None:
    out = write_partition("fixture", "scaffold", synthetic_partition, M=2, root=tmp_path)
    assert out == partition_path("fixture", "scaffold", M=2, root=tmp_path)
    assert out.exists()
    loaded = pd.read_parquet(out)
    PartitionSchema.validate(loaded)
    pd.testing.assert_frame_equal(loaded, synthetic_partition)


def test_partition_path_random_requires_b() -> None:
    with pytest.raises(ValueError, match="random"):
        partition_path("fixture", "random", M=5)


def test_partition_path_named_requires_M() -> None:
    with pytest.raises(ValueError, match="scaffold"):
        partition_path("fixture", "scaffold")
