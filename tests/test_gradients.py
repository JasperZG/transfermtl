"""Unit tests for src/transfermtl/gradients/.

Most tests use synthetic PyG `Data` graphs constructed in-process so they do
not depend on RDKit or A2's featurization cache.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from transfermtl.gradients.affinity import (
    GRAD_NORM_ZERO,
    cosine_affinity,
    dot_product_affinity,
)
from transfermtl.gradients.extract import (
    compute_gradient_vector,
    encoder_param_order_hash,
)
from transfermtl.gradients.io import region_affinity_path, write_region_affinity
from transfermtl.gradients.trajectory import compute_trajectory_affinity
from transfermtl.models.gcn import GCNEncoder
from transfermtl.models.multihead import MultiHeadModel
from transfermtl.utils.schemas import GradientAffinitySchema
from transfermtl.utils.seeding import set_seed

ATOM_FEAT = 74
HIDDEN = 32  # smaller encoder so tests run fast


def _build_model(tasks: list[str]) -> MultiHeadModel:
    encoder = GCNEncoder(hidden_dim=HIDDEN, n_layers=2, dropout=0.0, atom_feature_dim=ATOM_FEAT)
    return MultiHeadModel(encoder, tasks, head_hidden_dim=16, head_dropout=0.0)


def _make_data(n: int, n_tasks: int = 1, seed: int = 0, n_atoms: int = 4) -> list[Data]:
    rng = np.random.default_rng(seed)
    out: list[Data] = []
    for i in range(n):
        x = torch.tensor(rng.random((n_atoms, ATOM_FEAT)), dtype=torch.float32)
        # Linear chain edges (undirected).
        src = list(range(n_atoms - 1)) + list(range(1, n_atoms))
        dst = list(range(1, n_atoms)) + list(range(n_atoms - 1))
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.zeros((edge_index.shape[1], 12), dtype=torch.float32)
        labels = (rng.random(n_tasks) > 0.5).astype(np.float32)
        d = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.from_numpy(labels).unsqueeze(0),
            smi=f"smi_{i}",
        )
        out.append(d)
    return out


# ------------------------------------------------------------------
# Affinity primitives
# ------------------------------------------------------------------


def test_cosine_self_is_one() -> None:
    """Identity-style sanity check: cos(x, x) == 1."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=128)
    assert abs(cosine_affinity(g, g) - 1.0) < 1e-12


def test_cosine_undefined_when_zero_norm() -> None:
    """Either-side zero gradient norm yields NaN."""
    rng = np.random.default_rng(0)
    g = rng.normal(size=64)
    z = np.zeros_like(g)
    assert np.isnan(cosine_affinity(g, z))
    assert np.isnan(cosine_affinity(z, g))


def test_dot_product_affinity_basic() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert dot_product_affinity(a, b) == pytest.approx(1.0)
    assert dot_product_affinity(a, -b) == pytest.approx(-1.0)


# ------------------------------------------------------------------
# Gradient extraction
# ------------------------------------------------------------------


def test_identity_check() -> None:
    """G_ii(r) ≈ 1 — the §9.6 identity check baked into a unit test."""
    set_seed(0)
    model = _build_model(["t1"])
    data = _make_data(20, n_tasks=1, seed=0)
    g, n_used, norm = compute_gradient_vector(model, data, "t1", task_index=0)
    assert n_used > 0
    assert norm > GRAD_NORM_ZERO
    G_ii = cosine_affinity(g, g)
    assert abs(G_ii - 1.0) < 1e-3


def test_label_shuffle_drives_to_zero() -> None:
    """Shuffling task-2 labels independent of task-1 → cos(g_1, g_2) clusters near 0."""
    rng = np.random.default_rng(0)
    cosines = []
    for trial in range(5):
        set_seed(trial)
        model = _build_model(["t1", "t2"])
        data = _make_data(40, n_tasks=2, seed=trial, n_atoms=4)
        # Independent label assignment (simulating shuffled task-2 vs task-1).
        for d in data:
            d.y = torch.tensor(
                [[float(rng.integers(0, 2)), float(rng.integers(0, 2))]],
                dtype=torch.float32,
            )

        g1, _, _ = compute_gradient_vector(model, data, "t1", task_index=0)
        g2, _, _ = compute_gradient_vector(model, data, "t2", task_index=1)
        cosines.append(cosine_affinity(g1, g2))
    cosines_arr = np.asarray(cosines)
    assert abs(float(cosines_arr.mean())) < 0.20


def test_undefined_when_zero_grad_norm() -> None:
    """Empty / all-NaN region → returns zero vec, n_used=0, norm=0 (NaN affinity)."""
    set_seed(0)
    model = _build_model(["t1"])
    g, n_used, norm = compute_gradient_vector(model, [], "t1")
    assert n_used == 0
    assert norm == 0.0
    assert np.isnan(cosine_affinity(g, g))


def test_subsample_above_500() -> None:
    """Region with 1000 compounds → uses 500."""
    set_seed(0)
    model = _build_model(["t1"])
    data = _make_data(1000, n_tasks=1, seed=0)
    _, n_used, _ = compute_gradient_vector(model, data, "t1", max_subsample=500)
    assert n_used <= 500


def test_encoder_param_order_locked() -> None:
    """Building the model twice yields identical iteration order of encoder params."""
    a = encoder_param_order_hash(_build_model(["t1"]))
    b = encoder_param_order_hash(_build_model(["t1"]))
    assert a == b
    # Adding a head must not perturb encoder param order.
    c = encoder_param_order_hash(_build_model(["t1", "t2"]))
    assert a == c


def test_trajectory_three_checkpoints() -> None:
    """compute_trajectory_affinity returns the three required keys + summary."""
    rng = np.random.default_rng(0)
    cache = {
        "final": (rng.normal(size=64), rng.normal(size=64)),
        "0.8": (rng.normal(size=64), rng.normal(size=64)),
        "0.6": (rng.normal(size=64), rng.normal(size=64)),
    }
    out = compute_trajectory_affinity(lambda label: cache[label])
    assert set(out.keys()) == {"final", "0.8", "0.6", "_summary"}
    assert "G_ij" in out["final"]
    assert "mean" in out["_summary"]


# ------------------------------------------------------------------
# IO
# ------------------------------------------------------------------


def test_write_region_affinity_schema_valid(tmp_path: Path) -> None:
    rows = [
        {
            "region_id": 0,
            "G_ij": 0.7,
            "g_i_norm": 1.0,
            "g_j_norm": 1.0,
            "n_i_in_region": 50,
            "n_j_in_region": 50,
            "checkpoint_label": "final",
            "seed": 0,
        }
    ]
    path = write_region_affinity("dummy", "t1", "t2", 0, rows, root=tmp_path)
    assert path == region_affinity_path("dummy", "t1", "t2", 0, root=tmp_path)
    import pandas as pd

    df = pd.read_parquet(path)
    GradientAffinitySchema.validate(df)
