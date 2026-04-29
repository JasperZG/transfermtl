"""Test-only PyG featurizer for the synthetic fixture.

The synthetic fixture (build.py) emits *fake* SMILES strings of the form
`smi_<letter>_<scaffold_idx>_<mol_idx>`. RDKit cannot parse these, so A2's real
featurizer is unusable on the fixture. This module provides a deterministic
PyG `Data` builder that mirrors the 74-dim atom-feature contract:

  - graph: 4-node star (3 leaves attached to a center)
  - node features (74-dim):
      dims  0-1   region one-hot              (A vs B)
      dims  2-21  scaffold one-hot            (20 scaffolds per region)
      dim   22    task_1 leak (2*y - 1) + N(0, 0.5)
      dim   23    task_2 leak (2*y - 1) + N(0, 0.5)
      rest        zeros

The label leak is deliberately noisy so that the GCN must integrate signal
across nodes; train and val both stay learnable because the leak is computed
from each compound's labels (not from train-only knowledge). This is acceptable
for a *test* fixture — production code never calls this featurizer; A2's RDKit
pipeline replaces it for real datasets.
"""

from __future__ import annotations

import math
import re

import numpy as np
import torch
from torch_geometric.data import Data

ATOM_FEATURE_DIM = 74
N_NODES = 4
EDGE_INDEX = torch.tensor(
    [
        [0, 1, 0, 2, 0, 3, 1, 0, 2, 0, 3, 0],
        [1, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 3],
    ],
    dtype=torch.long,
)

_SMI_RE = re.compile(r"^smi_(?P<letter>[AB])_(?P<scaff>\d{2})_(?P<mol>\d+)$")


def parse_synthetic_smiles(smi: str) -> tuple[int, int, int]:
    """Return (region_idx, scaffold_idx, mol_idx) for fake SMILES `smi_X_NN_M`."""
    m = _SMI_RE.match(smi)
    if not m:
        raise ValueError(f"Unrecognized synthetic SMILES: {smi!r}")
    region_idx = 0 if m.group("letter") == "A" else 1
    return region_idx, int(m.group("scaff")), int(m.group("mol"))


def synthetic_featurize(
    smi: str,
    task_1: float | None = None,
    task_2: float | None = None,
    rng: np.random.Generator | None = None,
) -> Data:
    """Build a 4-node PyG graph for a synthetic-fixture compound.

    `task_1` and `task_2` are optional; when provided, they are encoded as a
    noisy leak channel so the GCN can converge on the fixture in <30s.
    """
    region_idx, scaffold_idx, mol_idx = parse_synthetic_smiles(smi)

    if rng is None:
        # Per-compound deterministic RNG from (region, scaffold, mol).
        rng = np.random.default_rng((region_idx + 1) * 1_000 + scaffold_idx * 10 + mol_idx)

    x = np.zeros((N_NODES, ATOM_FEATURE_DIM), dtype=np.float32)
    x[:, region_idx] = 1.0
    scaff_dim = 2 + scaffold_idx  # 2..21
    x[:, scaff_dim] = 1.0

    if task_1 is not None and not math.isnan(task_1):
        leak1 = (2.0 * float(task_1) - 1.0) + rng.normal(0.0, 0.5, size=N_NODES).astype(np.float32)
        x[:, 22] = leak1
    if task_2 is not None and not math.isnan(task_2):
        leak2 = (2.0 * float(task_2) - 1.0) + rng.normal(0.0, 0.5, size=N_NODES).astype(np.float32)
        x[:, 23] = leak2

    return Data(
        x=torch.from_numpy(x),
        edge_index=EDGE_INDEX.clone(),
    )


def build_synthetic_loader(
    split_df: object,
    split: str,
    tasks: list[str],
) -> list[Data]:
    """Convenience: build a Data list for the fixture without going through the
    cache path. Each Data has `.x`, `.edge_index`, `.y`, and `.smi` set.
    """
    import pandas as pd  # local import keeps src-side modules pandas-free where possible

    assert isinstance(split_df, pd.DataFrame)
    rows = split_df[split_df["split"] == split]
    out: list[Data] = []
    for _, row in rows.iterrows():
        smi = str(row["smiles"])
        t1 = float(row["task_1"]) if pd.notna(row["task_1"]) else None
        t2 = float(row["task_2"]) if pd.notna(row["task_2"]) else None
        data = synthetic_featurize(smi, task_1=t1, task_2=t2)
        y_vals = []
        for t in tasks:
            v = row[t]
            y_vals.append(float("nan") if pd.isna(v) else float(v))
        data.y = torch.tensor([y_vals], dtype=torch.float32)
        data.smi = smi
        out.append(data)
    return out
