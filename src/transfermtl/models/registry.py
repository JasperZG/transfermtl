"""Encoder factory entries.

Wave 2 registers `gcn`. Wave 5 / A12 will add `gat`, `chemberta`, `ecfp_mlp`
under the same `("encoder", name)` namespace.
"""

from __future__ import annotations

from typing import Any

from transfermtl.models.gcn import GCNEncoder
from transfermtl.utils.registry import register


@register("encoder", "gcn")
def build_gcn(
    hidden_dim: int = 256,
    n_layers: int = 3,
    dropout: float = 0.1,
    atom_feature_dim: int = 74,
    **_: Any,  # tolerate locked-yaml keys (activation, pool, bond_feature_dim, name)
) -> GCNEncoder:
    return GCNEncoder(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        atom_feature_dim=atom_feature_dim,
    )
