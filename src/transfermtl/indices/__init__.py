"""indices/ — sign heterogeneity (S), heterogeneity (H), cancellation (C). Plan §2.9."""

from transfermtl.indices.cancellation import compute_C
from transfermtl.indices.heterogeneity_index import compute_H
from transfermtl.indices.io import (
    PAIR_INDICES_ROOT,
    pair_indices_path,
    write_pair_indices,
)
from transfermtl.indices.sign_heterogeneity import (
    compute_S_pair,
    compute_S_task_specific,
)

__all__ = [
    "PAIR_INDICES_ROOT",
    "compute_C",
    "compute_H",
    "compute_S_pair",
    "compute_S_task_specific",
    "pair_indices_path",
    "write_pair_indices",
]
