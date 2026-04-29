"""Frozen dataclasses passed across module boundaries.

Wave 1 freezes these field names. Wave 2+ does not edit this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    ci_lower: float
    ci_upper: float
    samples: np.ndarray | None = None


@dataclass(frozen=True)
class ValidityFlag:
    valid: bool
    failed_reasons: tuple[str, ...]


@dataclass(frozen=True)
class HierarchicalSamples:
    values: np.ndarray
    scaffold_ids: np.ndarray
    seed_ids: np.ndarray | None = None


@dataclass(frozen=True)
class RegionStats:
    region_id: int
    n_train_i: int
    n_train_j: int
    n_test_i: int
    n_test_j: int
    g_i_norm: float
    g_j_norm: float


@dataclass(frozen=True)
class PairIndices:
    pair_id: str
    S_ij: bool
    S_i: bool
    S_j: bool
    H_ij: float
    C_ij: float
    n_valid_regions: int
