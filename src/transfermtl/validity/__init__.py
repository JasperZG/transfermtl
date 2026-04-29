"""Local-support and meaningful-pair validity criteria (plan §2.11-2.12)."""

from transfermtl.validity.io import meaningful_pairs_path, write_meaningful_pairs
from transfermtl.validity.local_support import (
    REASON_CI_WIDTH,
    REASON_GRAD_NORM,
    REASON_LABEL_BALANCE,
    REASON_N_TEST,
    REASON_N_TRAIN,
    ValidityConfig,
    check_local_support,
)
from transfermtl.validity.meaningful_pair import (
    COND_LABEL_DISTRIBUTION,
    COND_LARGE_DELTA,
    COND_TEST_SIZE,
    COND_VALID_REGIONS,
    check_meaningful,
)

__all__ = [
    "COND_LABEL_DISTRIBUTION",
    "COND_LARGE_DELTA",
    "COND_TEST_SIZE",
    "COND_VALID_REGIONS",
    "REASON_CI_WIDTH",
    "REASON_GRAD_NORM",
    "REASON_LABEL_BALANCE",
    "REASON_N_TEST",
    "REASON_N_TRAIN",
    "ValidityConfig",
    "check_local_support",
    "check_meaningful",
    "meaningful_pairs_path",
    "write_meaningful_pairs",
]
