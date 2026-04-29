"""Hierarchical bootstrap and percentile-CI utilities (plan §2.10)."""

from transfermtl.bootstrap.calibration import CalibrationError, run_calibration_check
from transfermtl.bootstrap.hierarchical import (
    bootstrap_result_to_dict,
    hierarchical_bootstrap,
)
from transfermtl.bootstrap.percentile import percentile_ci
from transfermtl.bootstrap.seed_mixing import draw_seeds
from transfermtl.bootstrap.within_region import within_region_bootstrap

__all__ = [
    "CalibrationError",
    "bootstrap_result_to_dict",
    "draw_seeds",
    "hierarchical_bootstrap",
    "percentile_ci",
    "run_calibration_check",
    "within_region_bootstrap",
]
