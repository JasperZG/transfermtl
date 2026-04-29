"""Frozen pandera schemas. Every artifact parquet is validated against one of these.

Wave 2+ agents stub against these schemas. Field names and dtypes are immutable;
contract changes require a coordination PR.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

SplitSchema = DataFrameSchema(
    {
        "smiles": Column(str),
        "scaffold": Column(str),
        "split": Column(str, Check.isin(["train", "val", "test"])),
        r"task_.*": Column(float, nullable=True, regex=True),
    },
    strict=True,
    coerce=True,
)

PartitionSchema = DataFrameSchema(
    {
        "smiles": Column(str),
        "region_id": Column(int, Check.greater_than_or_equal_to(0)),
    },
    strict=True,
    coerce=True,
)

PredictionSchema = DataFrameSchema(
    {
        "smiles": Column(str),
        "task": Column(str),
        "y_true": Column(float, nullable=True),
        "y_pred": Column(float, nullable=True),
        "seed": Column(int),
    },
    strict=True,
    coerce=True,
)

GradientAffinitySchema = DataFrameSchema(
    {
        "region_id": Column(int, Check.greater_than_or_equal_to(0)),
        "G_ij": Column(float, nullable=True),
        "g_i_norm": Column(float, nullable=True),
        "g_j_norm": Column(float, nullable=True),
        "n_i_in_region": Column(int, Check.greater_than_or_equal_to(0)),
        "n_j_in_region": Column(int, Check.greater_than_or_equal_to(0)),
        "checkpoint_label": Column(str, Check.isin(["final", "0.8", "0.6"])),
        "seed": Column(int),
    },
    strict=True,
    coerce=True,
)

RegionBenefitSchema = DataFrameSchema(
    {
        "region_id": Column(int, Check.greater_than_or_equal_to(0)),
        "delta_pair": Column(float, nullable=True),
        "delta_i_from_j": Column(float, nullable=True),
        "delta_j_from_i": Column(float, nullable=True),
        "delta_worst": Column(float, nullable=True),
        "ci_lo": Column(float, nullable=True),
        "ci_hi": Column(float, nullable=True),
        "n_test": Column(int, Check.greater_than_or_equal_to(0)),
    },
    strict=True,
    coerce=True,
)

PairIndicesSchema = DataFrameSchema(
    {
        "pair_id": Column(str),
        "S_ij": Column(bool),
        "S_i": Column(bool),
        "S_j": Column(bool),
        "H_ij": Column(float, nullable=True),
        "C_ij": Column(float, nullable=True),
        "n_valid_regions": Column(int, Check.greater_than_or_equal_to(0)),
    },
    strict=True,
    coerce=True,
)

MeaningfulPairSchema = DataFrameSchema(
    {
        "pair_id": Column(str),
        "is_meaningful": Column(bool),
        "failed_reasons": Column(object),
    },
    strict=True,
    coerce=True,
)

PredictorScoresSchema = DataFrameSchema(
    {
        "pair_id": Column(str),
        "region_id": Column(int, Check.greater_than_or_equal_to(0)),
        "feature_name": Column(str),
        "value": Column(float, nullable=True),
    },
    strict=True,
    coerce=True,
)


__all__ = [
    "GradientAffinitySchema",
    "MeaningfulPairSchema",
    "PairIndicesSchema",
    "PartitionSchema",
    "PredictionSchema",
    "PredictorScoresSchema",
    "RegionBenefitSchema",
    "SplitSchema",
    "pa",
]
