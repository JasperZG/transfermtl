"""Predictor package.

Wave 3 / A7 ships only the *pilot* baseline used by the §5.7 predictor
green-light check (pilot_baseline.py + criteria.py). A8 (Wave 4) replaces this
with the full Phase 2 predictor and the incremental-R² test.
"""

from transfermtl.predictor.criteria import (
    CriterionResult,
    Decision,
    PhenomenonResult,
    PredictorResult,
    decide,
    evaluate_phenomenon_criteria,
    evaluate_predictor_criteria,
)
from transfermtl.predictor.pilot_baseline import (
    PredictorScores,
    cross_seed_sign_agreement,
    embedding_distance_baseline,
    evaluate_pilot_predictor,
    label_correlation_baseline,
    scaffold_tanimoto_baseline,
)

__all__ = [
    "CriterionResult",
    "Decision",
    "PhenomenonResult",
    "PredictorResult",
    "PredictorScores",
    "cross_seed_sign_agreement",
    "decide",
    "embedding_distance_baseline",
    "evaluate_phenomenon_criteria",
    "evaluate_pilot_predictor",
    "evaluate_predictor_criteria",
    "label_correlation_baseline",
    "scaffold_tanimoto_baseline",
]
