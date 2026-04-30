"""Plan §5.7 green-light criteria + plan §5.8 decision tree.

A7 evaluates seven sub-criteria (3 phenomenon + 4 predictor) and walks the
decision tree to one of: ``PROCEED`` / ``WORKSHOP`` / ``INVESTIGATE`` /
``PIVOT`` / ``DROP``. Each sub-criterion is a pure function of pre-computed
inputs so unit tests can drive the logic with synthetic numbers.

Thresholds are pulled from plan §5.7 verbatim — do not relax them post-hoc.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

# ---------------------------------------------------------------------------
# Threshold constants (plan §5.7).
# ---------------------------------------------------------------------------

PHEN_MIN_PAIRS_WITH_SIGN_HET = 5  # criterion 1
PHEN_MIN_PCT_MEANINGFUL = 0.15  # criterion 2 (15%)
PHEN_NULL_PERCENTILE = 0.95  # criterion 3 (95th percentile)

PRED_AUROC_MIN = 0.70  # criterion 1
PRED_AUROC_OVER_BASELINE = 0.07  # criterion 2 (delta vs best baseline)
PRED_SPEARMAN_MIN = 0.45  # criterion 3
PRED_CROSS_SEED_AGREEMENT_MIN = 0.80  # criterion 4 (80%)


@dataclass(frozen=True)
class PhenomenonResult:
    """Pre-computed phenomenon-side measurements consumed by criteria 1-3."""

    n_pairs_with_sign_het: int
    n_meaningful_pairs: int
    pct_meaningful_with_sign_het: float
    observed_prevalence: float
    null_percentile_threshold: float  # 95th-percentile of random-partition null
    null_p_value: float


@dataclass(frozen=True)
class PredictorResult:
    """Pre-computed predictor-side measurements consumed by criteria 1-4."""

    auroc_g_ij: float
    auroc_best_baseline: float
    best_baseline_name: str
    spearman_rho: float
    cross_seed_agreement: float


@dataclass(frozen=True)
class CriterionResult:
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str


class Decision(StrEnum):
    PROCEED = "PROCEED"
    WORKSHOP = "WORKSHOP"
    INVESTIGATE = "INVESTIGATE"
    PIVOT = "PIVOT"
    DROP = "DROP"


# ---------------------------------------------------------------------------
# Phenomenon sub-criteria (3).
# ---------------------------------------------------------------------------


def evaluate_phenomenon_criteria(result: PhenomenonResult) -> list[CriterionResult]:
    out: list[CriterionResult] = [
        CriterionResult(
            name="phenomenon_1_n_pairs_with_sign_het",
            passed=result.n_pairs_with_sign_het >= PHEN_MIN_PAIRS_WITH_SIGN_HET,
            value=float(result.n_pairs_with_sign_het),
            threshold=float(PHEN_MIN_PAIRS_WITH_SIGN_HET),
            detail=(
                f"{result.n_pairs_with_sign_het} pairs show stable sign "
                f"heterogeneity (need ≥{PHEN_MIN_PAIRS_WITH_SIGN_HET})"
            ),
        ),
        CriterionResult(
            name="phenomenon_2_pct_meaningful",
            passed=result.pct_meaningful_with_sign_het >= PHEN_MIN_PCT_MEANINGFUL,
            value=result.pct_meaningful_with_sign_het,
            threshold=PHEN_MIN_PCT_MEANINGFUL,
            detail=(
                f"{result.pct_meaningful_with_sign_het:.1%} of "
                f"{result.n_meaningful_pairs} meaningful pairs exhibit sign "
                f"heterogeneity (need ≥{PHEN_MIN_PCT_MEANINGFUL:.0%})"
            ),
        ),
        CriterionResult(
            name="phenomenon_3_null_test",
            passed=result.observed_prevalence > result.null_percentile_threshold,
            value=result.observed_prevalence,
            threshold=result.null_percentile_threshold,
            detail=(
                f"observed prevalence {result.observed_prevalence:.3f} vs 95th "
                f"percentile of random null {result.null_percentile_threshold:.3f} "
                f"(empirical p={result.null_p_value:.3g})"
            ),
        ),
    ]
    return out


# ---------------------------------------------------------------------------
# Predictor sub-criteria (4).
# ---------------------------------------------------------------------------


def evaluate_predictor_criteria(result: PredictorResult) -> list[CriterionResult]:
    auroc_delta = result.auroc_g_ij - result.auroc_best_baseline
    out: list[CriterionResult] = [
        CriterionResult(
            name="predictor_1_auroc_floor",
            passed=result.auroc_g_ij >= PRED_AUROC_MIN,
            value=result.auroc_g_ij,
            threshold=PRED_AUROC_MIN,
            detail=f"G_ij(r) sign-of-Δ AUROC {result.auroc_g_ij:.3f} (need ≥{PRED_AUROC_MIN:.2f})",
        ),
        CriterionResult(
            name="predictor_2_baseline_lift",
            passed=auroc_delta >= PRED_AUROC_OVER_BASELINE,
            value=auroc_delta,
            threshold=PRED_AUROC_OVER_BASELINE,
            detail=(
                f"AUROC lift over best baseline ({result.best_baseline_name}, "
                f"{result.auroc_best_baseline:.3f}) is {auroc_delta:+.3f} "
                f"(need ≥{PRED_AUROC_OVER_BASELINE:.2f})"
            ),
        ),
        CriterionResult(
            name="predictor_3_spearman",
            passed=result.spearman_rho >= PRED_SPEARMAN_MIN,
            value=result.spearman_rho,
            threshold=PRED_SPEARMAN_MIN,
            detail=f"Spearman ρ(G_ij, Δ_ij) = {result.spearman_rho:.3f} (need ≥{PRED_SPEARMAN_MIN:.2f})",
        ),
        CriterionResult(
            name="predictor_4_cross_seed_agreement",
            passed=result.cross_seed_agreement >= PRED_CROSS_SEED_AGREEMENT_MIN,
            value=result.cross_seed_agreement,
            threshold=PRED_CROSS_SEED_AGREEMENT_MIN,
            detail=(
                f"sign(G_ij(r)) cross-seed agreement {result.cross_seed_agreement:.1%} "
                f"(need ≥{PRED_CROSS_SEED_AGREEMENT_MIN:.0%})"
            ),
        ),
    ]
    return out


# ---------------------------------------------------------------------------
# Plan §5.8 decision tree.
# ---------------------------------------------------------------------------


def _all_passed(results: list[CriterionResult]) -> bool:
    return all(r.passed for r in results)


def decide(
    phenomenon: list[CriterionResult],
    predictor: list[CriterionResult] | None,
) -> Decision:
    """Walk plan §5.8.

    - Both pass            → PROCEED
    - Phenomenon ✓, predictor ✗ → WORKSHOP/INVESTIGATE — distinguish on whether
      the predictor floor (criterion 1) failed; if floor passed but lift or
      consistency failed, INVESTIGATE; if floor itself failed, WORKSHOP.
    - Phenomenon ✗, predictor untestable (None) → PIVOT
    - Both fail (or phenomenon ✗ and predictor ✗) → DROP
    """
    phen_ok = _all_passed(phenomenon)
    if phen_ok and predictor is not None and _all_passed(predictor):
        return Decision.PROCEED
    if phen_ok and (predictor is None or not _all_passed(predictor)):
        if predictor is None:
            return Decision.WORKSHOP
        floor = next(r for r in predictor if r.name == "predictor_1_auroc_floor")
        if floor.passed:
            return Decision.INVESTIGATE
        return Decision.WORKSHOP
    if not phen_ok and predictor is None:
        return Decision.PIVOT
    return Decision.DROP
