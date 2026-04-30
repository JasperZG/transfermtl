"""Synthesise pilot results into a go/no-go decision document.

Reads measurements written by A6 + A7's pilot predictor and writes:

- ``outputs/analysis/pilot_decision.md`` — human-readable, rendered from the
  Jinja2 template at ``templates/pilot_decision_template.md``
- ``outputs/analysis/pilot_summary.parquet`` — machine-readable summary

Plan §5.7 thresholds are pulled from
``transfermtl.predictor.criteria`` (do not relax post-hoc — that's p-hacking).
The script accepts pre-computed results via ``--phenomenon-json`` and
``--predictor-json`` for reproducibility and ease of testing; the unit tests
exercise the rendering path with synthetic JSON.

Usage:
    python scripts/pilot_decision.py \
        --phenomenon-json outputs/analysis/phenomenon.json \
        --predictor-json outputs/analysis/predictor.json \
        [--per-dataset-json outputs/analysis/per_dataset.json] \
        [--examples-json outputs/analysis/pilot_examples.json] \
        [--out-md outputs/analysis/pilot_decision.md] \
        [--out-summary outputs/analysis/pilot_summary.parquet]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jinja2
import pandas as pd

from transfermtl.predictor.criteria import (
    CriterionResult,
    Decision,
    PhenomenonResult,
    PredictorResult,
    decide,
    evaluate_phenomenon_criteria,
    evaluate_predictor_criteria,
)
from transfermtl.utils.git import current_sha
from transfermtl.utils.io import write_parquet

log = logging.getLogger("pilot_decision")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "templates" / "pilot_decision_template.md"

DECISION_EXPLANATIONS: dict[Decision, str] = {
    Decision.PROCEED: (
        "Both phenomenon and predictor blocks satisfy all sub-criteria. "
        "Wave 4+ is green-lit pending PI sign-off."
    ),
    Decision.WORKSHOP: (
        "The phenomenon is real but the predictor's AUROC floor (criterion 1) "
        "did not clear 0.70. Plan §5.8 recommends a workshop framing focused "
        "on documenting the failure mode rather than building a predictor."
    ),
    Decision.INVESTIGATE: (
        "The phenomenon is real and the predictor clears the AUROC floor, but "
        "fails one of the lift / consistency / cross-seed sub-criteria. Plan "
        "§5.8 recommends investing in better predictor features (A8) before "
        "scaling utility experiments."
    ),
    Decision.PIVOT: (
        "The phenomenon was not stable enough to evaluate the predictor block. "
        "Plan §5.8 recommends pivoting to a weaker framing — 'conditions under "
        "which global affinity is sufficient' — or dropping the project."
    ),
    Decision.DROP: (
        "Phenomenon and predictor blocks both fail. Plan §5.8 recommends "
        "discontinuing the project."
    ),
}


def _criteria_to_dicts(crits: list[CriterionResult]) -> list[dict[str, Any]]:
    return [asdict(c) for c in crits]


def _decision_trace(
    phenomenon: list[CriterionResult],
    predictor: list[CriterionResult] | None,
    decision: Decision,
) -> list[str]:
    trace: list[str] = []
    phen_ok = all(c.passed for c in phenomenon)
    pred_ok = predictor is not None and all(c.passed for c in predictor)
    trace.append(
        f"Phenomenon block: {sum(c.passed for c in phenomenon)}/3 sub-criteria pass "
        f"→ {'PASS' if phen_ok else 'FAIL'}."
    )
    if predictor is None:
        trace.append("Predictor block: untestable (no meaningful pairs).")
    else:
        trace.append(
            f"Predictor block: {sum(c.passed for c in predictor)}/4 sub-criteria pass "
            f"→ {'PASS' if pred_ok else 'FAIL'}."
        )
    trace.append(f"Decision tree (§5.8) result: {decision.value}.")
    trace.append(DECISION_EXPLANATIONS[decision])
    return trace


def render_decision_document(
    phenomenon_inputs: PhenomenonResult,
    predictor_inputs: PredictorResult | None,
    per_dataset: list[dict[str, Any]] | None = None,
    examples: list[dict[str, Any]] | None = None,
    template_path: Path = TEMPLATE_PATH,
    git_sha: str | None = None,
) -> tuple[str, Decision, list[CriterionResult], list[CriterionResult] | None]:
    """Render the markdown decision document. Pure function — used by tests."""
    phenomenon = evaluate_phenomenon_criteria(phenomenon_inputs)
    predictor = evaluate_predictor_criteria(predictor_inputs) if predictor_inputs else None
    decision = decide(phenomenon, predictor)

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
    )
    template = env.get_template(template_path.name)
    rendered = template.render(
        decision=decision.value,
        decision_explanation=DECISION_EXPLANATIONS[decision],
        decision_trace=_decision_trace(phenomenon, predictor, decision),
        phenomenon=phenomenon,
        predictor=predictor,
        phenomenon_pass=all(c.passed for c in phenomenon),
        predictor_pass=(predictor is not None and all(c.passed for c in predictor)),
        phenomenon_inputs=phenomenon_inputs,
        predictor_inputs=predictor_inputs,
        baselines_by_name=(
            {}
            if predictor_inputs is None
            else {predictor_inputs.best_baseline_name: predictor_inputs.auroc_best_baseline}
        ),
        per_dataset=per_dataset or [],
        examples=examples or [],
        generated_at=dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        git_sha=git_sha or current_sha(),
    )
    return rendered, decision, phenomenon, predictor


def write_summary_parquet(
    path: Path,
    decision: Decision,
    phenomenon: list[CriterionResult],
    predictor: list[CriterionResult] | None,
) -> Path:
    rows: list[dict[str, Any]] = []
    for block, crits in (("phenomenon", phenomenon), ("predictor", predictor or [])):
        for c in crits:
            rows.append(
                {
                    "block": block,
                    "name": c.name,
                    "passed": bool(c.passed),
                    "value": float(c.value),
                    "threshold": float(c.threshold),
                    "detail": c.detail,
                }
            )
    rows.append(
        {
            "block": "decision",
            "name": "final",
            "passed": decision == Decision.PROCEED,
            "value": float("nan"),
            "threshold": float("nan"),
            "detail": decision.value,
        }
    )
    return write_parquet(path, pd.DataFrame(rows))


def _load_phenomenon(path: Path) -> PhenomenonResult:
    data = json.loads(path.read_text())
    return PhenomenonResult(**data)


def _load_predictor(path: Path | None) -> PredictorResult | None:
    if path is None:
        return None
    text = path.read_text().strip()
    if not text or text == "null":
        return None
    return PredictorResult(**json.loads(text))


def _load_optional_json(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return list(json.loads(path.read_text()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phenomenon-json", type=Path, required=True)
    parser.add_argument("--predictor-json", type=Path, default=None)
    parser.add_argument("--per-dataset-json", type=Path, default=None)
    parser.add_argument("--examples-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=Path("outputs/analysis/pilot_decision.md"))
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=Path("outputs/analysis/pilot_summary.parquet"),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    phenomenon_inputs = _load_phenomenon(args.phenomenon_json)
    predictor_inputs = _load_predictor(args.predictor_json)
    per_dataset = _load_optional_json(args.per_dataset_json)
    examples = _load_optional_json(args.examples_json)

    rendered, decision, phenomenon, predictor = render_decision_document(
        phenomenon_inputs=phenomenon_inputs,
        predictor_inputs=predictor_inputs,
        per_dataset=per_dataset,
        examples=examples,
    )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(rendered)
    write_summary_parquet(args.out_summary, decision, phenomenon, predictor)

    log.info("decision=%s md=%s summary=%s", decision.value, args.out_md, args.out_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
