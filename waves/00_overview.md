# Wave Plan — Overview

This folder contains the agent assignments for building the Transfer Sign
Heterogeneity codebase in parallel. Read this file first; then read the
specific wave file you are executing.

## Wave structure

6 waves, 13 agents total. Peak parallelism is 4 (Wave 2). Two hard gates
that block progression: A7 (pilot greenlight) and A8 (incremental R²).

| Wave | Agents | Theme | Gate? |
|---|---|---|---|
| 1 | A1 | Foundation, contract lock | — |
| 2 | A2, A3, A4, A5 | Independent building blocks | — |
| 3 | A6, A7 | Measurement pipeline + pilot integration | ⛔ A7 |
| 4 | A8, A9 | Post-pilot predictor + phenomenon expansion | ⛔ A8 |
| 5 | A10, A11, A12 | Sharing architecture + baselines + sweep harness | — |
| 6 | A13 | Utility eval + sweep execution + figures + paper assembly | — |

## Agent index

- **A1** — Foundation & contract lock (Wave 1)
- **A2** — Data pipeline (Wave 2)
- **A3** — Models + training (Wave 2)
- **A4** — Statistical core: bootstrap, validity, null (Wave 2)
- **A5** — Region partitioning (Wave 2)
- **A6** — Core measurements: gradients, benefits, indices (Wave 3)
- **A7** — Pilot integration & decision document ⛔ (Wave 3)
- **A8** — Predictor evaluation + incremental R² ⛔ (Wave 4)
- **A9** — Phenomenon expansion (datasets, partitions) (Wave 4)
- **A10** — Local-affinity sharing architecture (Wave 5)
- **A11** — Baselines (Wave 5)
- **A12** — Robustness sweep harness (Wave 5)
- **A13** — Utility eval + sweep execution + figures (Wave 6)

## How to read a wave file

Each wave file contains, for every agent in that wave:

1. **Mission** — one paragraph summary of what the agent owns
2. **Owned files** — explicit paths the agent creates
3. **Reads from** — inputs from earlier waves (schemas, parquets,
   checkpoints)
4. **Writes to** — outputs other agents will consume
5. **Module specifications** — per-`.py`-file contents and key signatures
6. **Tests** — every test file and what it asserts
7. **Acceptance criteria** — explicit checklist; PR cannot merge unless all
   items pass
8. **Out of scope** — what NOT to touch (prevents agent overlap)
9. **References** — section pointers into `plan.txt` and `codebase_plan.md`

A brief is intended to be self-contained: an agent should be able to
execute against the brief without reading other wave files (though they
must read A1's locked schemas and types).

## Critical conventions (apply to every agent)

### 1. Frozen contracts
A1 freezes pandera schemas under `src/transfermtl/utils/schemas.py` and
dataclasses under `src/transfermtl/utils/types.py`. **No agent in Wave 2+
edits these files.** Any contract change requires a coordination PR before
parallel work resumes.

### 2. Synthetic fixture
A1 ships `tests/conftest.py` with a 200-compound 2-task fixture with known
ground-truth sign heterogeneity (region A: aligned, region B: opposed).
Every Wave 2+ agent uses this fixture for unit tests rather than real
datasets.

### 3. No re-training in measurement code
Once A3 saves checkpoints + prediction parquets, downstream modules
(gradients, benefits, indices, predictor, eval) read those parquets. No
module re-runs training to compute metrics.

### 4. Pre-committed hyperparameters are locked
plan.txt §2.21 values live in `configs/_shared/`. CI hashes these files
against `_lock.yaml`; mismatches fail the run. This prevents post-hoc
hyperparameter changes (p-hacking).

### 5. Hierarchical bootstrap everywhere
Any reported scalar must come with a CI from
`bootstrap.hierarchical_bootstrap` (A4). Plain iid bootstrap is
incorrect — compounds within scaffolds are not independent.

### 6. Random-partition null is required
Any phenomenon-prevalence claim must be accompanied by the random-partition
null distribution and an empirical p-value (plan §2.13). A4 builds the
framework; A6/A7 wire it in.

### 7. Gates are hard stops
- **A7** writes a go/no-go decision document. If pilot fails the §5.7
  green-light criteria, Wave 4+ does not start.
- **A8** must clear incremental R² ≥ 0.05 over chemistry baselines on
  multiple datasets. If it fails, the project pivots to a workshop
  framing and Wave 5 is reconsidered (plan §13.2).

## File layout that emerges

```
src/transfermtl/
├── utils/        (A1)
├── data/         (A2 + A9)
├── partition/    (A5)
├── models/       (A3 + A12 architecture variants)
├── training/     (A3)
├── bootstrap/    (A4)
├── validity/     (A4)
├── null/         (A4)
├── gradients/    (A6)
├── benefits/     (A6)
├── indices/      (A6)
├── predictor/    (A7 stub, A8 full)
├── architecture/ (A10)
├── baselines/    (A11)
├── eval/         (A13)
└── analysis/     (A13)
```

Ownership is exclusive: only one agent writes to any given module
directory. Cross-agent communication is via parquet/npz/checkpoint files
with locked schemas.

## Compute budget per wave

Approximate H100 GPU-hour estimates (plan §2.18):

| Wave | Compute | Notes |
|---|---|---|
| 1 | ~0 | CPU only |
| 2 | ~5 | Smoke training on synthetic + 1-2 real Tox21 STL runs |
| 3 (pilot) | ~40 | STL × 39 tasks × 3 seeds + MTL × 25 pairs × 3 seeds + bootstrap |
| 4 | ~80 | Phenomenon expansion datasets + predictor eval |
| 5 | ~40 | Architecture training + baseline training |
| 6 | ~250 | Robustness sweeps, multi-seed final, all utility experiments |

Total: ~415 GPU-hours, well within the 400-600 GPU-hour estimate.

## Synchronization protocol between waves

1. Last agent in a wave (or the wave's gate agent) opens a "wave complete"
   PR that:
   - Updates `outputs/manifests/wave{N}_complete.json`
   - Lists all artifacts produced
   - Confirms acceptance criteria for every agent in the wave
2. Reviewer signs off; only then does the next wave open agent PRs.
3. For gated waves (3, 4), human review of the decision document is
   required before the next wave starts.

## Modifying these wave files

Once a wave file is approved, its agent briefs are immutable. If an agent
discovers their brief is wrong (e.g., a missing dependency, a mis-specified
schema), they file a coordination PR that updates the brief AND any
affected schemas, and notifies parallel agents. **Do not silently expand
scope.**
