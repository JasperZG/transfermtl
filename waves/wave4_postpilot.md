# Wave 4 — Post-pilot expansion

## Synchronization
- **2 agents in parallel: A8, A9**
- Both depend on A7's pilot greenlight (project gate)
- A8 and A9 do not depend on each other; they can be developed
  simultaneously after the pilot decision document is signed off
- ⛔ **MID-PROJECT GATE**: A8's incremental R² test (plan §2.14, §13.2)
  determines whether the project remains a NeurIPS candidate. If
  incremental R² < 0.05 over chemistry baselines, reframe as workshop
  discovery paper and reconsider Wave 5.

## Goals
- A8 evaluates all 10 predictors on Tox21 + SIDER + ToxCast, including
  the **incremental R² centerpiece**
- A9 expands to ToxCast, TDC ADMET, ChEMBL kinase, QM9, synthetic
  matched-pair; runs all three partition schemes; produces Phase 1
  prevalence tables, cross-partition agreement, canonical examples

## Cross-agent guardrails
- A8 and A9 both consume A6's measurement pipeline; neither modifies it
- A9 adds new datasets to `data/datasets.py`; A8 does not
- A9 's expanded datasets become available to A8 mid-wave; A8 should be
  designed so adding a new dataset is a config-only change

---

## Agent A8 — Predictor evaluation & incremental R² ⛔ MID-PROJECT GATE

### Mission
Build the full Phase 2 predictor stack: 10 predictors (3 primary, 6
non-gradient baselines, 1 joint), all evaluation metrics, calibration
analysis, cross-dataset transfer, and the **incremental R² test** that
determines whether gradient affinity adds value over chemistry baselines.
This is the project's centerpiece scientific result.

### Owned files

#### `src/transfermtl/predictor/` (full version, replaces A7's pilot stub)

- `features.py` — per-(pair, region) feature computation
  ```python
  def compute_pair_region_features(
      dataset: str, task_i: str, task_j: str, partition_path: Path,
      checkpoints: dict[str, Path],
  ) -> pd.DataFrame[pair_id, region_id, feature_name, value]
  ```
  Features (plan §7.3 + §2.14):
  1. `G_ij(r)` (local gradient cosine, from A6)
  2. `G_ij(r)` + local support sizes (combined feature)
  3. TAG-style local affinity: effect of one task's regional gradient
     step on the other task's regional loss
  4. **Scaffold similarity**: mean Tanimoto between `D_i(r)` scaffolds
     and `D_j(r)` scaffolds
  5. **Regional embedding distance**: L2 between centroid encoder
     representations of `D_i(r)` and `D_j(r)`
  6. **Regional label correlation**: Pearson on co-measured compounds
     in region `r` (NaN when fewer than 10 co-measured)
  7. Regional MMD between encoder representations of `D_i(r)` and
     `D_j(r)` (RBF kernel; bandwidth via median heuristic)
  8. Global gradient affinity `G_ij^global` from ICLR paper, applied
     as constant predictor across all regions of the pair
  9. Regional Task2Vec similarity (compute-permitting; flagged optional)
  10. Local support sizes (logged): `log|D_i(r)|`, `log|D_j(r)|`

- `methods.py` — predictor implementations
  - Single-feature threshold predictors (varying threshold over feature)
  - Joint logistic regression (sklearn, `class_weight=balanced`)
  - Joint linear regression for continuous targets
  - Targets:
    - Binary: `sign(Δ_ij(r))` with neutral exclusion when `|Δ| < 0.5`
    - Continuous: `Δ_ij(r)`
    - Three-class: positive / negative / no significant transfer
    - Task-specific: `sign(Δ_{i←j}(r))`, `sign(Δ_{j←i}(r))`

- `cv.py` — leave-task-pairs-out cross-validation
  - 5-fold CV; each fold holds out 20% of task pairs (with all their
    regions). Crucially, regions of the same pair never split across
    folds — this prevents leakage.
  - `cross_validate(features, targets, model_factory, n_folds=5)`
  - Returns per-fold predictions + aggregated metrics

- `incremental_r2.py` — **THE CENTERPIECE**
  ```python
  def incremental_r2_test(
      features: pd.DataFrame, target: pd.Series,
      n_bootstrap: int = 1000,
  ) -> dict
  ```
  Two regressions on continuous target `Δ_ij(r)`:
  - **Baseline**: `Δ ~ ScaffoldSim + LabelCorr + EmbeddingDist +
    log|D_i| + log|D_j|`
  - **Full**: baseline + `G_ij(r) + G_ij(r)^2`
  Reports:
  - `R²_full − R²_baseline` with hierarchical-bootstrap CI (uses A4)
  - Partial F-test on the addition of `G_ij(r)` terms
  - Bootstrap distribution histogram (saved as PNG for the appendix)
  - Per-dataset breakdown
  - Per-partition-scheme breakdown (within-scaffold-cluster sub-analysis
    addresses the §1.3 mitigation)

- `calibration.py`
  - ECE with 10 bins on share/separate decisions
  - Reliability diagrams as matplotlib figures
  - Brier score
  - Returns `CalibrationReport(ece, brier, reliability_curve)`

- `cross_dataset.py`
  - `cross_dataset_transfer(train_dataset, test_dataset, model)
    -> dict[metric, value]`
  - Tox21→SIDER, SIDER→ToxCast, etc.

- `eval.py` — top-level entry point
  - Runs all 10 predictors against all targets
  - Produces `outputs/analysis/predictor_comparison.parquet` with one
    row per (predictor, target, dataset, metric)
  - Writes the Phase 2 deliverable summary markdown

#### Phase 2 figure scripts
- `src/transfermtl/analysis/phase2_figures.py`
  - **Figure 4**: scatter of `G_ij(r)` vs `Δ_ij(r)` across all datasets,
    colored by dataset, with marginal histograms + regression line.
  - **Figure 5a**: bar chart of AUROC for regional sign prediction across
    all 10 predictors with hierarchical-bootstrap CIs as error bars.
  - **Figure 5b**: incremental R² centerpiece. Bar chart of incremental
    R² for `G_ij(r)` over baseline. **This panel addresses the primary
    scientific risk.**

#### Scripts
- `scripts/eval_predictors.py`:
  ```
  python scripts/eval_predictors.py --datasets tox21 sider toxcast
      --partitions scaffold latent
  ```

### Reads from
- A1: schemas, types, configs
- A2 + A9: split parquets for all datasets
- A3 + A11: encoder checkpoints (for embedding distance, MMD)
- A4: hierarchical bootstrap (for incremental R² CI)
- A5: partition parquets
- A6: gradient affinity, regional benefits, pair indices
- A7: pilot baseline predictor (A8 generalizes)

### Writes to
- `outputs/analysis/pair_region_features.parquet` (validated against
  `PredictorScoresSchema` extended with feature columns)
- `outputs/analysis/predictor_comparison.parquet`
- `outputs/analysis/incremental_r2_results.parquet`
- `outputs/analysis/cross_dataset_transfer.parquet`
- `outputs/analysis/calibration_reports/{predictor}.parquet`
- `outputs/figures/figure_4_g_vs_delta.pdf`
- `outputs/figures/figure_5_predictor_comparison.pdf`
- `outputs/figures/figure_5b_incremental_r2.pdf`
- `outputs/figures/appendix_incremental_r2_bootstrap_hist.pdf`
- `outputs/manifests/wave4_a8_complete.json`

### Tests
- `tests/test_predictor.py`:
  - `test_features_compute_on_synthetic` — all 10 features computed
    without error on fixture
  - `test_features_schema_validates`
  - `test_cv_no_pair_leakage` — held-out pairs in fold k do not appear
    in fold k's training pairs (parametrized over folds)
  - `test_logistic_regression_recovers_synthetic_signal` — when
    `G_ij(r)` perfectly predicts `sign(Δ)`, AUROC ≈ 1.0
  - `test_incremental_r2_zero_when_redundant` — when `G_ij(r)` is a
    linear combination of baseline features, incremental R² ≈ 0
  - `test_incremental_r2_positive_when_independent` — when `G_ij(r)`
    adds independent signal, incremental R² > 0
  - `test_partial_f_test_pvalue` — F-test on a known-positive case
    yields p < 0.01
  - `test_calibration_ece_zero_for_perfect_predictor` — perfect predictor
    has ECE = 0
  - `test_cross_dataset_runs_without_pair_overlap` — train pairs and
    test pairs are disjoint (cross-dataset, this is automatic)

### Acceptance criteria

- [ ] All 10 predictor methods evaluated on Tox21, SIDER, ToxCast
- [ ] **Incremental R² test produces a numeric value with bootstrap CI
  for each dataset.** This is the gating measurement.
- [ ] AUROC for sign prediction reported with hierarchical-bootstrap CIs
- [ ] Calibration (ECE, reliability) reported for top 3 predictors
- [ ] Cross-dataset transfer results for at least Tox21→SIDER and
  SIDER→ToxCast
- [ ] Figures 4, 5a, 5b produced
- [ ] Phase 2 deliverable summary markdown written
- [ ] All output parquets schema-valid
- [ ] Tests pass on synthetic fixture
- [ ] `outputs/manifests/wave4_a8_complete.json` exists

### ⛔ Mid-project gate (plan §13.2)
- **Incremental R² ≥ 0.05 across multiple datasets** → continue to Wave 5
  utility experiments
- **Incremental R² < 0.05** → kill NeurIPS target; reframe as workshop
  discovery paper. Wave 5 may still proceed with a smaller scope (focus
  on phenomenon and calibration rather than utility), but the framing
  changes substantially. **Human review required.**

### Out of scope for A8
- Adding new datasets (A9)
- Local-affinity sharing architecture (A10)
- Baselines beyond what's needed for predictor evaluation (A11)
- Utility metrics (negative transfer rate, share/separate calibration in
  the deployment sense) — A13
- Phase 4 robustness sweeps (A12 / A13)

### References
- `codebase_plan.md` §2.14, §5 (Milestone 2)
- `plan.txt` §2.14 (line ~3099), §7 entire (line ~336), §13.2

---

## Agent A9 — Phenomenon expansion

### Mission
Expand the phenomenon characterization beyond the pilot's Tox21 + SIDER.
Add ToxCast, TDC ADMET, ChEMBL kinase, QM9, and a synthetic matched-pair
dataset. Run all three partition schemes (scaffold, latent, kNN) on
every dataset. Produce the full Phase 1 deliverables: prevalence tables,
cross-partition agreement, canonical examples catalog.

### Owned files

#### Dataset additions to `src/transfermtl/data/datasets.py`
- `load_toxcast()` — MoleculeNet ToxCast, 17 endpoints, ~80% overlap.
  Per-task scaffold-stratified splits with shared global scaffold→bin.
- `load_tdc_admet()` — TDC ADMET subset with sufficient sample sizes.
  Use `tdc.benchmark_group.admet_group`. Filter to tasks with ≥1000
  compounds.
- `load_chembl_kinase()` — 21-kinase selectivity panel from the ICLR
  paper. Source: ChEMBL release version pinned by the ICLR paper. If
  the panel composition is documented in `plan.txt` references, follow
  that exactly.
- `load_qm9()` — 12 quantum properties, 100% overlap. Used as physical
  reference where heterogeneity may be lower (a control).
- `load_tox21_adme_matched()` — cross-domain matched dataset from the
  ICLR paper.
- `load_synthetic_matched_pair()` — A9 designs and generates this; see
  below.

#### Synthetic matched-pair dataset
- `scripts/generate_synthetic_dataset.py` — produces a deterministic
  synthetic dataset with known sign heterogeneity for sanity-checking
  Phase 1 results. Stronger than A1's tests/conftest fixture: realistic
  scale (~5000 compounds, 4 regions, 6 tasks) and saved as a frozen
  parquet under `data/synthetic_matched_pair/data.parquet`.
- The construction follows the same logic as A1's fixture but at scale:
  task pairs are designed to have prescribed regional sign patterns.
  Document the construction in a docstring + `data/synthetic_matched_pair/README.md`.

#### Phase 1 analysis
- `src/transfermtl/analysis/phase1.py`
  - `compute_phase1_outputs(datasets, partitions) -> dict`:
    - Prevalence table: % meaningful pairs with sign heterogeneity, per
      (dataset, partition scheme), with random-partition null comparison
      and empirical p-value
    - Task-specific heterogeneity prevalence (separate from pair-averaged)
    - Cancellation analysis: distribution of `C_ij` across pairs;
      Spearman correlation between low global affinity and high `C_ij`
    - Cross-partition agreement: for sign-heterogeneous pairs under
      scaffolds, what fraction also show heterogeneity under latent
      clusters?
    - Random-partition control: full null distribution comparison
- `src/transfermtl/analysis/case_studies.py`
  - Selects 5-10 canonical sign-heterogeneous task pairs across datasets
    for use in figures and case studies (Figure 2 in particular).
  - Selection criteria: maximum `H_ij`, both regions pass CI test, at
    least one pair per dataset, prefer pairs with intuitive chemical
    interpretation.
  - Outputs `outputs/analysis/canonical_examples.parquet`.

#### Phase 1 figures
- `src/transfermtl/analysis/phase1_figures.py`
  - **Figure 2**: empirical sign heterogeneity examples (3 task pairs:
    Tox21, SIDER, ToxCast). Regional benefit bar chart with CIs, both
    pair-averaged and task-specific.
  - **Figure 3a**: prevalence per (dataset, partition scheme), with
    random-partition null distribution as violin plots underneath.
  - **Figure 3b**: scatter of global affinity `G_ij^global` vs
    cancellation index `C_ij` showing low-global-high-cancellation
    correlation.

#### Scripts
- `scripts/generate_synthetic_dataset.py` (above)
- `scripts/run_phase1.sh` — SLURM array dispatching dataset preparation,
  STL/MTL training, partitioning (3 schemes), measurement, indices,
  and null distribution for every (dataset, partition scheme).

### Reads from
- All Wave 1+2+3 outputs
- Same A6 measurement pipeline; no modification

### Writes to
- `outputs/splits/{toxcast,tdc_admet,chembl_kinase,qm9,tox21_adme_matched,synthetic_matched_pair}/split.parquet`
- `outputs/partitions/{dataset}/{scheme}_M{M}.parquet` for all new
  datasets and all schemes
- `outputs/predictions/{dataset}/...` (re-uses A3 training scripts)
- `outputs/gradients/{dataset}/...` (re-uses A6)
- `outputs/benefits/{dataset}/...` (re-uses A6)
- `outputs/analysis/{dataset}/pair_indices.parquet` (re-uses A6)
- `outputs/analysis/phase1_prevalence.parquet`
- `outputs/analysis/phase1_cancellation.parquet`
- `outputs/analysis/phase1_cross_partition_agreement.parquet`
- `outputs/analysis/canonical_examples.parquet`
- `outputs/figures/figure_2_sign_heterogeneity_examples.pdf`
- `outputs/figures/figure_3a_prevalence.pdf`
- `outputs/figures/figure_3b_cancellation.pdf`
- `data/synthetic_matched_pair/data.parquet` (committed to git, frozen)
- `outputs/manifests/wave4_a9_complete.json`

### Tests
- `tests/test_phenomenon_expansion.py`:
  - `test_toxcast_loader_returns_expected_shape`
  - `test_tdc_admet_loader_returns_expected_shape`
  - `test_chembl_kinase_loader_returns_expected_shape`
  - `test_qm9_loader_returns_regression_targets`
  - `test_synthetic_matched_pair_has_known_heterogeneity` — generated
    synthetic dataset has at least 3 task pairs with verifiable sign
    heterogeneity (`S_ij = True` after running the full pipeline)
  - `test_cross_partition_agreement_metric` — synthetic dataset where
    scaffolds and latent clusters agree → agreement metric ≈ 1.0
  - `test_canonical_example_selection_deterministic`

### Acceptance criteria

- [ ] All 6 new datasets have schema-valid `split.parquet`
- [ ] Per dataset, all 3 partition schemes (scaffold, latent, kNN) +
  random null produce schema-valid parquets
- [ ] Per dataset, full STL+MTL training + measurement pipeline runs
  without error
- [ ] Phase 1 prevalence table includes p-values from random-partition
  null
- [ ] Cross-partition agreement reported for at least scaffolds + latent
- [ ] Canonical examples catalog has ≥5 sign-heterogeneous pairs across
  datasets
- [ ] Figures 2, 3a, 3b produced
- [ ] Synthetic matched-pair dataset frozen and committed
- [ ] `outputs/manifests/wave4_a9_complete.json` exists

### Out of scope for A9
- Predictor evaluation (A8)
- Architecture variants beyond GCN — A12
- Utility / sharing experiments — Wave 5
- Robustness sweeps — Wave 6

### References
- `codebase_plan.md` §5 (Milestone 3)
- `plan.txt` §6 entire (line ~271), §6.2 datasets, §6.3 partitioning,
  §6.6 outputs
