# Wave 6 — Assembly

## Synchronization
- **1 agent: A13** (sequential)
- Depends on all earlier waves; cannot start until A10/A11/A12 are merged
- This is the final wave before paper submission

## Goals
- Run all utility experiments (Phase 3) using A10's architecture and
  A11's baselines
- Execute all Phase 4 robustness sweeps using A12's harness
- Produce every figure listed in plan §11
- Assemble paper-ready tables and the final reproducibility manifest
- Write the public-implementation README for the codebase

---

## Agent A13 — Utility evaluation, sweep execution, figures, paper assembly

### Mission
Execute every remaining experiment, produce every figure, and ship the
codebase in a state where the paper can be written from saved artifacts
alone. After A13 lands:
- Every figure in `plan.txt` §11 exists at `outputs/figures/`
- Every table in the paper has a corresponding parquet at
  `outputs/tables/`
- `make reproduce` regenerates everything from scratch given checkpoints

### Owned files

#### `src/transfermtl/eval/` (utility metrics, plan §2.17 + §8.5)
- `utility_metrics.py`
  - Per-task AUC / RMSE averaged across tasks
  - Worst-region per-task performance (per task, take min over regions;
    then average across tasks)
  - Worst-task overall performance
  - Average regional performance
  - Sample-weighted average performance
- `negative_transfer.py` — plan §8.5 negative-transfer metrics:
  - `NT_rate(method_preds, stl_preds, threshold=1.5) -> float`
  - Severe NT rate at threshold 3.0
  - Average regional NT magnitude
  - Task-specific NT rate using `Δ_{i←j}` instead of pair average
- `calibration_utility.py` — for methods producing share/separate
  decisions:
  - ECE on these decisions vs ground-truth `sign(Δ_ij(r))`
  - Reliability diagrams with 10 bins
  - Brier score
- `holdout_calibration.py` — 5-fold leave-task-pairs-out CV:
  - Train decision rule on 80% of pairs
  - Evaluate calibration on 20%
  - Reports calibration on held-out pairs
- `statistical_compare.py` — paired Wilcoxon signed-rank between methods
  across (task, region) combinations; Bonferroni correction across the
  comparison family
- `efficiency.py` — training compute (GPU-hours), inference latency,
  parameter counts

#### `src/transfermtl/analysis/` (figures and decomposition, plan §2.20 + §11)
- `decomposition.py` — plan §2.20 global-vs-regional decomposition:
  - Computes `G_ij^global = cos(Σ p_i(r) g_i(r), Σ p_j(r) g_j(r))`
  - Naive linear approximation `bar_G_ij = Σ p(r) G_ij(r)`
  - Cancellation gap `|G_ij^global − bar_G_ij|`
  - Documents this with a worked example (one canonical pair)
- `figures.py` — orchestrates all 9 figures from plan §11. One function
  per figure, each reading from saved parquets/npy and writing PDFs.
- `tables.py` — pandas-to-LaTeX conversion for every table the paper
  needs. Output at `outputs/tables/{table_name}.tex`.

#### `scripts/` — execution scripts
- `scripts/eval_utility.py` — runs full Phase 3 utility evaluation:
  - For each method (A10's three sharing variants + A11's eight
    baselines + STL):
    - Compute all utility metrics on Tox21, SIDER, ToxCast, TDC ADMET,
      kinase panel
    - Compute calibration on share/separate decisions
    - Compute held-out calibration via leave-task-pairs-out CV
    - Compute efficiency metrics
  - Run paired Wilcoxon comparisons
  - Output: `outputs/analysis/phase3_utility.parquet`
- `scripts/run_phase4_sweeps.py` — executes A12's robustness harness:
  - Partition sweep (M ∈ {3, 5, 8, 10}, four schemes)
  - n_min sweep
  - Architecture sweep (GCN, GAT, ECFP, ChemBERTa if compute allows)
  - Gradient ablations (last-layer, dot product, Fisher-normalized,
    trajectory mean)
  - Negative controls (label permutation, random regions, identity check)
- `scripts/make_figures.py` — invokes `analysis.figures` to regenerate
  every figure deterministically. Idempotent. Should run in <10 minutes
  on a single machine given saved artifacts.
- `scripts/make_tables.py` — invokes `analysis.tables`.
- `scripts/audit_results.py` — validates plan §13.4 reproducibility
  checklist for every result in the paper:
  - Producing script exists
  - Config checked in
  - Git SHA captured
  - Seed list logged
  - Bootstrap CIs present
  - Random-partition null reported where relevant
- `scripts/reproduce.sh` — single command that regenerates all
  Phase 1-4 artifacts from scratch given access to checkpoints.

#### Documentation
- `README.md` — public-implementation README:
  - Project description (1-paragraph summary of the paper)
  - Installation
  - Quickstart: how to reproduce the pilot
  - How to reproduce each figure
  - Citation block
  - License
- `paper/` — directory for paper-side resources (LaTeX is in a separate
  repo, but `paper/figure_manifest.md` lists every figure and its
  source script + output path).

#### Phase 3 + 4 figures (plan §11)
- **Figure 1**: conceptual schematic. Two task gradients over input
  space partitioned into regions. In region A, gradients align (positive
  transfer); region B, gradients oppose. Hand-drawn or matplotlib
  schematic; not derived from data.
- **Figure 2**: empirical sign heterogeneity examples. Done in Wave 4
  (A9), refresh with final 5-seed data here.
- **Figure 3**: prevalence and cancellation across datasets. Done in
  Wave 4 (A9), refresh.
- **Figure 4**: scatter of `G_ij(r)` vs `Δ_ij(r)` across all datasets.
  Done in Wave 4 (A8), refresh.
- **Figure 5**: predictor comparison + incremental R² centerpiece. Done
  in Wave 4 (A8), refresh.
- **Figure 6**: identifiability and abstention. NEW in Wave 6:
  - Left: predictor accuracy as a function of regional sample size
  - Right: retained accuracy after abstention vs abstention rate
- **Figure 7**: utility — negative transfer reduction and calibration.
  NEW in Wave 6:
  - Left: NT rate across all sharing methods
  - Right: calibration reliability diagram for share/separate decisions
- **Figure 8**: robustness summary. NEW in Wave 6:
  - Phenomenon prevalence across architectures
  - Predictor AUROC across architectures
  - Sign agreement across partition schemes
  - Random partition negative control
- **Figure 9 (appendix)**: architecture diagram for local-affinity
  sharing model. Hand-drawn or matplotlib schematic.

### Reads from
- All earlier waves' outputs

### Writes to
- `outputs/analysis/phase3_utility.parquet`
- `outputs/analysis/phase4_robustness/{sweep}/*.parquet`
- `outputs/analysis/decomposition_examples.parquet`
- `outputs/analysis/audit_report.md`
- `outputs/figures/figure_1_schematic.pdf` through
  `outputs/figures/figure_9_architecture.pdf`
- `outputs/tables/*.tex` (one per paper table)
- `README.md`
- `outputs/manifests/wave6_a13_complete.json`
- `outputs/manifests/final_reproducibility_manifest.json` — captures
  everything needed for paper submission

### Tests
- `tests/test_eval_utility.py`:
  - `test_nt_rate_zero_when_method_matches_stl`
  - `test_nt_rate_one_when_method_uniformly_worse`
  - `test_severe_nt_rate_subset_of_nt_rate`
  - `test_paired_wilcoxon_runs`
  - `test_bonferroni_correction_applied`
- `tests/test_make_figures.py`:
  - `test_each_figure_function_runs_on_synthetic` — every figure-making
    function runs on synthetic mock data without error
  - `test_figures_idempotent` — regenerating produces byte-identical
    PDFs (matplotlib `rcParams` set deterministically)
- `tests/test_audit.py`:
  - `test_audit_passes_on_complete_run` — fixture run with all
    artifacts present passes audit
  - `test_audit_fails_when_missing_ci` — artifact without bootstrap CI
    fails audit
  - `test_audit_fails_when_missing_seeds` — artifact without 5 seeds
    in metadata fails audit
- `tests/test_reproduce_pipeline.py`:
  - `test_reproduce_synthetic_e2e` — `reproduce.sh` on synthetic fixture
    runs end-to-end and produces all expected outputs

### Acceptance criteria

- [ ] All Phase 3 utility metrics computed for all sharing methods on
  all five datasets (Tox21, SIDER, ToxCast, TDC ADMET, kinase panel)
- [ ] All Phase 4 sweeps executed:
  - Partition robustness across M ∈ {3, 5, 8, 10} and 4 schemes
  - n_min sweep at 4 thresholds
  - Architecture sweep across 3-4 architectures
  - All 4 gradient ablations
  - All 3 negative controls
- [ ] All 9 figures present at `outputs/figures/figure_*.pdf`
- [ ] All paper tables present at `outputs/tables/*.tex`
- [ ] `make_figures.py` regenerates all figures from saved artifacts in
  <10 minutes on a single machine
- [ ] `audit_results.py` reports zero violations of the §13.4
  reproducibility checklist
- [ ] `reproduce.sh` documented and tested on synthetic fixture
- [ ] README written
- [ ] `outputs/manifests/wave6_a13_complete.json` and
  `outputs/manifests/final_reproducibility_manifest.json` exist
- [ ] All tests pass

### Pre-submission decision (plan §13.3)
A13 records the result of the Phase 3 utility evaluation in
`outputs/analysis/phase3_summary.md`. The submission decision follows
plan §13.3:
- **Clear NT reduction with comparable or better average performance** →
  NeurIPS submission, primary framing on utility
- **Match adaptive baseline on average but with calibration advantage** →
  NeurIPS submission with framing around regional diagnosis and
  calibration
- **Substantially underperform adaptive baselines** → ICML or workshop
  submission with phenomenon-focused framing

A13 does not unilaterally decide submission venue. It writes the summary
and the team decides.

### Out of scope for A13
- Modifying any earlier-wave code (if a bug is found, file a coordination
  PR; do not fix in-place during Wave 6)
- Adding new datasets, architectures, or baselines
- LaTeX paper writing (separate repo)
- Slide decks or talks

### Practical notes
- A13 is the heaviest wave by compute (~250 GPU-hours estimated). Stage
  the work:
  1. Phase 3 utility on Tox21 first (fastest dataset); sanity-check
     the metrics pipeline
  2. Then SIDER, ToxCast, TDC ADMET, kinase panel
  3. Phase 4 sweeps last; partition + n_min are cheap, architecture is
     expensive
- Save sweep checkpoints: do NOT delete intermediate artifacts. The
  `outputs/` directory after A13 is the paper's reproducibility evidence.
- Match figure styling across all 9 figures (font, color palette,
  sizing). Use `analysis.figures._set_style()` as the central style
  setter; each figure function calls it.

### References
- `codebase_plan.md` §2.17, §2.20, §5 (Milestone 6)
- `plan.txt` §2.17 (line ~4599), §2.20 (line ~4661), §8 entire
  utility (line ~392), §9 robustness (line ~488), §11 figures
  (line ~585), §13.3 decision framework
