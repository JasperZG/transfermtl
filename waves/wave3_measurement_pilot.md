# Wave 3 — Measurement pipeline & pilot integration

## Synchronization
- **2 agents: A6, A7**
- A6 depends on all of Wave 2 (A2, A3, A4, A5)
- A7 depends on A6 (and indirectly on all of Wave 2)
- A7 can start scaffolding (script structure, decision-document template)
  while A6 finishes; full integration test waits for A6
- ⛔ **PROJECT GATE**: A7's go/no-go decision document determines whether
  Wave 4+ begins. Pilot failure = workshop pivot or termination.

## Goals
- Compute regional gradient affinity, regional benefits, sign heterogeneity
  / H / C indices end-to-end on Tox21 + SIDER
- Run the Phase 0 pilot (plan §5)
- Evaluate the green-light criteria (plan §5.7) explicitly
- Write `outputs/analysis/pilot_decision.md` with pass/fail per criterion

---

## Agent A6 — Core measurements

### Mission
Implement regional gradient affinity, regional transfer benefit, and
sign-heterogeneity / H / C indices. These are the central measurements
the paper reports. Every quantity must come with hierarchical-bootstrap
CIs from A4 — no point estimates without CIs.

### Owned files

#### `src/transfermtl/gradients/` (plan §2.7)
- `extract.py`
  ```python
  def compute_regional_gradient(
      checkpoint_path: Path,
      task: str,
      region_compounds: list[str],     # SMILES list
      max_subsample: int = 500,
      rng_seed: int = 0,
  ) -> tuple[np.ndarray, int, float]: ...
      # returns (g_vec, n_used, grad_norm)
  ```
  - Loads checkpoint produced by A3
  - Restricts to compounds in `region_compounds`
  - Subsamples to 500 if region is larger
  - Computes `g_i(r) = (1/|D_i(r)|) Σ ∇_θ ℓ_i(x; θ, φ_i)` over the
    subsample
  - Uses `torch.autograd.grad(loss, encoder.parameters(),
    retain_graph=True)`; flattens and concatenates into single vector
  - Encoder parameter order is locked (asserted in unit test) so two
    calls produce identical concatenation order
- `affinity.py`
  - `cosine_affinity(g_i, g_j) -> float | NaN`
  - Returns NaN if either norm < `grad_norm_zero` (1e-8 from config)
  - `dot_product_affinity(g_i, g_j) -> float` for ablation
- `trajectory.py`
  - `compute_trajectory_affinity(checkpoint_dir, ...) -> dict`
  - Returns `{"final": G, "0.8": G, "0.6": G}` per plan §2.7 step 4
  - Reports mean and std across the three checkpoints
- `io.py`
  - `write_region_affinity(...)` writes
    `outputs/gradients/{dataset}/{task_i}_{task_j}/seed{s}/region_affinity.parquet`
  - Validated against `GradientAffinitySchema`
  - One row per `(region_id, checkpoint_label)`

#### `src/transfermtl/benefits/` (plan §2.8)
- `perf.py`
  - `regional_perf(predictions: pd.DataFrame, region_compounds: list[str],
    task: str, task_type: Literal["clf", "reg"]) -> float | NaN`
  - ROC-AUC for clf, negative RMSE for reg
  - Returns NaN if region fails size or class-balance check (plan §2.11
    conditions 2-3)
- `delta.py`
  - `compute_deltas(stl_preds_i, stl_preds_j, mtl_preds_pair, partition,
    task_i, task_j) -> dict[int, RegionBenefit]`
  - Computes for each region:
    - `Δ_ij(r) = Perf_MTL(r) − 0.5*(Perf_STL_i(r) + Perf_STL_j(r))`
    - `Δ_{i←j}(r) = Perf_MTL_i(r) − Perf_STL_i(r)`
    - `Δ_{j←i}(r) = Perf_MTL_j(r) − Perf_STL_j(r)`
    - `Δ_ij^worst(r) = min(Δ_{i←j}, Δ_{j←i})`
- `aggregate.py`
  - Averages across seeds
  - Attaches hierarchical-bootstrap CIs via A4's `within_region`
  - Two-level resampling: scaffolds within region (level 1) + compounds
    within scaffold (level 2) + seed mixing (level 3)

#### `src/transfermtl/indices/` (plan §2.9)
- `sign_heterogeneity.py`
  - `compute_S_pair(deltas: dict[int, BootstrapResult],
    epsilon: float = 1.5) -> bool`
    - Returns True iff there exist two valid regions r_a, r_b such that:
      - `delta(r_a) > +epsilon` AND `CI(r_a)` excludes 0
      - `delta(r_b) < -epsilon` AND `CI(r_b)` excludes 0
  - `compute_S_task_specific(...)` — analogously using `Δ_{i←j}` and
    `Δ_{j←i}` separately
- `heterogeneity_index.py`
  - `compute_H(deltas: dict[int, float], region_test_sizes: dict[int, int])
    -> float`
  - Test-set-weighted mean absolute deviation from the test-set-weighted
    global mean
- `cancellation.py`
  - `compute_C(deltas, region_test_sizes, eta: float = 0.5) -> float`
  - Numerator: weighted sum of `|Δ(r)|`
  - Denominator: `|weighted sum of Δ(r)| + eta`
  - `C > 1` indicates regional magnitudes substantially exceed the
    global average
- `io.py`
  - Writes `outputs/analysis/{dataset}/pair_indices.parquet`
  - Validated against `PairIndicesSchema`

### Scripts
- `scripts/compute_gradients.py`:
  ```
  python scripts/compute_gradients.py --dataset tox21
      --task-i NR-AR --task-j NR-AR-LBD --partition scaffold_M5 --seed 0
  ```
- `scripts/compute_benefits.py`:
  ```
  python scripts/compute_benefits.py --dataset tox21
      --task-i NR-AR --task-j NR-AR-LBD --partition scaffold_M5
  ```
- `scripts/compute_indices.py`:
  ```
  python scripts/compute_indices.py --dataset tox21 --partition scaffold_M5
  ```

### Reads from
- A1: schemas, types, configs
- A2: splits
- A3: checkpoints + prediction parquets (STL, pairwise MTL)
- A4: `bootstrap.hierarchical_bootstrap`,
  `bootstrap.within_region`, `validity.local_support`
- A5: partition parquets

### Writes to
- `outputs/gradients/{dataset}/{task_i}_{task_j}/seed{s}/region_affinity.parquet`
- `outputs/benefits/{dataset}/{task_i}_{task_j}/region_benefits.parquet`
- `outputs/analysis/{dataset}/pair_indices.parquet`

### Tests
- `tests/test_gradients.py`:
  - `test_identity_check` — `G_ii(r) ≈ 1` to within 1e-3 (the §9.6
    sanity check, baked in as a unit test)
  - `test_label_shuffle_drives_to_zero` — shuffling task labels within
    a region makes `G_ij(r)` cluster around 0 (|mean| < 0.15 over 5 trials)
  - `test_undefined_when_zero_grad_norm` — synthetic case with zero
    gradient → returns NaN, not error
  - `test_subsample_above_500` — region with 1000 compounds → uses 500
  - `test_encoder_param_order_locked` — flatten order is identical
    across two calls (asserted via hash of concatenated index map)
  - `test_trajectory_three_checkpoints` — returns dict with keys
    `{"final", "0.8", "0.6"}`
- `tests/test_benefits.py`:
  - `test_delta_pair_recovers_known` — on synthetic fixture: region A
    has `Δ_pair > 0`; region B has `Δ_pair < 0`
  - `test_perf_undefined_for_small_region` — region with n<30 (clf) →
    NaN
  - `test_perf_undefined_for_single_class` — all-positive region → NaN
  - `test_task_specific_consistency` — `Δ_pair = (Δ_{i←j} + Δ_{j←i})/2`
    within fp tolerance
  - `test_bootstrap_ci_attached` — every output has `ci_lo`, `ci_hi`
- `tests/test_indices.py`:
  - `test_S_synthetic_fixture_true` — synthetic fixture → `S_12 = True`
  - `test_S_requires_ci_excludes_zero` — large `Δ` but CI crosses 0 →
    `S = False`
  - `test_H_zero_for_uniform_deltas` — equal `Δ` across regions →
    `H = 0`
  - `test_C_large_for_sign_flips` — synthetic fixture → `C_12 > 5`
  - `test_C_one_when_no_cancellation` — uniform-sign `Δ` → `C ≈ 1`

### Acceptance criteria

- [ ] All tests pass on synthetic fixture
- [ ] On Tox21 (NR-AR, NR-AR-LBD), 5 scaffold regions, 3 seeds:
  - All regions produce well-defined `G_ij(r)` (NaN only when region
    invalid)
  - Bootstrap CIs converge in <30s per region
  - At least one region has CI excluding 0 on `Δ_ij`
- [ ] Indices computed for all valid pairs without crashing
- [ ] All output parquets schema-valid
- [ ] `outputs/manifests/wave3_a6_complete.json` exists

### Out of scope for A6
- Predictor evaluation (uses A6's outputs but is A8)
- Pilot integration (A7)
- Phenomenon expansion to additional datasets (A9)
- Architecture variants (A12)

### References
- `codebase_plan.md` §2.7-2.9
- `plan.txt` §2.7 (line ~2950), §2.8 (line ~2977), §2.9 (line ~3008)

---

## Agent A7 — Pilot integration & decision ⛔ PROJECT GATE

### Mission
Wire A1-A6 together for the Phase 0 pilot (Tox21 + SIDER, 20-40 task
pairs, 3 seeds, M=5 scaffold regions). Evaluate the green-light criteria
from plan §5.7. Write a structured go/no-go decision document. **Wave 4+
does not start until this passes human review.**

### Owned files

#### Pilot orchestration
- `scripts/launch_pilot.sh` — SLURM array script with 7 stages:
  - **Stage 1**: `prepare_dataset.py --dataset tox21` and `--dataset sider`
  - **Stage 2**: STL training, parallelized over `(dataset, task, seed)`.
    Tox21: 12 tasks × 3 seeds = 36 jobs. SIDER: 27 tasks × 3 seeds = 81
    jobs.
  - **Stage 3**: pairwise MTL training, parallelized over `(dataset, pair,
    seed)`. ~25-30 pairs × 2 datasets × 3 seeds = ~150 jobs.
  - **Stage 4**: `compute_partitions.py --scheme scaffold --M 5` and
    `--scheme random --n-partitions 200` for each dataset.
  - **Stage 5**: `compute_gradients.py`, `compute_benefits.py`,
    `compute_indices.py` per (dataset, partition).
  - **Stage 6**: random-partition null distribution: re-run stages 5 over
    the 200 random partitions per dataset (this is the heaviest stage).
  - **Stage 7**: `pilot_decision.py` synthesizes everything.
- `scripts/select_pilot_pairs.py` — picks 20-40 task pairs per dataset:
  - **Tox21**: within-mechanism (NR vs NR pathways), cross-mechanism
    (NR vs SR), and a few random pairs. Examples: (NR-AR, NR-AR-LBD),
    (NR-AR, SR-p53), etc.
  - **SIDER**: SOC-grouped vs cross-SOC pairs. Use the SIDER ATC
    grouping if available.
  - Selection is deterministic given a seed; output saved to
    `outputs/analysis/pilot_pairs.parquet` with columns
    `[dataset, task_i, task_j, category]`.

#### Pilot-specific lightweight predictor
A7 does not build the full Phase 2 predictor (that is A8); it builds a
minimal predictor sufficient for the §5.7 predictor green-light check.

- `src/transfermtl/predictor/pilot_baseline.py`
  - Threshold-based AUROC of `G_ij(r) → sign(Δ_ij(r))`
  - Spearman ρ between `G_ij(r)` and `Δ_ij(r)`
  - Cross-seed sign agreement: fraction of (pair, region) where
    `sign(G_ij(r))` is identical across all 3 seeds
  - Compares against three non-gradient baselines:
    - Scaffold similarity: mean Tanimoto between `D_i(r)` and `D_j(r)`
      scaffolds
    - Regional embedding distance: L2 between mean encoder reps
    - Regional label correlation (where co-measured)
  - Each baseline implemented inline; A8 generalizes these.

#### Decision logic
- `scripts/pilot_decision.py` — evaluates plan §5.7 explicitly:
  - **Phenomenon check** (3 sub-criteria):
    1. ≥5 task pairs across both datasets show stable sign heterogeneity
       (passes `|Δ| > 1.5` AND CI test in at least one positive AND one
       negative region)
    2. ≥15% of meaningful task pairs (per the four-criteria definition)
       exhibit sign heterogeneity
    3. Scaffold-based heterogeneity prevalence exceeds 95th percentile
       of random-partition null distribution
  - **Predictor check** (4 sub-criteria):
    1. AUROC for regional sign prediction ≥ 0.70
    2. AUROC exceeds best non-gradient baseline by ≥ 0.07
    3. Spearman ρ between `G_ij(r)` and `Δ_ij(r)` ≥ 0.45
    4. Sign of `G_ij(r)` agrees in ≥80% of regions across 3 seeds
  - Writes `outputs/analysis/pilot_decision.md` and
    `outputs/analysis/pilot_summary.parquet` (machine-readable).

#### Decision document template
- `templates/pilot_decision_template.md` — Jinja2 template rendered by
  `pilot_decision.py`. Sections:
  1. Executive summary table: each criterion → pass/fail + value
  2. Phenomenon results: per-dataset prevalence with random-partition
     null violin plot reference
  3. Predictor results: AUROC table (G_ij vs each baseline), Spearman ρ,
     cross-seed agreement
  4. Three example task pairs with regional benefit bar charts
  5. Decision tree (plan §5.8) walked through with explicit reasoning
  6. Final decision: `PROCEED` / `WORKSHOP` / `INVESTIGATE` / `DROP`

### Reads from
- All Wave 1+2+3 outputs (A1-A6)

### Writes to
- `outputs/analysis/pilot_pairs.parquet`
- `outputs/analysis/pilot_decision.md` (auto-generated)
- `outputs/analysis/pilot_summary.parquet`
- `outputs/figures/pilot_figure_2_preview.pdf` — early version of paper
  Figure 2 (regional benefit bar chart for one canonical pair)
- `outputs/manifests/wave3_a7_complete.json`

### Tests
- `tests/test_pilot_e2e.py`:
  - `test_pilot_smoke_synthetic` — runs the entire pipeline on synthetic
    fixture (1 task pair) end-to-end:
    - Loads fixture
    - "Trains" via short STL+MTL (10 epochs each)
    - Computes scaffold partition
    - Extracts gradients, computes benefits, indices
    - Asserts `S_12 = True`, `G_12(A) > 0`, `G_12(B) < 0`
    - Runs in <60s on CPU
  - `test_pilot_decision_script_runs` — pilot decision script produces
    a non-empty markdown file when fed mock measurement outputs
  - `test_select_pilot_pairs_deterministic` — same seed → same pair list
  - `test_phenomenon_criteria_evaluation` — feed synthetic
    pre-computed indices, verify pass/fail logic for each criterion
  - `test_predictor_criteria_evaluation` — feed synthetic predictor
    scores, verify pass/fail logic

### Acceptance criteria

- [ ] Full pilot completes within ~40 GPU-hours on H100s (per plan §2.18)
- [ ] `pilot_decision.md` includes:
  - [ ] Green-light criteria table with explicit pass/fail per
    sub-criterion (3 + 4 = 7 rows)
  - [ ] Per-dataset heterogeneity prevalence + random-partition null
    p-value
  - [ ] Predictor AUROC for `G_ij(r)` vs scaffold similarity vs
    embedding distance vs label correlation
  - [ ] Spearman ρ
  - [ ] Cross-seed sign agreement percentage
  - [ ] Three example task pairs with regional benefit visualization
  - [ ] Walked-through decision tree with reasoning
  - [ ] Final decision: one of `PROCEED` / `WORKSHOP` / `INVESTIGATE` /
    `DROP`
- [ ] Smoke test on synthetic fixture passes in <60s
- [ ] All Tox21+SIDER pilot outputs schema-valid
- [ ] Number of trained models matches expected count (no silent failures)
- [ ] `outputs/manifests/wave3_a7_complete.json` enumerates everything

### Decision rules and downstream consequences (plan §5.8)
- **Both pass** → "PROCEED"; Wave 4+ green-lit
- **Phenomenon passes, predictor fails** → "WORKSHOP" or "INVESTIGATE";
  Wave 4 may proceed but A8's framing must be reconsidered (workshop
  paper or invest in better predictors first)
- **Phenomenon fails, predictor untestable** → "PIVOT" to weaker framing
  ("conditions under which global affinity is sufficient") or "DROP";
  Wave 4 blocked
- **Both fail** → "DROP"

A7 does not unilaterally proceed past the gate. It writes the document.
**Human review (PI sign-off) approves the next wave.**

### Out of scope for A7
- Building the full predictor (A8): A7's predictor is a 4-feature
  baseline only
- Phenomenon expansion to other datasets (A9)
- Any utility experiments (Wave 5)
- Robustness sweeps (Wave 6)
- Cross-dataset transfer evaluation (A8)

### References
- `codebase_plan.md` §5 (Milestone 1)
- `plan.txt` §5 entire (line ~202), §5.7 green-light criteria, §5.8
  decision tree, §13.1
