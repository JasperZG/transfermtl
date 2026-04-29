# Codebase Plan: Transfer Sign Heterogeneity in Multi-Task Learning

This document specifies the implementation architecture for the research described
in `plan.txt`. The research is a phenomenon-and-diagnostic paper about
**region-conditional task affinity** in molecular MTL, evaluated across five
phases (pilot → phenomenon → predictor → utility → robustness).

The codebase must support the following non-negotiable properties:

1. **Reproducibility**: every result tied to a config + git hash + seed.
2. **Pre-committed hyperparameters**: the values in `plan.txt` Section 2.21 are
   locked; changing them is p-hacking. The config system enforces this.
3. **Phase-gated compute**: pilot must be cheap and conclusive; later phases
   reuse pilot artifacts where possible.
4. **Honest baselines**: MMoE / Cross-Stitch implementations must be defensible.
   Bad baselines invalidate the utility claim.
5. **Hierarchical bootstrap everywhere**: scalar quantities without CIs are
   useless for this paper.

---

## 1. Top-level repository layout

```
transfermtl/
├── plan.txt                       # research plan (canonical)
├── codebase_plan.md               # this document
├── README.md                      # quickstart, written after Phase 0
├── pyproject.toml                 # package metadata, deps, ruff/black config
├── environment.yml                # conda env (rdkit + pyg + cuda)
├── .pre-commit-config.yaml        # black, ruff, mypy strict on src/
│
├── configs/                       # YAML configs, one per experiment
│   ├── _shared/
│   │   ├── encoder_gcn.yaml       # frozen ICLR-paper encoder
│   │   ├── train_default.yaml     # frozen optimizer/schedule
│   │   └── bootstrap.yaml         # frozen B=1000, B'=200
│   ├── pilot/
│   │   ├── tox21.yaml
│   │   └── sider.yaml
│   ├── phase1/                    # phenomenon characterization
│   ├── phase2/                    # predictor evaluation
│   ├── phase3/                    # utility / sharing
│   └── phase4/                    # robustness sweeps
│
├── src/transfermtl/               # importable package
│   ├── __init__.py
│   ├── data/                      # Section 2.2
│   ├── partition/                 # Section 2.6
│   ├── models/                    # Section 2.3 + ablation archs
│   ├── training/                  # Section 2.4, 2.5
│   ├── gradients/                 # Section 2.7
│   ├── benefits/                  # Section 2.8
│   ├── indices/                   # Section 2.9 (S, H, C)
│   ├── bootstrap/                 # Section 2.10
│   ├── validity/                  # Section 2.11, 2.12
│   ├── null/                      # Section 2.13
│   ├── predictor/                 # Section 2.14, Phase 2
│   ├── architecture/              # Section 2.15 — local-affinity sharing
│   ├── baselines/                 # Section 2.16
│   ├── eval/                      # Section 2.17, utility metrics
│   ├── analysis/                  # Section 2.20, 6.6, 7.6, 8.8 outputs
│   └── utils/                     # logging, seeding, io, registry
│
├── scripts/                       # entry points; no logic, just glue
│   ├── prepare_dataset.py
│   ├── train_stl.py
│   ├── train_pairwise_mtl.py
│   ├── train_all_task_mtl.py
│   ├── compute_partitions.py
│   ├── compute_gradients.py
│   ├── compute_benefits.py
│   ├── compute_indices.py
│   ├── run_bootstrap.py
│   ├── run_null_distribution.py
│   ├── eval_predictors.py
│   ├── train_local_affinity_sharing.py
│   ├── eval_utility.py
│   ├── make_figures.py
│   └── pilot_decision.py          # writes the go/no-go document
│
├── tests/                         # pytest, mirrors src/ layout
│   ├── conftest.py                # tiny synthetic 2-task dataset fixture
│   ├── test_data_pipeline.py
│   ├── test_partition.py
│   ├── test_gradients.py
│   ├── test_bootstrap.py
│   ├── test_validity.py
│   ├── test_indices.py
│   ├── test_predictor.py
│   ├── test_architecture.py       # critical: verifies gating spec
│   └── test_baselines.py
│
├── notebooks/                     # exploratory only; results in scripts/
│
├── outputs/                       # all artifacts (gitignored except manifests)
│   ├── splits/
│   ├── partitions/
│   ├── checkpoints/
│   ├── predictions/
│   ├── gradients/
│   ├── benefits/
│   ├── analysis/
│   └── figures/
│
└── logs/                          # W&B local cache, slurm logs
```

Two conventions:

- **Anything under `src/`** is library code with type hints + tests.
- **Anything under `scripts/`** is a thin CLI that loads a YAML config, calls
  library functions, and writes parquet/npz to `outputs/`. Scripts are themselves
  not unit-tested; they are smoke-tested via the synthetic fixture.

---

## 2. Module-by-module specification

The numbering matches `plan.txt` Section 2 so cross-referencing is direct.

### 2.1 `data/` — preprocessing pipeline (plan §2.2)

Files:

- `data/datasets.py` — registry: `tox21`, `sider`, `toxcast`, `qm9`,
  `tdc_admet`, `chembl_kinase`, `tox21_adme_matched`, `synthetic_matched_pair`.
  Each entry has a loader returning `pd.DataFrame[smiles, labels..., source_id]`.
- `data/standardize.py` — RDKit `MolStandardize.Cleanup` +
  `LargestFragmentChooser`, canonical SMILES.
- `data/scaffolds.py` — `MurckoScaffold.MurckoScaffoldSmiles` with
  `includeChirality=False`; empty-scaffold bucketing.
- `data/fingerprints.py` — Morgan radius=2, 2048 bits; cached on disk by
  scaffold hash.
- `data/featurize.py` — atom (74-d) + bond (12-d) features; PyG `Data` objects
  cached via `torch.save` keyed by SMILES hash.
- `data/splits.py` — scaffold-stratified 70/15/15 split, splitting seed = 42,
  shared global scaffold→bin assignment for sparse multi-task datasets.
- `data/manifest.py` — writes `outputs/splits/{dataset}/split.parquet` with
  columns `[smiles, scaffold, split, task_labels...]`. Idempotent: rerunning
  must produce identical output (asserted in test).

The pipeline is a single function `prepare_dataset(name, force=False)` invoked
by `scripts/prepare_dataset.py`. It is the **only** code path that produces
splits; downstream code reads from the parquet.

### 2.3 `models/` — encoders and heads (plan §2.3, §9.3)

- `models/gcn.py` — primary 3-layer GCN, hidden=256, ReLU, dropout=0.1, global
  mean pool. Implemented with `torch_geometric.nn.GCNConv`.
- `models/gat.py` — GAT variant for Phase 4 architecture robustness.
- `models/chemberta.py` — wraps HuggingFace ChemBERTa for Phase 4 (gated by
  compute availability).
- `models/ecfp_mlp.py` — fixed-feature baseline for Phase 4.
- `models/heads.py` — `[256, 128, 1]` MLP head with dropout 0.1.
- `models/registry.py` — `build_encoder(name, **kwargs)` factory.

All encoders expose `encode(batch) -> Tensor[B, 256]`. Heads consume that
representation and produce per-task logits/values.

### 2.4-2.5 `training/` — STL and MTL training (plan §2.4, §2.5)

- `training/loops.py` — generic train loop with early stopping, gradient
  clipping (max_norm 1.0), cosine LR schedule, AdamW.
- `training/stl.py` — `train_stl(dataset, task, seed)` → checkpoint at
  `outputs/checkpoints/{dataset}/stl/{task}/seed{s}.pt`.
- `training/pairwise_mtl.py` — `train_pairwise_mtl(dataset, task_i, task_j,
  seed)`. Batch construction matches plan §2.5: compounds with valid label for
  either task contribute via masked loss.
- `training/all_task_mtl.py` — used by §2.6.2 latent partitioning and by
  baselines.
- `training/checkpoint.py` — checkpoint format: `{model_state, optim_state,
  epoch, val_loss, config_dict, seed, git_sha}`.

Predictions on the test split are saved alongside the checkpoint as
`predictions/{dataset}/{stl|mtl}/.../seed{s}.parquet`. Downstream metric and
gradient computation reads from these files; **no module re-runs training to
compute downstream metrics.**

### 2.6 `partition/` — region partitioning (plan §2.6)

- `partition/scaffold.py` — hierarchical agglomerative clustering on
  Tanimoto distance, average linkage; M ∈ {3, 5, 8, 10} (default 5). Validates
  region size against `2 * n_min` and merges undersized regions with nearest
  neighbor.
- `partition/latent.py` — k-means on encoder representations from a single
  all-task MTL pretraining run (seed=0). M ∈ {3, 5, 8, 10}.
- `partition/knn.py` — assign by nearest scaffold-cluster centroid in encoder
  space.
- `partition/random_null.py` — generates B=200 random partitions matched to
  the scaffold partition's region-size distribution.
- `partition/io.py` — partitioning files at
  `outputs/partitions/{dataset}/{scheme}_M{M}.parquet` with columns
  `[smiles, region_id]`.

A partition is a deterministic function from a SMILES list to integer
region IDs. The partition file is the contract; everything downstream reads it.

### 2.7 `gradients/` — regional gradient affinity (plan §2.7)

- `gradients/extract.py` — given a checkpoint and a region, computes
  `g_i(r) = mean over D_i(r) of ∇_θ ℓ_i(x; θ, φ_i)`. Subsamples to 500 if region
  is larger.
  - Uses `torch.autograd.grad(loss, encoder.parameters(), retain_graph=True)`,
    flattens, concatenates.
  - Returns `(g_vec, n_used, grad_norm)`.
- `gradients/affinity.py` — `G_ij(r) = cos(g_i(r), g_j(r))`. Returns NaN if
  either norm < 1e-8. Also computes the unnormalized dot product (cached for
  ablations).
- `gradients/trajectory.py` — runs extraction at three checkpoints (final,
  80%, 60%) to report stability.
- `gradients/io.py` — writes
  `outputs/gradients/{dataset}/{task_i}_{task_j}/seed{s}/region_affinity.parquet`
  with `[region_id, G_ij, g_i_norm, g_j_norm, n_i_in_region, n_j_in_region,
  checkpoint_label]`.

**Test requirement:** `test_gradients.py` validates the identity check
`G_ii(r) ≈ 1` to within numerical tolerance and that label-shuffling drives
`G_ij(r)` toward zero (this is the §9.6 negative control wired as a unit test).

### 2.8 `benefits/` — regional transfer benefit (plan §2.8)

- `benefits/perf.py` — region-restricted ROC-AUC for classification, negative
  RMSE for regression. Returns NaN if region fails the §2.11 validity test
  (n<30 classification / n<50 regression / single-class).
- `benefits/delta.py` — computes `Δ_ij(r)`, `Δ_{i←j}(r)`, `Δ_{j←i}(r)`,
  `Δ_ij^worst(r)` from saved STL/MTL predictions.
- `benefits/aggregate.py` — averages across seeds; final per-region table at
  `outputs/benefits/{dataset}/{task_i}_{task_j}/region_benefits.parquet`.

This module never trains; it consumes prediction parquets and partition files.

### 2.9 `indices/` — sign heterogeneity, H, C (plan §2.9)

- `indices/sign_heterogeneity.py` — pair-averaged `S_ij` (binary) and
  task-specific `S_ij^i`, `S_ij^j`. Default ε = 1.5 AUC / 0.10 RMSE. Requires
  CI inputs (from §2.10) so the function signature is
  `compute_S(deltas: dict[r, BootstrapResult]) -> bool`.
- `indices/heterogeneity_index.py` — test-set-weighted `H_ij`.
- `indices/cancellation.py` — `C_ij` with η = 0.5.
- `indices/io.py` — final pair-level table at
  `outputs/analysis/{dataset}/pair_indices.parquet`.

### 2.10 `bootstrap/` — hierarchical bootstrap (plan §2.10)

This module is load-bearing. CIs are required for sign heterogeneity, validity
filter, and every predictor metric.

- `bootstrap/hierarchical.py` — exposes
  ```python
  def hierarchical_bootstrap(
      compute_fn: Callable,
      data: HierarchicalSamples,        # carries scaffold and seed structure
      n_iter: int = 1000,
      level1: Literal["scaffold", "cluster"] = "scaffold",
      seeds: list[int] | None = None,
  ) -> BootstrapResult                  # estimate, ci_lower, ci_upper, samples
  ```
- `bootstrap/within_region.py` — within-region scaffold-level resampling (used
  for `G_ij(r)` and `Δ_ij(r)` CIs since cluster-level resampling does not apply
  inside a single region).
- `bootstrap/seed_mixing.py` — for two-level (data + seed) CIs, randomly draws
  one seed per bootstrap iteration.
- `bootstrap/calibration.py` — pre-pilot calibration check from plan §2.10:
  asserts `G_ii(r)` CI width < 0.05 and Shapiro-Wilk normality on ≥80% of
  statistics. Run before the main pipeline.

Implementation note: `BootstrapResult` is a dataclass; serializing the full
`samples` array is optional and only kept when the user explicitly enables
`save_samples=True` (memory-heavy at scale).

### 2.11-2.12 `validity/` — local-support and meaningful pairs (plan §2.11, §2.12)

- `validity/local_support.py` — five-condition validity check on a single
  `(task_pair, region)`. Returns a structured `ValidityFlag(valid: bool,
  failed_reasons: list[str])`.
- `validity/meaningful_pair.py` — four-condition meaningful-pair filter
  (≥3 valid regions, ≥1 region with `|Δ| > 1.5` and CI excluding 0, etc.).
- `validity/io.py` — writes `outputs/analysis/{dataset}/meaningful_pairs.parquet`.
  The denominator for prevalence statistics is locked in this file.

### 2.13 `null/` — random-partition null distributions (plan §2.13)

- `null/run_null.py` — for each random partition (B=200), runs the same
  pipeline (gradient affinity → benefits → indices) and aggregates the
  scaffold-level statistic.
- `null/pvalue.py` — empirical p-value `(1 + #{S^(b) ≥ S^scaffold}) / (1 + B)`.
- `null/io.py` — writes per-statistic null arrays at
  `outputs/analysis/{dataset}/null_dist_{statistic}_M{M}.npy`.

### 2.14 `predictor/` — predictor evaluation (plan §2.14, §7)

- `predictor/features.py` — computes per-(pair, region) features:
  `G_ij(r)`, scaffold similarity (mean Tanimoto between region scaffold sets),
  embedding distance (centroid L2), label correlation (Pearson on co-measured
  compounds), regional MMD (RBF kernel; bandwidth via median heuristic),
  global gradient affinity (constant within pair), local support sizes,
  TAG-style local affinity.
- `predictor/methods.py` — single-feature threshold predictors and joint
  logistic-regression / linear-regression predictors. Targets:
  `sign(Δ_ij(r))` (binary, with neutral exclusion at `|Δ| < 0.5`),
  `Δ_ij(r)` (continuous), three-class, task-specific.
- `predictor/cv.py` — 5-fold leave-task-pairs-out cross-validation. Holds out
  20% of pairs (with all their regions) per fold so regions of the same pair
  do not leak across folds.
- `predictor/incremental_r2.py` — **centerpiece**. Two regressions:
  - Baseline: `Δ ~ ScaffoldSim + LabelCorr + EmbeddingDist + log|D_i| + log|D_j|`
  - Full:    baseline + `G_ij(r) + G_ij(r)^2`

  Reports `R²_full − R²_baseline` with hierarchical-bootstrap CI and partial
  F-test. This is the function whose number decides Phase 2.
- `predictor/calibration.py` — ECE with 10 bins, reliability diagrams, Brier.
- `predictor/cross_dataset.py` — train predictor on dataset A, evaluate on B
  (Tox21→SIDER, SIDER→ToxCast).
- `predictor/eval.py` — top-level entry that runs all predictors and writes
  the comparison table.

### 2.15 `architecture/` — local-affinity sharing (plan §2.15)

This is the highest-engineering-risk module. The plan calls out a "simplified
implementation" (region-conditional masking with an isolated-head trick) —
build that first; only attempt the full per-task gradient-buffer version if
the simple variant is publishable but underperforms.

- `architecture/local_affinity_sharing.py` — region-conditional gated MTL
  model. Key components:
  - Shared encoder + per-task heads + per-task **isolated** heads
    `φ_k^iso`. The isolated head receives gradients but does not propagate them
    to the encoder (achieved via `encoder.detach()` on the isolated path).
  - `Gates`: tensor of shape `[K, K, M]` with `γ_ij(r) ∈ {0,1}` for hard
    sharing or `[0,1]` for soft. Computed from pre-trained `G_ij(r)` via
    `σ(α · G_ij(r))` (default α = 5; hard at threshold 0).
  - Forward pass standard. Backward pass routes loss through shared vs.
    isolated path according to `γ_ij(π(x))`.
- `architecture/pretrain_gates.py` — runs 5 epochs of plain pairwise MTL with
  `γ = 1` everywhere, computes `G_ij(r)`, sets gates, then resumes training.
- `architecture/abstention.py` — wraps a trained sharing model. For regions
  failing §2.11 validity, falls back to the STL prediction.

**Critical test (`tests/test_architecture.py`):** verifies that for a synthetic
2-task problem with `γ_12(r=0) = 1` and `γ_12(r=1) = 0`, encoder gradients
from task 2 on region-1 inputs are exactly zero. This is the contract.

### 2.16 `baselines/` — competing methods (plan §2.16)

Each file is independently testable.

- `baselines/stl.py` — wraps `training/stl.py`.
- `baselines/all_task_mtl.py`.
- `baselines/pairwise_union.py` — ensembles pairwise MTL predictions per task
  by logit-averaging across pairs containing the task.
- `baselines/random_grouping.py` — 10 random group assignments, train all-task
  MTL within each group, average results.
- `baselines/global_affinity_grouping.py` — implements ICLR-paper grouping:
  `G_ij^global` from a pretrained all-task MTL, hierarchical clustering on
  `1 − G_ij^global`, dendrogram cut.
- `baselines/tag_grouping.py` — Fifty et al. update-affinity:
  `a_{i→j} = Perf_j(θ + step_i) − Perf_j(θ)`.
- `baselines/mmoe.py` — Ma et al. 2018: 4 experts (each 2-layer MLP, hidden
  256), per-task softmax gates over experts. Joint training.
- `baselines/cross_stitch.py` — Misra et al. 2016: per-task encoder columns
  + 2×2 cross-stitch matrices between layers. Default adaptive baseline.
- `baselines/dynashare.py` — optional second adaptive baseline. Decision made
  by Phase 3 start (engineering assessment); see §13.1 below.

Per-baseline unit test: trains for 3 epochs on a 2-task synthetic problem,
asserts validation loss decreases, asserts predictions are non-trivial
(std > 0.01 across compounds).

### 2.17 `eval/` — utility metrics (plan §2.17, §8.5)

- `eval/utility_metrics.py` — average AUC/RMSE, worst-region per-task,
  sample-weighted average.
- `eval/negative_transfer.py` — NT-rate at thresholds 1.5 (default) and 3.0
  (severe).
- `eval/calibration_utility.py` — share/separate decision calibration; ECE,
  reliability diagrams, Brier.
- `eval/holdout_calibration.py` — 5-fold leave-task-pairs-out for calibration.
- `eval/statistical_compare.py` — paired Wilcoxon signed-rank with Bonferroni
  correction over the comparison family.
- `eval/efficiency.py` — GPU-hours, inference latency, parameter count.

### 2.20 `analysis/` — global decomposition and figures (plan §2.20, §11)

- `analysis/decomposition.py` — `G_ij^global` vs naive linear `Σ p(r) G_ij(r)`
  vs cancellation gap.
- `analysis/figures.py` — produces every figure in plan §11 from saved
  parquets/npy (no model runs). One function per figure number.
- `analysis/case_studies.py` — selects 5–10 canonical sign-heterogeneous pairs
  for Figure 2 and the case-study appendix.

### `utils/`

- `utils/seeding.py` — sets PyTorch, NumPy, CUDA, cuDNN deterministic mode
  from a single seed.
- `utils/logging.py` — W&B integration with project tag `tsh-mtl`.
- `utils/io.py` — parquet/npz read/write helpers; manifest enforcement.
- `utils/git.py` — captures git SHA + dirty-tree warning at run start.
- `utils/registry.py` — string-keyed factories for datasets, encoders,
  baselines, predictors.
- `utils/types.py` — typed dataclasses: `BootstrapResult`, `RegionStats`,
  `ValidityFlag`, `PairIndices`.

---

## 3. Data formats and on-disk contracts

Every artifact is parquet (tabular) or npz (arrays). No pickle in the artifact
tree — pickle is reserved for checkpoints (where torch dictates it). All
column schemas are typed via `pandera` schemas under `src/transfermtl/utils/schemas.py`.

| Path | Schema | Producer | Consumers |
|------|--------|----------|-----------|
| `outputs/splits/{ds}/split.parquet` | `[smiles, scaffold, split, t1..tK]` | `prepare_dataset.py` | all |
| `outputs/partitions/{ds}/{scheme}_M{M}.parquet` | `[smiles, region_id]` | `compute_partitions.py` | gradients, benefits, indices, predictor |
| `outputs/checkpoints/{ds}/{stl\|mtl}/.../seed{s}.pt` | torch dict | training scripts | gradients, predictor, eval |
| `outputs/predictions/{ds}/.../seed{s}.parquet` | `[smiles, task, y_true, y_pred]` | training scripts | benefits |
| `outputs/gradients/.../region_affinity.parquet` | `[region_id, G_ij, g_i_norm, g_j_norm, n_i, n_j, ckpt]` | `compute_gradients.py` | predictor, indices |
| `outputs/benefits/.../region_benefits.parquet` | `[region_id, Δ_pair, Δ_i_from_j, Δ_j_from_i, Δ_worst, ci_lo, ci_hi, n_test]` | `compute_benefits.py` | indices, predictor |
| `outputs/analysis/{ds}/pair_indices.parquet` | `[pair_id, S_ij, S_i, S_j, H_ij, C_ij, n_valid_regions]` | `compute_indices.py` | analysis, figures |
| `outputs/analysis/{ds}/meaningful_pairs.parquet` | `[pair_id, is_meaningful, reasons]` | `validity/meaningful_pair.py` | prevalence stats |
| `outputs/analysis/{ds}/null_dist_*.npy` | float[B] | `null/run_null.py` | analysis, figures |

Idempotency: every script implements a `--force` flag; absent `--force` it
skips work whose output already exists at the expected path with a hash
matching the input.

---

## 4. Configuration system

YAML configs are layered with [Hydra](https://hydra.cc) (or OmegaConf if
Hydra is overkill). The frozen pre-committed parameters live in
`configs/_shared/`. Phase configs `extends:` the shared base.

Two rules:

1. **Locked params**: any key in `_shared/` is immutable. CI checks the file
   hashes before every run; mismatch fails the job. The corresponding values
   are listed in plan §2.21.
2. **Override surface**: only seeds, dataset name, task pair, and partition
   (M and scheme) are exposed on the CLI. Everything else must be a config
   change with a code review.

A typical pilot launch:

```
python scripts/train_pairwise_mtl.py \
    --config configs/pilot/tox21.yaml \
    --task-i NR-AR --task-j NR-AR-LBD --seed 0
```

---

## 5. Phase-by-phase implementation roadmap

The codebase is built bottom-up over six milestones. Each milestone is
gated on tests passing.

### Milestone 0: Foundations (week 1)

Modules: `utils/`, `data/`, `models/`, `training/`, `bootstrap/`, `validity/`.

Deliverables:
- `prepare_dataset` works for Tox21 + SIDER, persists splits.
- STL training of one Tox21 task converges (smoke test).
- Hierarchical bootstrap calibration check passes (plan §2.10).
- All Milestone 0 modules have unit tests on the synthetic 2-task fixture.

### Milestone 1: Pilot pipeline (week 2-3)

Modules add: `partition/`, `gradients/`, `benefits/`, `indices/`, `null/`.

Deliverables:
- End-to-end Phase 0 run on Tox21 + SIDER (20-40 task pairs, 3 seeds, M=5).
- `pilot_decision.py` reads results and writes a go/no-go document at
  `outputs/analysis/pilot_decision.md`.
- Both green-light criteria (plan §5.7) evaluated and explicit pass/fail.

**Phase 0 is the gating decision.** Do not begin Milestone 2 until pilot
passes both criteria. If only phenomenon passes → workshop framing branch.

### Milestone 2: Predictor (week 4-5)

Modules add: `predictor/`, `analysis/decomposition.py`, `analysis/figures.py`.

Deliverables:
- All 10 predictors evaluated on Tox21 + SIDER + ToxCast.
- Incremental R² test on three datasets with bootstrap CIs (plan §2.14).
- Cross-dataset transfer: Tox21→SIDER, SIDER→ToxCast.
- Figures 4 and 5 produced.

**Mid-project gate (plan §13.2):** if incremental R² < 0.05, kill the NeurIPS
target and reframe as a workshop discovery paper.

### Milestone 3: Phenomenon expansion (week 4-6, parallel to M2)

Adds the remaining datasets (TDC ADMET, ChEMBL kinase panel, Tox21+ADME
matched, QM9, synthetic matched-pair). Adds latent and kNN partitionings.
Produces Figures 2 and 3, plus Phase 1 deliverables.

### Milestone 4: Utility (week 7-9)

Modules add: `architecture/`, `baselines/`, `eval/`.

Deliverables:
- Local-affinity sharing model trains stably; `test_architecture.py` passes.
- All 11 sharing strategies evaluated on Tox21, SIDER, ToxCast, TDC ADMET,
  kinase panel.
- Negative transfer rate, calibration, paired Wilcoxon comparison.
- Figure 7.

The DynaShare-vs-Cross-Stitch decision is locked at the start of this
milestone via an engineering assessment recorded in
`outputs/analysis/baseline_decision.md`. Default is Cross-Stitch; switch only
if a clean DynaShare implementation exists or can be built within a 2-day
budget without compromising fidelity.

### Milestone 5: Robustness and ablations (week 10-11)

Phase 4 sweeps:

- Partition robustness (scaffolds, latent k=3,5,8,10, kNN, random null).
- `n_min` sweep: {25, 50, 100, 150}.
- Architecture sweep: GCN, GAT, ECFP+MLP; ChemBERTa if compute allows.
- 5 seeds for final runs (replaces pilot's 3).
- Gradient-computation ablations (last-layer vs all-encoder, cosine vs dot,
  Fisher-normalized, trajectory-mean vs final).
- Negative controls: label permutation, random partition, identity check.

Produces Figure 8 and the appendix robustness tables.

### Milestone 6: Paper assembly (week 12)

`make_figures.py` regenerates every figure from saved artifacts.
`scripts/build_paper.sh` runs the figure script + LaTeX. The paper repo is a
sibling directory; this codebase exports figures at
`outputs/figures/{fig_number}.pdf` and a `tables/` directory with
`csv → latex` conversion via `pandas.to_latex(longtable=True)`.

---

## 6. Compute and parallelization

From plan §2.18, total budget is 400-600 GPU-hours on H100s. The codebase
parallelizes via SLURM array jobs.

- `scripts/launch_pilot.sh` — SLURM array dispatching all (task, seed) STL
  jobs and (pair, seed) MTL jobs. Each job is one training run.
- Bootstrap iterations are intra-job parallel via `multiprocessing.Pool` —
  most bootstrap cost is trivially parallelizable since each iteration is
  independent.
- Predictor evaluation is single-GPU.
- Architecture experiments (Phase 4) gate on Phase 0-3 completion to avoid
  burning compute on a path that may be killed.

Each SLURM job logs the git SHA, config hash, and seed. A nightly health-check
script reads `outputs/manifests/` and reports any orphaned or duplicated
artifacts.

---

## 7. Testing strategy

The synthetic 2-task fixture in `tests/conftest.py` is a tiny graph dataset
with known ground-truth: tasks 1 and 2 align on region A and oppose on
region B. Every module is tested against this fixture, plus:

- **Property tests** (Hypothesis): bootstrap CIs are monotone in B; partition
  outputs are deterministic given a seed; validity flags correspond to their
  reasons.
- **Regression tests**: pilot decision document on a frozen synthetic
  dataset must match a checked-in expected output.
- **Architecture contract test**: gating-spec compliance for the local-affinity
  sharing model.
- **Negative-control test**: shuffled labels drive `G_ij(r)` toward zero
  within tolerance.

CI runs the full test suite + a 30-second pilot smoke-test on the synthetic
fixture for every PR.

---

## 8. Dependencies

Pinned in `environment.yml`:

- Python 3.11
- `torch >= 2.2`, `torch_geometric`, CUDA 12.x
- `rdkit-pypi`
- `pandas`, `numpy`, `scipy`, `scikit-learn`
- `pyarrow` (parquet)
- `pandera` (schema enforcement)
- `hydra-core` or `omegaconf`
- `wandb`
- `pytest`, `hypothesis`, `pytest-cov`
- `transformers` (only if ChemBERTa is in scope)
- `tdc` (TDC ADMET datasets)
- Plotting: `matplotlib`, `seaborn` for diagnostics; final figures via a
  `make_figures.py` that uses matplotlib only (publication-stable).

A second env file `environment-dev.yml` adds `black`, `ruff`, `mypy`,
`pre-commit`.

---

## 9. Reproducibility checklist

For every result reported in the paper, the following must be true (and
asserted by `scripts/audit_results.py`):

1. The producing script exists at `scripts/<name>.py` and is referenced by
   the figure caption / appendix.
2. The config used is checked into `configs/`.
3. The git SHA is captured in the artifact metadata.
4. The seed list (3 for pilot, 5 for final) is logged.
5. Hierarchical-bootstrap CIs are present for every scalar.
6. Random-partition null is reported wherever the phenomenon claim is made.

Re-running `scripts/reproduce.sh <phase>` from a fresh checkout regenerates
all artifacts for that phase deterministically.

---

## 10. Open engineering decisions (deferred to first PRs)

These are the choices that should be made in code review rather than in this
plan. None of them are scientifically pre-committed.

- **Hydra vs OmegaConf vs simple YAML loader.** Lean Hydra for composition,
  but Hydra's job sweep machinery is overkill if SLURM does the dispatching.
- **Parquet engine: pyarrow vs fastparquet.** pyarrow by default; switch only
  if a benchmark proves fastparquet is materially faster on the artifact mix.
- **W&B vs MLflow.** Plan §2.19 specifies W&B with project tag `tsh-mtl`;
  keep it unless the lab moves off W&B.
- **DynaShare implementation source.** Either re-implement faithfully from the
  paper or skip in favor of Cross-Stitch. Decision recorded in
  `baseline_decision.md` at start of Milestone 4 (see §5).
- **Encoder gradient flatten order.** `parameters()` order is implementation-
  defined; lock the order and assert it in a unit test so bootstrap iterations
  are bitwise-identical across reruns.

---

## 11. What this codebase plan does NOT cover

- Paper LaTeX source — lives in a separate repo.
- Slide decks for talks.
- Compute provisioning (SLURM cluster setup).
- Manual curation of the canonical case-study task pairs (selected by hand
  from `analysis/case_studies.py` candidate output).
- Synthetic matched-pair dataset generation — designed once, generated by a
  one-off script under `scripts/generate_synthetic_dataset.py` and committed
  as a frozen parquet.

These are out of scope for the implementation effort but are flagged here so
nothing falls through the cracks at submission time.
