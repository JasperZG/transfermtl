# Wave 2 — Independent building blocks

## Synchronization
- **4 agents in parallel: A2, A3, A4, A5**
- All depend on A1's contracts (schemas, types, configs, fixture)
- No agent depends on another's in-flight work, with one caveat:
  - A5's **latent** partitioning needs A3's all-task MTL checkpoint.
    A5 stubs the latent loader against a synthetic checkpoint; the real
    integration test waits for A3 to land.

## Goals
- A2 produces real splits for Tox21 + SIDER
- A3 produces converging STL + pairwise MTL training on Tox21
- A4 produces a calibrated hierarchical bootstrap that passes the §2.10
  pre-pilot calibration check
- A5 produces scaffold + kNN + random partitions; latent stub awaiting
  A3's checkpoint

## Cross-agent guardrails
- No agent edits `utils/schemas.py`, `utils/types.py`, or
  `configs/_shared/*` (locked by A1)
- All agents add tests using A1's synthetic fixture, not real datasets
- All agents merge against `main` only via PR; conflicts resolve in
  PR review, not silently
- If an agent discovers a missing schema field or type, file a
  coordination issue; do not silently extend A1's contracts

---

## Agent A2 — Data pipeline

### Mission
Build the preprocessing pipeline (plan §2.2) for Tox21 and SIDER. Produce
deterministic, schema-valid `splits/{dataset}/split.parquet` files that
every downstream module reads from. Cache featurized PyG `Data` objects
to disk so A3 doesn't re-featurize on every training run.

### Owned files

`src/transfermtl/data/`:
- `datasets.py` — registry of dataset loaders. Tox21 + SIDER only in
  this wave; ToxCast/QM9/TDC ADMET/kinase/synthetic added by A9.
  ```python
  @register("dataset", "tox21")
  def load_tox21() -> pd.DataFrame: ...
      # columns: smiles, NR-AR, NR-AR-LBD, ..., SR-p53 (12 task cols)

  @register("dataset", "sider")
  def load_sider() -> pd.DataFrame: ...
      # columns: smiles, 27 task columns
  ```
  Each loader downloads from MoleculeNet's canonical CSV URL (or uses a
  local cached copy under `data/raw/{dataset}/`) and returns a frame.
- `standardize.py`
  - `standardize_smiles(smi: str) -> str | None` — RDKit
    `MolStandardize.Cleanup` + `LargestFragmentChooser`, then canonical
    SMILES via `Chem.MolToSmiles`. Returns `None` on RDKit parse failure.
- `scaffolds.py`
  - `compute_scaffold(smi: str, include_chirality: bool = False) -> str`
    — `MurckoScaffold.MurckoScaffoldSmiles`. Empty scaffolds (acyclic
    molecules) bucket under the literal string `"<EMPTY>"`.
- `fingerprints.py`
  - `morgan_fingerprint(smi: str, radius: int = 2, n_bits: int = 2048)
    -> np.ndarray[np.uint8]` — bit vector via
    `AllChem.GetMorganFingerprintAsBitVect`.
  - `cache_scaffold_fingerprints(scaffolds: list[str]) -> dict[str,
    np.ndarray]` — disk-cached at
    `outputs/cache/scaffold_fps/{dataset}.parquet`.
- `featurize.py`
  - 74-dim atom features (atomic number one-hot, degree, formal charge,
    chirality, hybridization, aromaticity, hydrogens, etc.) and 12-dim
    bond features (bond type, conjugation, ring membership, stereo).
  - Uses the standard MoleculeNet GNN featurization; reference
    implementation: `chemprop` or DGL-LifeSci.
  - Returns `torch_geometric.data.Data(x, edge_index, edge_attr, y, smi)`.
  - Cached to disk at `outputs/cache/featurized/{smi_hash}.pt` (sha256
    of canonical SMILES → 16-char prefix).
- `splits.py`
  - `scaffold_stratified_split(df, train=0.70, val=0.15, test=0.15,
    seed=42) -> pd.Series[Literal["train","val","test"]]`
  - Algorithm (plan §2.2 step 6):
    1. Group molecules by scaffold (already computed)
    2. Sort scaffold groups by size, descending
    3. Greedily assign whole scaffold groups to train/val/test bins,
       always picking the bin furthest below its target fraction
  - For sparse multi-task datasets (added later by A9): splits computed
    per-task but use a shared global scaffold→bin assignment so no
    scaffold appears in train for one task and test for another.
- `manifest.py`
  - `write_manifest(dataset: str, df: pd.DataFrame, splits: pd.Series)
    -> Path` — writes parquet validated against `SplitSchema` and emits
    a JSON sidecar with: input file hash(es), RDKit version, pandas
    version, row count, scaffold count, per-split counts, git SHA.

### Scripts
- `scripts/prepare_dataset.py`:
  ```
  python scripts/prepare_dataset.py --dataset tox21 [--force]
  ```
  - Loads, standardizes, computes scaffolds, computes Morgan FPs,
    featurizes (cached), splits, writes manifest.
  - Idempotent: skips if output exists with matching input hash, unless
    `--force`.

### Reads from
- A1: `utils/schemas.py::SplitSchema`, `utils/registry.py`,
  `utils/io.py`, `configs/_shared/preprocess.yaml`
- Raw data from MoleculeNet (Tox21, SIDER CSVs)

### Writes to
- `outputs/splits/tox21/split.parquet` (validated via `SplitSchema`)
- `outputs/splits/sider/split.parquet`
- `outputs/cache/featurized/*.pt` (PyG Data objects, ~tens of thousands
  of files)
- `outputs/cache/scaffold_fps/{dataset}.parquet`
- `outputs/data_manifest/{dataset}.json`

### Tests
- `tests/test_data_pipeline.py`:
  - `test_standardize_idempotent` — `standardize(standardize(s)) ==
    standardize(s)` for a list of canonical SMILES
  - `test_standardize_handles_salts` — `"CCO.[Na+].[Cl-]"` → `"CCO"`
  - `test_standardize_returns_none_on_garbage` — `"not-a-smiles"` →
    `None`
  - `test_scaffold_empty_bucket` — methane, ethane → `"<EMPTY>"`
  - `test_scaffold_canonical` — same molecule, different SMILES form →
    same scaffold
  - `test_fingerprint_shape` — Morgan FP is `(2048,)` uint8
  - `test_split_no_scaffold_leakage` — no scaffold value appears in
    train AND val, train AND test, or val AND test
  - `test_split_fractions_within_tolerance` — actual fractions within
    ±2% of (0.70, 0.15, 0.15)
  - `test_split_deterministic` — same seed → identical row assignment
  - `test_prepare_dataset_idempotent` — running twice produces
    byte-identical `split.parquet`
  - `test_split_schema_validates` — saved parquet passes `SplitSchema`
  - `test_featurize_synthetic_roundtrip` — featurize a synthetic
    molecule, recover atom count via `data.x.shape[0]`

### Acceptance criteria

- [ ] `python scripts/prepare_dataset.py --dataset tox21` succeeds and
  produces a schema-valid parquet
- [ ] Same on SIDER
- [ ] Tox21 split has approximately 5,476 train / 1,174 val / 1,174 test
  (±5%); SIDER has ~995 / ~213 / ~214 (±5%)
- [ ] Re-running `prepare_dataset.py` is a no-op (idempotent)
- [ ] Manifest JSON includes RDKit version, raw-CSV sha256, git SHA
- [ ] Featurization cache populated; second run skips re-featurization
  (verified by timing)
- [ ] All tests pass
- [ ] `outputs/manifests/wave2_a2_complete.json` enumerates outputs

### Out of scope for A2
- Datasets beyond Tox21 + SIDER — A9
- Synthetic dataset generation — A9 / A1 fixture
- Region partitioning — A5
- Any model training — A3

### References
- `codebase_plan.md` §2.1
- `plan.txt` §2.2 (line ~2803), §2.21

---

## Agent A3 — Models and training

### Mission
Build the GCN encoder, MLP heads, and STL / pairwise-MTL / all-task-MTL
training loops. Save checkpoints + per-test-compound prediction parquets
that downstream measurement code consumes (without ever re-training).

### Owned files

`src/transfermtl/models/`:
- `gcn.py` — primary 3-layer GCN (plan §2.3)
  ```python
  class GCNEncoder(nn.Module):
      def __init__(self, hidden_dim=256, n_layers=3, dropout=0.1,
                   atom_feature_dim=74): ...
      def encode(self, data: PyGBatch) -> Tensor: ...  # [B, hidden_dim]
  ```
  Uses `torch_geometric.nn.GCNConv`, ReLU activations, dropout=0.1
  between layers, global mean pool.
- `heads.py`
  ```python
  class TaskHead(nn.Module):
      def __init__(self, in_dim=256, hidden_dim=128, dropout=0.1): ...
      def forward(self, h: Tensor) -> Tensor: ...  # [B, 1] (logits)
  ```
- `registry.py` — `@register("encoder", "gcn")` factory.
  A12 will register `gat`, `chemberta`, `ecfp_mlp`.

`src/transfermtl/training/`:
- `loops.py` — generic train loop:
  - AdamW, cosine LR schedule (1e-3 → 1e-5 over max_epochs)
  - Early stopping on validation loss (patience 25)
  - Gradient clipping max_norm=1.0
  - Returns `TrainOutcome(best_val_loss, best_epoch, final_model_state)`
- `stl.py`
  ```python
  def train_stl(dataset: str, task: str, seed: int,
                cfg: TrainConfig) -> Path: ...
      # returns checkpoint path
  ```
- `pairwise_mtl.py`
  ```python
  def train_pairwise_mtl(dataset, task_i, task_j, seed, cfg) -> Path
  ```
  Batch construction (plan §2.5):
  - Each batch contains compounds with valid label for at least one of
    `{i, j}`
  - Compounds with both labels contribute to both task losses
  - Loss: `mean(loss_i over valid_i) + mean(loss_j over valid_j)`
- `all_task_mtl.py`
  ```python
  def train_all_task_mtl(dataset, seed, cfg) -> Path
  ```
  Used by A5 latent partitioning and by A11 baselines.
- `checkpoint.py`
  - `save_checkpoint(path, model, optim, epoch, val_loss, cfg, seed)`
  - `load_checkpoint(path) -> CheckpointBundle`
  - Checkpoint dict keys: `model_state, optim_state, epoch, val_loss,
    cfg_dict, seed, git_sha, timestamp`
- `predict.py`
  - `save_predictions(model, dataset, split, path)` — writes
    `PredictionSchema`-valid parquet
  - For STL: one task per file
  - For MTL: one file per (pair, seed), with both task predictions
    stacked (one row per compound × task)

### Scripts
- `scripts/train_stl.py`:
  ```
  python scripts/train_stl.py --dataset tox21 --task NR-AR --seed 0
  ```
- `scripts/train_pairwise_mtl.py`:
  ```
  python scripts/train_pairwise_mtl.py --dataset tox21
      --task-i NR-AR --task-j NR-AR-LBD --seed 0
  ```
- `scripts/train_all_task_mtl.py`:
  ```
  python scripts/train_all_task_mtl.py --dataset tox21 --seed 0
  ```

### Reads from
- A1: `utils/types.py`, `utils/schemas.py`, `configs/_shared/`,
  `utils/seeding.py`, `utils/logging.py`
- A2: `outputs/splits/{dataset}/split.parquet`,
  `outputs/cache/featurized/*.pt`

### Writes to
- `outputs/checkpoints/{dataset}/stl/{task}/seed{s}.pt`
- `outputs/checkpoints/{dataset}/mtl/{task_i}_{task_j}/seed{s}.pt`
- `outputs/checkpoints/{dataset}/all_task/seed{s}.pt`
- `outputs/predictions/{dataset}/stl/{task}/seed{s}.parquet`
- `outputs/predictions/{dataset}/mtl/{task_i}_{task_j}/seed{s}.parquet`
- `outputs/predictions/{dataset}/all_task/seed{s}.parquet`

All prediction parquets validated against `PredictionSchema`.

### Tests
- `tests/test_models.py`:
  - `test_gcn_output_shape` — synthetic batch of 8 graphs → `[8, 256]`
  - `test_gcn_param_count_in_range` — params in [200K, 800K]
  - `test_head_output_shape` — head produces `[B, 1]`
  - `test_encoder_deterministic_with_seed`
- `tests/test_training.py`:
  - `test_stl_converges_on_fixture` — train on synthetic fixture for 20
    epochs; val AUC > 0.7 on held-out (the fixture is easy)
  - `test_mtl_converges_on_fixture` — pairwise MTL on synthetic fixture
  - `test_all_task_mtl_runs` — completes one epoch on fixture without
    error
  - `test_checkpoint_roundtrip` — save+load yields identical state dict
  - `test_predictions_match_schema` — saved parquet validates against
    `PredictionSchema`
  - `test_seeded_determinism` — same seed → identical val_loss across
    runs (within fp tolerance 1e-5)
  - `test_mtl_handles_missing_labels` — fixture with NaN labels in one
    task → loss masks correctly

### Acceptance criteria

- [ ] STL training on Tox21 NR-AR converges with val AUC > 0.75 (typical
  range 0.78-0.85)
- [ ] Pairwise MTL on (NR-AR, NR-AR-LBD) converges
- [ ] All-task MTL on Tox21 completes within ~20 GPU-min on H100
- [ ] Checkpoints round-trip
- [ ] All prediction parquets schema-valid
- [ ] W&B logging functional with project tag `tsh-mtl`
- [ ] Synthetic fixture STL run completes in <30s on CPU
- [ ] All tests pass
- [ ] One real Tox21 STL run completes in <5 GPU-min on H100

### Out of scope for A3
- Architecture variants (GAT, ChemBERTa, ECFP+MLP) — A12 / Wave 5
- Local-affinity sharing model — A10
- Baselines beyond STL + plain MTL — A11
- Data preprocessing — A2
- Partition-aware training — A10
- Gradient extraction (A6 reads from A3's checkpoints)

### References
- `codebase_plan.md` §2.3, §2.4, §2.5
- `plan.txt` §2.3 (line ~2828), §2.4 (line ~2847), §2.5 (line ~2868),
  §2.19 (reproducibility)

---

## Agent A4 — Statistical core

### Mission
Build the statistical infrastructure every downstream module depends on
for confidence intervals: hierarchical bootstrap, validity checks,
random-partition null distributions. Pure-math code with no ML training.
Treat this as a library; correctness here is load-bearing for every
result the paper reports.

### Owned files

`src/transfermtl/bootstrap/` (plan §2.10):
- `hierarchical.py`
  ```python
  def hierarchical_bootstrap(
      compute_fn: Callable[[HierarchicalSamples], float],
      data: HierarchicalSamples,
      n_iter: int = 1000,
      level1: Literal["scaffold", "cluster"] = "scaffold",
      seeds: Sequence[int] | None = None,
      save_samples: bool = False,
      rng_seed: int = 0,
  ) -> BootstrapResult: ...
  ```
  Two-level resampling:
  1. Level 1: resample scaffold IDs (or cluster IDs) with replacement
  2. Level 2: within each resampled scaffold, resample compounds with
     replacement
  3. Optional level 3 (seed mixing): randomly draw one seed per iteration
- `within_region.py` — variant for regional statistics. Resamples
  scaffolds within the region (not across regions); used for `G_ij(r)`
  and `Δ_ij(r)` CIs.
- `seed_mixing.py` — utility for drawing one seed per bootstrap iteration
  when `seeds` is provided to `hierarchical_bootstrap`.
- `calibration.py` — implements the §2.10 pre-pilot calibration check:
  ```python
  def run_calibration_check(synthetic_data: HierarchicalSamples) -> None
      # raises CalibrationError if any check fails
  ```
  Checks:
  - On `G_ii(r)` (identity): CI width < 0.05
  - On a known-positive task pair: non-degenerate CIs
  - Bootstrap distribution Shapiro-Wilk normality on ≥80% of statistics
- `percentile.py` — `percentile_ci(samples, alpha=0.05) -> tuple[float,
  float]`. Standard percentile bootstrap.

`src/transfermtl/validity/` (plan §2.11-2.12):
- `local_support.py`
  ```python
  def check_local_support(
      n_train_i: int, n_train_j: int,
      n_test_i: int, n_test_j: int,
      n_test_pos_i: int, n_test_neg_i: int,
      n_test_pos_j: int, n_test_neg_j: int,
      g_i_norm: float, g_j_norm: float,
      g_ij_ci_width: float,
      task_type: Literal["clf", "reg"],
      cfg: ValidityConfig,
  ) -> ValidityFlag: ...
  ```
  Five conditions per plan §2.11:
  1. `min(n_train_i, n_train_j) >= n_min`
  2. `min(n_test_i, n_test_j) >= test_min`
  3. min positives & negatives in test (clf only)
  4. `g_ij_ci_width <= w_max`
  5. `min(g_i_norm, g_j_norm) > grad_norm_min`
  Each failure recorded in `ValidityFlag.failed_reasons`.
- `meaningful_pair.py`
  ```python
  def check_meaningful(
      pair_id: str,
      region_validity: dict[int, ValidityFlag],
      region_deltas: dict[int, BootstrapResult],
      cfg: ValidityConfig,
  ) -> tuple[bool, list[str]]
  ```
  Four conditions per plan §2.12.
- `io.py`
  - `write_meaningful_pairs(dataset, results) -> Path` — writes
    `outputs/analysis/{dataset}/meaningful_pairs.parquet` validated
    against `MeaningfulPairSchema`.

`src/transfermtl/null/` (plan §2.13):
- `run_null.py`
  ```python
  def build_null_distribution(
      dataset: str, statistic: str, M: int, n_partitions: int = 200,
      compute_statistic: Callable,
  ) -> np.ndarray
  ```
  Skeleton in this wave; full integration with measurement pipeline
  happens in Wave 3 (A6 wires this in).
- `pvalue.py`
  ```python
  def empirical_pvalue(observed: float, null: np.ndarray) -> float:
      return (1 + np.sum(null >= observed)) / (1 + len(null))
  ```
- `io.py` — saves null arrays to
  `outputs/analysis/{dataset}/null_dist_{statistic}_M{M}.npy`.

### Reads from
- A1: `utils/types.py`, `utils/schemas.py`, `configs/_shared/bootstrap.yaml`
- A1: synthetic fixture (for calibration test)
- A5 (after partitioning lands): random partition factory
- A6 (in Wave 3): measurement compute functions for null distribution

### Writes to
- `outputs/analysis/{dataset}/meaningful_pairs.parquet` (when invoked by A6)
- `outputs/analysis/{dataset}/null_dist_*.npy` (when invoked by A6)
- The bootstrap module itself does not write directly; it's a library.

### Tests
- `tests/test_bootstrap.py`:
  - `test_bootstrap_recovers_known_mean` — bootstrap of mean of normal
    samples matches sample mean ±0.05
  - `test_bootstrap_ci_coverage` — over 100 trials of synthetic data
    with known mean, 95% CI contains true value in ≥93 trials
  - `test_hierarchical_resample_respects_clusters` — when scaffolds in
    level-1 resample appear, all their compounds appear together
  - `test_seed_mixing_uniform` — over 10000 iterations, each of 5
    seeds chosen 1500-2500 times
  - `test_calibration_check_passes_on_synthetic` — A1's fixture passes
    `run_calibration_check`
  - `test_save_samples_round_trip` — `BootstrapResult` with samples is
    JSON-serializable when samples are flattened
- `tests/test_validity.py`:
  - `test_local_support_all_pass`
  - Parametrized: `test_local_support_fails_each_condition` over the
    5 reasons (n_train, n_test, class balance, CI width, grad norm)
  - `test_meaningful_pair_4_conditions` — parametrized over 4 conditions
- `tests/test_null.py`:
  - `test_pvalue_extremes` — observed = max(null) → p ≈ 1/(B+1);
    observed < min(null) → p ≈ 1
  - `test_pvalue_smoothing` — empty null array → p = 1.0

### Acceptance criteria

- [ ] All tests pass on synthetic fixture
- [ ] `hierarchical_bootstrap(compute_mean, normal_samples, n_iter=1000)`
  matches `np.mean ± 1.96 * np.std/sqrt(n)` to within 0.02
- [ ] §2.10 calibration check passes on A1's synthetic fixture
- [ ] B=1000 bootstrap of a synthetic statistic completes in <30s on
  single CPU
- [ ] `BootstrapResult` is JSON-serializable (excluding samples)
- [ ] `outputs/manifests/wave2_a4_complete.json` exists

### Out of scope for A4
- Computing `G_ij(r)` or `Δ_ij(r)` — A6
- Predictor evaluation — A8
- Random partition generation — A5 owns the partitioning; A4 owns the
  null-distribution math

### References
- `codebase_plan.md` §2.10, §2.11-2.12, §2.13
- `plan.txt` §2.10 (line ~3030), §2.11 (line ~3059), §2.12 (line ~3071),
  §2.13 (line ~3082)

---

## Agent A5 — Region partitioning

### Mission
Implement the three partitioning schemes (scaffold primary, latent
secondary, kNN tertiary) and the random-partition negative control.
Output is per-dataset parquet files mapping each compound to a region
ID; downstream measurement code consumes these.

### Owned files

`src/transfermtl/partition/`:
- `scaffold.py` — primary scheme (plan §2.6.1)
  - Hierarchical agglomerative clustering on Tanimoto distance, average
    linkage
  - M ∈ {3, 5, 8, 10}, default 5
  - Distance matrix: `1 - Tanimoto(fp_a, fp_b)` over all unique
    scaffolds; uses `sklearn.cluster.AgglomerativeClustering`
  - Validates each region has ≥`2 * n_min` train compounds; merges
    undersized regions with their nearest-neighbor cluster (smallest
    centroid distance)
  - Function: `compute_scaffold_partition(dataset, M=5) ->
    pd.DataFrame[smiles, region_id]`
- `latent.py` — secondary scheme (plan §2.6.2)
  - K-means on encoder representations from a single all-task MTL run
    (seed=0); k-means++ init, 10 restarts
  - M ∈ {3, 5, 8, 10}
  - Reads checkpoint from A3 path:
    `outputs/checkpoints/{dataset}/all_task/seed0.pt`
  - **Stub policy**: until A3 lands, A5 implements with a path-injection
    parameter so a fake checkpoint can be passed in tests
- `knn.py` — tertiary scheme (plan §2.6.3)
  - Uses scaffold cluster centroids (from `scaffold.py`) as anchor
    points; assigns each compound to nearest centroid in encoder space
- `random_null.py` — negative control (plan §2.6.4)
  - `generate_random_partitions(dataset, scaffold_partition,
    n_partitions=200, seed=0) -> list[pd.DataFrame]`
  - Each random partition matches the scaffold partition's region-size
    distribution
  - Algorithm: shuffle compound indices, assign first n_1 to region 1,
    next n_2 to region 2, etc.
- `io.py`
  - `write_partition(dataset, scheme, M, df) -> Path`
  - Writes to `outputs/partitions/{dataset}/{scheme}_M{M}.parquet`
    (validated via `PartitionSchema`)
  - For random: `outputs/partitions/{dataset}/random_b{b}.parquet`

### Scripts
- `scripts/compute_partitions.py`:
  ```
  python scripts/compute_partitions.py --dataset tox21
      --scheme scaffold --M 5
  python scripts/compute_partitions.py --dataset tox21
      --scheme random --n-partitions 200
  ```

### Reads from
- A1: `utils/schemas.py::PartitionSchema`, `configs/_shared/preprocess.yaml`
- A2: `outputs/splits/{dataset}/split.parquet`,
  `outputs/cache/scaffold_fps/{dataset}.parquet`
- A3 (latent only): `outputs/checkpoints/{dataset}/all_task/seed0.pt`
  — A5 stubs latent partition until A3 lands

### Writes to
- `outputs/partitions/{dataset}/scaffold_M{M}.parquet`
- `outputs/partitions/{dataset}/latent_M{M}.parquet`
- `outputs/partitions/{dataset}/knn_M{M}.parquet`
- `outputs/partitions/{dataset}/random_b{b}.parquet` (b ∈ 1..200)

All validated via `PartitionSchema`.

### Tests
- `tests/test_partition.py`:
  - `test_scaffold_deterministic` — same data + M → same region_id
    column
  - `test_scaffold_min_size_enforced` — no region below `2 * n_min`
    after merging (parametrized over n_min ∈ {25, 50, 100})
  - `test_partition_schema_validates`
  - `test_random_partition_size_distribution_matches` — random partition
    region_id value_counts equal scaffold partition's
  - `test_latent_partition_clusters_synthetic_correctly` — on synthetic
    fixture (after stub all-task checkpoint provided), latent k-means
    recovers the 2 known regions with Adjusted Rand Index > 0.8
  - `test_knn_consistent_with_scaffold` — kNN with same M as scaffold
    has Jaccard overlap ≥ 0.5 on synthetic fixture (sanity check, not
    perfect agreement)
  - `test_M_variants` — parametrized over M ∈ {3, 5, 8, 10}, each
    produces M regions

### Acceptance criteria

- [ ] All four schemes produce schema-valid parquets
- [ ] Tox21 scaffold partition with M=5 yields 5 regions with sizes
  satisfying `2 * n_min` (after merging, possibly fewer than 5 regions)
- [ ] Random partitions match scaffold region-size distribution exactly
  (assertion on value_counts)
- [ ] On synthetic fixture, latent partition recovers known regions
  with ARI > 0.8 (using a stub checkpoint)
- [ ] `python scripts/compute_partitions.py --dataset tox21
  --scheme scaffold --M 5` completes in <2 minutes
- [ ] All tests pass
- [ ] `outputs/manifests/wave2_a5_complete.json` exists

### Out of scope for A5
- Computing gradients or benefits over regions — A6
- Building null statistics from random partitions — A4 owns the null
  framework, A6 wires it in (Wave 3)
- Data preprocessing — A2
- Training the all-task MTL needed for latent partitioning — A3

### References
- `codebase_plan.md` §2.6
- `plan.txt` §2.6 (line ~2894), §6.3
