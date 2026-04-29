# Wave 5 — Utility (architecture + baselines + sweep harness)

## Synchronization
- **3 agents in parallel: A10, A11, A12**
- All depend on A8's incremental R² clearing the kill criterion (≥0.05)
- A12 can begin scaffolding the sweep harness while A10/A11 build
  components; A12 doesn't run sweeps yet (that's Wave 6)
- A10 and A11 are independent; both depend on A3 + A8

## Goals
- A10 ships the local-affinity sharing architecture with the gating
  contract test passing
- A11 ships eight baselines (STL, all-task, pairwise union, random,
  global affinity, TAG, MMoE, Cross-Stitch/DynaShare)
- A12 ships the Phase 4 sweep harness — code that runs robustness sweeps
  but is not yet executed (execution is in Wave 6)

## Cross-agent guardrails
- A10 and A11 share zero source files; they only share encoder/training
  utilities from A3
- A11's baselines must each be independently testable
- The DynaShare-vs-Cross-Stitch decision is locked at the start of this
  wave via an engineering assessment recorded in
  `outputs/analysis/baseline_decision.md`. **Default is Cross-Stitch.**
  Switch to DynaShare only if a clean implementation can be built in a
  ≤2-day budget; bad baselines invalidate the utility claim (plan §8.2).

---

## Agent A10 — Local-affinity sharing architecture

### Mission
Implement the region-conditional gated MTL model from plan §2.15. The
"simplified implementation" (region-conditional masking with isolated-
head trick) is the recommended path; only attempt the full per-task
gradient-buffer version if the simple variant is publishable but
underperforms. Ship the architecture, three sharing variants (hard,
soft, abstention), and **the gating-contract test** that validates the
architecture's promise.

### Owned files

#### `src/transfermtl/architecture/`
- `local_affinity_sharing.py` — the region-conditional gated MTL model
  ```python
  class LocalAffinitySharingModel(nn.Module):
      def __init__(
          self,
          encoder: nn.Module,
          task_names: list[str],
          n_regions: int,
          gates: Tensor,  # shape [K, K, M], hard or soft
          gate_mode: Literal["hard", "soft"],
      ): ...
  ```
  - Shared encoder `f_θ: X → R^256` (from A3's `models/`)
  - Per-task heads `φ_k` for each task `k ∈ T` (from A3's `heads.py`)
  - Per-task **isolated** heads `φ_k^iso` (parallel copies). Isolated
    head receives gradients but does **not** propagate them back to the
    encoder (achieved via `encoder_output.detach()` on the isolated
    forward path).
  - Region assignment function `π: X → {1, ..., M}` (precomputed,
    deterministic from A5's partition)
  - Per-(task pair, region) gate `γ_ij(r)`:
    - Hard mode: `γ ∈ {0, 1}` based on `G_ij(r) > τ_+` (default τ = 0)
    - Soft mode: `γ = σ(α · G_ij(r))` (default α = 5)

- `forward.py` — forward pass logic
  - Standard forward: encode → task heads
  - For each compound x with `π(x) = r` and active task k:
    - Run shared head `φ_k(h)` (gradient flows through encoder)
    - For task pair (k, j) where `γ_kj(r) = 0`: route loss through
      isolated head instead

- `backward.py` — backward pass with regional gating
  ```python
  def compute_loss_with_gating(
      batch, model, gates, partition,
  ) -> Tensor: ...
  ```
  Implementation strategy (plan §2.15 "Simplified implementation"):
  - For each compound x in batch with `π(x) = r` and valid label for
    task k:
    - If `γ_kj(r) = 1` for all relevant pairs: contribute to `L_k`
      via shared head (encoder gradients flow normally)
    - If `γ_kj(r) = 0`: contribute to `L_k^iso` via isolated head
      (encoder treated as fixed for this loss term)
  - Total loss: `Σ_k (L_k + L_k^iso)`
  - **The isolated-head trick is the contract**: when the gate is closed,
    encoder gradients from that (task, region) are exactly zero.

- `pretrain_gates.py` — implements the §2.15 "Pre-training for gate
  values":
  - Train 5 epochs of standard pairwise MTL with `γ = 1` everywhere
  - Compute `G_ij(r)` from the resulting model (uses A6's gradient
    extraction)
  - Set gates: `γ_ij(r) = 1[G_ij(r) > τ_+]` (hard) or
    `σ(α · G_ij(r))` (soft)
  - Resume training with active gates for remaining epochs

- `abstention.py` — wraps a trained sharing model for inference-time
  abstention
  - For regions failing §2.11 validity, falls back to the corresponding
    STL prediction
  - `predict_with_abstention(model, stl_models, x) -> tuple[float, bool]`
    returns `(prediction, abstained)`

#### Sharing strategy variants
A10 also implements the three sharing strategies as composable functions:

- `strategies.py`
  - `local_affinity_hard_sharing(...)` — plan §8.2.9
  - `local_affinity_soft_sharing(...)` — plan §8.2.10
  - `local_affinity_with_abstention(...)` — plan §8.2.11

#### Scripts
- `scripts/train_local_affinity_sharing.py`:
  ```
  python scripts/train_local_affinity_sharing.py
      --dataset tox21 --gate-mode hard --tau 0.0 --seed 0
  ```

### Reads from
- A1: types, schemas, configs
- A3: encoder + heads + training loops + checkpoints
- A4: validity (for abstention region selection)
- A5: partition parquets
- A6: `G_ij(r)` from gradient affinity (for gate initialization)

### Writes to
- `outputs/checkpoints/{dataset}/local_affinity/{gate_mode}_tau{tau}/seed{s}.pt`
- `outputs/predictions/{dataset}/local_affinity/{gate_mode}/seed{s}.parquet`
- `outputs/manifests/wave5_a10_complete.json`

### Tests
- `tests/test_architecture.py`:
  - **`test_gating_contract_hard_zero_gate`** — **THE CRITICAL TEST**.
    Construct a synthetic 2-task problem with `γ_12(r=0) = 1` and
    `γ_12(r=1) = 0`. Train one step. Assert that encoder gradients from
    task 2 on region-1 inputs are exactly zero (within fp tolerance
    1e-7). This is the architecture's contract; if this test fails, the
    architecture is broken.
  - `test_gating_contract_hard_one_gate` — when `γ_12(r=0) = 1`,
    encoder gradients from task 2 on region-0 inputs are nonzero.
  - `test_soft_gate_equivalence` — when α → ∞ and threshold = 0, soft
    gates approximate hard gates (within tolerance).
  - `test_pretrain_gate_initialization` — after pretrain, gates match
    `1[G_ij(r) > τ_+]` for hard mode.
  - `test_abstention_uses_stl` — for an "invalid" region (per validity
    flag), prediction equals STL prediction exactly.
  - `test_isolated_head_no_encoder_gradient` — isolated head's
    backward does not write to encoder gradient buffers.
  - `test_full_training_converges_on_synthetic` — training on synthetic
    fixture for 30 epochs, val loss decreases monotonically (or at
    least achieves loss < initial × 0.5).

### Acceptance criteria

- [ ] **Gating contract test passes** — closed-gate gradients are
  numerically zero. This is the non-negotiable acceptance criterion.
- [ ] All three sharing variants train without error on Tox21 with
  M=5 scaffold partition
- [ ] Pretrain-then-train pipeline produces deterministic gates given
  a seed
- [ ] Soft sharing converges as well as hard sharing on Tox21 (val loss
  within 5%)
- [ ] Abstention falls back to STL for regions failing validity (verified
  on a deliberately-invalid synthetic region)
- [ ] All tests pass
- [ ] `outputs/manifests/wave5_a10_complete.json` exists

### Out of scope for A10
- Baselines (A11)
- Robustness sweeps (A12 / A13)
- Utility evaluation metrics (A13)
- Cross-architecture experiments (A12 / A13)

### References
- `codebase_plan.md` §2.15
- `plan.txt` §2.15 (line ~3143, continued at ~4522), §8.2 sharing
  strategies, §8.3 implementation guidance

---

## Agent A11 — Baselines

### Mission
Implement eight baseline methods for the utility comparison. Each must be
faithful to its source paper; bad implementations are worse than no
implementation (plan §8.2). Each baseline is independently testable on a
2-task synthetic problem.

### Owned files

#### `src/transfermtl/baselines/`
- `stl.py` — wraps `training/stl.py` so baselines have a uniform API
  ```python
  @register("baseline", "stl")
  def train_stl_baseline(dataset, seed, cfg) -> dict[task, predictions]
  ```
- `all_task_mtl.py` — single shared encoder, K task heads, joint training
  on all tasks. Loss = sum over per-task losses with masking for missing
  labels.
- `pairwise_union.py` — train pairwise MTL for each pair (plan §2.5);
  at test, ensemble predictions for task k by averaging logits across
  all pairs containing k.
- `random_grouping.py` — for G groups, randomly assign each task to a
  group (10 trials averaged). Train all-task MTL within each group.
  Tasks in different groups do not share parameters.
- `global_affinity_grouping.py` — ICLR-paper grouping:
  1. Compute `G_ij^global` for all pairs from an all-task MTL pretrained
     model (use A3's all-task checkpoint)
  2. Hierarchically cluster tasks using `1 - G_ij^global` as distance
  3. Cut dendrogram to produce G groups
  4. Train all-task MTL within each group
- `tag_grouping.py` — Fifty et al. update-affinity:
  ```
  a_{i→j} = Perf_j(θ + step_i) − Perf_j(θ)
  ```
  Implementation:
  1. Pretrain all-task MTL
  2. For each task pair (i, j): take a small gradient step using only
     task i's loss; measure change in task j's validation loss
  3. Cluster tasks based on the asymmetric `a` matrix (symmetrize via
     `(a_{i→j} + a_{j→i})/2`, then hierarchical clustering)
- `mmoe.py` — Ma et al. 2018:
  - Shared expert tower: 4 experts, each a 2-layer MLP with hidden=256
  - Per-task gating network: input is encoder representation, output is
    softmax over 4 experts
  - Per-task head on top of gated expert mixture
  - Joint training across all tasks
- `cross_stitch.py` — Misra et al. 2016 (default adaptive baseline):
  - Per-task encoder columns with shared cross-stitch units between
    layers
  - Cross-stitch units are 2×2 learned matrices that mix per-task layer
    activations:
    ```
    [x_i'] = [α_AA α_BA] [x_i]
    [x_j']   [α_AB α_BB] [x_j]
    ```
  - Trained jointly with sum-of-task-losses
- `dynashare.py` — optional. Built only if Cross-Stitch is judged
  insufficient and a clean DynaShare implementation can be made in a
  ≤2-day engineering budget.

#### Decision document
- `outputs/analysis/baseline_decision.md` — written at start of Wave 5:
  - Engineering assessment of DynaShare implementation difficulty
  - Decision: Cross-Stitch (default) or DynaShare
  - If DynaShare: document of fidelity-checking approach (synthetic
    benchmarks, comparison to original paper's reported numbers, etc.)

### Reads from
- A1: schemas, types, configs
- A2 + A9: split parquets
- A3: encoder, heads, training utilities
- A6: `G_ij^global` (for global_affinity_grouping)

### Writes to
- `outputs/checkpoints/{dataset}/baselines/{baseline_name}/seed{s}.pt`
- `outputs/predictions/{dataset}/baselines/{baseline_name}/seed{s}.parquet`
- `outputs/analysis/baseline_decision.md`
- `outputs/manifests/wave5_a11_complete.json`

### Tests
Per-baseline contract: each baseline must pass a unit test that:
- Trains for 3 epochs on a 2-task synthetic problem
- Asserts validation loss strictly decreases over the 3 epochs
- Asserts predictions are non-trivial (`std > 0.01` across compounds)

- `tests/test_baselines.py`:
  - `test_stl_baseline_smoke`
  - `test_all_task_mtl_baseline_smoke`
  - `test_pairwise_union_baseline_smoke` + assert ensemble averaging
    works correctly
  - `test_random_grouping_smoke` + assert 10 trials produce different
    group assignments
  - `test_global_affinity_grouping_smoke` + assert dendrogram cut
    produces G groups
  - `test_tag_grouping_smoke` + assert affinity matrix is asymmetric
  - `test_mmoe_smoke` + assert gating weights sum to 1 per task
  - `test_cross_stitch_smoke` + assert cross-stitch matrices update
- `tests/test_baseline_fidelity.py`:
  - `test_mmoe_matches_paper_param_count` — within 5% of Ma et al.'s
    reported parameter count for an equivalent setup
  - `test_cross_stitch_matches_paper_param_count`

### Acceptance criteria

- [ ] All 8 baselines (or 7 if DynaShare skipped) train without error
  on Tox21
- [ ] Each baseline produces schema-valid prediction parquets
- [ ] Per-baseline smoke test passes (loss decreases over 3 epochs)
- [ ] Per-baseline fidelity test passes (parameter count, architectural
  spec)
- [ ] Cross-Stitch vs DynaShare decision documented
- [ ] All tests pass
- [ ] `outputs/manifests/wave5_a11_complete.json` exists

### Out of scope for A11
- Local-affinity sharing architecture (A10)
- Utility evaluation metrics (A13)
- Phase 4 robustness sweeps (A12)

### References
- `codebase_plan.md` §2.16
- `plan.txt` §2.16 (line ~4558), §8.2 sharing strategies (full list)

---

## Agent A12 — Robustness sweep harness

### Mission
Build the Phase 4 robustness sweep infrastructure. **A12 does not run
sweeps in this wave; it ships the harness ready to be run by A13 in
Wave 6.** Decoupling sweep code from sweep execution prevents A12 from
blocking on compute availability and lets A13 schedule sweeps once all
dependencies are stable.

### Owned files

#### `src/transfermtl/eval/robustness/`
- `partition_sweep.py` — sweeps M ∈ {3, 5, 8, 10} for scaffold + latent
  + kNN + random partitions. For each (dataset, scheme, M), runs the
  full measurement + indices pipeline. Aggregates prevalence as a
  function of M.
- `n_min_sweep.py` — sweeps `n_min ∈ {25, 50, 100, 150}` and reports:
  - Phenomenon prevalence as a function of `n_min`
  - Predictor AUROC as a function of `n_min`
  - Abstention rate as a function of `n_min`
  - Retained accuracy after abstention
- `architecture_sweep.py` — re-runs key experiments (gradient affinity,
  prevalence, predictor) with:
  - GCN (primary, already done)
  - GAT
  - ChemBERTa (compute-permitting; flagged optional)
  - ECFP+MLP (fixed-feature baseline)
  - Reports Jaccard similarity of "positive transfer regions" across
    architectures (plan §9.3 expectation: r=0.71-0.81 across learned
    architectures, lower for ECFP)
- `gradient_ablations.py` — per plan §9.5:
  - Last-shared-layer-only vs all-encoder gradients
  - Cosine vs unnormalized dot product
  - Fisher-normalized gradients
  - Trajectory mean vs final-checkpoint gradient
- `negative_controls.py` — per plan §9.6:
  - Random label permutation: shuffle task labels within each region;
    `G_ij(r)` should drop to noise
  - Random region assignment: matched-size random partitions, full null
    distribution; should not produce sign heterogeneity at scaffold rate
  - Single-task identity: `G_ii(r) ≈ 1` for all r (also tested as unit
    test in A6)

#### Architecture variants (delegated to A12)
A12 implements the architecture variants needed for `architecture_sweep`
(the variants are referenced by A11/A12 but implemented here):
- `models/gat.py` — graph attention network variant. Spec: 3-layer GAT
  with 4 attention heads, hidden 256, similar parameter budget to GCN.
- `models/chemberta.py` — wraps HuggingFace ChemBERTa-77M. Optional;
  gated by config flag `--use-chemberta`.
- `models/ecfp_mlp.py` — ECFP fingerprint + 3-layer MLP, hidden 256.

#### Scripts (harnesses, not yet executed)
- `scripts/sweep_partition_robustness.py`
- `scripts/sweep_n_min.py`
- `scripts/sweep_architecture.py`
- `scripts/sweep_gradient_ablations.py`
- `scripts/run_negative_controls.py`

Each script reads a `configs/phase4/{sweep}.yaml`, dispatches a SLURM
array, and writes results to `outputs/analysis/robustness/{sweep}/`.

### Reads from
- All Wave 1+2+3+4 outputs
- A10 (for sharing-architecture robustness, if needed)
- A11 (for baseline robustness, if needed)

### Writes to
- `configs/phase4/{sweep}.yaml` (sweep configurations)
- Sweep harness implementations under `src/transfermtl/eval/robustness/`
- Architecture variant implementations under `src/transfermtl/models/`
- `outputs/manifests/wave5_a12_complete.json`

A12 does NOT write to `outputs/analysis/robustness/{sweep}/*.parquet`
in this wave — those are produced when A13 executes the sweeps.

### Tests
- `tests/test_robustness_harness.py`:
  - `test_partition_sweep_dispatches` — script enumerates all
    (dataset, scheme, M) combos correctly
  - `test_n_min_sweep_dispatches`
  - `test_architecture_sweep_smoke_per_architecture` — each architecture
    completes 1 epoch on synthetic fixture
  - `test_gradient_ablations_dispatches`
  - `test_negative_controls_label_shuffle_drives_to_zero` — shuffled
    labels → `G_ij(r)` mean within ±0.15 of 0
  - `test_negative_controls_random_partition_low_prevalence` — random
    partitions show heterogeneity at < 5th percentile of scaffold-based
    prevalence (uses synthetic fixture)
- `tests/test_architecture_variants.py`:
  - `test_gat_output_shape`
  - `test_gat_param_count`
  - `test_ecfp_mlp_output_shape`
  - `test_chemberta_output_shape` (skipped if `transformers` not installed)

### Acceptance criteria

- [ ] All sweep harnesses defined and dispatchable (verified by dry-run
  flag)
- [ ] All 4 architecture variants train for 1 epoch on Tox21 NR-AR
  without error
- [ ] Negative-control tests pass on synthetic fixture
- [ ] Sweep configs (`configs/phase4/*.yaml`) reviewed and committed
- [ ] All tests pass
- [ ] `outputs/manifests/wave5_a12_complete.json` exists

### Out of scope for A12
- Executing the sweeps (Wave 6 / A13)
- Aggregating and visualizing sweep results (A13)
- Local-affinity sharing architecture (A10)
- Baselines beyond architecture variants (A11)
- Final paper figures (A13)

### References
- `codebase_plan.md` §5 (Milestone 5)
- `plan.txt` §9 entire (line ~488), §9.1-9.7 robustness specifications
