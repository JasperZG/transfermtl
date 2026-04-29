# Wave 1 — Foundation

## Synchronization
- **1 agent: A1** (sequential, blocks all later waves)
- No external dependencies; this is the bootstrap wave
- Wave 2 cannot start until A1's PR is merged

## Goals of this wave
- Lock all data contracts (pandera schemas, Python dataclasses)
- Stand up build/lint/test infrastructure
- Create the synthetic 2-task fixture used by every later agent
- Freeze pre-committed hyperparameters in `configs/_shared/`
- Establish CI workflow that prevents accidental schema/config drift

After A1 lands, four agents in Wave 2 can develop in parallel without
interface collisions.

---

## Agent A1 — Foundation & contract lock

### Mission
Establish the repo skeleton, freeze every cross-cutting contract, and ship
a working test harness on a synthetic molecular fixture. After A1 lands,
A2/A3/A4/A5 can develop in parallel.

### Owned files

#### Build & tooling
- `pyproject.toml` — package metadata, dependencies (per
  `codebase_plan.md` §8), ruff/black/mypy configuration. Python 3.11.
- `environment.yml` — conda env with rdkit, torch, torch_geometric, CUDA 12.x.
- `environment-dev.yml` — adds black, ruff, mypy, pre-commit.
- `.pre-commit-config.yaml` — runs ruff, black, mypy on staged files.
- `.gitignore` — `outputs/`, `logs/`, `*.pkl`, `*.pt`, `wandb/`,
  `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `*.egg-info/`,
  but **keep** `outputs/manifests/`.
- `Makefile` — targets: `test`, `test-fast`, `lint`, `format`, `typecheck`,
  `clean`.
- `.github/workflows/ci.yml` — runs `make test`, `make lint`, `make
  typecheck` on every PR. Includes a job that verifies
  `configs/_shared/_lock.yaml` matches actual file checksums.

#### Source skeleton (`src/transfermtl/`)
- `__init__.py` — sets `__version__ = "0.0.1"`.
- `utils/__init__.py` — re-exports key types and helpers.
- `utils/seeding.py` — `set_seed(seed: int)` controls torch + numpy + cuda
  + cuDNN deterministic mode. Returns the active seed for logging.
- `utils/logging.py` — W&B init with project tag `tsh-mtl`. Wraps
  `wandb.init` with config-snapshot logic, captures git SHA via
  `utils.git`. Provides `log_metrics(step, **kw)` and a no-op fallback
  when W&B is offline.
- `utils/io.py` — typed parquet/npz read/write helpers:
  - `read_parquet(path, schema=None) -> pd.DataFrame`
  - `write_parquet(path, df, schema=None) -> Path` (validates if schema)
  - `read_npy(path) -> np.ndarray`, `write_npy(path, arr)`
  - `ensure_parent(path)` — `mkdir -p`.
- `utils/git.py` — `current_sha() -> str`, `dirty_tree_warning() -> bool`.
  Captured at run start by every script.
- `utils/registry.py` — string-keyed factory pattern:
  ```python
  REGISTRY: dict[str, dict[str, Callable]] = defaultdict(dict)
  def register(kind: str, name: str): ...  # decorator
  def build(kind: str, name: str, **cfg): ...
  ```
  Kinds: `"dataset"`, `"encoder"`, `"baseline"`, `"predictor"`,
  `"partition"`.
- `utils/types.py` — frozen dataclasses (pin field names exactly):
  ```python
  @dataclass(frozen=True)
  class BootstrapResult:
      estimate: float
      ci_lower: float
      ci_upper: float
      samples: np.ndarray | None = None  # opt-in via save_samples=True

  @dataclass(frozen=True)
  class ValidityFlag:
      valid: bool
      failed_reasons: tuple[str, ...]

  @dataclass(frozen=True)
  class HierarchicalSamples:
      values: np.ndarray
      scaffold_ids: np.ndarray
      seed_ids: np.ndarray | None = None

  @dataclass(frozen=True)
  class RegionStats:
      region_id: int
      n_train_i: int
      n_train_j: int
      n_test_i: int
      n_test_j: int
      g_i_norm: float
      g_j_norm: float

  @dataclass(frozen=True)
  class PairIndices:
      pair_id: str
      S_ij: bool
      S_i: bool
      S_j: bool
      H_ij: float
      C_ij: float
      n_valid_regions: int
  ```
- `utils/schemas.py` — pandera schemas (see "Locked schemas" section
  below). Every dataframe parquet in this project is validated against
  one of these.
- `utils/config.py` — Hydra/OmegaConf wrappers. `load_config(path) ->
  DictConfig`. Hash-checks `_shared/_lock.yaml` on every load.

#### Empty package directories (importable)
Create these with `__init__.py` files (empty docstring placeholder):
- `data/`, `partition/`, `models/`, `training/`, `bootstrap/`,
  `validity/`, `null/`, `gradients/`, `benefits/`, `indices/`,
  `predictor/`, `architecture/`, `baselines/`, `eval/`, `analysis/`

This prevents Wave 2+ agents from racing on directory creation.

#### Configuration
- `configs/_shared/encoder_gcn.yaml` — frozen encoder hparams
- `configs/_shared/train_default.yaml` — frozen optimizer/schedule
- `configs/_shared/bootstrap.yaml` — B=1000, B'=200, alpha=0.05
- `configs/_shared/preprocess.yaml` — split seed=42, n_min=50, ε, w_max
- `configs/_shared/_lock.yaml` — sha256 checksums of the four files
  above. CI verifies these match.

#### Tests
- `tests/__init__.py`
- `tests/conftest.py` — synthetic fixture (see "Synthetic fixture"
  section below). Uses pytest fixtures so any test can request
  `synthetic_dataset`, `synthetic_split`, `synthetic_partition`.
- `tests/synthetic_fixture/build.py` — generates the fixture parquet.
  Run once at import time; cached at
  `tests/synthetic_fixture/data.parquet`.
- `tests/test_utils.py` — smoke tests for seeding, registry, schemas,
  io, git.
- `tests/test_synthetic_fixture.py` — verifies fixture matches its
  contract (see test list below).

### Locked schemas (`src/transfermtl/utils/schemas.py`)

Every Wave 2+ agent stubs against these. Field names and types are immutable.

| Schema | Producer | Columns |
|---|---|---|
| `SplitSchema` | A2 | `smiles: str`, `scaffold: str`, `split: Category[train\|val\|test]`, `task_*: float` (NaN for missing labels) |
| `PartitionSchema` | A5 | `smiles: str`, `region_id: int` |
| `PredictionSchema` | A3 | `smiles: str`, `task: str`, `y_true: float`, `y_pred: float`, `seed: int` |
| `GradientAffinitySchema` | A6 | `region_id: int`, `G_ij: float`, `g_i_norm: float`, `g_j_norm: float`, `n_i_in_region: int`, `n_j_in_region: int`, `checkpoint_label: Category[final\|0.8\|0.6]`, `seed: int` |
| `RegionBenefitSchema` | A6 | `region_id: int`, `delta_pair: float`, `delta_i_from_j: float`, `delta_j_from_i: float`, `delta_worst: float`, `ci_lo: float`, `ci_hi: float`, `n_test: int` |
| `PairIndicesSchema` | A6 | `pair_id: str`, `S_ij: bool`, `S_i: bool`, `S_j: bool`, `H_ij: float`, `C_ij: float`, `n_valid_regions: int` |
| `MeaningfulPairSchema` | A4 | `pair_id: str`, `is_meaningful: bool`, `failed_reasons: list[str]` |
| `PredictorScoresSchema` | A8 | `pair_id: str`, `region_id: int`, `feature_name: str`, `value: float` |

Use `pandera.DataFrameSchema` with `strict=True` so any extra column
fails validation. NaN-allowed columns must be explicitly marked
`nullable=True`.

### Synthetic fixture (`tests/conftest.py`)

200 molecules with two binary tasks, partitioned into 2 known regions:

- **Region A** (100 molecules): a single scaffold cluster of simple
  benzene/aniline derivatives. Generated by attaching random small
  substituents to a benzene core.
- **Region B** (100 molecules): a different scaffold cluster of pyrrole
  / pyridine derivatives.
- **Task 1**: random binary label, 50% positive.
- **Task 2**:
  - In region A: identical to task 1 → perfect positive transfer.
  - In region B: opposite of task 1 → perfect negative transfer.
- Train/val/test: 70/15/15 scaffold-stratified split.

Ground-truth values that synthetic-fixture-using tests can rely on:

| Quantity | Expected value | Tolerance |
|---|---|---|
| `G_12(A)` | +1 | within 0.1 after small training run |
| `G_12(B)` | −1 | within 0.1 |
| `Δ_12(A)` | > 0 | CI excludes 0 |
| `Δ_12(B)` | < 0 | CI excludes 0 |
| `S_12` | True | exact |
| `H_12` | ≈ \|Δ(A) − Δ(B)\|/2 | within 20% |
| `C_12` | very large (≥ 5) | exact comparison |

Implementation note: A1 generates the molecules deterministically with
seed=12345. The fixture is checked into git as a parquet so tests don't
re-generate every run.

### Pre-committed YAML configs

`configs/_shared/encoder_gcn.yaml`:
```yaml
name: gcn
n_layers: 3
hidden_dim: 256
activation: relu
dropout: 0.1
pool: mean
atom_feature_dim: 74
bond_feature_dim: 12
```

`configs/_shared/train_default.yaml`:
```yaml
optimizer: adamw
lr: 1.0e-3
weight_decay: 1.0e-2
batch_size: 32
max_epochs: 100
patience: 25
lr_schedule: cosine
lr_min: 1.0e-5
grad_clip: 1.0
```

`configs/_shared/bootstrap.yaml`:
```yaml
n_iter: 1000
random_partitions: 200
ci_alpha: 0.05
seed_mixing: true
level1_default: scaffold
```

`configs/_shared/preprocess.yaml`:
```yaml
split_seed: 42
train_frac: 0.70
val_frac: 0.15
test_frac: 0.15
n_min: 50
test_min_clf: 30
test_min_reg: 50
test_min_pos: 5
test_min_neg: 5
epsilon_clf: 1.5
epsilon_reg: 0.10
ci_width_max: 0.4
grad_norm_min: 1.0e-6
grad_norm_zero: 1.0e-8
```

### Tests A1 must add

- `tests/test_utils.py::test_seeding_deterministic` — same seed →
  identical `torch.randn` and `np.random.rand` outputs across calls.
- `tests/test_utils.py::test_registry_lookup` — `register("encoder",
  "x")` then `build("encoder", "x")` returns the registered factory.
- `tests/test_utils.py::test_schemas_validate_good` — minimal valid
  frame passes each schema.
- `tests/test_utils.py::test_schemas_reject_bad` — frame missing a
  required column or with wrong dtype fails validation.
- `tests/test_utils.py::test_lock_file_matches_shared` — the checksums
  in `_lock.yaml` equal `sha256` of each file in `_shared/`.
- `tests/test_utils.py::test_io_roundtrip_parquet` — write/read parquet
  with schema validation.
- `tests/test_utils.py::test_io_roundtrip_npy` — write/read npy.
- `tests/test_synthetic_fixture.py::test_fixture_shape` — 200 rows,
  2 task columns.
- `tests/test_synthetic_fixture.py::test_fixture_two_regions` —
  `region_id` column has values {0, 1}.
- `tests/test_synthetic_fixture.py::test_fixture_ground_truth` —
  task_2 == task_1 on region 0; task_2 != task_1 on region 1.
- `tests/test_synthetic_fixture.py::test_fixture_split_no_leakage` —
  no scaffold appears in more than one split.
- `tests/test_synthetic_fixture.py::test_fixture_passes_split_schema` —
  saved fixture parquet validates against `SplitSchema`.

### Acceptance criteria

- [ ] `pip install -e .[dev]` succeeds from a fresh Python 3.11 venv
- [ ] `pytest` runs the test suite (~12 tests) and they all pass
- [ ] `ruff check src/ tests/` clean
- [ ] `mypy --strict src/transfermtl` clean (or the strict subset agreed
  with the team)
- [ ] `make test`, `make lint`, `make format`, `make typecheck` all
  defined and working
- [ ] `python -c "from transfermtl.utils.schemas import SplitSchema,
  PartitionSchema, PredictionSchema, GradientAffinitySchema,
  RegionBenefitSchema, PairIndicesSchema, MeaningfulPairSchema"` works
- [ ] `python -c "from transfermtl.utils.types import BootstrapResult,
  ValidityFlag, HierarchicalSamples, RegionStats, PairIndices"` works
- [ ] CI workflow runs on a draft PR and verifies the lock file
- [ ] Synthetic fixture parquet exists at
  `tests/synthetic_fixture/data.parquet` and round-trips
- [ ] `outputs/manifests/wave1_complete.json` lists all owned files

### Out of scope for A1

- Real dataset loaders — A2
- Any model code (encoder, heads, training) — A3
- Bootstrap math — A4 (A1 only ships `BootstrapResult` dataclass)
- Partitioning logic — A5
- RDKit calls beyond what the synthetic fixture needs (atom/bond
  featurization is A2's domain; A1's fixture can use a simplified
  feature set just enough to make the synthetic data round-trip)

If A1 starts touching `data/datasets.py`, `bootstrap/hierarchical.py`,
or `models/gcn.py`, that is scope creep and should be split into a
follow-up PR.

### References

- `codebase_plan.md` §1 (top-level layout), §3 (data formats), §4
  (configuration system), §7 (testing), §8 (dependencies), §9
  (reproducibility)
- `plan.txt` §2.21 — pre-committed hyperparameter table (line ~4675).
  This table is the authoritative source for the values in
  `configs/_shared/`.
- `plan.txt` §2.19 — reproducibility & logging (line ~4648). W&B project
  tag `tsh-mtl`, deterministic CUDA, git SHA capture all originate here.
