#!/usr/bin/env bash
# scripts/launch_pilot.sh — Phase 0 pilot orchestration on SLURM.
#
# Stages (plan §5):
#   1. prepare_dataset.py for tox21 + sider
#   2. STL training: (dataset, task, seed) array
#   3. pairwise MTL training: (dataset, pair, seed) array
#   4. compute_partitions.py for scaffold (M=5) + 200 random null partitions
#   5. compute_gradients/benefits/indices per (dataset, partition)
#   6. random-partition null: re-run stage 5 over 200 random partitions
#   7. pilot_decision.py — synthesises everything into the gate document
#
# A6 owns the compute_gradients / compute_benefits / compute_indices entry
# points; this script invokes them by name. If A6 has not landed, stages 5–7
# will fail fast with "command not found", which is the desired behaviour.
#
# Usage:
#   sbatch scripts/launch_pilot.sh [stage]
#   STAGES="1 2 3 4 5 6 7" bash scripts/launch_pilot.sh    # run locally (CPU/single-GPU)

#SBATCH --job-name=tsh-pilot
#SBATCH --partition=scu-gpu
#SBATCH --output=outputs/logs/pilot_%j.out
#SBATCH --error=outputs/logs/pilot_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

set -euo pipefail

# Under sbatch, BASH_SOURCE points into the SLURM spool dir (unwritable).
# SLURM_SUBMIT_DIR is the dir you ran sbatch from; fall back to BASH_SOURCE
# only when running outside SLURM.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"
export WANDB_MODE=offline

# Activate the conda env explicitly — sbatch shells don't inherit it.
CONDA_ENV="${CONDA_ENV:-transfermtl}"
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "${CONDA_DEFAULT_ENV}" != "${CONDA_ENV}" ]]; then
  # conda's profile script trips set -u; relax it just for the activation.
  set +u
  source /etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
  set -u
fi

PYTHON="${PYTHON:-python}"
DATASETS="${DATASETS:-tox21 sider}"
SEEDS="${SEEDS:-0 1 2}"
M="${M:-5}"
N_RANDOM_PARTITIONS="${N_RANDOM_PARTITIONS:-200}"
STAGES="${STAGES:-${1:-1 2 3 4 5 6 7}}"

mkdir -p outputs/logs outputs/analysis

run_stage() {
  local n="$1"; shift
  echo "[stage ${n}] $*"
  "$@"
}

# ---------------------------------------------------------------------------
# Stage 1: prepare datasets
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 1 "* ]]; then
  for ds in ${DATASETS}; do
    run_stage 1 "${PYTHON}" scripts/prepare_dataset.py --dataset "${ds}"
  done
fi

# ---------------------------------------------------------------------------
# Stage 2: STL training over (dataset, task, seed)
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 2 "* ]]; then
  for ds in ${DATASETS}; do
    n_tasks=$("${PYTHON}" -c "
import pandas as pd
df = pd.read_parquet('outputs/splits/${ds}/split.parquet')
print(sum(1 for c in df.columns if c.startswith('task_')))
")
    for ti in $(seq 1 "${n_tasks}"); do
      task="task_${ti}"
      for seed in ${SEEDS}; do
        run_stage 2 "${PYTHON}" scripts/train_stl.py \
          --dataset "${ds}" --task "${task}" --seed "${seed}"
      done
    done
  done
fi

# ---------------------------------------------------------------------------
# Stage 3: pairwise MTL training over (dataset, pair, seed)
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 3 "* ]]; then
  if [[ ! -f outputs/analysis/pilot_pairs.parquet ]]; then
    run_stage 3 "${PYTHON}" scripts/select_pilot_pairs.py
  fi
  while IFS=$'\t' read -r dataset task_i task_j _category; do
    [[ "${dataset}" == "dataset" ]] && continue
    for seed in ${SEEDS}; do
      run_stage 3 "${PYTHON}" scripts/train_pairwise_mtl.py \
        --dataset "${dataset}" --task-i "${task_i}" --task-j "${task_j}" --seed "${seed}"
    done
  done < <("${PYTHON}" -c "
import pandas as pd
df = pd.read_parquet('outputs/analysis/pilot_pairs.parquet')
for r in df.itertuples(index=False):
    print(f'{r.dataset}\t{r.task_i}\t{r.task_j}\t{r.category}')
")
fi

# ---------------------------------------------------------------------------
# Stage 4: partitions (scaffold M=5 + random null)
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 4 "* ]]; then
  for ds in ${DATASETS}; do
    run_stage 4 "${PYTHON}" scripts/compute_partitions.py \
      --dataset "${ds}" --scheme scaffold --M "${M}"
    run_stage 4 "${PYTHON}" scripts/compute_partitions.py \
      --dataset "${ds}" --scheme random --n-partitions "${N_RANDOM_PARTITIONS}"
  done
fi

# ---------------------------------------------------------------------------
# Stage 5: gradients + benefits + indices (per dataset, scaffold partition)
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 5 "* ]]; then
  for ds in ${DATASETS}; do
    while IFS=$'\t' read -r dataset task_i task_j _; do
      [[ "${dataset}" != "${ds}" ]] && continue
      [[ "${dataset}" == "dataset" ]] && continue
      for seed in ${SEEDS}; do
        run_stage 5 "${PYTHON}" scripts/compute_gradients.py \
          --dataset "${ds}" --task-i "${task_i}" --task-j "${task_j}" \
          --partition "scaffold_M${M}" --seed "${seed}"
      done
      run_stage 5 "${PYTHON}" scripts/compute_benefits.py \
        --dataset "${ds}" --task-i "${task_i}" --task-j "${task_j}" \
        --partition "scaffold_M${M}"
    done < <("${PYTHON}" -c "
import pandas as pd
df = pd.read_parquet('outputs/analysis/pilot_pairs.parquet')
for r in df.itertuples(index=False):
    print(f'{r.dataset}\t{r.task_i}\t{r.task_j}\t{r.category}')
")
    run_stage 5 "${PYTHON}" scripts/compute_indices.py \
      --dataset "${ds}" --partition "scaffold_M${M}"
  done
fi

# ---------------------------------------------------------------------------
# Stage 6: random-partition null distribution (heaviest stage)
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 6 "* ]]; then
  for ds in ${DATASETS}; do
    for b in $(seq 1 "${N_RANDOM_PARTITIONS}"); do
      run_stage 6 "${PYTHON}" scripts/compute_indices.py \
        --dataset "${ds}" --partition "random_b${b}" --null-mode
    done
  done
fi

# ---------------------------------------------------------------------------
# Stage 7: pilot_decision.py — synthesise + render the gate document
# ---------------------------------------------------------------------------
if [[ " ${STAGES} " == *" 7 "* ]]; then
  run_stage 7 "${PYTHON}" scripts/pilot_decision.py \
    --phenomenon-json outputs/analysis/phenomenon.json \
    --predictor-json outputs/analysis/predictor.json \
    --per-dataset-json outputs/analysis/per_dataset.json \
    --examples-json outputs/analysis/pilot_examples.json
fi

echo "pilot stages complete: ${STAGES}"
