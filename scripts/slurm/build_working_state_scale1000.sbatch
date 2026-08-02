#!/bin/bash
#SBATCH --job-name=ws-retro-scale1000
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-ws-scale1000-%A_%a.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory}
CONTROLLER_ENV_FILE=${CONTROLLER_ENV_FILE:-/file_storage01/home/juanliu/.config/cl-gism/controller.env}
RUN_DIR=${RUN_DIR:-${PROJECT_ROOT}/data/working-state-labels/scale1000}
EXCLUDE_QIDS_FILE=${EXCLUDE_QIDS_FILE:-${RUN_DIR}/existing_qids.json}
NUM_SHARDS=${NUM_SHARDS:-32}
SHARD_ID=${SLURM_ARRAY_TASK_ID:?submit this file as a Slurm array}

set -a
source "$CONTROLLER_ENV_FILE"
set +a
source "${PROJECT_ROOT}/vendor/openresearcher/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
mkdir -p "$RUN_DIR"

OUTPUT="${RUN_DIR}/working_state_retrospective_mimo25pro_shard${SHARD_ID}.jsonl"

python "${PROJECT_ROOT}/scripts/build_working_state_retrospective.py" \
  --input-glob "${PROJECT_ROOT}/data/openresearcher-dataset/seed_*/*.parquet" \
  --output "$OUTPUT" \
  --target-trajectories "${TARGET_TRAJECTORIES_PER_SHARD:-25}" \
  --min-loops-per-trajectory 2 \
  --min-continues-per-trajectory 2 \
  --unique-qids \
  --exclude-qids-file "$EXCLUDE_QIDS_FILE" \
  --qid-modulus "$NUM_SHARDS" \
  --qid-remainder "$SHARD_ID" \
  --min-decision-points 4 \
  --max-decision-points 10 \
  --max-trajectory-chars 70000 \
  --seed "$((20260900 + SHARD_ID))" \
  --model "${DATA_MODEL:-mimo-v2.5-pro}" \
  --preview-count 6
