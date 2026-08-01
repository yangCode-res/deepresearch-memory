#!/bin/bash
#SBATCH --job-name=working-state-retro-20
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-working-state-retro-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory}
CONTROLLER_ENV_FILE=${CONTROLLER_ENV_FILE:-/file_storage01/home/juanliu/.config/cl-gism/controller.env}
set -a
source "$CONTROLLER_ENV_FILE"
set +a
source "${PROJECT_ROOT}/vendor/openresearcher/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

OUTPUT=${OUTPUT:-${PROJECT_ROOT}/data/working-state-labels/working_state_retrospective_20.jsonl}

python "${PROJECT_ROOT}/scripts/build_working_state_retrospective.py" \
  --input-glob "${PROJECT_ROOT}/data/openresearcher-dataset/seed_*/*.parquet" \
  --output "$OUTPUT" \
  --num-samples "${NUM_SAMPLES:-20}" \
  --min-decision-points "${MIN_DECISION_POINTS:-3}" \
  --max-decision-points "${MAX_DECISION_POINTS:-10}" \
  --max-trajectory-chars "${MAX_TRAJECTORY_CHARS:-70000}" \
  --seed "${DATASET_SEED:-20260801}" \
  --model "${DATA_MODEL:-mimo-v2.5}" \
  --preview-count "${PREVIEW_COUNT:-20}"
