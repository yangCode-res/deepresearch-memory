#!/bin/bash
#SBATCH --job-name=build-working-state-200
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-build-working-state-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory}
CONTROLLER_ENV_FILE=${CONTROLLER_ENV_FILE:-/file_storage01/home/juanliu/.config/cl-gism/controller.env}
if [[ ! -f "$CONTROLLER_ENV_FILE" ]]; then
  echo "Controller environment file not found: $CONTROLLER_ENV_FILE" >&2
  exit 1
fi
set -a
source "$CONTROLLER_ENV_FILE"
set +a

source "${PROJECT_ROOT}/vendor/openresearcher/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export CL_GISM_DATA_MODEL=${CL_GISM_DATA_MODEL:-mimo-v2.5}

OUTPUT=${OUTPUT:-${PROJECT_ROOT}/data/working-state-labels/working_state_200_mimo_v2_5.jsonl}

python "${PROJECT_ROOT}/scripts/build_working_state_dataset.py" \
  --input-glob "${PROJECT_ROOT}/data/openresearcher-dataset/seed_*/*.parquet" \
  --output "$OUTPUT" \
  --num-samples "${NUM_SAMPLES:-200}" \
  --max-steps-per-trajectory "${MAX_STEPS_PER_TRAJECTORY:-10}" \
  --model "$CL_GISM_DATA_MODEL"
