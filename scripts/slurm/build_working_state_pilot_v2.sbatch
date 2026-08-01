#!/bin/bash
#SBATCH --job-name=working-state-v2-20
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-working-state-v2-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory}
CONTROLLER_ENV_FILE=${CONTROLLER_ENV_FILE:-/file_storage01/home/juanliu/.config/cl-gism/controller.env}
set -a
source "$CONTROLLER_ENV_FILE"
set +a
source "${PROJECT_ROOT}/vendor/openresearcher/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/scripts/build_working_state_pilot_v2.py" \
  --input-glob "${PROJECT_ROOT}/data/openresearcher-dataset/seed_*/*.parquet" \
  --output "${PROJECT_ROOT}/data/working-state-labels/working_state_pilot_v2_20.jsonl" \
  --continue-target 10 \
  --switch-target 6 \
  --ready-target 4 \
  --max-steps-per-trajectory 40 \
  --max-trajectory-decision-points 12 \
  --model mimo-v2.5 \
  --preview-count 20
