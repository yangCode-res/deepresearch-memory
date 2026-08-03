#!/bin/bash
#SBATCH --job-name=eval-controller-8b
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-eval-controller-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory}
MODEL=${MODEL:-/file_storage01/home/juanliu/25_ymj/model/Llama-3.1-8B-Instruct}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/controller-sft/current}
ADAPTER=${ADAPTER:?submit with ADAPTER=/path/to/final_adapter}
OUTPUT=${OUTPUT:-${ADAPTER%/}/heldout_predictions.jsonl}
PYTHON_ENV=${PYTHON_ENV:-${PROJECT_ROOT}/vendor/openresearcher/.venv}

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
cd "$PROJECT_ROOT"

python scripts/evaluate_controller_lora.py \
  --model "$MODEL" \
  --adapter "$ADAPTER" \
  --validation "${DATA_DIR}/validation.jsonl" \
  --output "$OUTPUT" \
  --max-samples "${MAX_SAMPLES:-64}"
