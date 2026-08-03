#!/bin/bash
#SBATCH --job-name=controller-lora-8b
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-controller-lora-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory}
MODEL=${MODEL:-/file_storage01/home/juanliu/25_ymj/model/Llama-3.1-8B-Instruct}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data/controller-sft/current}
OUTPUT_DIR=${OUTPUT_DIR:-${PROJECT_ROOT}/artifacts/controller-lora-8b/${SLURM_JOB_ID}}
PYTHON_ENV=${PYTHON_ENV:-${PROJECT_ROOT}/vendor/openresearcher/.venv}

source "${PYTHON_ENV}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN

cd "$PROJECT_ROOT"
echo "model=$MODEL"
echo "data_dir=$DATA_DIR"
echo "output_dir=$OUTPUT_DIR"
echo "gpus=${SLURM_GPUS_ON_NODE:-6}"

torchrun --standalone --nproc_per_node="${SLURM_GPUS_ON_NODE:-6}" \
  scripts/train_controller_lora.py \
  --model "$MODEL" \
  --train "${DATA_DIR}/train.jsonl" \
  --validation "${DATA_DIR}/validation.jsonl" \
  --output-dir "$OUTPUT_DIR" \
  --max-length "${MAX_LENGTH:-6144}" \
  --epochs "${EPOCHS:-2}" \
  --learning-rate "${LEARNING_RATE:-2e-4}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation "${GRADIENT_ACCUMULATION:-2}" \
  --lora-r "${LORA_R:-32}" \
  --lora-alpha "${LORA_ALPHA:-64}"
