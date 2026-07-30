#!/bin/bash
#SBATCH --job-name=openresearcher-bcplus-dense
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-openresearcher-%j.out

set -euo pipefail

# Submit examples:
#   sbatch scripts/slurm/openresearcher_browsecomp_plus_dense.sbatch
#   sbatch --export=ALL,RUN_MODE=full,MAX_CONCURRENCY=8 \
#     scripts/slurm/openresearcher_browsecomp_plus_dense.sbatch
#
# Optional path overrides:
#   OPENRESEARCHER_ROOT=/path/to/vendor/openresearcher
#   OPENRESEARCHER_MODEL_REF=/path/to/OpenResearcher-30B-A3B
#   DENSE_MODEL_REF=/path/to/Qwen3-Embedding-8B
#   OUTPUT_DIR=/path/to/results

OPENRESEARCHER_ROOT=${OPENRESEARCHER_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory/vendor/openresearcher}
OPENRESEARCHER_MODEL_REF=${OPENRESEARCHER_MODEL_REF:-${OPENRESEARCHER_ROOT}/models/OpenResearcher-30B-A3B}
RUN_MODE=${RUN_MODE:-smoke}
SMOKE_SIZE=${SMOKE_SIZE:-10}
OFFLINE_MODE=${OFFLINE_MODE:-1}

if [[ "$RUN_MODE" != "smoke" && "$RUN_MODE" != "full" ]]; then
  echo "RUN_MODE must be 'smoke' or 'full', got: $RUN_MODE" >&2
  exit 1
fi

if [[ "$RUN_MODE" == "smoke" ]]; then
  MAX_CONCURRENCY=${MAX_CONCURRENCY:-2}
  OUTPUT_DIR=${OUTPUT_DIR:-${OPENRESEARCHER_ROOT}/results/smoke/baseline_dense}
else
  MAX_CONCURRENCY=${MAX_CONCURRENCY:-8}
  OUTPUT_DIR=${OUTPUT_DIR:-${OPENRESEARCHER_ROOT}/results/browsecomp-plus/baseline_dense}
fi

if [[ ! -d "$OPENRESEARCHER_ROOT" ]]; then
  echo "OpenResearcher directory not found: $OPENRESEARCHER_ROOT" >&2
  exit 1
fi

cd "$OPENRESEARCHER_ROOT"

if [[ ! -d ".venv" ]]; then
  echo "Python environment not found: ${OPENRESEARCHER_ROOT}/.venv" >&2
  exit 1
fi

if [[ ! -d "$OPENRESEARCHER_MODEL_REF" ]]; then
  echo "OpenResearcher model not found: $OPENRESEARCHER_MODEL_REF" >&2
  echo "Set OPENRESEARCHER_MODEL_REF to the downloaded model directory." >&2
  exit 1
fi

source /file_storage01/home/juanliu/miniconda3/etc/profile.d/conda.sh
conda activate openresearcher312
CONDA_ENV_ROOT=$CONDA_PREFIX
source .venv/bin/activate
VENV_PYTHON=${OPENRESEARCHER_ROOT}/.venv/bin/python

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtual-environment Python is not executable: $VENV_PYTHON" >&2
  exit 1
fi

export JAVA_HOME=$CONDA_ENV_ROOT
export PATH="${JAVA_HOME}/bin:/persist_data/home/juanliu/.local/bin:${PATH}"
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-600}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}

if [[ "$OFFLINE_MODE" == "1" ]]; then
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

# PyJNIus does not discover Conda's OpenJDK layout reliably. Locate libjvm.so
# explicitly so importing Pyserini works for both BM25 and Dense backends.
JVM_PATH=$(find -L "$JAVA_HOME" -type f -name libjvm.so -print -quit)
if [[ -z "$JVM_PATH" ]]; then
  echo "Unable to find libjvm.so below JAVA_HOME=$JAVA_HOME" >&2
  exit 1
fi
export JVM_PATH
export LD_LIBRARY_PATH="$(dirname "$JVM_PATH"):${LD_LIBRARY_PATH:-}"

LOCAL_DENSE_MODEL=${OPENRESEARCHER_ROOT}/models/Qwen3-Embedding-8B
if [[ -d "$LOCAL_DENSE_MODEL" ]]; then
  DEFAULT_DENSE_MODEL_REF=$LOCAL_DENSE_MODEL
else
  DEFAULT_DENSE_MODEL_REF=Qwen/Qwen3-Embedding-8B
fi
DENSE_MODEL_REF=${DENSE_MODEL_REF:-$DEFAULT_DENSE_MODEL_REF}

shopt -s nullglob
TEST_FILES=(Tevatron/browsecomp-plus/data/*.parquet)
CORPUS_FILES=(Tevatron/browsecomp-plus-corpus/data/*.parquet)
DENSE_INDEX_FILES=(Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl)
shopt -u nullglob

if (( ${#TEST_FILES[@]} == 0 )); then
  echo "BrowseComp-Plus test parquet files are missing." >&2
  exit 1
fi
if (( ${#CORPUS_FILES[@]} == 0 )); then
  echo "BrowseComp-Plus corpus parquet files are missing." >&2
  exit 1
fi
if (( ${#DENSE_INDEX_FILES[@]} == 0 )); then
  echo "BrowseComp-Plus dense index files are missing." >&2
  exit 1
fi

# Slurm may expose physical GPU IDs rather than renumbering them. Preserve the
# allocation and select five GPUs from it instead of assuming physical IDs.
ALLOCATED_GPU_CSV=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_CSV"
if (( ${#ALLOCATED_GPUS[@]} < 6 )); then
  echo "Expected 6 allocated GPUs, got ${#ALLOCATED_GPUS[@]}: $ALLOCATED_GPU_CSV" >&2
  exit 1
fi

VLLM1_GPUS=${ALLOCATED_GPUS[0]},${ALLOCATED_GPUS[1]}
VLLM2_GPUS=${ALLOCATED_GPUS[2]},${ALLOCATED_GPUS[3]}
DENSE_GPU=${ALLOCATED_GPUS[4]}
RESERVED_GPU=${ALLOCATED_GPUS[5]}

# Derive ports from the Slurm job ID to reduce collisions when several jobs
# share a node. Values remain stable for the lifetime of this job.
JOB_NUMBER=${SLURM_JOB_ID:-0}
PORT_OFFSET=$((JOB_NUMBER % 10000))
SEARCH_PORT=$((20000 + PORT_OFFSET))
VLLM1_PORT=$((SEARCH_PORT + 1))
VLLM2_PORT=$((SEARCH_PORT + 2))

SERVICE_LOG_DIR=${OPENRESEARCHER_ROOT}/logs/slurm-${JOB_NUMBER}
mkdir -p "$SERVICE_LOG_DIR" "$OUTPUT_DIR"

echo "===== OPENRESEARCHER BROWSECOMP-PLUS CONFIG ====="
echo "job_id=$JOB_NUMBER"
echo "run_mode=$RUN_MODE"
echo "project_root=$OPENRESEARCHER_ROOT"
echo "openresearcher_model=$OPENRESEARCHER_MODEL_REF"
echo "dense_model=$DENSE_MODEL_REF"
echo "java_home=$JAVA_HOME"
echo "jvm_path=$JVM_PATH"
echo "allocated_gpus=$ALLOCATED_GPU_CSV"
echo "vllm_1_gpus=$VLLM1_GPUS port=$VLLM1_PORT"
echo "vllm_2_gpus=$VLLM2_GPUS port=$VLLM2_PORT"
echo "dense_gpu=$DENSE_GPU port=$SEARCH_PORT"
echo "reserved_gpu=$RESERVED_GPU"
echo "max_concurrency_per_worker=$MAX_CONCURRENCY"
echo "output_dir=$OUTPUT_DIR"
echo "================================================="

nvidia-smi
"$VENV_PYTHON" -c "import sys; print('Python executable:', sys.executable); print('Python version:', sys.version)"
"$VENV_PYTHON" -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available(), 'visible GPU count:', torch.cuda.device_count())"
java -version

SERVICE_PIDS=()

cleanup() {
  exit_code=$?
  trap - EXIT INT TERM
  echo "Stopping background services..."
  for pid in "${SERVICE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "Cleanup complete (exit code: $exit_code)."
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

wait_for_service() {
  service_name=$1
  service_url=$2
  service_pid=$3
  service_log=$4
  max_attempts=${5:-360}

  echo "Waiting for $service_name at $service_url ..."
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    if curl --fail --silent --show-error "$service_url" >/dev/null 2>&1; then
      echo "$service_name is ready."
      return 0
    fi
    if ! kill -0 "$service_pid" 2>/dev/null; then
      echo "$service_name exited before becoming ready." >&2
      tail -n 100 "$service_log" >&2 || true
      return 1
    fi
    if (( attempt % 12 == 0 )); then
      echo "Still waiting for $service_name ($((attempt * 5)) seconds elapsed)..."
    fi
    sleep 5
  done

  echo "Timed out waiting for $service_name." >&2
  tail -n 100 "$service_log" >&2 || true
  return 1
}

# Dense retrieval service on the fifth allocated GPU.
export LUCENE_EXTRA_DIR=${OPENRESEARCHER_ROOT}/tevatron
export CORPUS_PARQUET_PATH="${OPENRESEARCHER_ROOT}/Tevatron/browsecomp-plus-corpus/data/*.parquet"
export DENSE_INDEX_PATH="${OPENRESEARCHER_ROOT}/Tevatron/browsecomp-plus-indexes/qwen3-embedding-8b/*.pkl"
export DENSE_MODEL_NAME=$DENSE_MODEL_REF
export SEARCHER_TYPE=dense
export GPU_IDS=0

SEARCH_LOG=${SERVICE_LOG_DIR}/dense-search.log
CUDA_VISIBLE_DEVICES=$DENSE_GPU \
  "$VENV_PYTHON" -m uvicorn scripts.deploy_search_service:app \
    --host 127.0.0.1 \
    --port "$SEARCH_PORT" \
    >"$SEARCH_LOG" 2>&1 &
SEARCH_PID=$!
SERVICE_PIDS+=("$SEARCH_PID")

# Two OpenResearcher model servers, each using tensor parallelism across two GPUs.
VLLM1_LOG=${SERVICE_LOG_DIR}/vllm-${VLLM1_PORT}.log
CUDA_VISIBLE_DEVICES=$VLLM1_GPUS \
  "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$OPENRESEARCHER_MODEL_REF" \
    --host 127.0.0.1 \
    --port "$VLLM1_PORT" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-prefix-caching \
    >"$VLLM1_LOG" 2>&1 &
VLLM1_PID=$!
SERVICE_PIDS+=("$VLLM1_PID")

VLLM2_LOG=${SERVICE_LOG_DIR}/vllm-${VLLM2_PORT}.log
CUDA_VISIBLE_DEVICES=$VLLM2_GPUS \
  "$VENV_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$OPENRESEARCHER_MODEL_REF" \
    --host 127.0.0.1 \
    --port "$VLLM2_PORT" \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-prefix-caching \
    >"$VLLM2_LOG" 2>&1 &
VLLM2_PID=$!
SERVICE_PIDS+=("$VLLM2_PID")

wait_for_service "Dense search" "http://127.0.0.1:${SEARCH_PORT}/" "$SEARCH_PID" "$SEARCH_LOG"
wait_for_service "vLLM server 1" "http://127.0.0.1:${VLLM1_PORT}/v1/models" "$VLLM1_PID" "$VLLM1_LOG"
wait_for_service "vLLM server 2" "http://127.0.0.1:${VLLM2_PORT}/v1/models" "$VLLM2_PID" "$VLLM2_LOG"

if [[ "$RUN_MODE" == "smoke" ]]; then
  SMOKE_DIR=${OPENRESEARCHER_ROOT}/Tevatron/browsecomp-plus-smoke/data
  SMOKE_FILE=${SMOKE_DIR}/smoke-${SMOKE_SIZE}.parquet
  mkdir -p "$SMOKE_DIR"

  if [[ ! -f "$SMOKE_FILE" ]]; then
    SMOKE_SIZE="$SMOKE_SIZE" SMOKE_FILE="$SMOKE_FILE" "$VENV_PYTHON" - <<'PY'
import glob
import os

import pyarrow.dataset as ds
import pyarrow.parquet as pq

source_files = sorted(glob.glob("Tevatron/browsecomp-plus/data/*.parquet"))
size = int(os.environ["SMOKE_SIZE"])
output_file = os.environ["SMOKE_FILE"]
table = ds.dataset(source_files, format="parquet").head(size)
pq.write_table(table, output_file)
print(f"Created smoke dataset: {output_file} ({table.num_rows} rows)")
PY
  fi
  DATA_PATH=$SMOKE_FILE
else
  DATA_PATH="${OPENRESEARCHER_ROOT}/Tevatron/browsecomp-plus/data/*.parquet"
fi

VLLM_SERVER_URLS="http://127.0.0.1:${VLLM1_PORT}/v1,http://127.0.0.1:${VLLM2_PORT}/v1"

echo "Starting OpenResearcher agent..."
"$VENV_PYTHON" deploy_agent.py \
  --output_dir "$OUTPUT_DIR" \
  --model_name_or_path "$OPENRESEARCHER_MODEL_REF" \
  --search_url "http://127.0.0.1:${SEARCH_PORT}" \
  --dataset_name browsecomp_plus \
  --data_path "$DATA_PATH" \
  --browser_backend local \
  --reasoning_effort high \
  --vllm_server_url "$VLLM_SERVER_URLS" \
  --max_concurrency_per_worker "$MAX_CONCURRENCY"

echo "Agent run finished. Result files:"
find "$OUTPUT_DIR" -maxdepth 1 -name '*.jsonl' -type f -print
