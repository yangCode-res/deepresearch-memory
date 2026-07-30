#!/bin/bash
#SBATCH --job-name=cl-gism-bcplus-dense
#SBATCH --partition=gre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/file_storage01/home/juanliu/25_ymj/memory/memory/slurm-cl-gism-%j.out

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/file_storage01/home/juanliu/25_ymj/memory/memory}
export PROJECT_ROOT
export OPENRESEARCHER_ROOT=${OPENRESEARCHER_ROOT:-${PROJECT_ROOT}/vendor/openresearcher}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-cl_gism_dense}
export AGENT_ENTRYPOINT=${AGENT_ENTRYPOINT:-${PROJECT_ROOT}/integrations/openresearcher_memory_agent.py}
export CL_GISM_TRACE_DIR=${CL_GISM_TRACE_DIR:-${OPENRESEARCHER_ROOT}/results/memory_traces/${SLURM_JOB_ID:-manual}}
export CL_GISM_MEMORY_TOP_K=${CL_GISM_MEMORY_TOP_K:-4}
export CL_GISM_MEMORY_TEXT_LIMIT=${CL_GISM_MEMORY_TEXT_LIMIT:-1800}
export CL_GISM_CONTROLLER_TIMEOUT_SECONDS=${CL_GISM_CONTROLLER_TIMEOUT_SECONDS:-300}
export CL_GISM_CONTROLLER_MAX_TOKENS=${CL_GISM_CONTROLLER_MAX_TOKENS:-2048}

exec bash "${PROJECT_ROOT}/scripts/slurm/openresearcher_browsecomp_plus_dense.sbatch"
