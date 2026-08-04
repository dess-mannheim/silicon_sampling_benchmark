#!/bin/bash
#SBATCH --job-name=silicon_sampling_benchmark_t1_individual_qwen36
#SBATCH --output=slurm/logs/%A/%x_%a.out
#SBATCH --error=slurm/logs/%A/%x_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --array=0-1%1
#SBATCH --nodelist=compute2

set -euo pipefail

MODEL_IDS=(
    "Qwen/Qwen3.6-35B-A3B"
    "Qwen/Qwen3.6-27B"
)

REASONING_START_TOKENS=(
    "<think>"
    "<think>"
)

REASONING_END_TOKENS=(
    "</think>"
    "</think>"
)

if (( ${#MODEL_IDS[@]} != ${#REASONING_START_TOKENS[@]} ||
      ${#MODEL_IDS[@]} != ${#REASONING_END_TOKENS[@]} )); then
    echo "Model configuration arrays must have identical lengths." >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 ||
      SLURM_ARRAY_TASK_ID >= ${#MODEL_IDS[@]} )); then
    echo "Array index ${SLURM_ARRAY_TASK_ID} is outside the configured model list." >&2
    exit 1
fi

MODEL_ID="${MODEL_IDS[$SLURM_ARRAY_TASK_ID]}"
REASONING_START_TOKEN="${REASONING_START_TOKENS[$SLURM_ARRAY_TASK_ID]}"
REASONING_END_TOKEN="${REASONING_END_TOKENS[$SLURM_ARRAY_TASK_ID]}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0

ARGS=(
    --model-id "$MODEL_ID"
    --reasoning-start-token "$REASONING_START_TOKEN"
    --reasoning-end-token "$REASONING_END_TOKEN"
    --tensor-parallel-size 2
    --max-num-seqs 100
    --gpu-memory-utilization 0.95
    --enforce-eager
    --disable-custom-all-reduce
    --enable-thinking
)

uv run python src/main.py --experiment individual "${ARGS[@]}"
