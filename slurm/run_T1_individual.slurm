#!/bin/bash
#SBATCH --job-name=silicon_sampling_benchmark_t1_individual
#SBATCH --output=slurm/logs/%A/%x_%a.out
#SBATCH --error=slurm/logs/%A/%x_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --array=0-8%3
#SBATCH --nodelist=compute2

set -euo pipefail

# Keep these four arrays in matching order. Update the Slurm array range above
# whenever the number of configured models changes.
MODEL_IDS=(
    "google/gemma-4-E2B-it"
    "google/gemma-4-E4B-it"
    "google/gemma-4-31B-it"
    "google/gemma-4-26B-A4B-it"
    "Qwen/Qwen3.5-9B"
    "Qwen/Qwen3.5-4B"
    "Qwen/Qwen3.5-2B"
    "openai/gpt-oss-20b"
    "openai/gpt-oss-120b"
)

ENABLE_THINKING_BY_MODEL=(
    true
    true
    true
    true
    true
    true
    true
    true
    true
)

REASONING_START_TOKENS=(
    "<|channel>thought"
    "<|channel>thought"
    "<|channel>thought"
    "<|channel>thought"
    "<think>"
    "<think>"
    "<think>"
    "analysis"
    "analysis"
)

REASONING_END_TOKENS=(
    "<channel|>"
    "<channel|>"
    "<channel|>"
    "<channel|>"
    "</think>"
    "</think>"
    "</think>"
    "assistantfinal"
    "assistantfinal"
)

# Fail before model loading when the manually maintained arrays are inconsistent.
if (( ${#MODEL_IDS[@]} != ${#ENABLE_THINKING_BY_MODEL[@]} ||
      ${#MODEL_IDS[@]} != ${#REASONING_START_TOKENS[@]} ||
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
ENABLE_THINKING="${ENABLE_THINKING_BY_MODEL[$SLURM_ARRAY_TASK_ID]}"
REASONING_START_TOKEN="${REASONING_START_TOKENS[$SLURM_ARRAY_TASK_ID]}"
REASONING_END_TOKEN="${REASONING_END_TOKENS[$SLURM_ARRAY_TASK_ID]}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

ARGS=(
    --model-id "$MODEL_ID"
    --reasoning-start-token "$REASONING_START_TOKEN"
    --reasoning-end-token "$REASONING_END_TOKEN"
)

if [[ "$ENABLE_THINKING" == "true" ]]; then
    ARGS+=(--enable-thinking)
else
    ARGS+=(--no-enable-thinking)
fi

# Model-specific resource and quantization settings remain manual by design.
export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0

uv run python src/main.py --experiment individual "${ARGS[@]}"
