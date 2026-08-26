#!/bin/bash
#SBATCH --job-name=silicon_sampling_benchmark_t1_probability_debug_gemma
#SBATCH --output=slurm/logs/%j/%x.out
#SBATCH --error=slurm/logs/%j/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=compute2

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_ROOT"

export VLLM_USE_DEEP_GEMM=0
export VLLM_USE_FLASHINFER_SAMPLER=0

uv run python src/main.py \
    --experiment individual-probability \
    --individual-probability-config qstn_data/individual_probability_debug_config.json \
    --model-id "google/gemma-4-E2B-it" \
    --reasoning-start-token "<|channel>thought" \
    --reasoning-end-token "<channel|>" \
    --enable-thinking
