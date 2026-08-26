#!/bin/bash
#SBATCH --job-name=silicon_sampling_benchmark_t1_individual_qwen36
#SBATCH --output=slurm/logs/%j/%x.out
#SBATCH --error=slurm/logs/%j/%x.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --nodelist=compute2
#SBATCH --mail-type=END
#SBATCH --mail-user=maximilian.kreutner@uni-mannheim.de

set -euo pipefail

MODEL_ID="Qwen/Qwen3.6-27B"
REASONING_START_TOKEN="<think>"
REASONING_END_TOKEN="</think>"

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
