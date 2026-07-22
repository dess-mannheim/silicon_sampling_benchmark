import argparse, logging
from group_level.qstn_setup import run_tier2_survey
from group_level.result_saving import save_tier2_results


def args():
    p = argparse.ArgumentParser(
        description="Run Tier 2 predictions with a local vLLM model."
    )
    p.add_argument("--model-id", required=True)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--enable-thinking", action="store_true", default=False)
    g.add_argument("--no-enable-thinking", action="store_false", dest="enable_thinking")
    p.add_argument("--reasoning-start-token", default="<think>")
    p.add_argument("--reasoning-end-token", default="</think>")
    p.add_argument("--max-model-len", type=int, default=5000)
    p.add_argument("--max-tokens", type=int, default=5000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def main():
    a = args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Starting Tier 2 run: %s", vars(a))
    results, metadata = run_tier2_survey(
        model_id=a.model_id,
        max_model_len=a.max_model_len,
        gpu_memory_utilization=a.gpu_memory_utilization,
        dtype=a.dtype,
        max_tokens=a.max_tokens,
        enable_thinking=a.enable_thinking,
        reasoning_start_token=a.reasoning_start_token,
        reasoning_end_token=a.reasoning_end_token,
    )
    for name, path in save_tier2_results(
        model_id=a.model_id, survey_results=results, prompt_metadata=metadata
    ).items():
        logging.info("Wrote %s: %s", name, path)


if __name__ == "__main__":
    main()
