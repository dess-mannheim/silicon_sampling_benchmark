import argparse
import logging

from ccam_probability.qstn_setup import run_ccam_probability_survey
from ccam_probability.result_saving import save_ccam_probability_results
from group_level.qstn_setup import run_tier2_survey
from group_level.result_saving import save_tier2_results
from individual_level.qstn_setup import run_tier1_survey
from individual_level.result_saving import save_tier1_results
from individual_probability.qstn_setup import load_probability_config, run_probability_tier1_survey
from individual_probability.result_saving import save_probability_tier1_results


def args():
    p = argparse.ArgumentParser(description="Run benchmark predictions with a local vLLM model.")
    p.add_argument("--model-id", required=True)
    p.add_argument(
        "--experiment",
        choices=("group", "individual", "individual-probability", "ccam-probability"),
        default="group",
    )
    p.add_argument("--individual-config", default=None)
    p.add_argument("--individual-probability-config", default=None)
    p.add_argument("--ccam-probability-config", default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--enable-thinking", action="store_true", default=False)
    g.add_argument("--no-enable-thinking", action="store_false", dest="enable_thinking")
    p.add_argument("--reasoning-start-token", default="<think>")
    p.add_argument("--reasoning-end-token", default="</think>")
    p.add_argument("--max-model-len", type=int, default=15000)
    p.add_argument("--max-tokens", type=int, default=15000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--max-num-seqs", type=int, default=None)
    h = p.add_mutually_exclusive_group()
    h.add_argument("--enforce-eager", action="store_true", default=False)
    h.add_argument("--no-enforce-eager", action="store_false", dest="enforce_eager")
    r = p.add_mutually_exclusive_group()
    r.add_argument("--disable-custom-all-reduce", action="store_true", default=False)
    r.add_argument("--no-disable-custom-all-reduce", action="store_false", dest="disable_custom_all_reduce")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def main():
    a = args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Starting %s-level run: %s", a.experiment, vars(a))
    common = dict(model_id=a.model_id, max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization, tensor_parallel_size=a.tensor_parallel_size, max_num_seqs=a.max_num_seqs, dtype=a.dtype, max_tokens=a.max_tokens, enforce_eager=a.enforce_eager, disable_custom_all_reduce=a.disable_custom_all_reduce, enable_thinking=a.enable_thinking, reasoning_start_token=a.reasoning_start_token, reasoning_end_token=a.reasoning_end_token)
    if a.experiment == "individual":
        results, metadata = run_tier1_survey(**common, config_path=a.individual_config)
        paths = save_tier1_results(model_id=a.model_id, survey_results=results, prompt_metadata=metadata)
    elif a.experiment == "individual-probability":
        config = load_probability_config(a.individual_probability_config)
        results, metadata = run_probability_tier1_survey(
            **common, config_path=a.individual_probability_config
        )
        paths = save_probability_tier1_results(
            model_id=a.model_id,
            survey_results=results,
            prompt_metadata=metadata,
            config=config,
        )
    elif a.experiment == "ccam-probability":
        results, metadata = run_ccam_probability_survey(
            **common, config_path=a.ccam_probability_config
        )
        paths = save_ccam_probability_results(
            model_id=a.model_id,
            survey_results=results,
            prompt_metadata=metadata,
        )
    else:
        results, metadata = run_tier2_survey(**common)
        paths = save_tier2_results(model_id=a.model_id, survey_results=results, prompt_metadata=metadata)
    for name, path in paths.items():
        logging.info("Wrote %s: %s", name, path)


if __name__ == "__main__":
    main()
