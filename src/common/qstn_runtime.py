"""vLLM construction shared by group- and individual-level experiments."""
from __future__ import annotations

from vllm import LLM


def qwen36_compilation_config(model_id: str) -> dict[str, object] | None:
    if "Qwen3.6" not in model_id:
        return None
    return {"cudagraph_mode": "NONE", "inductor_compile_config": {
        "combo_kernels": False, "benchmark_combo_kernel": False}}


def build_llm(*, model_id, max_model_len, gpu_memory_utilization, dtype,
              tensor_parallel_size, max_num_seqs, enforce_eager,
              disable_custom_all_reduce):
    """Create the benchmark's consistently configured vLLM instance."""
    kwargs = {
        "model": model_id,
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": dtype,
        "enforce_eager": enforce_eager,
        "disable_custom_all_reduce": disable_custom_all_reduce,
    }
    if max_num_seqs is not None:
        kwargs["max_num_seqs"] = max_num_seqs
    compilation_config = qwen36_compilation_config(model_id)
    if compilation_config is not None:
        kwargs["compilation_config"] = compilation_config
    return LLM(
        **kwargs,
        attention_config={"backend": "TRITON_ATTN"},
        moe_backend="triton",
        linear_backend="triton",
        gdn_prefill_backend="triton",
        kernel_config={
            "enable_flashinfer_autotune": False,
            "enable_cutedsl_warmup": False,
        },
    )
