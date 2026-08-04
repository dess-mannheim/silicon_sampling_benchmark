from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from qstn.prompt_builder import LLMPrompt
from qstn.survey_manager import conduct_survey_battery
from qstn.utilities import placeholder
from common.qstn_runtime import build_llm

SYSTEM_PROMPT = (
    "You will be given a demographic and a set of questions. Your task is to predict the mean answer of this demographic for every question.\n"
    "ONLY respond in a JSON, that includes your prediction for every question in the following format:\n"
    "{\n"
    'trust_competence_1: "Your prediction e.g., 50"\n'
    "//All Other Questions\n"
    'behavior_donate: "Your prediction e.g., 50"\n'
    "}"
)

PROMPT = "Predict the mean answer for each question for a representative sample of 5000 people of this demographic. {demographic_name}: {demographic_value}.\nBefore the participants were asked these questions, they were instructed to read:\n\n{text}\n\n Format is QUESTION_ID: QUESTION? ANSWER_OPTIONS\n{questions}"


@dataclass(frozen=True)
class PromptMetadata:
    condition: str
    condition_variant: int
    moderator: str
    moderator_level: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "condition_variant": self.condition_variant,
            "moderator": self.moderator,
            "moderator_level": self.moderator_level,
        }


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_study_inputs(root: Path | None = None):
    root = repository_root() if root is None else root
    data = root / "qstn_data"
    with (data / "conditions.json").open() as f:
        conditions = json.load(f)
    with (data / "moderators.json").open() as f:
        moderators = json.load(f)
    return conditions, moderators, data / "questionnaire.csv"


def build_prompts(root: Path | None = None):
    conditions, moderators, questionnaire = load_study_inputs(root)
    prompts = []
    metadata = []
    for condition, variants in conditions.items():
        for moderator, levels in moderators.items():
            for variant, text in enumerate(variants):
                for level in levels:
                    prompt = LLMPrompt(
                        questionnaire_name=f"{condition}_{moderator}_{level}",
                        questionnaire_source=str(questionnaire),
                        system_prompt=SYSTEM_PROMPT,
                        prompt=PROMPT.format(
                            demographic_name=moderator,
                            demographic_value=level,
                            text=text,
                            questions=placeholder.PROMPT_QUESTIONS,
                        ),
                    )
                    questions = list(prompt.get_questions())
                    prompt.prepare_prompt(
                        question_stem=[
                            f"{q.item_id}: {placeholder.QUESTION_CONTENT} {placeholder.PROMPT_OPTIONS}"
                            for q in questions
                        ],
                        answer_options={q.item_id: q.answer_options for q in questions},
                    )
                    prompts.append(prompt)
                    metadata.append(
                        PromptMetadata(condition, variant, moderator, level)
                    )
    return prompts, metadata


def qwen36_compilation_config(model_id: str) -> dict[str, object] | None:
    """Return the narrower compile workaround for Qwen 3.6 models only."""
    if "Qwen3.6" not in model_id:
        return None
    return {
        "cudagraph_mode": "NONE",
        "inductor_compile_config": {
            "combo_kernels": False,
            "benchmark_combo_kernel": False,
        },
    }


def run_tier2_survey(
    *,
    model_id,
    max_model_len,
    gpu_memory_utilization,
    dtype,
    tensor_parallel_size,
    max_num_seqs,
    max_tokens,
    enforce_eager,
    disable_custom_all_reduce,
    enable_thinking,
    reasoning_start_token,
    reasoning_end_token,
    root=None,
):
    prompts, metadata = build_prompts(root)
    model = build_llm(
        model_id=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        dtype=dtype,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    return conduct_survey_battery(
        model,
        prompts,
        max_tokens=max_tokens,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
    ), metadata
