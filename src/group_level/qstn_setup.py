from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from qstn.prompt_builder import LLMPrompt
from qstn.survey_manager import conduct_survey_battery
from qstn.utilities import placeholder
from vllm import LLM

SYSTEM_PROMPT = "You will be given a demographic and a set of questions. Your task is to predict the mean answer of this demographic for every question. ONLY respond in a JSON object with every question ID as a numeric key-value prediction."
PROMPT = "Predict the mean answer for each question for a representative sample of 5000 people of this demographic. {demographic_name}: {demographic_value}.\nBefore the participants were asked these questions, they were instructed to read:\n\n{text}\n\nQUESTION_ID: QUESTION? ANSWER_OPTIONS\n{questions}"


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


def run_tier2_survey(
    *,
    model_id,
    max_model_len,
    gpu_memory_utilization,
    dtype,
    max_tokens,
    enable_thinking,
    reasoning_start_token,
    reasoning_end_token,
    root=None,
):
    prompts, metadata = build_prompts(root)
    model = LLM(
        model=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
    )
    return conduct_survey_battery(
        model,
        prompts,
        max_tokens=max_tokens,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
    ), metadata
