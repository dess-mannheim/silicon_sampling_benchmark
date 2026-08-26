"""Prompt construction and execution for the CCAM probability battery."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.qstn_runtime import build_llm
from group_level.qstn_setup import load_study_inputs, repository_root
from individual_level.qstn_setup import (
    Persona,
    format_persona_answers,
    load_individual_config,
    load_or_create_persona_assignments,
)
from qstn.prompt_builder import LLMPrompt
from qstn.survey_manager import conduct_survey_battery
from qstn.utilities import placeholder


SYSTEM_PROMPT = """You are simulating one exact survey respondent. Predict that respondent's uncertainty over their answer to EVERY question as probabilities over the supplied response options.

Return ONLY one valid JSON object. Do not include explanation, reasoning, Markdown, a code fence, or any text before or after the JSON.

The JSON object's keys must be exactly the question IDs supplied below, with every ID appearing exactly once. Each value must be an object whose keys are exactly the numeric response codes printed for that question. Every probability must be a finite JSON number greater than or equal to 0, and the probabilities within each question must sum to exactly 1.

For example, a four-option item with response codes 1 through 4 is written as:
{"worry":{"1":0.10,"2":0.20,"3":0.40,"4":0.30}}

Use every required question ID and response code literally. Do not rename them, add keys, omit keys, use percentages, or give a single-point answer. Base each probability distribution on the stated persona."""

PROMPT = """The survey respondent has already answered the following questions:
{persona}

For every question below, return a probability object over its original answer options.
QUESTION_ID: QUESTION? ORIGINAL_ANSWER_OPTIONS
{questions}"""


@dataclass(frozen=True)
class CCAMPromptMetadata:
    persona: Persona

    def as_dict(self) -> dict[str, Any]:
        return {"persona_id": self.persona.profile_id, **self.persona.demographics}


def load_ccam_config(path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root = repository_root() if root is None else root
    path = root / "qstn_data" / "ccam_probability_config.json" if path is None else Path(path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    n_individuals = config.get("n_individuals")
    if not isinstance(n_individuals, int) or isinstance(n_individuals, bool) or n_individuals <= 0:
        raise ValueError("n_individuals must be a positive integer")
    source = config.get("source_persona_file")
    if not isinstance(source, str) or not source:
        raise ValueError("source_persona_file must be a non-empty path")
    return config


def load_ccam_personas(config: dict[str, Any], root: Path) -> list[Persona]:
    conditions, moderators, _ = load_study_inputs(root)
    source_config = {
        **load_individual_config(root=root),
        "n_individuals": config["n_individuals"],
        "persona_file": config["source_persona_file"],
    }
    assignments = load_or_create_persona_assignments(
        source_config, moderators, list(conditions), root
    )
    return [persona for persona, _condition in assignments]


def build_ccam_prompts(root: Path | None = None, config_path: Path | None = None):
    root = repository_root() if root is None else root
    config = load_ccam_config(config_path, root)
    questionnaire = root / "qstn_data" / "questionnaire_ccam.csv"
    prompts, metadata = [], []
    for persona in load_ccam_personas(config, root):
        prompt = LLMPrompt(
            questionnaire_name=f"ccam_probability__{persona.profile_id}",
            questionnaire_source=str(questionnaire),
            system_prompt=SYSTEM_PROMPT,
            prompt=PROMPT.format(
                persona=format_persona_answers(persona),
                questions=placeholder.PROMPT_QUESTIONS,
            ),
        )
        questions = list(prompt.get_questions())
        prompt.prepare_prompt(
            question_stem=[
                f"{question.item_id}: {placeholder.QUESTION_CONTENT} {placeholder.PROMPT_OPTIONS}"
                for question in questions
            ],
            answer_options={question.item_id: question.answer_options for question in questions},
        )
        prompts.append(prompt)
        metadata.append(CCAMPromptMetadata(persona))
    return prompts, metadata


def run_ccam_probability_survey(
    *, model_id, max_model_len, gpu_memory_utilization, dtype,
    tensor_parallel_size, max_num_seqs, max_tokens, enforce_eager,
    disable_custom_all_reduce, enable_thinking, reasoning_start_token,
    reasoning_end_token, root=None, config_path=None,
):
    prompts, metadata = build_ccam_prompts(root, config_path)
    model = build_llm(
        model_id=model_id,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_num_seqs=max_num_seqs,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=disable_custom_all_reduce,
    )
    results = conduct_survey_battery(
        model,
        prompts,
        max_tokens=max_tokens,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
    )
    return results, metadata
