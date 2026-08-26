"""Battery prompt construction and execution for probability-based Tier-1 runs."""
from __future__ import annotations

import json
import math
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


SYSTEM_PROMPT = """You are simulating one exact survey respondent. Predict that respondent's uncertainty over their answer to EVERY question as probabilities over the required response ranges.

Return ONLY one valid JSON object. Do not include explanation, reasoning, Markdown, a code fence, or any text before or after the JSON.

The JSON object's keys must be exactly the question IDs supplied below, with every ID appearing exactly once. Each value must use the probability-object structure from the matching example below. Every probability must be a finite JSON number greater than or equal to 0, and the probabilities within each question must sum to exactly 1.

For example, one 0-100 item is written as:
{"trust_post":{"0-20":0.10,"21-40":0.20,"41-60":0.30,"61-80":0.25,"81-100":0.15}}
The $10 donation item is written as:
{"donation_ams":{"0-2":0.30,"3-4":0.25,"5-6":0.20,"7-8":0.15,"9-10":0.10}}
The binary newsletter item uses output key "0" for No and output key "1" for Yes. It is written as:
{"newsletter_signup":{"0":0.35,"1":0.65}}
Use the required keys literally: do not rename them, add keys, omit keys, use percentages, or give a single-point answer. Base the probability distribution on the stated persona and the text they read."""

PROMPT = """The survey respondent has already answered the following questions:
{persona}
\n
Before answering the remaining questions, the respondent was instructed to read:

{text}

For every question below, return the appropriate probability object shown in the system examples.
QUESTION_ID: QUESTION? ORIGINAL_ANSWER_OPTIONS
{questions}"""


@dataclass(frozen=True)
class ProbabilityPromptMetadata:
    condition: str
    condition_variant: int
    persona: Persona

    def as_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "condition_variant": self.condition_variant,
            "persona_id": self.persona.profile_id,
            **self.persona.demographics,
        }


def load_probability_config(path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root = repository_root() if root is None else root
    path = root / "qstn_data" / "individual_probability_config.json" if path is None else Path(path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    sampling_seed = config.get("sampling_seed")
    if not isinstance(sampling_seed, int) or isinstance(sampling_seed, bool):
        raise ValueError("sampling_seed must be an integer")
    softmax_temperature = config.get("softmax_temperature")
    if (
        isinstance(softmax_temperature, bool)
        or not isinstance(softmax_temperature, (int, float))
        or not math.isfinite(softmax_temperature)
        or softmax_temperature <= 0
    ):
        raise ValueError("softmax_temperature must be positive and finite")
    n_individuals = config.get("n_individuals")
    if not isinstance(n_individuals, int) or isinstance(n_individuals, bool) or n_individuals <= 0:
        raise ValueError("n_individuals must be a positive integer")
    return config


def load_probability_persona_assignments(config: dict[str, Any], moderators: dict[str, list[str]], conditions: list[str], root: Path):
    """Reuse the direct Tier-1 manifest; debug runs take its deterministic prefix."""
    source_config = load_individual_config(root=root)
    source_manifest = config.get("source_persona_file")
    if source_manifest is not None:
        source_config = {**source_config, "persona_file": source_manifest}
    assignments = load_or_create_persona_assignments(source_config, moderators, conditions, root)
    requested = config["n_individuals"]
    if requested > len(assignments):
        raise ValueError(f"n_individuals={requested} exceeds the shared manifest size {len(assignments)}")
    return assignments[:requested]


def build_probability_prompts(root: Path | None = None, config_path: Path | None = None):
    """Build one complete qstn battery prompt per probability-sampled persona."""
    root = repository_root() if root is None else root
    conditions, moderators, questionnaire = load_study_inputs(root)
    if any(len(variants) != 1 for variants in conditions.values()):
        raise ValueError("Individual-level assignment requires exactly one text variant per condition.")
    config = load_probability_config(config_path, root)
    assignments = load_probability_persona_assignments(config, moderators, list(conditions), root)
    prompts, metadata = [], []
    for persona, condition in assignments:
        prompt = LLMPrompt(
            questionnaire_name=f"probability__{condition}__{persona.profile_id}",
            questionnaire_source=str(questionnaire),
            system_prompt=SYSTEM_PROMPT,
            prompt=PROMPT.format(
                persona=format_persona_answers(persona),
                text=conditions[condition][0],
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
        metadata.append(ProbabilityPromptMetadata(condition, 0, persona))
    return prompts, metadata


def run_probability_tier1_survey(*, model_id, max_model_len, gpu_memory_utilization, dtype,
                                  tensor_parallel_size, max_num_seqs, max_tokens, enforce_eager,
                                  disable_custom_all_reduce, enable_thinking, reasoning_start_token,
                                  reasoning_end_token, root=None, config_path=None):
    prompts, metadata = build_probability_prompts(root, config_path)
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
    return conduct_survey_battery(
        model,
        prompts,
        max_tokens=max_tokens,
        chat_template_kwargs={"enable_thinking": enable_thinking},
        reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token,
    ), metadata
