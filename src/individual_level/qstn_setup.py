from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.qstn_runtime import build_llm
from group_level.qstn_setup import load_study_inputs, repository_root
from qstn.prompt_builder import LLMPrompt
from qstn.survey_manager import conduct_survey_battery
from qstn.utilities import placeholder

from .sampling import assign_conditions, draw_personas, fit_population, load_population_source

SYSTEM_PROMPT = (
    "You will be given one respondent persona and a set of questions. Predict that "
    "person's exact answer to every question. ONLY respond with a JSON object mapping "
    "every question ID to its answer."
)
PROMPT = (
    "The survey respondent has already answered the following questions:\n{persona}\n\n"
    "Predict an individual, do not predict a group mean. Before answering the remaining questions, the respondent was instructed to read:\n\n{text}\n\n"
    "QUESTION_ID: QUESTION? ANSWER_OPTIONS\n{questions}"
)
DEMO_COLUMNS = ["gender", "age_band", "race", "education", "income", "party"]
MANIFEST_COLUMNS = ["profile_id", "condition", *DEMO_COLUMNS]


@dataclass(frozen=True)
class Persona:
    profile_id: str
    demographics: dict[str, str]


@dataclass(frozen=True)
class PromptMetadata:
    condition: str
    condition_variant: int
    persona: Persona

    def as_dict(self) -> dict[str, Any]:
        return {"condition": self.condition, "condition_variant": self.condition_variant,
                "persona_id": self.persona.profile_id, **self.persona.demographics}


def load_individual_config(path: Path | None = None, root: Path | None = None) -> dict[str, Any]:
    root = repository_root() if root is None else root
    path = root / "qstn_data" / "individual_level_config.json" if path is None else path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _path_from_config(config: dict[str, Any], root: Path, key: str, default: str) -> Path:
    value = config.get(key, default)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty path")
    path = Path(value)
    return path if path.is_absolute() else root / path


def generate_personas(config: dict[str, Any], moderators: dict[str, list[str]], root: Path) -> list[Persona]:
    n_individuals, seed = config.get("n_individuals"), config.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    source_path = _path_from_config(config, root, "population_distribution_file", "")
    population = fit_population(load_population_source(source_path), moderators)
    profiles = draw_personas(population, n_individuals, seed)
    return [Persona(f"synthetic_{index + 1}", profile) for index, profile in enumerate(profiles)]


def _load_manifest(path: Path, *, n_individuals: int, moderators: dict[str, list[str]], conditions: list[str]) -> list[tuple[Persona, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MANIFEST_COLUMNS:
            raise ValueError(f"{path} must have exactly these columns: {', '.join(MANIFEST_COLUMNS)}")
        rows = list(reader)
    if len(rows) != n_individuals:
        raise ValueError(f"{path} contains {len(rows)} rows; config requests n_individuals={n_individuals}")
    profile_ids = [row["profile_id"] for row in rows]
    if any(not profile_id for profile_id in profile_ids) or len(set(profile_ids)) != len(profile_ids):
        raise ValueError(f"{path} profile_id values must be non-empty and unique")
    assignments = []
    for row in rows:
        if row["condition"] not in conditions:
            raise ValueError(f"{path} contains unknown condition {row['condition']!r}")
        demographics = {name: row[name] for name in moderators}
        for name, levels in moderators.items():
            if demographics[name] not in levels:
                raise ValueError(f"{path} contains invalid {name} level {demographics[name]!r}")
        assignments.append((Persona(row["profile_id"], demographics), row["condition"]))
    return assignments


def _write_manifest(path: Path, assignments: list[tuple[Persona, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for persona, condition in assignments:
            writer.writerow({"profile_id": persona.profile_id, "condition": condition, **persona.demographics})


def load_or_create_persona_assignments(config: dict[str, Any], moderators: dict[str, list[str]], conditions: list[str], root: Path) -> list[tuple[Persona, str]]:
    n_individuals = config.get("n_individuals")
    seed = config.get("seed")
    if not isinstance(n_individuals, int) or isinstance(n_individuals, bool) or n_individuals <= 0:
        raise ValueError("n_individuals must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    manifest = _path_from_config(config, root, "persona_file", "personas/individual_level_personas.csv")
    if manifest.exists():
        return _load_manifest(manifest, n_individuals=n_individuals, moderators=moderators, conditions=conditions)
    personas = generate_personas(config, moderators, root)
    assignments = list(zip(personas, assign_conditions([p.demographics for p in personas], conditions, seed), strict=True))
    _write_manifest(manifest, assignments)
    return assignments


def format_persona_answers(persona: Persona) -> str:
    """Render the six modeled demographics as the respondent's survey answers."""
    demographics = persona.demographics
    questions = (
        ("What is your gender? (Male, Female, Other)", "gender"),
        ("What is your age group? (18-29, 30-44, 45-59, 60+)", "age_band"),
        (
            "Please select which race / ethnicity you most identify as. "
            "(White / Caucasian, Black / African American, Hispanic / Latino, "
            "Asian / Asian American, Other)",
            "race",
        ),
        (
            "What is the highest level of school that you have completed? "
            "(Less than high school, High school diploma / GED, Some college or "
            "Associate's degree, Bachelor's degree, Master's degree / Professional "
            "degree, Doctorate degree / Ph.D.)",
            "education",
        ),
        (
            "What is your total yearly family/household income before taxes? "
            "(Less than $30,000, $30,000 to $55,999, $56,000 to $99,999, "
            "$100,000 to $167,999, $168,000 or more)",
            "income",
        ),
        (
            "Generally speaking, do you usually think of yourself as a Republican, "
            "a Democrat, an Independent, or what? (Republican, Democrat, Independent, Other)",
            "party",
        ),
    )
    return "\n".join(f"Q: {question}\nA: {demographics[key]}" for question, key in questions)


def build_prompts(root: Path | None = None, config_path: Path | None = None):
    root = repository_root() if root is None else root
    conditions, moderators, questionnaire = load_study_inputs(root)
    if any(len(variants) != 1 for variants in conditions.values()):
        raise ValueError("Individual-level assignment requires exactly one text variant per condition.")
    assignments = load_or_create_persona_assignments(load_individual_config(config_path, root), moderators, list(conditions), root)
    prompts, metadata = [], []
    for persona, condition in assignments:
        prompt = LLMPrompt(questionnaire_name=f"{condition}__{persona.profile_id}", questionnaire_source=str(questionnaire), system_prompt=SYSTEM_PROMPT,
            prompt=PROMPT.format(persona=format_persona_answers(persona), text=conditions[condition][0], questions=placeholder.PROMPT_QUESTIONS))
        questions = list(prompt.get_questions())
        prompt.prepare_prompt(question_stem=[f"{q.item_id}: {placeholder.QUESTION_CONTENT} {placeholder.PROMPT_OPTIONS}" for q in questions],
                              answer_options={q.item_id: q.answer_options for q in questions})
        prompts.append(prompt)
        metadata.append(PromptMetadata(condition, 0, persona))
    return prompts, metadata


def run_tier1_survey(*, model_id, max_model_len, gpu_memory_utilization, dtype,
                      tensor_parallel_size, max_num_seqs, max_tokens, enforce_eager,
                      disable_custom_all_reduce, enable_thinking, reasoning_start_token,
                      reasoning_end_token, root=None, config_path=None):
    prompts, metadata = build_prompts(root, config_path)
    model = build_llm(model_id=model_id, max_model_len=max_model_len,
                      gpu_memory_utilization=gpu_memory_utilization, dtype=dtype,
                      tensor_parallel_size=tensor_parallel_size, max_num_seqs=max_num_seqs,
                      enforce_eager=enforce_eager, disable_custom_all_reduce=disable_custom_all_reduce)
    return conduct_survey_battery(model, prompts, max_tokens=max_tokens,
        chat_template_kwargs={"enable_thinking": enable_thinking}, reasoning_start_token=reasoning_start_token,
        reasoning_end_token=reasoning_end_token), metadata
