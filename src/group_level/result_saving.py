"""Parse, aggregate, and persist Tier 2 model predictions."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from json_repair import repair_json

from .imputation import impute_missing_predictions
from .qstn_setup import PromptMetadata, repository_root

OUTCOME_COMPONENTS: dict[str, list[str]] = {
    "trust_multidimensional": [
        "trust_competence_1",
        "trust_competence_2",
        "trust_competence_3",
        "trust_integrity_1",
        "trust_integrity_2",
        "trust_integrity_3",
        "trust_benevolence_1",
        "trust_benevolence_2",
        "trust_benevolence_3",
        "trust_openness_1",
        "trust_openness_2",
        "trust_openness_3",
    ],
    "inst_trust_mean": [
        "inst_trust_epa",
        "inst_trust_nasa",
        "inst_trust_noaa",
        "inst_trust_universities",
        "inst_trust_federal_gov",
    ],
    "policy_role_mean": [
        "policy_role_1",
        "policy_role_2",
        "policy_role_3",
        "policy_role_4",
    ],
    "concern_mean": ["concern_1", "concern_2", "concern_3"],
    "policy_specific_mean": [f"policy_specific_{index}" for index in range(1, 8)],
    "behavior_mean": [
        "behavior_meat",
        "behavior_transport",
        "behavior_solar",
        "behavior_fly",
        "behavior_talk",
        "behavior_donate",
    ],
}

OUTCOMES = [
    *OUTCOME_COMPONENTS,
    "trust_post",
    "distrust_post",
    "donation_ams",
    "newsletter_signup",
    "funding_perceptions",
    "belief_post",
    "policy_general",
]


def model_basename(model_id: str) -> str:
    """Convert a Hugging Face model identifier into a safe filename prefix."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def parse_llm_json(response: str) -> tuple[dict[str, Any] | None, bool, str | None]:
    """Return parsed JSON, whether repair was needed, and a parse error."""
    try:
        parsed = json.loads(response)
        repaired = False
    except json.JSONDecodeError:
        try:
            parsed = repair_json(response, return_objects=True)
            repaired = True
        except Exception as error:
            return None, False, f"{type(error).__name__}: {error}"

    if not isinstance(parsed, dict):
        return None, repaired, f"Expected a JSON object, got {type(parsed).__name__}"
    return parsed, repaired, None


def as_number(value: Any) -> float | None:
    """Extract a numeric prediction from an API value or a textual response."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group())
    return None


def newsletter_as_number(value: Any) -> float | None:
    """Convert newsletter labels to the benchmark's 0/1 coding."""
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"yes", "true", "1"}:
            return 1.0
        if normalized in {"no", "false", "0", "2"}:
            return 0.0
    return as_number(value)


def mean_of_answers(answers: dict[str, Any], item_ids: list[str]) -> float | None:
    """Return a composite only when every constituent item has a numeric value."""
    values = [as_number(answers.get(item_id)) for item_id in item_ids]
    return (
        sum(values) / len(values)
        if all(value is not None for value in values)
        else None
    )


def outcome_values(answers: dict[str, Any]) -> dict[str, float]:
    """Construct the benchmark's 13 outcomes from item-level answers."""
    funding_response = as_number(answers.get("funding_perceptions"))
    outcomes = {
        outcome: mean_of_answers(answers, item_ids)
        for outcome, item_ids in OUTCOME_COMPONENTS.items()
    }
    outcomes.update(
        {
            "trust_post": as_number(answers.get("trust_post")),
            "distrust_post": as_number(answers.get("distrust_post")),
            "donation_ams": as_number(answers.get("donation_ams")),
            "newsletter_signup": newsletter_as_number(answers.get("newsletter_signup")),
            "funding_perceptions": (
                100 - funding_response if funding_response is not None else None
            ),
            "belief_post": as_number(answers.get("belief_post")),
            "policy_general": as_number(answers.get("policy_general")),
        }
    )
    return {outcome: value for outcome, value in outcomes.items() if value is not None}


def current_run_id() -> str:
    """Return the Slurm run id used for per-run JSON artifacts."""
    return (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or "local"
    )


def output_paths(model_id: str, root: Path, run_id: str | None = None) -> dict[str, Path]:
    """Return model-specific paths for raw, parsed, main, and moderator outputs."""
    basename = model_basename(model_id)
    run_id = current_run_id() if run_id is None else run_id
    return {
        "raw": root / "raw_results" / run_id / f"{basename}_T2_primary_raw.json",
        "parsed": root / "results" / run_id / f"{basename}_T2_primary_parsed.json",
        "main": root / "predictions" / f"{basename}_T2_primary_main.csv",
        "moderator": root / "predictions" / f"{basename}_T2_primary_moderator.csv",
    }


def save_tier2_results(
    *,
    model_id: str,
    survey_results: Iterable[Any],
    prompt_metadata: Iterable[PromptMetadata],
    root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    """Persist Tier 2 outputs for one model and return their paths."""
    root = repository_root() if root is None else root
    run_id = current_run_id() if run_id is None else run_id
    paths = output_paths(model_id, root, run_id)
    for directory in {path.parent for path in paths.values()}:
        directory.mkdir(parents=True, exist_ok=True)

    results = list(survey_results)
    metadata = list(prompt_metadata)
    if len(results) != len(metadata):
        raise ValueError("Survey result count does not match prompt metadata count.")

    raw_records: list[dict[str, Any]] = []
    parsed_records: list[dict[str, Any]] = []
    for prompt_info, inference_result in zip(metadata, results, strict=True):
        for item_id, response in inference_result.results.items():
            answers, repaired, parse_error = parse_llm_json(response.llm_response)
            raw_records.append(
                {
                    **prompt_info.as_dict(),
                    "questionnaire_name": inference_result.questionnaire.questionnaire_name,
                    "item_id": item_id,
                    "question": response.question,
                    "reasoning": response.reasoning,
                    "llm_response": response.llm_response,
                    "logprobs": response.logprobs,
                    "parsed_with_repair": repaired,
                    "parse_error": parse_error,
                }
            )
            if answers is not None:
                parsed_records.append({**prompt_info.as_dict(), "answers": answers})

    timestamp = datetime.now(timezone.utc).isoformat()
    paths["raw"].write_text(
        json.dumps(
            {
                "model": model_id,
                "run_id": run_id,
                "created_at": timestamp,
                "records": raw_records,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["parsed"].write_text(
        json.dumps(
            {
                "model": model_id,
                "run_id": run_id,
                "created_at": timestamp,
                "records": parsed_records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    prediction_rows = []
    for record in parsed_records:
        cell = {
            key: record[key] for key in ("condition", "moderator", "moderator_level")
        }
        for outcome, mean in outcome_values(record["answers"]).items():
            prediction_rows.append({**cell, "outcome": outcome, "mean": mean})

    values = pd.DataFrame(prediction_rows)
    moderator = impute_missing_predictions(values, root=root, outcomes=OUTCOMES)
    main = (
        moderator.groupby(["condition", "outcome"], as_index=False)["mean"]
        .mean()
        .sort_values(["condition", "outcome"])
    )

    main.to_csv(paths["main"], index=False)
    moderator.to_csv(paths["moderator"], index=False)
    return paths
