"""Validate, sample, and persist probability-based Tier-1 predictions."""
from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from json_repair import repair_json

from .demographic_tilt import (
    CONSTRUCT_TEMPERATURES,
    DEMOGRAPHIC_TILT_RIDGE,
    DEMOGRAPHIC_TILTS,
    GLOBAL_TEMPERATURE,
    apply_ordinal_tilt,
    demographic_feature_vector,
    normalized_bin_scores,
)
from .qstn_setup import ProbabilityPromptMetadata
from .specs import ProbabilitySpec, load_probability_specs


COMPOSITES = {
    "trust_multidimensional": [f"trust_{dimension}_{i}" for dimension in ("competence", "integrity", "benevolence", "openness") for i in range(1, 4)],
    "policy_role_mean": [f"policy_role_{i}" for i in range(1, 5)],
    "inst_trust_mean": ["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa", "inst_trust_universities", "inst_trust_federal_gov"],
    "concern_mean": [f"concern_{i}" for i in range(1, 4)],
    "policy_specific_mean": [f"policy_specific_{i}" for i in range(1, 8)],
    "behavior_mean": [f"behavior_{name}" for name in ("meat", "transport", "solar", "fly", "talk", "donate")],
}
DEMO_COLUMNS = ["gender", "age_band", "race", "education", "income", "party"]
TIER1_COLUMNS = [
    "profile_id", "condition", *DEMO_COLUMNS, "trust_multidimensional", *COMPOSITES["trust_multidimensional"],
    "trust_post", "distrust_post", "funding_perceptions", "policy_role_mean", "inst_trust_mean",
    "belief_post", "concern_mean", "policy_general", "policy_specific_mean", "behavior_mean",
    "donation_ams", "newsletter_signup",
]
PROBABILITY_TOLERANCE = 1e-6
RESPONSE_ERROR_KEY = "__response__"
SOFTMAX_EPSILON = 1e-8
DEFAULT_SOFTMAX_TEMPERATURE = GLOBAL_TEMPERATURE
DEFAULT_CONSTRUCT_TEMPERATURES = CONSTRUCT_TEMPERATURES


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def current_run_id() -> str:
    return os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"


def model_basename(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def output_paths(model_id: str, root: Path, run_id: str | None = None) -> dict[str, Path]:
    base = model_basename(model_id)
    run_id = current_run_id() if run_id is None else run_id
    return {
        "raw": root / "raw_results" / run_id / f"{base}_T1_probability_raw.json",
        "parsed": root / "results" / run_id / f"{base}_T1_probability_parsed.json",
        "prediction": root / "predictions" / f"{base}_T1_probability_v1.csv",
        "prediction_softmax": root / "predictions" / f"{base}_T1_probability_softmax_v1.csv",
        "prediction_construct": root / "predictions" / f"{base}_T1_probability_construct_v1.csv",
        "prediction_final": root / "predictions" / f"{base}_T1_probability_construct_demographic_v1.csv",
    }


def parse_probability_json(response: str | None) -> tuple[dict[str, Any] | None, bool, str | None]:
    if not isinstance(response, str):
        return None, False, "Expected an LLM response string"
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


def validate_probability_answers(answers: dict[str, Any] | None, specs: dict[str, ProbabilitySpec]) -> tuple[dict[str, list[float]], dict[str, str], dict[str, float]]:
    """Return valid vectors, concise structural errors, and normalized totals.

    A truncated battery can still contain useful, valid early item vectors. Keep
    those vectors and report the missing/unexpected question IDs once under the
    response-level key, rather than copying one long error to every missing item.
    """
    valid: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    normalization_totals: dict[str, float] = {}
    if answers is None:
        return valid, {RESPONSE_ERROR_KEY: "No JSON answer object was parsed."}, normalization_totals

    expected_item_ids, actual_item_ids = set(specs), set(answers)
    missing_item_ids = sorted(expected_item_ids - actual_item_ids)
    unexpected_item_ids = sorted(actual_item_ids - expected_item_ids)
    if missing_item_ids or unexpected_item_ids:
        parts = []
        if missing_item_ids:
            parts.append(f"Missing question IDs: {', '.join(missing_item_ids)}")
        if unexpected_item_ids:
            parts.append(f"Unexpected question IDs: {', '.join(unexpected_item_ids)}")
        errors[RESPONSE_ERROR_KEY] = "; ".join(parts)
    for item_id, spec in specs.items():
        if item_id not in answers:
            continue
        value = answers[item_id]
        if not isinstance(value, dict):
            errors[item_id] = "Expected a probability object"
            continue
        missing_probability_keys = [key for key in spec.keys if key not in value]
        unexpected_probability_keys = sorted(set(value) - set(spec.keys))
        if missing_probability_keys or unexpected_probability_keys:
            parts = []
            if missing_probability_keys:
                parts.append(f"missing: {', '.join(missing_probability_keys)}")
            if unexpected_probability_keys:
                parts.append(f"unexpected: {', '.join(unexpected_probability_keys)}")
            errors[item_id] = f"Probability keys mismatch ({'; '.join(parts)})"
            continue
        values = [value[key] for key in spec.keys]
        if any(isinstance(number, bool) or not isinstance(number, (int, float)) for number in values):
            errors[item_id] = "Probabilities must be JSON numbers"
            continue
        vector = [float(number) for number in values]
        if not all(math.isfinite(number) and number >= 0 for number in vector):
            errors[item_id] = "Probabilities must be finite and non-negative"
            continue
        total = sum(vector)
        if total <= 0:
            errors[item_id] = "At least one probability must be positive"
            continue
        if not math.isclose(total, 1.0, abs_tol=PROBABILITY_TOLERANCE):
            vector = [number / total for number in vector]
            normalization_totals[item_id] = total
        valid[item_id] = vector
    return valid, errors, normalization_totals


def impute_probability_vectors(records: list[dict[str, Any]], specs: dict[str, ProbabilitySpec]) -> list[dict[str, dict[str, Any]]]:
    """Resolve every persona-item vector using condition, global, then uniform fallback."""
    resolved: list[dict[str, dict[str, Any]]] = []
    for record in records:
        item_vectors: dict[str, dict[str, Any]] = {}
        for item_id, spec in specs.items():
            vector = record["probabilities"].get(item_id)
            if vector is not None:
                item_vectors[item_id] = {"probabilities": vector, "source": "model"}
                continue
            same_condition = [
                peer["probabilities"][item_id] for peer in records
                if peer["condition"] == record["condition"] and item_id in peer["probabilities"]
            ]
            all_vectors = [peer["probabilities"][item_id] for peer in records if item_id in peer["probabilities"]]
            if same_condition:
                vector, source = np.mean(same_condition, axis=0).tolist(), "condition_mean"
            elif all_vectors:
                vector, source = np.mean(all_vectors, axis=0).tolist(), "global_mean"
            else:
                vector, source = [1 / len(spec.bins)] * len(spec.bins), "uniform"
            item_vectors[item_id] = {"probabilities": vector, "source": source}
        resolved.append(item_vectors)
    return resolved


def temperature_scale_probabilities(probabilities: Iterable[float], temperature: float) -> np.ndarray:
    """Apply standard log-probability temperature scaling to one vector."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    values = np.asarray(list(probabilities), dtype=float)
    if values.ndim != 1 or not len(values) or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite non-negative vector")
    total = values.sum()
    if total <= 0:
        raise ValueError("probabilities must have positive mass")
    normalized = values / total
    if temperature == 1.0:
        return normalized
    logits = np.log(normalized + SOFTMAX_EPSILON) / temperature
    logits -= logits.max()
    scaled = np.exp(logits)
    return scaled / scaled.sum()


def sample_answers(
    resolved: list[dict[str, dict[str, Any]]], specs: dict[str, ProbabilitySpec],
    sampling_seed: int,
    temperature: float | Mapping[str, float] = 1.0,
    demographic_records: list[dict[str, Any]] | None = None,
    demographic_tilts: Mapping[str, Iterable[float]] | None = None,
) -> list[dict[str, float]]:
    """Draw one discrete answer from each resolved distribution and its selected range."""
    if demographic_tilts is not None and (
        demographic_records is None or len(demographic_records) != len(resolved)
    ):
        raise ValueError(
            "Demographic records must align with resolved probabilities"
        )
    rng = np.random.default_rng(sampling_seed)
    sampled: list[dict[str, float]] = []
    for record_index, item_vectors in enumerate(resolved):
        answers: dict[str, float] = {}
        feature_values = (
            demographic_feature_vector(demographic_records[record_index])
            if demographic_tilts is not None
            else None
        )
        for item_id, spec in specs.items():
            probabilities = item_vectors[item_id]["probabilities"]
            item_temperature = (
                float(
                    temperature.get(
                        item_id,
                        temperature.get("*", DEFAULT_SOFTMAX_TEMPERATURE),
                    )
                )
                if isinstance(temperature, Mapping)
                else float(temperature)
            )
            if item_temperature != 1.0:
                probabilities = temperature_scale_probabilities(
                    probabilities, item_temperature
                )
            if demographic_tilts is not None and item_id in demographic_tilts:
                probabilities = apply_ordinal_tilt(
                    probabilities,
                    normalized_bin_scores(spec.bins),
                    feature_values,
                    demographic_tilts[item_id],
                )
            bin_index = int(rng.choice(len(spec.bins), p=probabilities))
            _, lower, upper = spec.bins[bin_index]
            answer = lower if lower == upper else int(rng.integers(lower, upper + 1))
            answers[item_id] = float(answer)
        sampled.append(answers)
    return sampled


def _tier1_row(record: dict[str, Any], answers: dict[str, float]) -> dict[str, Any]:
    row = {"profile_id": record["persona_id"], "condition": record["condition"], **{key: record[key] for key in DEMO_COLUMNS}}
    row.update(answers)
    row["funding_perceptions"] = 100 - row["funding_perceptions"]
    for name, members in COMPOSITES.items():
        row[name] = sum(row[item] for item in members) / len(members)
    return {column: row[column] for column in TIER1_COLUMNS}


def _prediction_frame(
    records: list[dict[str, Any]], resolved: list[dict[str, dict[str, Any]]],
    specs: dict[str, ProbabilitySpec], sampling_seed: int,
    temperature: float | Mapping[str, float],
    demographic_tilts: Mapping[str, Iterable[float]] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    sampled = sample_answers(
        resolved, specs, sampling_seed, temperature,
        demographic_records=records,
        demographic_tilts=demographic_tilts,
    )
    frame = pd.DataFrame([
        _tier1_row(record, answers)
        for record, answers in zip(records, sampled, strict=True)
    ])
    return frame, sampled


def save_probability_prediction_csvs_from_parsed(
    parsed_path: Path, *, root: Path | None = None,
    softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
) -> dict[str, Path]:
    """Recreate raw and calibrated Tier-1 CSVs without rerunning the model."""
    root = repository_root() if root is None else root
    with Path(parsed_path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Parsed probability JSON must contain a non-empty records list")
    try:
        resolved = [record["resolved_probabilities"] for record in records]
        model_id = payload["model"]
        run_id = str(payload["run_id"])
        sampling_seed = int(payload["sampling_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Parsed probability JSON is missing reproducibility metadata") from error
    specs = load_probability_specs(root)
    paths = output_paths(model_id, root, run_id)
    paths["prediction"].parent.mkdir(parents=True, exist_ok=True)
    raw_frame, _ = _prediction_frame(records, resolved, specs, sampling_seed, 1.0)
    softmax_frame, _ = _prediction_frame(records, resolved, specs, sampling_seed, softmax_temperature)
    construct_temperature_map = {
        **DEFAULT_CONSTRUCT_TEMPERATURES, "*": softmax_temperature
    }
    construct_frame, _ = _prediction_frame(
        records, resolved, specs, sampling_seed, construct_temperature_map
    )
    final_frame, _ = _prediction_frame(
        records, resolved, specs, sampling_seed, construct_temperature_map,
        DEMOGRAPHIC_TILTS,
    )
    raw_frame.to_csv(paths["prediction"], index=False)
    softmax_frame.to_csv(paths["prediction_softmax"], index=False)
    construct_frame.to_csv(paths["prediction_construct"], index=False)
    final_frame.to_csv(paths["prediction_final"], index=False)
    return {
        name: paths[name]
        for name in (
            "prediction", "prediction_softmax", "prediction_construct",
            "prediction_final",
        )
    }


def _response_for_result(result: Any) -> Any:
    responses = list(result.results.values())
    if not responses:
        raise ValueError("Battery result does not contain an LLM response")
    return responses[0]


def save_probability_tier1_results(*, model_id: str, survey_results: Iterable[Any], prompt_metadata: Iterable[ProbabilityPromptMetadata], config: dict[str, Any], root: Path | None = None, run_id: str | None = None) -> dict[str, Path]:
    """Persist probability artifacts and the sampled Tier-1 prediction CSV."""
    root = repository_root() if root is None else root
    run_id = current_run_id() if run_id is None else run_id
    sampling_seed = config["sampling_seed"]
    softmax_temperature = float(config["softmax_temperature"])
    specs = load_probability_specs(root)
    paths = output_paths(model_id, root, run_id)
    for directory in {path.parent for path in paths.values()}:
        directory.mkdir(parents=True, exist_ok=True)
    results, metadata = list(survey_results), list(prompt_metadata)
    if len(results) != len(metadata):
        raise ValueError("Survey result count does not match prompt metadata count.")

    raw_records, parsed_records = [], []
    for info, result in zip(metadata, results, strict=True):
        response = _response_for_result(result)
        answers, repaired, parse_error = parse_probability_json(response.llm_response)
        probabilities, validation_errors, normalization_totals = validate_probability_answers(answers, specs)
        record = {
            **info.as_dict(), "questionnaire_name": result.questionnaire.questionnaire_name,
            "probabilities": probabilities, "normalization_totals": normalization_totals,
            "validation_errors": validation_errors,
            "parsed_with_repair": repaired, "parse_error": parse_error,
        }
        parsed_records.append(record)
        raw_records.append({
            **info.as_dict(), "questionnaire_name": result.questionnaire.questionnaire_name,
            "item_id": next(iter(result.results)), "question": response.question,
            "reasoning": response.reasoning, "llm_response": response.llm_response,
            "logprobs": response.logprobs, "parsed_with_repair": repaired,
            "parse_error": parse_error, "normalization_totals": normalization_totals,
            "validation_errors": validation_errors,
        })

    resolved = impute_probability_vectors(parsed_records, specs)
    raw_frame, sampled = _prediction_frame(parsed_records, resolved, specs, sampling_seed, 1.0)
    softmax_frame, sampled_softmax = _prediction_frame(
        parsed_records, resolved, specs, sampling_seed, softmax_temperature
    )
    construct_temperature_map = {
        **DEFAULT_CONSTRUCT_TEMPERATURES, "*": softmax_temperature
    }
    construct_frame, sampled_construct = _prediction_frame(
        parsed_records, resolved, specs, sampling_seed,
        construct_temperature_map,
    )
    final_frame, sampled_final = _prediction_frame(
        parsed_records, resolved, specs, sampling_seed,
        construct_temperature_map, DEMOGRAPHIC_TILTS,
    )
    for (
        record, item_vectors, answers, softmax_answers,
        construct_answers, final_answers,
    ) in zip(
        parsed_records, resolved, sampled, sampled_softmax,
        sampled_construct, sampled_final, strict=True
    ):
        record["resolved_probabilities"] = item_vectors
        record["sampled_answers"] = answers
        record["sampled_answers_softmax"] = softmax_answers
        record["sampled_answers_construct"] = construct_answers
        record["sampled_answers_final"] = final_answers
    validation = {
        "respondents": len(parsed_records),
        "complete_valid_responses": sum(not record["parse_error"] and not record["validation_errors"] for record in parsed_records),
        "strict_valid_responses": sum(not record["parsed_with_repair"] and not record["parse_error"] and not record["validation_errors"] and not record["normalization_totals"] for record in parsed_records),
        "repaired_json_responses": sum(record["parsed_with_repair"] for record in parsed_records),
        "normalized_responses": sum(bool(record["normalization_totals"]) for record in parsed_records),
        "normalized_item_vectors": sum(len(record["normalization_totals"]) for record in parsed_records),
        "responses_requiring_imputation": sum(bool(record["validation_errors"]) for record in parsed_records),
    }
    created_at = datetime.now(timezone.utc).isoformat()
    common = {"model": model_id, "run_id": run_id, "created_at": created_at, "sampling_seed": sampling_seed,
              "softmax_temperature": softmax_temperature,
              "construct_temperatures": DEFAULT_CONSTRUCT_TEMPERATURES,
              "demographic_tilt_ridge": DEMOGRAPHIC_TILT_RIDGE,
              "validation": validation}
    paths["raw"].write_text(json.dumps({**common, "records": raw_records}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    paths["parsed"].write_text(json.dumps({**common, "records": parsed_records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw_frame.to_csv(paths["prediction"], index=False)
    softmax_frame.to_csv(paths["prediction_softmax"], index=False)
    construct_frame.to_csv(paths["prediction_construct"], index=False)
    final_frame.to_csv(paths["prediction_final"], index=False)
    return paths
