"""Validate and persist CCAM probability elicitation results without sampling."""
from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from json_repair import repair_json

from .qstn_setup import CCAMPromptMetadata
from .specs import CCAMProbabilitySpec, load_ccam_specs


PROBABILITY_TOLERANCE = 1e-6
RESPONSE_ERROR_KEY = "__response__"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def current_run_id() -> str:
    return os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"


def model_basename(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def output_paths(model_id: str, root: Path, run_id: str | None = None) -> dict[str, Path]:
    run_id = current_run_id() if run_id is None else run_id
    base = model_basename(model_id)
    return {
        "raw": root / "raw_results" / run_id / f"{base}_CCAM_probability_raw.json",
        "parsed": root / "results" / run_id / f"{base}_CCAM_probability_parsed.json",
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


def validate_probability_answers(
    answers: dict[str, Any] | None,
    specs: dict[str, CCAMProbabilitySpec],
) -> tuple[dict[str, list[float]], dict[str, str], dict[str, float]]:
    valid: dict[str, list[float]] = {}
    errors: dict[str, str] = {}
    normalization_totals: dict[str, float] = {}
    if answers is None:
        return valid, {RESPONSE_ERROR_KEY: "No JSON answer object was parsed."}, normalization_totals
    missing = sorted(set(specs) - set(answers))
    unexpected = sorted(set(answers) - set(specs))
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"Missing question IDs: {', '.join(missing)}")
        if unexpected:
            parts.append(f"Unexpected question IDs: {', '.join(unexpected)}")
        errors[RESPONSE_ERROR_KEY] = "; ".join(parts)
    for item_id, spec in specs.items():
        if item_id not in answers:
            continue
        value = answers[item_id]
        if not isinstance(value, dict):
            errors[item_id] = "Expected a probability object"
            continue
        missing_keys = [key for key in spec.keys if key not in value]
        unexpected_keys = sorted(set(value) - set(spec.keys))
        if missing_keys or unexpected_keys:
            errors[item_id] = (
                f"Probability keys mismatch (missing: {', '.join(missing_keys) or 'none'}; "
                f"unexpected: {', '.join(unexpected_keys) or 'none'})"
            )
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


def impute_probability_vectors(
    records: list[dict[str, Any]], specs: dict[str, CCAMProbabilitySpec]
) -> list[dict[str, dict[str, Any]]]:
    """Resolve missing vectors with the global model mean, then uniform fallback."""
    resolved: list[dict[str, dict[str, Any]]] = []
    global_means: dict[str, list[float] | None] = {}
    for item_id in specs:
        vectors = [record["probabilities"][item_id] for record in records if item_id in record["probabilities"]]
        global_means[item_id] = np.mean(vectors, axis=0).tolist() if vectors else None
    for record in records:
        item_vectors: dict[str, dict[str, Any]] = {}
        for item_id, spec in specs.items():
            vector = record["probabilities"].get(item_id)
            if vector is not None:
                source = "model"
            elif global_means[item_id] is not None:
                vector, source = global_means[item_id], "global_mean"
            else:
                vector, source = [1 / len(spec.keys)] * len(spec.keys), "uniform"
            item_vectors[item_id] = {"probabilities": vector, "source": source}
        resolved.append(item_vectors)
    return resolved


def _response_for_result(result: Any) -> Any:
    responses = list(result.results.values())
    if not responses:
        raise ValueError("Battery result does not contain an LLM response")
    return responses[0]


def save_ccam_probability_results(
    *, model_id: str, survey_results: Iterable[Any],
    prompt_metadata: Iterable[CCAMPromptMetadata], root: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    root = repository_root() if root is None else root
    effective_run_id = current_run_id() if run_id is None else run_id
    paths = output_paths(model_id, root, effective_run_id)
    for directory in {path.parent for path in paths.values()}:
        directory.mkdir(parents=True, exist_ok=True)
    results, metadata = list(survey_results), list(prompt_metadata)
    if len(results) != len(metadata):
        raise ValueError("Survey result count does not match prompt metadata count.")
    specs = load_ccam_specs(root)
    raw_records, parsed_records = [], []
    for info, result in zip(metadata, results, strict=True):
        response = _response_for_result(result)
        answers, repaired, parse_error = parse_probability_json(response.llm_response)
        probabilities, validation_errors, totals = validate_probability_answers(answers, specs)
        common = {
            **info.as_dict(),
            "questionnaire_name": result.questionnaire.questionnaire_name,
            "parsed_with_repair": repaired,
            "parse_error": parse_error,
            "normalization_totals": totals,
            "validation_errors": validation_errors,
        }
        parsed_records.append({**common, "probabilities": probabilities})
        raw_records.append({
            **common,
            "item_id": next(iter(result.results)),
            "question": response.question,
            "reasoning": response.reasoning,
            "llm_response": response.llm_response,
            "logprobs": response.logprobs,
        })
    resolved = impute_probability_vectors(parsed_records, specs)
    for record, vectors in zip(parsed_records, resolved, strict=True):
        record["resolved_probabilities"] = vectors
    validation = {
        "respondents": len(parsed_records),
        "complete_valid_responses": sum(not r["parse_error"] and not r["validation_errors"] for r in parsed_records),
        "strict_valid_responses": sum(not r["parsed_with_repair"] and not r["parse_error"] and not r["validation_errors"] and not r["normalization_totals"] for r in parsed_records),
        "repaired_json_responses": sum(r["parsed_with_repair"] for r in parsed_records),
        "normalized_responses": sum(bool(r["normalization_totals"]) for r in parsed_records),
        "normalized_item_vectors": sum(len(r["normalization_totals"]) for r in parsed_records),
        "responses_requiring_imputation": sum(bool(r["validation_errors"]) for r in parsed_records),
    }
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {"model": model_id, "run_id": effective_run_id,
               "created_at": created_at, "validation": validation}
    paths["raw"].write_text(json.dumps({**payload, "records": raw_records}, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    paths["parsed"].write_text(json.dumps({**payload, "records": parsed_records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths
