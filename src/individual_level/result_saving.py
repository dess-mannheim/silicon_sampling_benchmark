"""Persist individual responses in the benchmark's Tier-1 submission schema."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from group_level.result_saving import as_number, model_basename, parse_llm_json
from group_level.qstn_setup import repository_root
from .qstn_setup import PromptMetadata

COMPOSITES = {
    "trust_multidimensional": [f"trust_{dimension}_{i}" for dimension in ("competence", "integrity", "benevolence", "openness") for i in range(1, 4)],
    "policy_role_mean": [f"policy_role_{i}" for i in range(1, 5)],
    "inst_trust_mean": ["inst_trust_epa", "inst_trust_nasa", "inst_trust_noaa", "inst_trust_universities", "inst_trust_federal_gov"],
    "concern_mean": [f"concern_{i}" for i in range(1, 4)],
    "policy_specific_mean": [f"policy_specific_{i}" for i in range(1, 8)],
    "behavior_mean": [f"behavior_{name}" for name in ("meat", "transport", "solar", "fly", "talk", "donate")],
}
DEMO_COLUMNS = ["gender", "age_band", "race", "education", "income", "party"]


def current_run_id() -> str:
    return os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID") or "local"


def questionnaire_scales(root: Path) -> dict[str, tuple[float, float]]:
    frame = pd.read_csv(root / "qstn_data" / "questionnaire.csv")
    scales = {}
    for row in frame.itertuples(index=False):
        codes = json.loads(row.answer_codes)
        scales[row.questionnaire_item_id] = (float(codes[0]), float(codes[1]))
    return scales


def _newsletter(value: Any) -> float | None:
    """Normalize the questionnaire's canonical 0=No, 1=Yes coding."""
    if isinstance(value, str):
        value = value.strip().casefold()
        if value in {"yes", "true"}:
            return 1.0
        if value in {"no", "false"}:
            return 0.0
    value = as_number(value)
    # Accept the former displayed option 2=No for already-generated direct runs.
    if value == 2.0:
        return 0.0
    return float(value) if value in {0.0, 1.0} else None


def normalized_answer(item_id: str, value: Any, scales: dict[str, tuple[float, float]]) -> float | None:
    value = _newsletter(value) if item_id == "newsletter_signup" else as_number(value)
    if value is None or not math.isfinite(value):
        return None
    lower, upper = scales[item_id]
    return float(value) if lower <= value <= upper else None


def impute_answers(records: list[dict[str, Any]], item_ids: list[str], scales: dict[str, tuple[float, float]]) -> list[dict[str, float]]:
    """Impute each expected persona-condition-item cell from increasingly broad peers."""
    valid = [{item: normalized_answer(item, record["answers"].get(item), scales) for item in item_ids} for record in records]
    for index, record in enumerate(records):
        for item in item_ids:
            if valid[index][item] is not None:
                continue
            same_condition = [row[item] for row, peer in zip(valid, records, strict=True)
                              if peer["condition"] == record["condition"] and row[item] is not None]
            all_values = [row[item] for row in valid if row[item] is not None]
            midpoint = 0.5 if item == "newsletter_signup" else sum(scales[item]) / 2
            valid[index][item] = sum(same_condition) / len(same_condition) if same_condition else (sum(all_values) / len(all_values) if all_values else midpoint)
    return valid


def _tier1_row(record: dict[str, Any], answers: dict[str, float], item_ids: list[str], condition_index: int) -> dict[str, Any]:
    row = {"profile_id": record["persona_id"], "condition": record["condition"], **{key: record[key] for key in DEMO_COLUMNS}}
    row.update(answers)
    row["funding_perceptions"] = 100 - row["funding_perceptions"]
    for name, members in COMPOSITES.items():
        row[name] = sum(row[item] for item in members) / len(members)
    columns = ["profile_id", "condition", *DEMO_COLUMNS, "trust_multidimensional", *COMPOSITES["trust_multidimensional"], "trust_post", "distrust_post", "funding_perceptions", "policy_role_mean", "inst_trust_mean", "belief_post", "concern_mean", "policy_general", "policy_specific_mean", "behavior_mean", "donation_ams", "newsletter_signup"]
    return {column: row[column] for column in columns}


def save_tier1_results(*, model_id: str, survey_results: Iterable[Any], prompt_metadata: Iterable[PromptMetadata], root: Path | None = None, run_id: str | None = None) -> dict[str, Path]:
    root, run_id = (repository_root() if root is None else root), (current_run_id() if run_id is None else run_id)
    base = model_basename(model_id)
    paths = {"raw": root / "raw_results" / run_id / f"{base}_T1_primary_raw.json", "parsed": root / "results" / run_id / f"{base}_T1_primary_parsed.json", "prediction": root / "predictions" / f"{base}_T1_primary_v1.csv"}
    for directory in {path.parent for path in paths.values()}:
        directory.mkdir(parents=True, exist_ok=True)
    results, metadata = list(survey_results), list(prompt_metadata)
    if len(results) != len(metadata):
        raise ValueError("Survey result count does not match prompt metadata count.")
    raw, parsed = [], []
    for info, result in zip(metadata, results, strict=True):
        info_dict, first_answers = info.as_dict(), None
        for item_id, response in result.results.items():
            answers, repaired, error = parse_llm_json(response.llm_response)
            raw.append({**info_dict, "questionnaire_name": result.questionnaire.questionnaire_name, "item_id": item_id, "question": response.question, "reasoning": response.reasoning, "llm_response": response.llm_response, "logprobs": response.logprobs, "parsed_with_repair": repaired, "parse_error": error})
            if first_answers is None and answers is not None:
                first_answers = answers
        parsed.append({**info_dict, "answers": first_answers or {}})
    created_at = datetime.now(timezone.utc).isoformat()
    for name, records in (("raw", raw), ("parsed", parsed)):
        paths[name].write_text(json.dumps({"model": model_id, "run_id": run_id, "created_at": created_at, "records": records}, indent=2, default=str) + "\n", encoding="utf-8")
    scales = questionnaire_scales(root)
    item_ids = list(scales)
    answers = impute_answers(parsed, item_ids, scales)
    condition_order = {condition: i + 1 for i, condition in enumerate(dict.fromkeys(record["condition"] for record in parsed))}
    rows = [_tier1_row(record, answer, item_ids, condition_order[record["condition"]]) for record, answer in zip(parsed, answers, strict=True)]
    pd.DataFrame(rows).to_csv(paths["prediction"], index=False)
    return paths
