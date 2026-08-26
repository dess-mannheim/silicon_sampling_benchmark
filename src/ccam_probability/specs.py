"""Categorical response specifications for the CCAM calibration battery."""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FAMILIES = {
    "belief_consensus": ("happening", "cause_recoded", "sci_consensus"),
    "concern_harm": (
        "worry", "harm_personally", "harm_US", "harm_dev_countries",
        "harm_future_gen", "harm_plants_animals", "when_harm_US",
    ),
    "climate_policy": (
        "reg_CO2_pollutant", "fund_research", "reduce_tax",
        "priority_cleanenergy", "priority", "generate_renewable", "teach_gw",
        "transition_economy",
    ),
    "engagement_exposure": ("discuss_GW", "hear_GW_media"),
}


@dataclass(frozen=True)
class CCAMProbabilitySpec:
    item_id: str
    codes: tuple[str, ...]
    labels: tuple[str, ...]
    family: str

    @property
    def keys(self) -> tuple[str, ...]:
        return self.codes

    @property
    def scores(self) -> tuple[float, ...]:
        """Unit-spaced ordinal scores; category labels remain authoritative."""
        if len(self.codes) == 1:
            return (0.0,)
        return tuple(index / (len(self.codes) - 1) for index in range(len(self.codes)))


def _literal_list(value: object, column: str, item_id: str) -> list[object]:
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"Invalid {column} for {item_id!r}: {value!r}") from error
    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"{column} for {item_id!r} must be a non-empty list")
    return parsed


def load_ccam_specs(root: Path) -> dict[str, CCAMProbabilitySpec]:
    questionnaire = pd.read_csv(root / "qstn_data" / "questionnaire_ccam.csv")
    expected_columns = [
        "questionnaire_item_id", "question_content", "answer_codes", "answer_texts",
        "likert_only_from_to_scale", "likert_n", "likert_start_idx", "scale_prompt_template",
    ]
    if list(questionnaire.columns) != expected_columns:
        raise ValueError("questionnaire_ccam.csv must use the same eight-column schema as questionnaire.csv")
    family_by_item = {
        item_id: family for family, item_ids in FAMILIES.items() for item_id in item_ids
    }
    specs: dict[str, CCAMProbabilitySpec] = {}
    for row in questionnaire.itertuples(index=False):
        item_id = str(row.questionnaire_item_id)
        codes = tuple(str(value) for value in _literal_list(row.answer_codes, "answer_codes", item_id))
        labels = tuple(str(value) for value in _literal_list(row.answer_texts, "answer_texts", item_id))
        if len(codes) != len(labels):
            raise ValueError(f"Mismatched answer codes and labels for {item_id!r}")
        if len(set(codes)) != len(codes):
            raise ValueError(f"Duplicate answer codes for {item_id!r}")
        if "-1" in codes:
            raise ValueError(f"Refused code -1 must not be prompted for {item_id!r}")
        if item_id not in family_by_item:
            raise ValueError(f"No calibration family assigned to {item_id!r}")
        specs[item_id] = CCAMProbabilitySpec(item_id, codes, labels, family_by_item[item_id])
    expected = set(family_by_item)
    if set(specs) != expected:
        raise ValueError(
            f"CCAM questionnaire mismatch; missing={sorted(expected-set(specs))}, "
            f"unexpected={sorted(set(specs)-expected)}"
        )
    return specs
