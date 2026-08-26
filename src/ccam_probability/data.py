"""Load CCAM ground truth and probability predictions for calibration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .specs import CCAMProbabilitySpec


CCAM_WAVES = tuple(range(22, 32))
WAVE_LABELS = {
    22: "Apr 2020", 23: "Dec 2020", 24: "Mar 2021", 25: "Sep 2021",
    26: "Apr 2022", 27: "Dec 2022", 28: "Apr 2023", 29: "Oct 2023",
    30: "Apr 2024", 31: "Dec 2024",
}
DEMOGRAPHICS = ("gender", "age_band", "race", "education", "income", "party")


@dataclass(frozen=True)
class ProbabilityDataset:
    personas: pd.DataFrame
    probabilities: dict[str, np.ndarray]

    def subset(self, mask: np.ndarray | pd.Series) -> "ProbabilityDataset":
        indices = np.asarray(mask, dtype=bool)
        return ProbabilityDataset(
            self.personas.loc[indices].reset_index(drop=True),
            {item: values[indices] for item, values in self.probabilities.items()},
        )


def _canonical_code(value: Any) -> str | None:
    if pd.isna(value):
        return None
    number = float(value)
    return str(int(number)) if number.is_integer() else str(number)


def harmonize_personas(personas: pd.DataFrame) -> pd.DataFrame:
    result = personas.copy()
    result["gender_h"] = result["gender"].where(result["gender"].isin(["Male", "Female"]), "Other")
    result["age_band_h"] = result["age_band"]
    result["race_h"] = result["race"].map({
        "White / Caucasian": "White", "Black / African American": "Black",
        "Hispanic / Latino": "Hispanic", "Asian / Asian American": "Other", "Other": "Other",
    })
    result["education_h"] = result["education"].map({
        "Less than high school": "Less than high school",
        "High school diploma / GED": "High school",
        "Some college or Associate's degree": "Some college",
        "Bachelor's degree": "Bachelor's degree or higher",
        "Master's degree / Professional degree": "Bachelor's degree or higher",
        "Doctorate degree / Ph.D.": "Bachelor's degree or higher",
    })
    result["income_h"] = result["income"].map({
        "Less than $30,000": "Less than $50,000",
        "$30,000 to $55,999": "Less than $50,000",
        "$56,000 to $99,999": "$50,000 to $99,999",
        "$100,000 to $167,999": "$100,000 or more",
        "$168,000 or more": "$100,000 or more",
    })
    result["party_h"] = result["party"]
    return result


def load_prediction_probabilities(
    path: Path, specs: dict[str, CCAMProbabilitySpec]
) -> ProbabilityDataset:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read parsed CCAM probabilities from {path}: {error}") from error
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Parsed CCAM JSON must contain a non-empty records list")
    rows: list[dict[str, Any]] = []
    matrices = {item: [] for item in specs}
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"CCAM prediction record {index} is not an object")
        row = {"persona_id": record.get("persona_id")}
        row.update({name: record.get(name) for name in DEMOGRAPHICS})
        if not row["persona_id"] or any(row[name] is None for name in DEMOGRAPHICS):
            raise ValueError(f"CCAM prediction record {index} has incomplete persona metadata")
        resolved = record.get("resolved_probabilities")
        if not isinstance(resolved, dict):
            raise ValueError(f"CCAM prediction record {index} has no resolved_probabilities")
        for item_id, spec in specs.items():
            try:
                vector = np.asarray(resolved[item_id]["probabilities"], dtype=float)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Record {index} has no valid vector for {item_id}") from error
            if vector.shape != (len(spec.codes),) or not np.all(np.isfinite(vector)) or np.any(vector < 0) or vector.sum() <= 0:
                raise ValueError(f"Record {index} has an invalid vector for {item_id}")
            matrices[item_id].append(vector / vector.sum())
        rows.append(row)
    personas = pd.DataFrame(rows)
    if personas.persona_id.duplicated().any():
        raise ValueError("CCAM prediction persona_id values must be unique")
    return ProbabilityDataset(
        harmonize_personas(personas),
        {item: np.vstack(vectors) for item, vectors in matrices.items()},
    )


def harmonize_ground_truth(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["gender_h"] = result["gender"].map({1: "Male", 2: "Female"})
    result["age_band_h"] = pd.cut(
        pd.to_numeric(result["age"], errors="coerce"),
        bins=[-np.inf, 29, 44, 59, np.inf],
        labels=["18-29", "30-44", "45-59", "60+"],
    ).astype("string")
    result["race_h"] = result["race"].map({1: "White", 2: "Black", 3: "Other", 4: "Hispanic", 5: "Other"})
    result["education_h"] = result["educ_category"].map({
        1: "Less than high school", 2: "High school", 3: "Some college",
        4: "Bachelor's degree or higher",
    })
    income = pd.to_numeric(result["income"], errors="coerce")
    result["income_h"] = np.select(
        [income.between(1, 11), income.between(12, 15), income.between(16, 21)],
        ["Less than $50,000", "$50,000 to $99,999", "$100,000 or more"],
        default=None,
    )
    result["party_h"] = result["party"].map({
        1: "Republican", 2: "Democrat", 3: "Independent", 4: "Other", 5: "Other",
    })
    return result


def load_ccam_ground_truth(
    path: Path,
    specs: dict[str, CCAMProbabilitySpec],
    waves: tuple[int, ...] = CCAM_WAVES,
) -> pd.DataFrame:
    try:
        import pyreadstat
    except ImportError as error:
        raise RuntimeError("Reading CCAM ground truth requires pyreadstat") from error
    needed = [
        "case_ID", "wave", "weight_wave", "weight_aggregate",
        "gender", "age", "educ_category", "income", "race", "party", *specs,
    ]
    data, _metadata = pyreadstat.read_sav(str(path), usecols=needed, apply_value_formats=False)
    data["wave"] = pd.to_numeric(data["wave"], errors="coerce").astype("Int64")
    data = data.loc[data.wave.isin(waves)].copy()
    if data.empty:
        raise ValueError(f"No CCAM records found for waves {waves}")
    for item_id, spec in specs.items():
        valid = set(spec.codes)
        data[item_id] = pd.array(
            [code if code in valid else None for code in map(_canonical_code, data[item_id])],
            dtype="string",
        )
    for weight in ("weight_wave", "weight_aggregate"):
        data[weight] = pd.to_numeric(data[weight], errors="coerce")
        if data[weight].isna().any() or (data[weight] <= 0).any():
            raise ValueError(f"CCAM {weight} must be positive and complete in selected waves")
    data["weight_wave_norm"] = data["weight_wave"] / data.groupby("wave")["weight_wave"].transform("sum")
    data["weight_aggregate_norm"] = data["weight_aggregate"] / data["weight_aggregate"].sum()
    data["wave_label"] = data["wave"].map(WAVE_LABELS)
    return harmonize_ground_truth(data).reset_index(drop=True)
