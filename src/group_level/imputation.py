"""Impute missing Tier 2 prediction cells from model-provided predictions."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Iterable

import pandas as pd

PREDICTION_COLUMNS = [
    "condition",
    "moderator",
    "moderator_level",
    "outcome",
    "mean",
]

MIDPOINTS = {
    "newsletter_signup": 0.5,
    "donation_ams": 5.0,
}


def native_midpoint(outcome: str) -> float:
    """Return the native-scale midpoint used only as a final fallback."""
    return MIDPOINTS.get(outcome, 50.0)


def load_prediction_grid(root: Path, outcomes: Iterable[str]) -> pd.DataFrame:
    """Return every expected condition x moderator level x outcome cell."""
    data = root / "qstn_data"
    with (data / "conditions.json").open(encoding="utf-8") as f:
        conditions = list(json.load(f))
    with (data / "moderators.json").open(encoding="utf-8") as f:
        moderators = json.load(f)

    rows = []
    for condition, (moderator, levels), outcome in product(
        conditions, moderators.items(), outcomes
    ):
        for level in levels:
            rows.append(
                {
                    "condition": condition,
                    "moderator": moderator,
                    "moderator_level": level,
                    "outcome": outcome,
                }
            )
    return pd.DataFrame(rows)


def _observed_values(values: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate model-provided cells and drop absent/non-finite means."""
    if values.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    observed = values.loc[:, PREDICTION_COLUMNS].copy()
    observed["mean"] = pd.to_numeric(observed["mean"], errors="coerce")
    observed = observed.dropna(subset=["mean"])
    observed = observed[observed["mean"].map(math.isfinite)]
    if observed.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    return observed.groupby(
        ["condition", "moderator", "moderator_level", "outcome"],
        as_index=False,
    )["mean"].mean()


def _lookup_mean(frame: pd.DataFrame, filters: dict[str, str]) -> float | None:
    mask = pd.Series(True, index=frame.index)
    for column, value in filters.items():
        mask &= frame[column] == value
    matches = frame.loc[mask, "mean"]
    if matches.empty:
        return None
    return float(matches.mean())


def impute_missing_predictions(
    values: pd.DataFrame,
    *,
    root: Path,
    outcomes: Iterable[str],
) -> pd.DataFrame:
    """Return a full moderator prediction grid, preserving provided values."""
    grid = load_prediction_grid(root, outcomes)
    observed = _observed_values(values)

    full = grid.merge(
        observed,
        on=["condition", "moderator", "moderator_level", "outcome"],
        how="left",
    )
    if observed.empty:
        full["mean"] = full["outcome"].map(native_midpoint)
        return full.loc[:, PREDICTION_COLUMNS].sort_values(PREDICTION_COLUMNS[:-1])

    missing = full["mean"].isna()
    for index, row in full.loc[missing].iterrows():
        fallbacks = [
            {
                "condition": row.condition,
                "outcome": row.outcome,
                "moderator": row.moderator,
            },
            {"condition": row.condition, "outcome": row.outcome},
            {
                "outcome": row.outcome,
                "moderator": row.moderator,
                "moderator_level": row.moderator_level,
            },
            {"outcome": row.outcome},
        ]
        value = None
        for filters in fallbacks:
            value = _lookup_mean(observed, filters)
            if value is not None:
                break
        full.at[index, "mean"] = native_midpoint(row.outcome) if value is None else value

    return full.loc[:, PREDICTION_COLUMNS].sort_values(PREDICTION_COLUMNS[:-1])
