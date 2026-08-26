"""Regularized ordinal demographic calibration for CCAM probability vectors."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data import ProbabilityDataset
from .specs import CCAMProbabilitySpec


HARMONIZED_LEVELS = {
    "gender_h": ("Male", "Female", "Other"),
    "age_band_h": ("18-29", "30-44", "45-59", "60+"),
    "race_h": ("White", "Black", "Hispanic", "Other"),
    "education_h": (
        "Less than high school", "High school", "Some college",
        "Bachelor's degree or higher",
    ),
    "income_h": (
        "Less than $50,000", "$50,000 to $99,999", "$100,000 or more",
    ),
    "party_h": ("Republican", "Democrat", "Independent", "Other"),
}
REFERENCE_LEVELS = {
    "gender_h": "Male",
    "age_band_h": "18-29",
    "race_h": "White",
    "education_h": "Less than high school",
    "income_h": "Less than $50,000",
    "party_h": "Republican",
}
FEATURE_DEFINITIONS = tuple(
    (column, level)
    for column, levels in HARMONIZED_LEVELS.items()
    for level in levels
    if level != REFERENCE_LEVELS[column]
)
EPSILON = 1e-8


def demographic_feature_matrix(personas: pd.DataFrame) -> np.ndarray:
    """Encode additive demographic main effects using fixed reference levels."""
    missing = sorted({column for column, _ in FEATURE_DEFINITIONS} - set(personas))
    if missing:
        raise ValueError(f"Missing harmonized demographic columns: {missing}")
    return np.column_stack([
        personas[column].astype(str).eq(level).to_numpy(float)
        for column, level in FEATURE_DEFINITIONS
    ])


def demographic_group_masks(personas: pd.DataFrame) -> dict[str, np.ndarray]:
    masks = {"overall": np.ones(len(personas), dtype=bool)}
    for column, levels in HARMONIZED_LEVELS.items():
        for level in levels:
            mask = personas[column].astype(str).eq(level).to_numpy()
            if mask.any():
                masks[f"{column}={level}"] = mask
    return masks


def _weighted_category_shares(
    data: pd.DataFrame,
    item_id: str,
    spec: CCAMProbabilitySpec,
    weight_column: str,
) -> np.ndarray | None:
    valid = data[item_id].notna() & data[weight_column].notna()
    if not valid.any():
        return None
    values = data.loc[valid, item_id]
    weights = data.loc[valid, weight_column].to_numpy(float, copy=True)
    valid_weight = np.isfinite(weights) & (weights > 0)
    if not valid_weight.any():
        return None
    values = values.iloc[np.flatnonzero(valid_weight)]
    weights = weights[valid_weight]
    weights /= weights.sum()
    return np.array([
        weights[values.eq(code).to_numpy()].sum()
        for code in spec.codes
    ])


def demographic_target_cells(
    ground_truth: pd.DataFrame,
    specs: Mapping[str, CCAMProbabilitySpec],
    waves: Iterable[int],
    *,
    min_group_n: int = 30,
) -> pd.DataFrame:
    """Average survey-weighted item/group targets equally across selected waves."""
    rows: list[dict[str, Any]] = []
    for wave in sorted(set(int(value) for value in waves)):
        wave_data = ground_truth.loc[ground_truth.wave.eq(wave)]
        for item_id, spec in specs.items():
            overall = _weighted_category_shares(
                wave_data, item_id, spec, "weight_wave"
            )
            if overall is not None:
                rows.append({
                    "wave": wave,
                    "item_id": item_id,
                    "group_key": "overall",
                    "is_overall": True,
                    "shares": overall,
                })
            for column, levels in HARMONIZED_LEVELS.items():
                for level in levels:
                    group = wave_data.loc[wave_data[column].astype(str).eq(level)]
                    if int(group[item_id].notna().sum()) < min_group_n:
                        continue
                    shares = _weighted_category_shares(
                        group, item_id, spec, "weight_wave"
                    )
                    if shares is not None:
                        rows.append({
                            "wave": wave,
                            "item_id": item_id,
                            "group_key": f"{column}={level}",
                            "is_overall": False,
                            "shares": shares,
                        })
    if not rows:
        raise ValueError("No eligible demographic calibration cells")
    result = []
    frame = pd.DataFrame(rows)
    for (item_id, group_key, is_overall), group in frame.groupby(
        ["item_id", "group_key", "is_overall"], sort=False
    ):
        result.append({
            "item_id": item_id,
            "group_key": group_key,
            "is_overall": bool(is_overall),
            "n_waves": len(group),
            "shares": np.mean(np.stack(group.shares), axis=0),
        })
    return pd.DataFrame(result)


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    values = values / values.sum(axis=1, keepdims=True)
    if temperature == 1.0:
        return values.copy()
    logits = np.log(values + EPSILON) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def ordinal_tilt_probabilities(
    probabilities: np.ndarray,
    scores: Sequence[float],
    temperature: float,
    features: np.ndarray,
    coefficients: Sequence[float],
) -> np.ndarray:
    """Apply temperature scaling and a transferable linear ordinal logit tilt."""
    scaled = temperature_scale(probabilities, temperature)
    score_values = np.asarray(scores, dtype=float)
    score_values -= score_values.mean()
    coefficient_values = np.asarray(coefficients, dtype=float)
    if features.shape != (len(scaled), len(coefficient_values)):
        raise ValueError("Feature and coefficient dimensions do not match")
    logits = np.log(scaled + EPSILON)
    logits += (features @ coefficient_values)[:, None] * score_values[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _prepared_cells(
    predictions: ProbabilityDataset,
    specs: Mapping[str, CCAMProbabilitySpec],
    target_cells: pd.DataFrame,
    item_ids: Sequence[str],
    temperature: float,
    group_masks: Mapping[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]]:
    prepared = []
    for row in target_cells.loc[target_cells.item_id.isin(item_ids)].itertuples(
        index=False
    ):
        if row.group_key not in group_masks:
            continue
        scores = np.asarray(specs[row.item_id].scores, dtype=float)
        scores -= scores.mean()
        prepared.append((
            temperature_scale(predictions.probabilities[row.item_id], temperature),
            scores,
            group_masks[row.group_key],
            np.asarray(row.shares, dtype=float),
            bool(row.is_overall),
        ))
    if not prepared:
        raise ValueError("No eligible cells for requested construct")
    return prepared


def _loss_and_gradient(
    coefficients: np.ndarray,
    features: np.ndarray,
    prepared: Sequence[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]
    ],
    ridge: float,
) -> tuple[float, np.ndarray]:
    eta = features @ coefficients
    overall_loss = demographic_loss = 0.0
    overall_gradient = np.zeros_like(coefficients)
    demographic_gradient = np.zeros_like(coefficients)
    n_overall = n_demographic = 0
    for base, scores, mask, target, is_overall in prepared:
        logits = np.log(base + EPSILON) + eta[:, None] * scores[None, :]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        group_probabilities = probabilities[mask]
        predicted = group_probabilities.mean(axis=0)
        cell_loss = float(
            -target @ np.log(np.clip(predicted, EPSILON, 1))
        )
        derivative = group_probabilities * (
            scores[None, :]
            - (group_probabilities @ scores)[:, None]
        )
        share_derivative = features[mask].T @ derivative / mask.sum()
        cell_gradient = -share_derivative @ (
            target / np.clip(predicted, EPSILON, 1)
        )
        if is_overall:
            overall_loss += cell_loss
            overall_gradient += cell_gradient
            n_overall += 1
        else:
            demographic_loss += cell_loss
            demographic_gradient += cell_gradient
            n_demographic += 1
    data_loss = 0.5 * overall_loss / n_overall + 0.5 * demographic_loss / n_demographic
    data_gradient = (
        0.5 * overall_gradient / n_overall
        + 0.5 * demographic_gradient / n_demographic
    )
    penalty = ridge * float(np.mean(coefficients**2))
    penalty_gradient = 2 * ridge * coefficients / len(coefficients)
    return data_loss + penalty, data_gradient + penalty_gradient


def fit_demographic_tilt(
    predictions: ProbabilityDataset,
    specs: Mapping[str, CCAMProbabilitySpec],
    target_cells: pd.DataFrame,
    item_ids: Sequence[str],
    temperature: float,
    ridge: float,
    *,
    initial: Sequence[float] | None = None,
) -> dict[str, Any]:
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and non-negative")
    features = demographic_feature_matrix(predictions.personas)
    masks = demographic_group_masks(predictions.personas)
    prepared = _prepared_cells(
        predictions, specs, target_cells, item_ids, temperature, masks
    )
    start = (
        np.zeros(features.shape[1], dtype=float)
        if initial is None
        else np.asarray(initial, dtype=float)
    )
    result = minimize(
        lambda values: _loss_and_gradient(values, features, prepared, ridge),
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success:
        raise RuntimeError(result.message)
    return {
        "coefficients": np.asarray(result.x, dtype=float),
        "penalized_loss": float(result.fun),
        "iterations": int(result.nit),
        "coefficient_norm": float(np.linalg.norm(result.x)),
    }


def demographic_tilt_loss(
    predictions: ProbabilityDataset,
    specs: Mapping[str, CCAMProbabilitySpec],
    target_cells: pd.DataFrame,
    item_ids: Sequence[str],
    temperature: float,
    coefficients: Sequence[float],
) -> float:
    features = demographic_feature_matrix(predictions.personas)
    masks = demographic_group_masks(predictions.personas)
    prepared = _prepared_cells(
        predictions, specs, target_cells, item_ids, temperature, masks
    )
    return float(
        _loss_and_gradient(
            np.asarray(coefficients, dtype=float), features, prepared, 0.0
        )[0]
    )
