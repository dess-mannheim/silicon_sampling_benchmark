"""Temperature calibration against survey-weighted CCAM category distributions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from .data import ProbabilityDataset
from .specs import CCAMProbabilitySpec, FAMILIES


EPSILON = 1e-8


@dataclass(frozen=True)
class CalibrationFit:
    global_temperature: float
    family_temperatures: dict[str, float]
    regularization: float
    objective: float


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    values = np.asarray(probabilities, dtype=float)
    if values.ndim not in (1, 2) or np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("probabilities must be a finite non-negative vector or matrix")
    totals = values.sum(axis=-1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("each probability vector must have positive mass")
    normalized = values / totals
    logits = np.log(np.clip(normalized, EPSILON, None)) / temperature
    logits -= logits.max(axis=-1, keepdims=True)
    output = np.exp(logits)
    return output / output.sum(axis=-1, keepdims=True)


def ground_truth_targets(
    ground_truth: pd.DataFrame,
    specs: dict[str, CCAMProbabilitySpec],
    *, waves: Iterable[int] | None = None,
) -> pd.DataFrame:
    selected = ground_truth if waves is None else ground_truth.loc[ground_truth.wave.isin(list(waves))]
    rows = []
    for wave, wave_data in selected.groupby("wave", sort=True):
        for item_id, spec in specs.items():
            valid = wave_data[item_id].notna()
            if not valid.any():
                continue
            values = wave_data.loc[valid, item_id]
            weights = wave_data.loc[valid, "weight_wave"].to_numpy(float, copy=True)
            weights /= weights.sum()
            shares = np.array([weights[values.eq(code).to_numpy()].sum() for code in spec.codes])
            rows.append({
                "wave": int(wave), "item_id": item_id, "family": spec.family,
                "n": int(valid.sum()), "shares": shares,
            })
    return pd.DataFrame(rows)


def pooled_ground_truth_targets(
    ground_truth: pd.DataFrame, specs: dict[str, CCAMProbabilitySpec]
) -> pd.DataFrame:
    rows = []
    for item_id, spec in specs.items():
        valid = ground_truth[item_id].notna()
        values = ground_truth.loc[valid, item_id]
        weights = ground_truth.loc[valid, "weight_aggregate"].to_numpy(float, copy=True)
        weights /= weights.sum()
        shares = np.array([weights[values.eq(code).to_numpy()].sum() for code in spec.codes])
        rows.append({"item_id": item_id, "family": spec.family, "n": int(valid.sum()), "shares": shares})
    return pd.DataFrame(rows)


def predicted_share(
    predictions: ProbabilityDataset, item_id: str, temperature: float,
    mask: np.ndarray | pd.Series | None = None,
) -> np.ndarray:
    values = predictions.probabilities[item_id]
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool)]
    if len(values) == 0:
        raise ValueError(f"No predicted personas available for {item_id}")
    return temperature_scale(values, temperature).mean(axis=0)


def cross_entropy(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(-np.sum(np.asarray(target) * np.log(np.clip(prediction, EPSILON, 1))))


def _target_loss(
    targets: pd.DataFrame, predictions: ProbabilityDataset,
    temperatures: dict[str, float],
) -> float:
    item_families = targets[["item_id", "family"]].drop_duplicates()
    predicted_by_item = {
        row.item_id: predicted_share(predictions, row.item_id, temperatures[row.family])
        for row in item_families.itertuples(index=False)
    }
    losses = [
        cross_entropy(row.shares, predicted_by_item[row.item_id])
        for row in targets.itertuples(index=False)
    ]
    if not losses:
        raise ValueError("No ground-truth item-wave targets are available")
    return float(np.mean(losses))


def fit_global_temperature(
    targets: pd.DataFrame, predictions: ProbabilityDataset
) -> CalibrationFit:
    families = tuple(FAMILIES)
    result = minimize_scalar(
        lambda log_t: _target_loss(targets, predictions, {family: float(np.exp(log_t)) for family in families}),
        bounds=(-3.0, 3.0), method="bounded",
    )
    temperature = float(np.exp(result.x))
    return CalibrationFit(temperature, {family: temperature for family in families}, 0.0, float(result.fun))


def fit_family_temperatures(
    targets: pd.DataFrame, predictions: ProbabilityDataset, regularization: float
) -> CalibrationFit:
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative")
    families = tuple(FAMILIES)

    def objective(parameters: np.ndarray) -> float:
        global_log_t, deviations = parameters[0], parameters[1:]
        temperatures = {
            family: float(np.exp(global_log_t + deviations[index]))
            for index, family in enumerate(families)
        }
        return _target_loss(targets, predictions, temperatures) + regularization * float(np.mean(deviations**2))

    global_fit = fit_global_temperature(targets, predictions)
    initial = np.array([np.log(global_fit.global_temperature), *([0.0] * len(families))])
    result = minimize(objective, initial, method="L-BFGS-B", bounds=[(-3, 3)] * len(initial))
    global_temperature = float(np.exp(result.x[0]))
    family_temperatures = {
        family: float(np.exp(result.x[0] + result.x[index + 1]))
        for index, family in enumerate(families)
    }
    return CalibrationFit(global_temperature, family_temperatures, regularization, float(result.fun))


def leave_one_wave_out_cv(
    targets: pd.DataFrame,
    predictions: ProbabilityDataset,
    regularization_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0),
) -> tuple[pd.DataFrame, float]:
    waves = sorted(targets.wave.unique())
    rows = []
    raw_temperatures = {family: 1.0 for family in FAMILIES}
    for held_out in waves:
        train = targets.loc[targets.wave.ne(held_out)]
        test = targets.loc[targets.wave.eq(held_out)]
        rows.append({"held_out_wave": held_out, "model": "raw", "regularization": np.nan,
                     "cross_entropy": _target_loss(test, predictions, raw_temperatures)})
        global_fit = fit_global_temperature(train, predictions)
        rows.append({"held_out_wave": held_out, "model": "global", "regularization": 0.0,
                     "cross_entropy": _target_loss(test, predictions, global_fit.family_temperatures),
                     "global_temperature": global_fit.global_temperature})
        for penalty in regularization_grid:
            fit = fit_family_temperatures(train, predictions, penalty)
            row = {"held_out_wave": held_out, "model": "family", "regularization": penalty,
                   "cross_entropy": _target_loss(test, predictions, fit.family_temperatures),
                   "global_temperature": fit.global_temperature}
            row.update({f"temperature_{family}": value for family, value in fit.family_temperatures.items()})
            rows.append(row)
    results = pd.DataFrame(rows)
    family_rows = results.loc[results.model.eq("family")]
    best = float(family_rows.groupby("regularization").cross_entropy.mean().idxmin())
    return results, best


def distribution_metrics(
    target: np.ndarray, prediction: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    midpoint = (target + prediction) / 2
    js = 0.5 * cross_entropy(target, midpoint) - 0.5 * cross_entropy(target, target)
    js += 0.5 * cross_entropy(prediction, midpoint) - 0.5 * cross_entropy(prediction, prediction)
    target_mean = float(target @ scores)
    prediction_mean = float(prediction @ scores)
    target_variance = float(target @ (scores - target_mean) ** 2)
    prediction_variance = float(prediction @ (scores - prediction_mean) ** 2)
    return {
        "cross_entropy": cross_entropy(target, prediction),
        "jensen_shannon": float(js),
        "total_variation": float(0.5 * np.abs(target - prediction).sum()),
        "mean_error": prediction_mean - target_mean,
        "ground_truth_variance": target_variance,
        "predicted_variance": prediction_variance,
        "variance_ratio": prediction_variance / target_variance if target_variance > 0 else np.nan,
        "absolute_log_variance_error": abs(float(np.log((prediction_variance + EPSILON) / (target_variance + EPSILON)))),
    }


def evaluate_temperatures(
    targets: pd.DataFrame,
    predictions: ProbabilityDataset,
    specs: dict[str, CCAMProbabilitySpec],
    models: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows = []
    for target in targets.itertuples(index=False):
        spec = specs[target.item_id]
        for model, temperatures in models.items():
            predicted = predicted_share(predictions, target.item_id, temperatures[target.family])
            rows.append({
                "wave": getattr(target, "wave", pd.NA), "item_id": target.item_id,
                "family": target.family, "model": model,
                **distribution_metrics(target.shares, predicted, np.asarray(spec.scores)),
            })
    return pd.DataFrame(rows)


def variance_decomposition(
    probabilities: np.ndarray, scores: np.ndarray
) -> dict[str, float]:
    values = np.asarray(probabilities, dtype=float)
    scores = np.asarray(scores, dtype=float)
    expectations = values @ scores
    within = np.sum(values * (scores[None, :] - expectations[:, None]) ** 2, axis=1)
    between_variance = float(np.var(expectations, ddof=0))
    within_variance = float(np.mean(within))
    return {
        "between_person_variance": between_variance,
        "within_person_sampling_variance": within_variance,
        "total_sampled_variance": between_variance + within_variance,
    }


def variance_table(
    pooled_targets: pd.DataFrame,
    predictions: ProbabilityDataset,
    specs: dict[str, CCAMProbabilitySpec],
    models: dict[str, dict[str, float]],
) -> pd.DataFrame:
    target_by_item = pooled_targets.set_index("item_id")
    rows = []
    for item_id, spec in specs.items():
        scores = np.asarray(spec.scores)
        target = target_by_item.loc[item_id, "shares"]
        target_mean = float(target @ scores)
        ground_truth_variance = float(target @ (scores - target_mean) ** 2)
        for model, temperatures in models.items():
            scaled = temperature_scale(predictions.probabilities[item_id], temperatures[spec.family])
            rows.append({"item_id": item_id, "family": spec.family, "model": model,
                         "ground_truth_variance": ground_truth_variance,
                         **variance_decomposition(scaled, scores)})
    return pd.DataFrame(rows)
