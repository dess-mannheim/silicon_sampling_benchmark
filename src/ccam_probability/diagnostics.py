"""Distribution and demographic diagnostics for the CCAM calibration notebook."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from .calibration import distribution_metrics, predicted_share, temperature_scale
from .data import ProbabilityDataset
from .specs import CCAMProbabilitySpec


MODERATOR_COLUMNS = {
    "gender": "gender_h", "age_band": "age_band_h", "race": "race_h",
    "education": "education_h", "income": "income_h", "party": "party_h",
}


def _weighted_shares(
    data: pd.DataFrame, item_id: str, spec: CCAMProbabilitySpec, weight: str
) -> np.ndarray | None:
    valid = data[item_id].notna() & data[weight].notna()
    if not valid.any():
        return None
    values = data.loc[valid, item_id]
    weights = data.loc[valid, weight].to_numpy(float, copy=True)
    weights /= weights.sum()
    return np.array([weights[values.eq(code).to_numpy()].sum() for code in spec.codes])


def subgroup_distribution_diagnostics(
    ground_truth: pd.DataFrame,
    predictions: ProbabilityDataset,
    specs: dict[str, CCAMProbabilitySpec],
    temperatures: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for moderator, column in MODERATOR_COLUMNS.items():
        truth_levels = set(ground_truth[column].dropna().astype(str))
        prediction_levels = set(predictions.personas[column].dropna().astype(str))
        for level in sorted(truth_levels & prediction_levels):
            truth_group = ground_truth.loc[ground_truth[column].astype(str).eq(level)]
            prediction_mask = predictions.personas[column].astype(str).eq(level).to_numpy()
            for item_id, spec in specs.items():
                target = _weighted_shares(truth_group, item_id, spec, "weight_aggregate")
                if target is None or not prediction_mask.any():
                    continue
                predicted = predicted_share(
                    predictions, item_id, temperatures[spec.family], prediction_mask
                )
                rows.append({
                    "moderator": moderator, "level": level, "item_id": item_id,
                    "family": spec.family, "n_ground_truth": int(truth_group[item_id].notna().sum()),
                    "n_prediction": int(prediction_mask.sum()),
                    **distribution_metrics(target, predicted, np.asarray(spec.scores)),
                })
    return pd.DataFrame(rows)


def parity_gap(subgroup_diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (moderator, item_id), group in subgroup_diagnostics.groupby(["moderator", "item_id"]):
        errors = group.assign(abs_mean_error=group.mean_error.abs())
        worst = errors.loc[errors.abs_mean_error.idxmax()]
        best = errors.loc[errors.abs_mean_error.idxmin()]
        rows.append({
            "moderator": moderator, "item_id": item_id, "usable_groups": len(errors),
            "parity_gap": float(worst.abs_mean_error - best.abs_mean_error),
            "worst_group": worst.level, "worst_abs_mean_error": float(worst.abs_mean_error),
            "best_group": best.level, "best_abs_mean_error": float(best.abs_mean_error),
        })
    return pd.DataFrame(rows)


def _weighted_coefficients(values: np.ndarray, groups: pd.Series, weights: np.ndarray):
    work = pd.DataFrame({"value": values, "group": groups, "weight": weights}).dropna()
    levels = sorted(work.group.astype(str).unique())
    if len(levels) < 2:
        return {}, levels
    group_values = work.group.astype(str)
    design = np.column_stack([
        np.ones(len(work)),
        *[group_values.eq(level).to_numpy(float) for level in levels[1:]],
    ])
    sqrt_weight = np.sqrt(work.weight.to_numpy(float))
    coefficients = np.linalg.lstsq(
        design * sqrt_weight[:, None], work.value.to_numpy(float) * sqrt_weight, rcond=None
    )[0]
    return {level: float(coefficients[index + 1]) for index, level in enumerate(levels[1:])}, levels


def demographic_coefficient_recovery(
    ground_truth: pd.DataFrame,
    predictions: ProbabilityDataset,
    specs: dict[str, CCAMProbabilitySpec],
    temperatures: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for moderator, column in MODERATOR_COLUMNS.items():
        common = set(ground_truth[column].dropna().astype(str)) & set(predictions.personas[column].dropna().astype(str))
        truth_mask = ground_truth[column].astype(str).isin(common)
        prediction_mask = predictions.personas[column].astype(str).isin(common)
        for item_id, spec in specs.items():
            code_scores = dict(zip(spec.codes, spec.scores, strict=True))
            truth_scores = ground_truth.loc[truth_mask, item_id].map(code_scores).to_numpy(float)
            truth_groups = ground_truth.loc[truth_mask, column]
            truth_weights = ground_truth.loc[truth_mask, "weight_aggregate"].to_numpy(float)
            scaled = temperature_scale(predictions.probabilities[item_id], temperatures[spec.family])
            prediction_scores = (scaled @ np.asarray(spec.scores))[prediction_mask.to_numpy()]
            prediction_groups = predictions.personas.loc[prediction_mask, column]
            truth_coefficients, truth_levels = _weighted_coefficients(
                truth_scores, truth_groups, truth_weights
            )
            prediction_coefficients, prediction_levels = _weighted_coefficients(
                prediction_scores, prediction_groups, np.ones(len(prediction_scores))
            )
            common_coefficients = sorted(set(truth_coefficients) & set(prediction_coefficients))
            reference = truth_levels[0] if truth_levels and truth_levels == prediction_levels else pd.NA
            for level in common_coefficients:
                rows.append({
                    "moderator": moderator, "item_id": item_id, "reference_level": reference,
                    "level": level, "ground_truth_coefficient": truth_coefficients[level],
                    "prediction_coefficient": prediction_coefficients[level],
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["coefficient_error"] = result.prediction_coefficient - result.ground_truth_coefficient
    return result


def coefficient_recovery_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for moderator, group in coefficients.groupby("moderator"):
        usable = group.dropna(subset=["ground_truth_coefficient", "prediction_coefficient"])
        rho = spearmanr(usable.ground_truth_coefficient, usable.prediction_coefficient).statistic if len(usable) >= 3 else np.nan
        rows.append({
            "moderator": moderator, "coefficient_pairs": len(usable), "spearman_rho": rho,
            "mean_absolute_error": usable.coefficient_error.abs().mean(),
        })
    return pd.DataFrame(rows)


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 3:
        return np.nan
    x, y, weights = x[valid], y[valid], weights[valid]
    weights = weights / weights.sum()
    x_mean, y_mean = weights @ x, weights @ y
    covariance = weights @ ((x - x_mean) * (y - y_mean))
    denominator = np.sqrt((weights @ (x - x_mean) ** 2) * (weights @ (y - y_mean) ** 2))
    return float(covariance / denominator) if denominator > 0 else np.nan


def question_correlation_diagnostics(
    ground_truth: pd.DataFrame,
    predictions: ProbabilityDataset,
    specs: dict[str, CCAMProbabilitySpec],
    temperatures: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    items = list(specs)
    truth_scores = pd.DataFrame(index=ground_truth.index)
    prediction_scores = pd.DataFrame(index=predictions.personas.index)
    for item_id, spec in specs.items():
        code_scores = dict(zip(spec.codes, spec.scores, strict=True))
        truth_scores[item_id] = ground_truth[item_id].map(code_scores)
        scaled = temperature_scale(predictions.probabilities[item_id], temperatures[spec.family])
        prediction_scores[item_id] = scaled @ np.asarray(spec.scores)
    rows = []
    weights = ground_truth.weight_aggregate.to_numpy(float)
    for first_index, first in enumerate(items):
        for second in items[first_index + 1:]:
            truth_rho = _weighted_correlation(
                rankdata(truth_scores[first].to_numpy(float), nan_policy="omit"),
                rankdata(truth_scores[second].to_numpy(float), nan_policy="omit"), weights,
            )
            prediction_rho = prediction_scores[[first, second]].corr(method="spearman").iloc[0, 1]
            rows.append({"question_1": first, "question_2": second,
                         "ground_truth_rho": truth_rho, "prediction_rho": prediction_rho,
                         "rho_error": prediction_rho - truth_rho})
    pairs = pd.DataFrame(rows)
    usable = pairs.dropna()
    summary = pd.DataFrame([{
        "question_pairs": len(usable),
        "structure_spearman_rho": spearmanr(usable.ground_truth_rho, usable.prediction_rho).statistic if len(usable) >= 3 else np.nan,
        "mean_absolute_rho_error": usable.rho_error.abs().mean(),
        "maximum_absolute_rho_error": usable.rho_error.abs().max(),
    }])
    return pairs, summary


def split_half_baseline(
    ground_truth: pd.DataFrame,
    specs: dict[str, CCAMProbabilitySpec],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for wave, wave_data in ground_truth.groupby("wave", sort=True):
        assignment = rng.permutation(len(wave_data)) % 2
        first, second = wave_data.iloc[np.flatnonzero(assignment == 0)], wave_data.iloc[np.flatnonzero(assignment == 1)]
        for item_id, spec in specs.items():
            first_shares = _weighted_shares(first, item_id, spec, "weight_wave")
            second_shares = _weighted_shares(second, item_id, spec, "weight_wave")
            if first_shares is None or second_shares is None:
                continue
            rows.append({"wave": int(wave), "item_id": item_id, "family": spec.family,
                         **distribution_metrics(first_shares, second_shares, np.asarray(spec.scores))})
    return pd.DataFrame(rows)


def concept_map() -> pd.DataFrame:
    """Document the intended later transfer; this study does not apply it."""
    rows = [
        ("belief_post", "cause_recoded", "belief_consensus", "direct"),
        ("concern_1, concern_2, concern_3", "worry; harm_*", "concern_harm", "direct/broad"),
        ("funding_perceptions", "fund_research", "climate_policy", "broad"),
        ("policy_general", "priority; transition_economy", "climate_policy", "direct/broad"),
        ("policy_specific_1", "reduce_tax", "climate_policy", "direct"),
        ("policy_specific_3", "generate_renewable", "climate_policy", "direct"),
        ("policy_specific_6", "priority_cleanenergy", "climate_policy", "broad"),
        ("behavior_talk", "discuss_GW", "engagement_exposure", "direct"),
        ("newsletter_signup", "discuss_GW; hear_GW_media", "engagement_exposure", "broad"),
        ("remaining policy_specific_*", "CCAM policy family", "climate_policy", "broad"),
        ("trust_*, distrust_post, inst_trust_*, policy_role_*", None, "global", "fallback"),
        ("donation_ams, remaining behavior_*", None, "global", "fallback"),
    ]
    return pd.DataFrame(rows, columns=["benchmark_items", "ccam_items", "temperature", "match_strength"])
