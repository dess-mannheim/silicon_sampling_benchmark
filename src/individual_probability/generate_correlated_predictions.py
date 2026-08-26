"""Create correlated Tier-1 draws from direct and probability predictions.

This is an offline, CPU-only post-processing program.  It deliberately does
not invoke an LLM or share result-writing code with either prediction method.
It uses direct predictions only to estimate a rank-normal within-condition
correlation matrix and to locate each persona's anchor in that persona's own
probability distribution.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd


# Allow ``uv run python src/individual_probability/generate_correlated_predictions.py``.
SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from individual_level.result_saving import normalized_answer, questionnaire_scales  # noqa: E402
from individual_probability.result_saving import COMPOSITES, DEMO_COLUMNS, TIER1_COLUMNS  # noqa: E402
from individual_probability.specs import ProbabilitySpec, load_probability_specs  # noqa: E402


NORMAL = NormalDist()
EPSILON = 1e-6
SYNTHETIC_ID = re.compile(r"^synthetic_0*(\d+)$")


def canonical_persona_id(persona_id: Any) -> str:
    """Make padded and unpadded synthetic IDs comparable without changing output IDs."""
    value = str(persona_id)
    match = SYNTHETIC_ID.fullmatch(value)
    return f"synthetic_{int(match.group(1))}" if match else value


def _load_records(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label} JSON from {path}: {error}") from error
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{label} JSON must contain a list of object records under 'records'.")
    if not records:
        raise ValueError(f"{label} JSON contains no records.")
    return records


def _records_by_id(records: list[dict[str, Any]], label: str) -> dict[str, tuple[int, dict[str, Any]]]:
    indexed: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, record in enumerate(records):
        if "persona_id" not in record:
            raise ValueError(f"{label} record {index} is missing persona_id.")
        key = canonical_persona_id(record["persona_id"])
        if key in indexed:
            raise ValueError(f"{label} has duplicate persona ID after canonicalization: {key!r}.")
        indexed[key] = (index, record)
    return indexed


def _assert_target_matches_direct(
    direct_by_id: dict[str, tuple[int, dict[str, Any]]], probability_records: list[dict[str, Any]],
) -> list[int]:
    """Match probability personas to direct personas and reject assignment drift."""
    direct_indices: list[int] = []
    identity_keys = ("condition", "condition_variant", *DEMO_COLUMNS)
    probability_ids: set[str] = set()
    for probability_index, probability_record in enumerate(probability_records):
        if "persona_id" not in probability_record:
            raise ValueError(f"Probability record {probability_index} is missing persona_id.")
        persona_key = canonical_persona_id(probability_record["persona_id"])
        if persona_key in probability_ids:
            raise ValueError(f"Probability records duplicate persona ID {persona_key!r}.")
        probability_ids.add(persona_key)
        matched = direct_by_id.get(persona_key)
        if matched is None:
            raise ValueError(f"No direct prediction exists for probability persona {persona_key!r}.")
        direct_index, direct_record = matched
        for key in identity_keys:
            if probability_record.get(key) != direct_record.get(key):
                raise ValueError(
                    f"Persona {persona_key!r} has different {key!r} values in direct and probability inputs: "
                    f"{direct_record.get(key)!r} != {probability_record.get(key)!r}."
                )
        direct_indices.append(direct_index)
    return direct_indices


def _direct_answer_matrix(
    records: list[dict[str, Any]], item_ids: list[str], scales: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, int]:
    """Read direct answers and fill missing cells with condition, global, then scale medians."""
    matrix = np.full((len(records), len(item_ids)), np.nan, dtype=float)
    for row_index, record in enumerate(records):
        answers = record.get("answers", {})
        if not isinstance(answers, dict):
            continue
        for column_index, item_id in enumerate(item_ids):
            answer = normalized_answer(item_id, answers.get(item_id), scales)
            if answer is not None:
                matrix[row_index, column_index] = answer

    imputed = 0
    conditions = np.asarray([str(record.get("condition", "")) for record in records], dtype=object)
    for column_index, item_id in enumerate(item_ids):
        column = matrix[:, column_index]
        valid = column[~np.isnan(column)]
        global_median = float(np.median(valid)) if valid.size else (0.5 if item_id == "newsletter_signup" else sum(scales[item_id]) / 2)
        for condition in dict.fromkeys(conditions):
            indices = np.flatnonzero(conditions == condition)
            missing = np.isnan(column[indices])
            if not missing.any():
                continue
            peers = column[indices][~missing]
            fill = float(np.median(peers)) if peers.size else global_median
            column[indices[missing]] = fill
            imputed += int(missing.sum())
        matrix[:, column_index] = column
    return matrix, imputed


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (one-indexed) without requiring scipy."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def _normal_scores_within_condition(matrix: np.ndarray, conditions: list[str]) -> np.ndarray:
    scores = np.empty_like(matrix, dtype=float)
    condition_array = np.asarray(conditions, dtype=object)
    for condition in dict.fromkeys(condition_array):
        indices = np.flatnonzero(condition_array == condition)
        for column_index in range(matrix.shape[1]):
            uniforms = (_average_ranks(matrix[indices, column_index]) - 0.5) / len(indices)
            scores[indices, column_index] = np.fromiter(
                (NORMAL.inv_cdf(float(value)) for value in uniforms), dtype=float, count=len(indices)
            )
    return scores


def _nearest_correlation(scores: np.ndarray, shrinkage: float) -> np.ndarray:
    """Return a shrunk, positive-semidefinite correlation matrix."""
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1.")
    centered = scores - scores.mean(axis=0, keepdims=True)
    standard_deviations = np.sqrt(np.mean(centered**2, axis=0))
    standardized = np.divide(
        centered, standard_deviations, out=np.zeros_like(centered), where=standard_deviations > 1e-12
    )
    correlation = standardized.T @ standardized / scores.shape[0]
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    correlation = (1 - shrinkage) * correlation + shrinkage * np.eye(correlation.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh((correlation + correlation.T) / 2)
    positive = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-10, None)) @ eigenvectors.T
    scales = np.sqrt(np.diag(positive))
    positive = positive / np.outer(scales, scales)
    positive = (positive + positive.T) / 2
    np.fill_diagonal(positive, 1.0)
    return positive


def estimate_within_condition_correlation(
    direct_matrix: np.ndarray, direct_records: list[dict[str, Any]], shrinkage: float,
) -> np.ndarray:
    """Estimate pooled rank-normal item correlation; never use this rank as an anchor."""
    if len(direct_records) < 2:
        raise ValueError("At least two direct records are required to estimate correlation.")
    scores = _normal_scores_within_condition(direct_matrix, [record["condition"] for record in direct_records])
    return _nearest_correlation(scores, shrinkage)


def _resolved_vector(record: dict[str, Any], item_id: str, spec: ProbabilitySpec) -> np.ndarray:
    try:
        vector = record["resolved_probabilities"][item_id]["probabilities"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            f"Probability persona {record.get('persona_id')!r} has no resolved vector for {item_id!r}."
        ) from error
    if not isinstance(vector, list) or len(vector) != len(spec.bins):
        raise ValueError(f"Probability persona {record.get('persona_id')!r} has an invalid vector length for {item_id!r}.")
    result = np.asarray(vector, dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result < 0) or result.sum() <= 0:
        raise ValueError(f"Probability persona {record.get('persona_id')!r} has invalid resolved probabilities for {item_id!r}.")
    return result / result.sum()


def _bin_index_for_answer(answer: float, spec: ProbabilitySpec) -> int:
    # Direct prompts require discrete values.  Round malformed fractional output to the
    # nearest legal integer so every valid scale answer still has an unambiguous bin.
    integer_answer = int(math.floor(answer + 0.5))
    for index, (_, lower, upper) in enumerate(spec.bins):
        if lower <= integer_answer <= upper:
            return index
    raise ValueError(f"Direct answer {answer!r} does not fit the bins for {spec.item_id!r}.")


def anchor_percentile(answer: float, probabilities: np.ndarray, spec: ProbabilitySpec) -> tuple[float, bool]:
    """Locate an exact direct answer in its own probability CDF.

    If its bin has zero probability, use the midpoint of the closest positive bin.
    """
    answer_bin = _bin_index_for_answer(answer, spec)
    positive_bins = np.flatnonzero(probabilities > 0)
    if not len(positive_bins):  # guarded by _resolved_vector, retained for direct use/tests
        raise ValueError(f"No positive probability bins for {spec.item_id!r}.")
    used_fallback = probabilities[answer_bin] == 0
    bin_index = answer_bin if not used_fallback else int(min(positive_bins, key=lambda index: (abs(index - answer_bin), index)))
    cumulative_before = float(probabilities[:bin_index].sum())
    probability = float(probabilities[bin_index])
    if used_fallback:
        within_bin = 0.5
    else:
        _, lower, upper = spec.bins[bin_index]
        rounded_answer = int(math.floor(answer + 0.5))
        within_bin = (rounded_answer - lower + 0.5) / (upper - lower + 1)
    return float(np.clip(cumulative_before + probability * within_bin, EPSILON, 1 - EPSILON)), used_fallback


def _draw_answer(percentile: float, probabilities: np.ndarray, spec: ProbabilitySpec) -> float:
    cumulative = np.cumsum(probabilities)
    bin_index = int(np.searchsorted(cumulative, percentile, side="left"))
    bin_index = min(bin_index, len(spec.bins) - 1)
    before = float(cumulative[bin_index - 1]) if bin_index else 0.0
    within = (percentile - before) / probabilities[bin_index]
    _, lower, upper = spec.bins[bin_index]
    answer = lower + int(math.floor(within * (upper - lower + 1)))
    return float(min(max(answer, lower), upper))


def _tier1_row(record: dict[str, Any], answers: dict[str, float]) -> dict[str, Any]:
    row = {"profile_id": record["persona_id"], "condition": record["condition"], **{key: record[key] for key in DEMO_COLUMNS}}
    row.update(answers)
    row["funding_perceptions"] = 100 - row["funding_perceptions"]
    for name, members in COMPOSITES.items():
        row[name] = sum(row[item] for item in members) / len(members)
    return {column: row[column] for column in TIER1_COLUMNS}


def generate_predictions(
    direct_path: Path, probability_path: Path, output_path: Path, *, alpha: float = 0.40,
    seed: int = 20260804, shrinkage: float = 0.20, root: Path | None = None,
) -> dict[str, Any]:
    """Generate one correlated discrete response vector per probability persona."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    root = Path.cwd() if root is None else root
    specs = load_probability_specs(root)
    item_ids = list(specs)
    scales = questionnaire_scales(root)
    direct_records = _load_records(direct_path, "Direct prediction")
    probability_records = _load_records(probability_path, "Probability prediction")
    direct_by_id = _records_by_id(direct_records, "Direct prediction")
    target_direct_indices = _assert_target_matches_direct(direct_by_id, probability_records)
    direct_matrix, direct_imputations = _direct_answer_matrix(direct_records, item_ids, scales)
    correlation = estimate_within_condition_correlation(direct_matrix, direct_records, shrinkage)

    target_count = len(probability_records)
    anchors = np.empty((target_count, len(item_ids)), dtype=float)
    probability_vectors: list[list[np.ndarray]] = []
    fallback_counts: Counter[str] = Counter()
    for target_index, (probability_record, direct_index) in enumerate(zip(probability_records, target_direct_indices, strict=True)):
        vectors_for_persona: list[np.ndarray] = []
        for item_index, item_id in enumerate(item_ids):
            vector = _resolved_vector(probability_record, item_id, specs[item_id])
            percentile, used_fallback = anchor_percentile(direct_matrix[direct_index, item_index], vector, specs[item_id])
            anchors[target_index, item_index] = NORMAL.inv_cdf(percentile)
            vectors_for_persona.append(vector)
            if used_fallback:
                fallback_counts[item_id] += 1
        probability_vectors.append(vectors_for_persona)

    rng = np.random.default_rng(seed)
    residuals = rng.multivariate_normal(np.zeros(len(item_ids)), correlation, size=target_count)
    latent = alpha * anchors + math.sqrt(1 - alpha**2) * residuals
    percentiles = np.clip(
        np.fromiter((NORMAL.cdf(float(value)) for value in latent.ravel()), dtype=float, count=latent.size).reshape(latent.shape),
        EPSILON, 1 - EPSILON,
    )
    sampled = [
        {
            item_id: _draw_answer(percentiles[row_index, item_index], probability_vectors[row_index][item_index], specs[item_id])
            for item_index, item_id in enumerate(item_ids)
        }
        for row_index in range(target_count)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_tier1_row(record, answers) for record, answers in zip(probability_records, sampled, strict=True)]).to_csv(output_path, index=False)
    return {
        "direct_records": len(direct_records),
        "probability_records": target_count,
        "direct_imputed_cells": direct_imputations,
        "zero_probability_anchor_fallbacks": dict(sorted(fallback_counts.items())),
        "alpha": alpha,
        "seed": seed,
        "shrinkage": shrinkage,
        "output": str(output_path),
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-parsed", type=Path, required=True, help="Direct Tier-1 parsed JSON.")
    parser.add_argument("--probability-parsed", type=Path, required=True, help="Probability Tier-1 parsed JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Combined Tier-1 prediction CSV to write.")
    parser.add_argument("--alpha", type=float, default=0.40, help="Direct-anchor weight in [0, 1] (default: 0.40).")
    parser.add_argument("--seed", type=int, default=20260804, help="Random sampling seed (default: 20260804).")
    parser.add_argument("--shrinkage", type=float, default=0.20, help="Correlation shrinkage toward identity in [0, 1] (default: 0.20).")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root containing qstn_data (default: current directory).")
    return parser


def main() -> None:
    args = _argument_parser().parse_args()
    summary = generate_predictions(
        args.direct_parsed, args.probability_parsed, args.output, alpha=args.alpha,
        seed=args.seed, shrinkage=args.shrinkage, root=args.root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
