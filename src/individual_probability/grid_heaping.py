"""Resample exact slider values while preserving previously selected bins."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from .specs import SLIDER_BINS, ProbabilitySpec


ORDINARY_WEIGHT = 1.0
MULTIPLE_OF_FIVE_WEIGHT = 1.7
MULTIPLE_OF_TEN_WEIGHT = 3.5
DEFAULT_GRID_SAMPLING_SEED = 20260805


def grid_value_probabilities(lower: int, upper: int) -> tuple[np.ndarray, np.ndarray]:
    """Return legal integer values and normalized 1:1.7:3.5 grid-line weights."""
    if lower > upper:
        raise ValueError("lower must not exceed upper")
    values = np.arange(lower, upper + 1, dtype=int)
    weights = np.full(values.shape, ORDINARY_WEIGHT, dtype=float)
    weights[values % 5 == 0] = MULTIPLE_OF_FIVE_WEIGHT
    weights[values % 10 == 0] = MULTIPLE_OF_TEN_WEIGHT
    return values, weights / weights.sum()


def bin_index_for_answer(answer: float, spec: ProbabilitySpec) -> int:
    """Locate an integer answer in exactly one of an item's declared bins."""
    value = float(answer)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError(f"{spec.item_id!r} has a non-integer sampled answer: {answer!r}")
    integer = int(value)
    for index, (_, lower, upper) in enumerate(spec.bins):
        if lower <= integer <= upper:
            return index
    raise ValueError(f"{spec.item_id!r} answer {answer!r} is outside its declared bins")


def resample_grid_heaped_answers(
    sampled_answers: Sequence[Mapping[str, float]],
    specs: Mapping[str, ProbabilitySpec],
    *,
    seed: int = DEFAULT_GRID_SAMPLING_SEED,
) -> list[dict[str, float]]:
    """Redraw 0--100 slider values inside fixed bins; copy other outcomes."""
    rng = np.random.default_rng(seed)
    resampled: list[dict[str, float]] = []
    slider_item_ids = [item_id for item_id, spec in specs.items() if spec.bins == SLIDER_BINS]

    for record_index, answers in enumerate(sampled_answers):
        missing = [item_id for item_id in specs if item_id not in answers]
        if missing:
            raise ValueError(
                f"Sampled answer record {record_index} is missing: {', '.join(missing)}"
            )
        updated = {item_id: float(answer) for item_id, answer in answers.items()}
        for item_id in slider_item_ids:
            spec = specs[item_id]
            bin_index = bin_index_for_answer(updated[item_id], spec)
            _, lower, upper = spec.bins[bin_index]
            values, probabilities = grid_value_probabilities(lower, upper)
            updated[item_id] = float(rng.choice(values, p=probabilities))
        resampled.append(updated)
    return resampled
