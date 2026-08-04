"""Representative demographic sampling from supplied marginal and pairwise constraints."""
from __future__ import annotations

import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PopulationModel:
    attributes: tuple[str, ...]
    levels: dict[str, tuple[str, ...]]
    probabilities: np.ndarray


def load_population_source(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _numbers(values: dict[str, Any], levels: tuple[str, ...], name: str) -> np.ndarray:
    if set(values) != set(levels):
        raise ValueError(f"{name} levels do not match the benchmark")
    result = np.asarray([float(values[level]) for level in levels], dtype=float)
    if not np.all(np.isfinite(result)) or np.any(result < 0):
        raise ValueError(f"{name} must contain finite, non-negative values")
    return result


def validate_source(source: dict[str, Any], moderators: dict[str, list[str]], *, tolerance: float = 2e-4) -> None:
    """Validate supplied margins and pairwise constraints against benchmark levels."""
    levels = {name: tuple(values) for name, values in moderators.items()}
    demographics = source.get("demographics")
    tables = source.get("pairwise_joint_distributions")
    if not isinstance(demographics, dict) or set(demographics) != set(levels):
        raise ValueError("source demographics must contain exactly the benchmark attributes")
    if not isinstance(tables, dict) or not tables:
        raise ValueError("source must contain pairwise_joint_distributions")
    marginals = {name: _numbers(demographics[name], attribute_levels, name) / 100
                 for name, attribute_levels in levels.items()}
    for name, marginal in marginals.items():
        if not math.isclose(float(marginal.sum()), 1.0, abs_tol=tolerance):
            raise ValueError(f"{name} marginal must sum to 100 percent")
    for name, table in tables.items():
        dimensions = table.get("dimensions")
        if not isinstance(dimensions, list) or len(dimensions) != 2 or any(d not in levels for d in dimensions):
            raise ValueError(f"{name} must declare two benchmark dimensions")
        first, second = dimensions
        joint = table.get("joint_percent")
        if not isinstance(joint, dict):
            raise ValueError(f"{name} is missing joint_percent")
        matrix = np.asarray([_numbers(joint[first_level], levels[second], name)
                             for first_level in levels[first]], dtype=float) / 100
        if not math.isclose(float(matrix.sum()), 1.0, abs_tol=tolerance):
            raise ValueError(f"{name} joint distribution must sum to 100 percent")
        if not np.allclose(matrix.sum(axis=1), marginals[first], atol=tolerance):
            raise ValueError(f"{name} disagrees with the {first} marginal")
        if not np.allclose(matrix.sum(axis=0), marginals[second], atol=tolerance):
            raise ValueError(f"{name} disagrees with the {second} marginal")


def fit_population(source: dict[str, Any], moderators: dict[str, list[str]], *, tolerance: float = 2e-6, max_iterations: int = 10_000) -> PopulationModel:
    """Fit the maximum-entropy full joint distribution with iterative proportional fitting."""
    validate_source(source, moderators)
    attributes = tuple(moderators)
    levels = {name: tuple(values) for name, values in moderators.items()}
    shape = tuple(len(levels[name]) for name in attributes)
    probabilities = np.full(shape, 1 / np.prod(shape), dtype=float)
    targets: list[tuple[tuple[int, ...], np.ndarray]] = []
    for axis, name in enumerate(attributes):
        targets.append(((axis,), _numbers(source["demographics"][name], levels[name], name) / 100))
    for name, table in source["pairwise_joint_distributions"].items():
        first, second = table["dimensions"]
        axes = (attributes.index(first), attributes.index(second))
        matrix = np.asarray([_numbers(table["joint_percent"][first_level], levels[second], name)
                             for first_level in levels[first]], dtype=float) / 100
        targets.append((axes, matrix))
    for _ in range(max_iterations):
        maximum_error = 0.0
        for axes, target in targets:
            other_axes = tuple(axis for axis in range(probabilities.ndim) if axis not in axes)
            moved = np.moveaxis(probabilities, axes, range(len(axes)))
            observed = moved.sum(axis=tuple(range(len(axes), probabilities.ndim))) if other_axes else moved
            maximum_error = max(maximum_error, float(np.max(np.abs(observed - target))))
            ratio = np.divide(target, observed, out=np.ones_like(target), where=observed > 0)
            sorted_axes = tuple(sorted(axes))
            permutation = [axes.index(axis) for axis in sorted_axes]
            reordered_ratio = np.transpose(ratio, permutation)
            reshape = [1] * probabilities.ndim
            for source_axis, probability_axis in enumerate(sorted_axes):
                reshape[probability_axis] = reordered_ratio.shape[source_axis]
            probabilities *= reordered_ratio.reshape(reshape)
        probabilities /= probabilities.sum()
        if maximum_error <= tolerance:
            return PopulationModel(attributes, levels, probabilities)
    raise ValueError(f"IPF did not converge within {max_iterations} iterations")


def draw_personas(model: PopulationModel, n_individuals: int, seed: int) -> list[dict[str, str]]:
    if not isinstance(n_individuals, int) or isinstance(n_individuals, bool) or n_individuals <= 0:
        raise ValueError("n_individuals must be a positive integer")
    generator = np.random.default_rng(seed)
    choices = generator.choice(model.probabilities.size, size=n_individuals, p=model.probabilities.ravel())
    coordinates = np.array(np.unravel_index(choices, model.probabilities.shape)).T
    return [{attribute: model.levels[attribute][coordinate[index]]
             for index, attribute in enumerate(model.attributes)} for coordinate in coordinates]


def condition_counts(conditions: list[str], n_individuals: int, seed: int) -> dict[str, int]:
    base, remainder = divmod(n_individuals, len(conditions))
    extra = list(conditions)
    random.Random(f"{seed}:condition-counts").shuffle(extra)
    return {condition: base + int(condition in extra[:remainder]) for condition in conditions}


def _targets(personas: list[dict[str, str]], conditions: list[str], capacities: dict[str, int], attributes: tuple[str, ...]) -> dict[str, dict[str, dict[str, float]]]:
    total = len(personas)
    return {condition: {
        attribute: {level: sum(persona[attribute] == level for persona in personas) * capacities[condition] / total
        for level in {persona[attribute] for persona in personas}}
        for attribute in attributes
    } for condition in conditions}


def assign_conditions(personas: list[dict[str, str]], conditions: list[str], seed: int) -> list[str]:
    """Assign every persona once while minimizing six marginal imbalance objectives."""
    if not personas:
        return []
    attributes = tuple(personas[0])
    capacities = condition_counts(conditions, len(personas), seed)
    targets = _targets(personas, conditions, capacities, attributes)
    counts = {condition: {attribute: {level: 0 for level in targets[condition][attribute]}
                          for attribute in attributes} for condition in conditions}
    rarity = [{attribute: sum(peer[attribute] == persona[attribute] for peer in personas)
               for attribute in attributes} for persona in personas]
    order = list(range(len(personas)))
    tie_breaker = random.Random(f"{seed}:assignment-order")
    tie_breaker.shuffle(order)
    order.sort(key=lambda index: sum(1 / rarity[index][attribute] for attribute in attributes), reverse=True)
    assigned = [""] * len(personas)
    chooser = random.Random(f"{seed}:assignment-ties")
    remaining = capacities.copy()
    for index in order:
        persona = personas[index]
        scores: dict[str, float] = {}
        for condition in conditions:
            if not remaining[condition]:
                continue
            score = 0.0
            for attribute in attributes:
                level = persona[attribute]
                before = counts[condition][attribute][level] - targets[condition][attribute][level]
                after = before + 1
                score += (after * after - before * before) / max(targets[condition][attribute][level], 1.0)
            scores[condition] = score
        best = min(scores.values())
        candidates = [condition for condition, score in scores.items() if math.isclose(score, best, abs_tol=1e-12)]
        condition = chooser.choice(candidates)
        assigned[index] = condition
        remaining[condition] -= 1
        for attribute in attributes:
            counts[condition][attribute][persona[attribute]] += 1
    return assigned


def balance_error(personas: list[dict[str, str]], assignments: list[str], conditions: list[str]) -> float:
    """Normalized sum-of-squares error of condition demographic margins."""
    capacities = {condition: assignments.count(condition) for condition in conditions}
    attributes = tuple(personas[0]) if personas else ()
    targets = _targets(personas, conditions, capacities, attributes)
    error = 0.0
    for condition in conditions:
        for attribute in attributes:
            for level, target in targets[condition][attribute].items():
                observed = sum(persona[attribute] == level and assignment == condition
                               for persona, assignment in zip(personas, assignments, strict=True))
                error += (observed - target) ** 2 / max(target, 1.0)
    return error
