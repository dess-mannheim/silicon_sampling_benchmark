"""Create raw and softmax grid-heaped candidates from probability run 6830.

The existing uniform-within-bin files are regenerated and verified in memory.
Only the exact 0--100 slider value is then redrawn; every selected probability
bin, donation answer, and newsletter answer remains fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from individual_probability.grid_heaping import (  # noqa: E402
    DEFAULT_GRID_SAMPLING_SEED,
    MULTIPLE_OF_FIVE_WEIGHT,
    MULTIPLE_OF_TEN_WEIGHT,
    ORDINARY_WEIGHT,
    bin_index_for_answer,
    resample_grid_heaped_answers,
)
from individual_probability.result_saving import (  # noqa: E402
    DEMO_COLUMNS,
    _prediction_frame,
    _tier1_row,
)
from individual_probability.specs import SLIDER_BINS, load_probability_specs  # noqa: E402


DEFAULT_PARSED = Path("results/6830/Qwen_Qwen3.6-27B_T1_probability_parsed.json")
DEFAULT_RAW_BASELINE = Path("predictions/Qwen_Qwen3.6-27B_T1_probability_v1.csv")
DEFAULT_SOFTMAX_BASELINE = Path("predictions/team_6_T1_primary_v1.csv")
DEFAULT_RAW_OUTPUT = Path("predictions/Qwen_Qwen3.6-27B_T1_probability_grid_v1.csv")
DEFAULT_SOFTMAX_OUTPUT = Path(
    "predictions/Qwen_Qwen3.6-27B_T1_probability_softmax_grid_v1.csv"
)
DEFAULT_SOFTMAX_TEMPERATURE = 1.4064310604266295
NUMERIC_TOLERANCE = 1e-12


def _load_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read parsed probability JSON {path}: {error}") from error
    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Parsed probability JSON must contain a non-empty records list")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every parsed probability record must be an object")
    return payload


def _assert_frame_matches(actual: pd.DataFrame, expected_path: Path, label: str) -> None:
    expected = pd.read_csv(expected_path)
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise ValueError(
            f"Regenerated {label} shape/columns do not match {expected_path}: "
            f"{actual.shape} != {expected.shape}"
        )
    numeric = list(actual.select_dtypes(include="number").columns)
    text_columns = [column for column in actual.columns if column not in numeric]
    if text_columns and not actual[text_columns].equals(expected[text_columns]):
        raise ValueError(f"Regenerated {label} identity/text columns do not match {expected_path}")
    if numeric and not np.allclose(
        actual[numeric].to_numpy(),
        expected[numeric].to_numpy(),
        rtol=0,
        atol=NUMERIC_TOLERANCE,
        equal_nan=True,
    ):
        largest = float(
            np.nanmax(np.abs(actual[numeric].to_numpy() - expected[numeric].to_numpy()))
        )
        raise ValueError(
            f"Regenerated {label} values do not match {expected_path}; "
            f"largest absolute difference is {largest}"
        )


def _assert_bins_preserved(before, after, specs, label: str) -> None:
    if len(before) != len(after):
        raise ValueError(f"{label} respondent count changed during grid resampling")
    for row_index, (old_answers, new_answers) in enumerate(zip(before, after, strict=True)):
        for item_id, spec in specs.items():
            if spec.bins == SLIDER_BINS:
                old_bin = bin_index_for_answer(old_answers[item_id], spec)
                new_bin = bin_index_for_answer(new_answers[item_id], spec)
                if old_bin != new_bin:
                    raise ValueError(
                        f"{label} changed {item_id!r} bin for respondent {row_index}: "
                        f"{old_bin} != {new_bin}"
                    )
            elif float(old_answers[item_id]) != float(new_answers[item_id]):
                raise ValueError(
                    f"{label} changed non-slider item {item_id!r} for respondent {row_index}"
                )


def _frame_from_answers(records, sampled_answers) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _tier1_row(record, answers)
            for record, answers in zip(records, sampled_answers, strict=True)
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_grid_heaped_predictions(
    *,
    root: Path,
    parsed_path: Path,
    raw_baseline_path: Path,
    softmax_baseline_path: Path,
    raw_output_path: Path,
    softmax_output_path: Path,
    softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
    grid_seed: int = DEFAULT_GRID_SAMPLING_SEED,
) -> dict[str, Any]:
    """Verify the uniform baselines and write two bucket-preserving candidates."""
    root = root.resolve()

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    parsed_path = resolve(parsed_path)
    raw_baseline_path = resolve(raw_baseline_path)
    softmax_baseline_path = resolve(softmax_baseline_path)
    raw_output_path = resolve(raw_output_path)
    softmax_output_path = resolve(softmax_output_path)

    payload = _load_payload(parsed_path)
    records = payload["records"]
    try:
        resolved = [record["resolved_probabilities"] for record in records]
        sampling_seed = int(payload["sampling_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Parsed probability JSON lacks resolved vectors or sampling_seed"
        ) from error

    specs = load_probability_specs(root)
    raw_frame, raw_answers = _prediction_frame(
        records, resolved, specs, sampling_seed, 1.0
    )
    softmax_frame, softmax_answers = _prediction_frame(
        records, resolved, specs, sampling_seed, softmax_temperature
    )
    _assert_frame_matches(raw_frame, raw_baseline_path, "raw baseline")
    _assert_frame_matches(softmax_frame, softmax_baseline_path, "softmax baseline")

    raw_grid_answers = resample_grid_heaped_answers(raw_answers, specs, seed=grid_seed)
    softmax_grid_answers = resample_grid_heaped_answers(
        softmax_answers, specs, seed=grid_seed
    )
    _assert_bins_preserved(raw_answers, raw_grid_answers, specs, "raw grid candidate")
    _assert_bins_preserved(
        softmax_answers, softmax_grid_answers, specs, "softmax grid candidate"
    )

    raw_grid_frame = _frame_from_answers(records, raw_grid_answers)
    softmax_grid_frame = _frame_from_answers(records, softmax_grid_answers)
    identity_columns = ["profile_id", "condition", *DEMO_COLUMNS]
    for label, candidate in (
        ("raw grid candidate", raw_grid_frame),
        ("softmax grid candidate", softmax_grid_frame),
    ):
        if not candidate[identity_columns].equals(raw_frame[identity_columns]):
            raise ValueError(f"{label} changed respondent identity or demographics")

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    softmax_output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_grid_frame.to_csv(raw_output_path, index=False)
    softmax_grid_frame.to_csv(softmax_output_path, index=False)

    return {
        "parsed_input": str(parsed_path),
        "respondents": len(records),
        "baseline_sampling_seed": sampling_seed,
        "grid_sampling_seed": grid_seed,
        "softmax_temperature": softmax_temperature,
        "weights": {
            "ordinary": ORDINARY_WEIGHT,
            "multiple_of_5_not_10": MULTIPLE_OF_FIVE_WEIGHT,
            "multiple_of_10": MULTIPLE_OF_TEN_WEIGHT,
        },
        "slider_items": sum(spec.bins == SLIDER_BINS for spec in specs.values()),
        "raw_output": {
            "path": str(raw_output_path),
            "sha256": _sha256(raw_output_path),
        },
        "softmax_output": {
            "path": str(softmax_output_path),
            "sha256": _sha256(softmax_output_path),
        },
    }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--parsed", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--raw-baseline", type=Path, default=DEFAULT_RAW_BASELINE)
    parser.add_argument("--softmax-baseline", type=Path, default=DEFAULT_SOFTMAX_BASELINE)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW_OUTPUT)
    parser.add_argument("--softmax-output", type=Path, default=DEFAULT_SOFTMAX_OUTPUT)
    parser.add_argument(
        "--softmax-temperature", type=float, default=DEFAULT_SOFTMAX_TEMPERATURE
    )
    parser.add_argument("--grid-seed", type=int, default=DEFAULT_GRID_SAMPLING_SEED)
    return parser


def main() -> None:
    args = _argument_parser().parse_args()
    summary = generate_grid_heaped_predictions(
        root=args.root,
        parsed_path=args.parsed,
        raw_baseline_path=args.raw_baseline,
        softmax_baseline_path=args.softmax_baseline,
        raw_output_path=args.raw_output,
        softmax_output_path=args.softmax_output,
        softmax_temperature=args.softmax_temperature,
        grid_seed=args.grid_seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
