"""Response-bin specifications for probability-based Tier-1 predictions."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SLIDER_BINS = (("0-20", 0, 20), ("21-40", 21, 40), ("41-60", 41, 60),
               ("61-80", 61, 80), ("81-100", 81, 100))
DONATION_BINS = (("0-2", 0, 2), ("3-4", 3, 4), ("5-6", 5, 6),
                 ("7-8", 7, 8), ("9-10", 9, 10))
NEWSLETTER_BINS = (("0", 0, 0), ("1", 1, 1))


@dataclass(frozen=True)
class ProbabilitySpec:
    """Exact JSON keys and corresponding discrete outcome values for one item."""

    item_id: str
    bins: tuple[tuple[str, int, int], ...]

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(bin_[0] for bin_ in self.bins)


def load_probability_specs(root: Path) -> dict[str, ProbabilitySpec]:
    """Derive the supported response bins from the questionnaire's native scales."""
    import pandas as pd

    questionnaire = pd.read_csv(root / "qstn_data" / "questionnaire.csv")
    specs: dict[str, ProbabilitySpec] = {}
    for row in questionnaire.itertuples(index=False):
        codes = tuple(json.loads(row.answer_codes))
        if row.questionnaire_item_id == "newsletter_signup":
            bins = NEWSLETTER_BINS
        elif codes == (0, 100):
            bins = SLIDER_BINS
        elif codes == (0, 10):
            bins = DONATION_BINS
        else:
            raise ValueError(
                f"No probability-bin configuration for {row.questionnaire_item_id} "
                f"with answer codes {codes}."
            )
        specs[row.questionnaire_item_id] = ProbabilitySpec(row.questionnaire_item_id, bins)
    return specs
