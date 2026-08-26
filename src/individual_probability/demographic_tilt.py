"""Frozen construct and demographic calibration parameters for benchmark sampling."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


GLOBAL_TEMPERATURE = 1.4064310604266295
CONSTRUCT_TEMPERATURES = {
    "belief_post": 0.36271534931013,
    "concern_1": 1.9554423583950935,
    "concern_2": 1.5940889004083494,
    "concern_3": 9.82736152780313,
    "policy_general": 0.6849888675516411,
    "policy_specific_1": 0.14778915020858457,
    "policy_specific_3": 1.5425426142131766,
    "behavior_talk": 3.5024018832607413,
}
BENCHMARK_ITEM_TO_CONSTRUCT = {
    "belief_post": "belief_causation",
    "concern_1": "concern_worry",
    "concern_2": "concern_harm",
    "concern_3": "concern_priority",
    "policy_general": "broad_policy",
    "policy_specific_1": "fossil_tax",
    "policy_specific_3": "renewables",
    "behavior_talk": "discussion",
}
FEATURE_DEFINITIONS = (
    ("gender_h", "Female"),
    ("gender_h", "Other"),
    ("age_band_h", "30-44"),
    ("age_band_h", "45-59"),
    ("age_band_h", "60+"),
    ("race_h", "Black"),
    ("race_h", "Hispanic"),
    ("race_h", "Other"),
    ("education_h", "High school"),
    ("education_h", "Some college"),
    ("education_h", "Bachelor's degree or higher"),
    ("income_h", "$50,000 to $99,999"),
    ("income_h", "$100,000 or more"),
    ("party_h", "Democrat"),
    ("party_h", "Independent"),
    ("party_h", "Other"),
)
DEMOGRAPHIC_TILT_RIDGE = 0.01
DEMOGRAPHIC_TILT_SELECTION_WAVE = 31
CONSTRUCT_TILT_COEFFICIENTS = {'belief_causation': (0.23995698622530162,
                      -0.029514527881765097,
                      -0.08772620882878449,
                      -0.38940448990314447,
                      -0.7832277473812658,
                      0.045660770283321946,
                      0.4031611146367538,
                      0.0649057097476391,
                      -0.9487977783224555,
                      -0.5596132921241737,
                      -0.24789490824863128,
                      -0.40387331189051434,
                      -0.705376846520105,
                      -0.18338648309732877,
                      0.6949704752948961,
                      -0.07693543665926285),
 'concern_worry': (-0.07818428435747292,
                   -0.01511878007641213,
                   0.07844218525586273,
                   -0.0603200165620662,
                   0.06337588713752378,
                   -0.11821079109877322,
                   0.14397883860865393,
                   0.46218021384538926,
                   -0.13194839397815883,
                   -0.1828161572796112,
                   0.05023680635485183,
                   -0.11964625567400335,
                   -0.00686178830551956,
                   0.0908272854534161,
                   -0.00582120425007099,
                   -0.22961646817602177),
 'concern_harm': (-0.30234699973920104,
                  -0.0008564814808873373,
                  0.15833718114694423,
                  -0.3541444643948399,
                  -0.897853597619672,
                  -0.29310035095627,
                  0.4454673243881217,
                  0.3020992384095349,
                  -1.616352111144164,
                  0.18325025965809622,
                  1.3720396814178388,
                  -0.4461221033530217,
                  0.36702646999247684,
                  1.6719783364870286,
                  0.5109330031661614,
                  -0.2569049784634856),
 'concern_priority': (0.07886912368412304,
                      -0.019090269530405206,
                      -0.13070502340532592,
                      -0.38248596006319174,
                      -0.4467746811260457,
                      0.39133758796889223,
                      0.29508161491517493,
                      0.4586398710570185,
                      -0.2953739884655138,
                      -0.3199140645692172,
                      0.09127467215147368,
                      -0.34754443873586327,
                      -0.3368122656876739,
                      1.4472593152407613,
                      0.3428680797859679,
                      0.27917314071048116),
 'broad_policy': (0.04632907529952581,
                  0.030357229029794264,
                  0.45500031152538234,
                  0.5311687010475294,
                  1.0918325439284495,
                  -0.26484497299699633,
                  0.21139714978929516,
                  0.3243711887391438,
                  0.6575626026021508,
                  0.2779769781145414,
                  0.1186021117199814,
                  0.3309693676341824,
                  0.6009437863231213,
                  -1.6480969640848695,
                  -0.19034393803810526,
                  -0.5327479660179392),
 'fossil_tax': (0.7205082226160675,
                0.04011868821459131,
                0.4578964301624514,
                0.39151843989957696,
                -0.25961490176252733,
                0.013326728321692472,
                0.3589703227198039,
                0.16536946169871208,
                0.01568650629415657,
                0.23176186372012156,
                0.6002316446793646,
                0.18785629730543227,
                0.8303164065436729,
                -0.8925356113379571,
                0.7154092901600939,
                0.23513933894059133),
 'renewables': (-0.06907642518240086,
                -0.0064850971491666135,
                0.07175454790720502,
                -0.013831090708927242,
                0.12080641696193556,
                0.05256057769105887,
                0.09198049890884781,
                0.25235324269046566,
                0.09532888715196552,
                -0.05417115680537689,
                -0.0014055428740136237,
                -0.033085239242231515,
                0.013478900770832714,
                -0.17123859377710116,
                -0.05581958691235112,
                -0.17415456370122995),
 'discussion': (-0.2773616428293995,
                -0.1171419031178121,
                -0.2958045962949945,
                -0.27393386651928747,
                -0.2842378596389539,
                -0.3743349747389589,
                -0.3138088110819354,
                -0.046854831009195366,
                -0.568770062665474,
                -0.49998073805187526,
                -0.33811467376663296,
                -0.2872322934085741,
                -0.28737287325299765,
                -0.1501719592247761,
                -0.19321085308270963,
                -0.4148070002472071)}
DEMOGRAPHIC_TILTS = {
    item_id: CONSTRUCT_TILT_COEFFICIENTS[construct]
    for item_id, construct in BENCHMARK_ITEM_TO_CONSTRUCT.items()
}


def harmonized_demographics(record: Mapping[str, Any]) -> dict[str, str]:
    race = {
        "White / Caucasian": "White", "White": "White",
        "Black / African American": "Black", "Black": "Black",
        "Hispanic / Latino": "Hispanic", "Hispanic": "Hispanic",
        "Asian / Asian American": "Other", "Other": "Other",
    }
    education = {
        "Less than high school": "Less than high school",
        "High school diploma / GED": "High school",
        "Some college or Associate's degree": "Some college",
        "Bachelor's degree": "Bachelor's degree or higher",
        "Master's degree / Professional degree": "Bachelor's degree or higher",
        "Doctorate degree / Ph.D.": "Bachelor's degree or higher",
        "High school": "High school",
        "Some college": "Some college",
        "Bachelor's degree or higher": "Bachelor's degree or higher",
    }
    income = {
        "Less than $30,000": "Less than $50,000",
        "$30,000 to $55,999": "Less than $50,000",
        "$56,000 to $99,999": "$50,000 to $99,999",
        "$100,000 to $167,999": "$100,000 or more",
        "$168,000 or more": "$100,000 or more",
        "Less than $50,000": "Less than $50,000",
        "$50,000-$99,999": "$50,000 to $99,999",
        "$50,000 to $99,999": "$50,000 to $99,999",
        "$100,000+": "$100,000 or more",
        "$100,000 or more": "$100,000 or more",
    }
    values = {
        "gender_h": {
            "Male": "Male", "Man": "Male",
            "Female": "Female", "Woman": "Female",
        }.get(record["gender"], "Other"),
        "age_band_h": record["age_band"],
        "race_h": race[record["race"]],
        "education_h": education[record["education"]],
        "income_h": income[record["income"]],
        "party_h": record["party"],
    }
    return values


def demographic_feature_vector(record: Mapping[str, Any]) -> np.ndarray:
    values = harmonized_demographics(record)
    return np.array([
        float(values[column] == level)
        for column, level in FEATURE_DEFINITIONS
    ])


def normalized_bin_scores(bins: Sequence[tuple[str, int, int]]) -> np.ndarray:
    midpoints = np.array([(lower + upper) / 2 for _, lower, upper in bins], dtype=float)
    span = float(midpoints.max() - midpoints.min())
    if span <= 0:
        return np.zeros(len(midpoints), dtype=float)
    return (midpoints - midpoints.min()) / span


def apply_ordinal_tilt(
    probabilities: Sequence[float],
    bin_scores: Sequence[float],
    feature_values: Sequence[float],
    coefficients: Sequence[float],
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    scores = np.asarray(bin_scores, dtype=float)
    scores -= scores.mean()
    features = np.asarray(feature_values, dtype=float)
    coefficient_values = np.asarray(coefficients, dtype=float)
    if features.shape != coefficient_values.shape:
        raise ValueError("Demographic feature and coefficient dimensions differ")
    logits = np.log(values + 1e-8)
    logits += float(features @ coefficient_values) * scores
    logits -= logits.max()
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum()
