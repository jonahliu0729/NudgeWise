"""
predict.py

NudgeWise AI v2.6
Production soft-probability prediction and explainability interface.

The model predicts a probability distribution across six
digital wellbeing interventions.

This module provides:
- primary recommendation
- full probability distribution
- alternative recommendation
- uncertainty measures
- model certainty category
- local model sensitivity explanations
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "nudge_model.pkl"
)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"NudgeWise model not found: {MODEL_PATH}"
    )


# ============================================================
# Load trained bundle
# ============================================================

BUNDLE = joblib.load(
    MODEL_PATH
)

MODEL = BUNDLE["model"]

CLASSES = list(
    BUNDLE["classes"]
)

FEATURES = list(
    BUNDLE["features"]
)


# ============================================================
# Feature reference values
# ============================================================

# These are neutral/reference states used only for local
# sensitivity analysis.
#
# They are NOT clinical targets and are NOT used to override
# the model prediction.

REFERENCE_VALUES = {
    "sleep": 8.0,
    "stress": 3,
    "mood": 3,
    "energy": 3,
    "screen_time": 3.0,
    "activity": "Relaxing",
    "social": "Medium",
    "hour": 15,
    "day_type": "Weekday",
}


DISPLAY_NAMES = {
    "sleep": "sleep",
    "stress": "stress",
    "mood": "mood",
    "energy": "energy",
    "screen_time": "recreational screen time",
    "activity": "current activity",
    "social": "social interaction",
    "hour": "time of day",
    "day_type": "day context",
}


# ============================================================
# Probability normalisation
# ============================================================

def normalise_probabilities(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim == 1:

        values = values.reshape(
            1,
            -1,
        )

    values = np.clip(
        values,
        0.0,
        None,
    )

    values += 1e-12

    row_sums = values.sum(
        axis=1,
        keepdims=True,
    )

    return (
        values
        / row_sums
    )


# ============================================================
# Input builder
# ============================================================

def build_input(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity,
    day_type,
    social,
    hour,
):

    dataframe = pd.DataFrame(
        {
            "sleep": [sleep],
            "stress": [stress],
            "mood": [mood],
            "energy": [energy],
            "screen_time": [screen_time],
            "activity": [activity],
            "social": [social],
            "hour": [hour],
            "day_type": [day_type],
        }
    )

    return dataframe[
        FEATURES
    ]


# ============================================================
# Full probability distribution
# ============================================================

def predict_probabilities(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity,
    day_type,
    social,
    hour,
):

    user_data = build_input(
        sleep=sleep,
        stress=stress,
        mood=mood,
        energy=energy,
        screen_time=screen_time,
        activity=activity,
        day_type=day_type,
        social=social,
        hour=hour,
    )

    raw_prediction = MODEL.predict(
        user_data
    )

    probabilities = (
        normalise_probabilities(
            raw_prediction
        )[0]
    )

    return {
        class_name: float(probability)
        for class_name, probability
        in zip(
            CLASSES,
            probabilities,
        )
    }


# ============================================================
# Uncertainty calculations
# ============================================================

def calculate_uncertainty(
    probabilities: dict[str, float],
):

    ordered = sorted(
        probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    top_probability = (
        ordered[0][1]
    )

    if len(ordered) >= 2:

        probability_margin = (
            ordered[0][1]
            - ordered[1][1]
        )

    else:

        probability_margin = 1.0

    values = np.asarray(
        list(
            probabilities.values()
        ),
        dtype=float,
    )

    entropy = float(
        -np.sum(
            values
            * np.log(
                values + 1e-12
            )
        )
    )

    maximum_entropy = float(
        np.log(
            max(
                len(values),
                1,
            )
        )
    )

    if maximum_entropy > 0:

        normalised_entropy = (
            entropy
            / maximum_entropy
        )

    else:

        normalised_entropy = 0.0

    return {
        "top_probability":
            float(
                top_probability
            ),

        "probability_margin":
            float(
                probability_margin
            ),

        "entropy":
            entropy,

        "normalised_entropy":
            float(
                normalised_entropy
            ),
    }


# ============================================================
# Certainty category
# ============================================================

def classify_certainty(
    top_probability: float,
    probability_margin: float,
    normalised_entropy: float,
) -> str:
    """
    Convert distribution shape into a restrained certainty label.

    This is a product-level interpretation of model uncertainty.

    It is NOT a claim that the recommendation is clinically
    correct or has a certain probability of being effective.
    """

    certainty_score = (
        0.45 * top_probability
        + 0.35 * min(
            probability_margin * 2.0,
            1.0,
        )
        + 0.20 * (
            1.0
            - normalised_entropy
        )
    )

    if certainty_score >= 0.58:
        return "High"

    if certainty_score >= 0.38:
        return "Moderate"

    return "Low"


# ============================================================
# Main compatibility function
# ============================================================

def predict_nudge(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity,
    day_type,
    social,
    hour,
):
    """
    Return:
        primary recommendation
        associated model probability

    Retained for compatibility with existing NudgeWise code.
    """

    details = get_prediction_details(
        sleep=sleep,
        stress=stress,
        mood=mood,
        energy=energy,
        screen_time=screen_time,
        activity=activity,
        day_type=day_type,
        social=social,
        hour=hour,
        include_explanation=False,
    )

    return (
        details["prediction"],
        details["confidence"],
    )


# ============================================================
# Local model sensitivity
# ============================================================

def calculate_local_sensitivity(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity,
    day_type,
    social,
    hour,
    prediction,
):
    """
    Estimate which inputs most influenced the current prediction.

    Method:
    Each feature is individually replaced with a neutral reference
    value. The reduction in the selected recommendation probability
    is measured.

    A larger reduction means the original feature value was more
    supportive of that recommendation.

    This is local model sensitivity, NOT causation.
    """

    original_values = {
        "sleep": sleep,
        "stress": stress,
        "mood": mood,
        "energy": energy,
        "screen_time": screen_time,
        "activity": activity,
        "day_type": day_type,
        "social": social,
        "hour": hour,
    }

    original_probabilities = (
        predict_probabilities(
            **original_values
        )
    )

    original_probability = (
        original_probabilities[
            prediction
        ]
    )

    sensitivities = []

    for feature_name in FEATURES:

        if (
            feature_name
            not in REFERENCE_VALUES
        ):
            continue

        perturbed = (
            original_values.copy()
        )

        perturbed[
            feature_name
        ] = REFERENCE_VALUES[
            feature_name
        ]

        perturbed_probabilities = (
            predict_probabilities(
                **perturbed
            )
        )

        perturbed_probability = (
            perturbed_probabilities.get(
                prediction,
                0.0,
            )
        )

        effect = (
            original_probability
            - perturbed_probability
        )

        sensitivities.append(
            {
                "feature":
                    feature_name,

                "display_name":
                    DISPLAY_NAMES.get(
                        feature_name,
                        feature_name,
                    ),

                "effect":
                    float(
                        effect
                    ),

                "original_value":
                    original_values[
                        feature_name
                    ],

                "reference_value":
                    REFERENCE_VALUES[
                        feature_name
                    ],
            }
        )

    sensitivities.sort(
        key=lambda item:
            abs(
                item["effect"]
            ),
        reverse=True,
    )

    return sensitivities


# ============================================================
# Human-readable model explanation
# ============================================================

def explain_prediction(
    prediction,
    sleep,
    stress,
    screen_time,
    hour,
    energy,
    mood,
    activity="Relaxing",
    social="Medium",
    day_type="Weekday",
):
    """
    Return model-linked explanation text.

    Explanations describe local model sensitivity rather than
    claiming that an input caused the recommendation.
    """

    sensitivities = (
        calculate_local_sensitivity(
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity=activity,
            day_type=day_type,
            social=social,
            hour=hour,
            prediction=prediction,
        )
    )

    positive = [
        item
        for item
        in sensitivities
        if item[
            "effect"
        ] > 0.005
    ]

    if not positive:

        return [
            (
                "This recommendation reflects the "
                "combined pattern across your latest check-in."
            )
        ]

    reasons = []

    for item in positive[:3]:

        name = item[
            "display_name"
        ]

        reasons.append(
            (
                f"Your {name} was one of the inputs "
                f"that increased the model's preference "
                f"for this recommendation."
            )
        )

    return reasons


# ============================================================
# Full prediction details
# ============================================================

def get_prediction_details(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity,
    day_type,
    social,
    hour,
    include_explanation=True,
):

    probabilities = (
        predict_probabilities(
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity=activity,
            day_type=day_type,
            social=social,
            hour=hour,
        )
    )

    ordered = sorted(
        probabilities.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    prediction = (
        ordered[0][0]
    )

    confidence = (
        ordered[0][1]
    )

    uncertainty = (
        calculate_uncertainty(
            probabilities
        )
    )

    certainty = (
        classify_certainty(
            top_probability=(
                uncertainty[
                    "top_probability"
                ]
            ),
            probability_margin=(
                uncertainty[
                    "probability_margin"
                ]
            ),
            normalised_entropy=(
                uncertainty[
                    "normalised_entropy"
                ]
            ),
        )
    )

    result = {
        "prediction":
            prediction,

        # Compatibility name.
        # UI should call this model probability.
        "confidence":
            confidence,

        "probabilities":
            probabilities,

        "ordered_probabilities":
            ordered,

        "top_two":
            ordered[:2],

        "alternative":
            (
                ordered[1]
                if len(ordered) > 1
                else None
            ),

        "probability_margin":
            uncertainty[
                "probability_margin"
            ],

        "entropy":
            uncertainty[
                "entropy"
            ],

        "normalised_entropy":
            uncertainty[
                "normalised_entropy"
            ],

        "certainty":
            certainty,
    }

    if include_explanation:

        result[
            "sensitivities"
        ] = calculate_local_sensitivity(
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity=activity,
            day_type=day_type,
            social=social,
            hour=hour,
            prediction=prediction,
        )

        result[
            "reasons"
        ] = explain_prediction(
            prediction=prediction,
            sleep=sleep,
            stress=stress,
            screen_time=screen_time,
            hour=hour,
            energy=energy,
            mood=mood,
            activity=activity,
            social=social,
            day_type=day_type,
        )

    return result


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    details = get_prediction_details(
        sleep=7.5,
        stress=1,
        mood=1,
        energy=1,
        screen_time=3.0,
        activity="Studying",
        day_type="Weekday",
        social="Low",
        hour=23,
    )

    print()
    print("=" * 70)
    print("NUDGEWISE AI V2.6")
    print("=" * 70)

    print(
        f"\nPrimary recommendation: "
        f"{details['prediction']}"
    )

    print(
        f"Model probability: "
        f"{details['confidence']:.1%}"
    )

    print(
        f"Model certainty: "
        f"{details['certainty']}"
    )

    if details["alternative"]:

        print(
            f"Alternative: "
            f"{details['alternative'][0]} "
            f"({details['alternative'][1]:.1%})"
        )

    print(
        "\nProbability distribution:"
    )

    for (
        class_name,
        probability,
    ) in details[
        "ordered_probabilities"
    ]:

        print(
            f"{class_name:<24}"
            f"{probability:>8.1%}"
        )

    print(
        f"\nTop-two margin: "
        f"{details['probability_margin']:.3f}"
    )

    print(
        f"Normalised entropy: "
        f"{details['normalised_entropy']:.3f}"
    )

    print(
        "\nLocal explanation:"
    )

    for reason in details[
        "reasons"
    ]:

        print(
            f"- {reason}"
        )

    print()
    print("=" * 70)