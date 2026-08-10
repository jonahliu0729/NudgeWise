"""
predict.py

NudgeWise AI v2.5
Production soft-probability prediction interface.

Instead of predicting a single hard class directly, NudgeWise
predicts a six-intervention probability distribution and selects
the recommendation with the highest estimated probability.
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


# ============================================================
# Load production bundle
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"NudgeWise model not found: "
        f"{MODEL_PATH}"
    )


BUNDLE = joblib.load(
    MODEL_PATH
)


MODEL = BUNDLE[
    "model"
]

CLASSES = BUNDLE[
    "classes"
]

FEATURES = BUNDLE[
    "features"
]


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


    values += 1e-9


    return (
        values
        / values.sum(
            axis=1,
            keepdims=True,
        )
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
            "sleep":
                [sleep],

            "stress":
                [stress],

            "mood":
                [mood],

            "energy":
                [energy],

            "screen_time":
                [screen_time],

            "activity":
                [activity],

            "social":
                [social],

            "hour":
                [hour],

            "day_type":
                [day_type],
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
        class_name:
            float(
                probability
            )

        for (
            class_name,
            probability,
        )
        in zip(
            CLASSES,
            probabilities,
        )
    }


# ============================================================
# Main application function
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
    Return the intervention with the highest predicted probability.

    confidence is the model's predicted probability associated
    with that recommendation.

    It is NOT model accuracy.
    """

    probabilities = predict_probabilities(
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


    prediction = max(
        probabilities,
        key=probabilities.get,
    )


    confidence = (
        probabilities[
            prediction
        ]
    )


    return (
        prediction,
        confidence,
    )


# ============================================================
# Human-readable explanation
# ============================================================

def explain_prediction(
    prediction,
    sleep,
    stress,
    screen_time,
    hour,
    energy,
    mood,
):
    """
    Provide restrained contextual explanations.

    These statements describe relevant user-reported context.
    They are NOT claims of causation or clinical interpretation.
    """

    reasons = []


    if mood <= 2:

        reasons.append(
            "Your reported mood was relatively low."
        )


    if energy <= 2:

        reasons.append(
            "Your reported energy was relatively low."
        )


    if stress >= 4:

        reasons.append(
            "Your reported stress was relatively high."
        )


    if sleep < 7:

        reasons.append(
            "Your reported sleep was below your usual target range."
        )


    if screen_time >= 6:

        reasons.append(
            "Recreational screen use was relatively high in this check-in."
        )


    if hour >= 21:

        reasons.append(
            "The check-in was completed later in the day."
        )


    if not reasons:

        reasons.append(
            "This recommendation reflects the combined pattern "
            "across your latest check-in."
        )


    return reasons[:3]


# ============================================================
# Research detail helper
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
):

    probabilities = predict_probabilities(
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


    ordered = sorted(
        probabilities.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )


    prediction = ordered[
        0
    ][0]


    confidence = ordered[
        0
    ][1]


    # --------------------------------------------------------
    # Margin measures how separated the top two options are.
    # --------------------------------------------------------

    if len(
        ordered
    ) >= 2:

        probability_margin = (
            ordered[0][1]
            - ordered[1][1]
        )

    else:

        probability_margin = 1.0


    # --------------------------------------------------------
    # Entropy is another uncertainty indicator.
    # --------------------------------------------------------

    probability_values = np.array(
        list(
            probabilities.values()
        )
    )


    entropy = float(
        -np.sum(
            probability_values
            * np.log(
                probability_values
                + 1e-12
            )
        )
    )


    return {
        "prediction":
            prediction,

        "confidence":
            confidence,

        "probabilities":
            probabilities,

        "top_two":
            ordered[:2],

        "probability_margin":
            probability_margin,

        "entropy":
            entropy,
    }


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
    print("=" * 66)
    print(
        "NUDGEWISE AI V2.5 TEST"
    )
    print("=" * 66)


    print()
    print(
        "Recommendation:"
    )

    print(
        details[
            "prediction"
        ]
    )


    print()
    print(
        "Predicted probability:"
    )

    print(
        f"{details['confidence']:.1%}"
    )


    print()
    print(
        "Probability distribution:"
    )


    for (
        class_name,
        probability,
    ) in sorted(
        details[
            "probabilities"
        ].items(),
        key=lambda item:
            item[1],
        reverse=True,
    ):

        print(
            f"{class_name:<24}"
            f"{probability:>8.1%}"
        )


    print()
    print(
        f"Top-two margin: "
        f"{details['probability_margin']:.3f}"
    )


    print(
        f"Prediction entropy: "
        f"{details['entropy']:.3f}"
    )


    print()
    print("=" * 66)