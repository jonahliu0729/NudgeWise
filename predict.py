"""
predict.py

NudgeWise AI v2.6
Production soft-probability prediction and explainability interface.

The trained model predicts a probability distribution across six
digital wellbeing interventions.

Production features:
- sleep
- stress
- mood
- energy
- screen_time
- activity_minutes
- connectedness
- hour
- activity
- day_type

This module provides:
- primary recommendation
- full probability distribution
- alternative recommendation
- uncertainty measures
- model certainty category
- local model sensitivity explanations

IMPORTANT
---------
Model probabilities describe the preference distribution learned from
the v2.6 synthetic decision generator.

They are NOT probabilities that an intervention will improve wellbeing.
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


MODEL = BUNDLE[
    "model"
]


CLASSES = list(
    BUNDLE[
        "classes"
    ]
)


FEATURES = list(
    BUNDLE[
        "features"
    ]
)


MODEL_VERSION = str(
    BUNDLE.get(
        "version",
        "unknown",
    )
)


# ============================================================
# Production schema validation
# ============================================================

EXPECTED_FEATURES = [
    "sleep",
    "stress",
    "mood",
    "energy",
    "screen_time",
    "activity_minutes",
    "connectedness",
    "hour",
    "activity",
    "day_type",
]


if FEATURES != EXPECTED_FEATURES:

    raise ValueError(
        "Loaded NudgeWise model does not match the "
        "expected v2.6 production feature schema.\n"
        f"Expected: {EXPECTED_FEATURES}\n"
        f"Loaded:   {FEATURES}"
    )


# ============================================================
# Feature reference values
# ============================================================

# These values are used ONLY for local sensitivity analysis.
#
# They are not clinical targets and do not override the model.

REFERENCE_VALUES = {
    "sleep": 8.0,

    "stress": 3,

    "mood": 3,

    "energy": 3,

    "screen_time": 3.0,

    "activity_minutes": 60,

    "connectedness": 3,

    "hour": 15,

    "activity": "Relaxing",

    "day_type": "Weekday",
}


DISPLAY_NAMES = {
    "sleep":
        "sleep",

    "stress":
        "stress",

    "mood":
        "mood",

    "energy":
        "energy",

    "screen_time":
        "recreational screen time",

    "activity_minutes":
        "physical activity",

    "connectedness":
        "social connectedness",

    "hour":
        "time of day",

    "activity":
        "current context",

    "day_type":
        "day context",
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
# Input validation
# ============================================================

def validate_input(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity_minutes,
    connectedness,
    activity,
    day_type,
    hour,
) -> None:
    """
    Validate production inputs before passing them to the model.

    Validation protects against malformed values but deliberately
    avoids silently changing unusual user responses.
    """

    if not 0.0 <= float(sleep) <= 16.0:

        raise ValueError(
            "sleep must be between 0 and 16 hours."
        )


    if int(stress) not in range(
        1,
        6,
    ):

        raise ValueError(
            "stress must be between 1 and 5."
        )


    if int(mood) not in range(
        1,
        6,
    ):

        raise ValueError(
            "mood must be between 1 and 5."
        )


    if int(energy) not in range(
        1,
        6,
    ):

        raise ValueError(
            "energy must be between 1 and 5."
        )


    if not 0.0 <= float(screen_time) <= 24.0:

        raise ValueError(
            "screen_time must be between 0 and 24 hours."
        )


    if not 0 <= int(activity_minutes) <= 300:

        raise ValueError(
            "activity_minutes must be between 0 and 300."
        )


    if int(connectedness) not in range(
        1,
        6,
    ):

        raise ValueError(
            "connectedness must be between 1 and 5."
        )


    if not 0 <= int(hour) <= 23:

        raise ValueError(
            "hour must be between 0 and 23."
        )


    valid_activities = {
        "Phone",
        "Studying",
        "Working",
        "Relaxing",
        "Exercise",
    }


    if activity not in valid_activities:

        raise ValueError(
            f"Unknown activity context: {activity}"
        )


    valid_day_types = {
        "Weekday",
        "Weekend",
    }


    if day_type not in valid_day_types:

        raise ValueError(
            f"Unknown day_type: {day_type}"
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
    activity_minutes,
    connectedness,
    activity,
    day_type,
    hour,
):

    validate_input(
        sleep=sleep,
        stress=stress,
        mood=mood,
        energy=energy,
        screen_time=screen_time,
        activity_minutes=activity_minutes,
        connectedness=connectedness,
        activity=activity,
        day_type=day_type,
        hour=hour,
    )


    dataframe = pd.DataFrame(
        {
            "sleep":
                [float(sleep)],

            "stress":
                [int(stress)],

            "mood":
                [int(mood)],

            "energy":
                [int(energy)],

            "screen_time":
                [float(screen_time)],

            "activity_minutes":
                [int(activity_minutes)],

            "connectedness":
                [int(connectedness)],

            "hour":
                [int(hour)],

            "activity":
                [activity],

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
    activity_minutes,
    connectedness,
    activity,
    day_type,
    hour,
):

    user_data = build_input(
        sleep=sleep,
        stress=stress,
        mood=mood,
        energy=energy,
        screen_time=screen_time,
        activity_minutes=activity_minutes,
        connectedness=connectedness,
        activity=activity,
        day_type=day_type,
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
# Uncertainty calculations
# ============================================================

def calculate_uncertainty(
    probabilities: dict[str, float],
):

    ordered = sorted(
        probabilities.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )


    top_probability = (
        ordered[
            0
        ][1]
    )


    if len(
        ordered
    ) >= 2:

        probability_margin = (
            ordered[
                0
            ][1]
            - ordered[
                1
            ][1]
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
                values
                + 1e-12
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
    Convert probability-distribution shape into a restrained
    certainty category.

    This is a product-level interpretation of model ambiguity.

    It does not represent clinical certainty.
    """

    certainty_score = (
        0.45
        * top_probability

        + 0.35
        * min(
            probability_margin
            * 2.0,
            1.0,
        )

        + 0.20
        * (
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
# Main prediction function
# ============================================================

def predict_nudge(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity_minutes,
    connectedness,
    activity,
    day_type,
    hour,
):
    """
    Return:
        primary recommendation
        associated model probability
    """

    details = get_prediction_details(
        sleep=sleep,
        stress=stress,
        mood=mood,
        energy=energy,
        screen_time=screen_time,
        activity_minutes=activity_minutes,
        connectedness=connectedness,
        activity=activity,
        day_type=day_type,
        hour=hour,
        include_explanation=False,
    )


    return (
        details[
            "prediction"
        ],

        details[
            "confidence"
        ],
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
    activity_minutes,
    connectedness,
    activity,
    day_type,
    hour,
    prediction,
):
    """
    Estimate which inputs most supported the selected
    recommendation.

    Method
    ------
    Each feature is independently replaced with a neutral
    reference value.

    The change in the selected recommendation's probability
    is measured.

    Positive effect:
        original value supported the recommendation

    Negative effect:
        original value reduced the recommendation

    This is local model sensitivity.

    It is NOT causal inference.
    """

    original_values = {
        "sleep":
            sleep,

        "stress":
            stress,

        "mood":
            mood,

        "energy":
            energy,

        "screen_time":
            screen_time,

        "activity_minutes":
            activity_minutes,

        "connectedness":
            connectedness,

        "hour":
            hour,

        "activity":
            activity,

        "day_type":
            day_type,
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

        if feature_name not in REFERENCE_VALUES:

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
                item[
                    "effect"
                ]
            ),
        reverse=True,
    )


    return sensitivities


# ============================================================
# Human-readable explanation helper
# ============================================================

def _human_reason(
    feature: str,
    value,
) -> str:
    """
    Convert influential features into restrained user-facing
    explanation text.
    """

    if feature == "sleep":

        return (
            f"Your reported sleep was {float(value):.1f} hours, "
            "which influenced the model's preference."
        )


    if feature == "stress":

        return (
            f"Your stress rating was {int(value)}/5, "
            "which influenced the model's preference."
        )


    if feature == "mood":

        return (
            f"Your mood rating was {int(value)}/5, "
            "which influenced the model's preference."
        )


    if feature == "energy":

        return (
            f"Your energy rating was {int(value)}/5, "
            "which influenced the model's preference."
        )


    if feature == "screen_time":

        return (
            f"You reported {float(value):.1f} hours of recreational "
            "screen time, which influenced the model's preference."
        )


    if feature == "activity_minutes":

        return (
            f"You reported about {int(value)} minutes of physical "
            "activity, which influenced the model's preference."
        )


    if feature == "connectedness":

        return (
            f"Your connectedness rating was {int(value)}/5, "
            "which influenced the model's preference."
        )


    if feature == "activity":

        return (
            f"Your current context was {str(value).lower()}, "
            "which affected how appropriate different suggestions "
            "were at this moment."
        )


    if feature == "hour":

        return (
            "The time of day influenced which suggestions "
            "were most contextually appropriate."
        )


    if feature == "day_type":

        return (
            "Whether this was a weekday or weekend had a small "
            "contextual influence on the model."
        )


    return (
        f"Your {DISPLAY_NAMES.get(feature, feature)} "
        "influenced the model's preference."
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
    activity_minutes,
    connectedness,
    activity="Relaxing",
    day_type="Weekday",
):
    """
    Return local model-sensitivity explanations.

    The wording intentionally avoids causal claims.
    """

    sensitivities = (
        calculate_local_sensitivity(
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity_minutes=activity_minutes,
            connectedness=connectedness,
            activity=activity,
            day_type=day_type,
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
                "This recommendation reflects the combined "
                "pattern across your latest check-in rather than "
                "one dominant input."
            )
        ]


    reasons = []


    for item in positive[
        :3
    ]:

        reasons.append(
            _human_reason(
                feature=item[
                    "feature"
                ],

                value=item[
                    "original_value"
                ],
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
    activity_minutes,
    connectedness,
    activity,
    day_type,
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
            activity_minutes=activity_minutes,
            connectedness=connectedness,
            activity=activity,
            day_type=day_type,
            hour=hour,
        )
    )


    ordered = sorted(
        probabilities.items(),
        key=lambda item:
            item[
                1
            ],
        reverse=True,
    )


    prediction = (
        ordered[
            0
        ][0]
    )


    confidence = (
        ordered[
            0
        ][1]
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

        # Compatibility key.
        #
        # UI wording should call this:
        # model probability
        #
        # NOT accuracy.
        "confidence":
            confidence,

        "probabilities":
            probabilities,

        "ordered_probabilities":
            ordered,

        "top_two":
            ordered[
                :2
            ],

        "alternative":
            (
                ordered[
                    1
                ]

                if len(
                    ordered
                ) > 1

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

        "model_version":
            MODEL_VERSION,
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
            activity_minutes=activity_minutes,
            connectedness=connectedness,
            activity=activity,
            day_type=day_type,
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
            activity_minutes=activity_minutes,
            connectedness=connectedness,
            activity=activity,
            day_type=day_type,
        )


    return result


# ============================================================
# Manual test
# ============================================================

if __name__ == "__main__":

    details = get_prediction_details(
        sleep=6.5,
        stress=4,
        mood=3,
        energy=2,
        screen_time=5.0,
        activity_minutes=20,
        connectedness=2,
        activity="Studying",
        day_type="Weekday",
        hour=21,
    )


    print()
    print("=" * 72)
    print(
        "NUDGEWISE AI V2.6 PRODUCTION TEST"
    )
    print("=" * 72)


    print()
    print(
        f"Model version: "
        f"{details['model_version']}"
    )


    print(
        f"Primary recommendation: "
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


    if details[
        "alternative"
    ]:

        print(
            f"Alternative: "
            f"{details['alternative'][0]} "
            f"({details['alternative'][1]:.1%})"
        )


    print()
    print(
        "Probability distribution:"
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


    print()
    print(
        f"Top-two margin: "
        f"{details['probability_margin']:.3f}"
    )


    print(
        f"Normalised entropy: "
        f"{details['normalised_entropy']:.3f}"
    )


    print()
    print(
        "Local explanation:"
    )


    for reason in details[
        "reasons"
    ]:

        print(
            f"- {reason}"
        )


    print()
    print("=" * 72)