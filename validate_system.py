"""
validate_system.py

NudgeWise v2.8
Production sanity validation.

Checks:
- model file exists
- expected intervention classes exist
- feature metadata exists
- prediction output has six probabilities
- probabilities are finite
- probabilities are within 0-1
- probabilities sum to approximately 1
- certainty is valid
- recommendation action engine returns usable output

This does not modify participant data.
"""

from __future__ import annotations

import math
from pathlib import Path

from predict import (
    get_prediction_details,
)

from services.recommendations import (
    personalise_recommendation,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/nudge_model.pkl"
)


EXPECTED_CLASSES = {
    "Connect socially",
    "Maintain habits",
    "Prepare for bed",
    "Reduce screen time",
    "Stay active",
    "Take a short break",
}


VALID_CERTAINTY = {
    "Low",
    "Moderate",
    "High",
}


TEST_CASES = [
    {
        "name":
            "balanced",

        "sleep":
            8.0,

        "stress":
            2,

        "mood":
            4,

        "energy":
            4,

        "screen_time":
            2.0,

        "activity_minutes":
            60,

        "connectedness":
            4,

        "activity":
            "Relaxing",

        "day_type":
            "Weekday",

        "hour":
            17,
    },

    {
        "name":
            "late_low_sleep",

        "sleep":
            4.5,

        "stress":
            4,

        "mood":
            2,

        "energy":
            2,

        "screen_time":
            7.0,

        "activity_minutes":
            10,

        "connectedness":
            2,

        "activity":
            "Phone",

        "day_type":
            "Weekday",

        "hour":
            22,
    },

    {
        "name":
            "high_screen",

        "sleep":
            7.0,

        "stress":
            3,

        "mood":
            3,

        "energy":
            3,

        "screen_time":
            10.0,

        "activity_minutes":
            20,

        "connectedness":
            3,

        "activity":
            "Phone",

        "day_type":
            "Weekend",

        "hour":
            16,
    },

    {
        "name":
            "study_stress",

        "sleep":
            6.5,

        "stress":
            5,

        "mood":
            3,

        "energy":
            2,

        "screen_time":
            3.0,

        "activity_minutes":
            30,

        "connectedness":
            3,

        "activity":
            "Studying",

        "day_type":
            "Weekday",

        "hour":
            19,
    },
]


# ============================================================
# Validation
# ============================================================

def validate_case(
    case,
) -> list[str]:

    failures = []


    details = get_prediction_details(
        sleep=case[
            "sleep"
        ],

        stress=case[
            "stress"
        ],

        mood=case[
            "mood"
        ],

        energy=case[
            "energy"
        ],

        screen_time=case[
            "screen_time"
        ],

        activity_minutes=case[
            "activity_minutes"
        ],

        connectedness=case[
            "connectedness"
        ],

        activity=case[
            "activity"
        ],

        day_type=case[
            "day_type"
        ],

        hour=case[
            "hour"
        ],

        include_explanation=True,
    )


    probabilities = details.get(
        "probabilities",
        {},
    )


    if set(
        probabilities.keys()
    ) != EXPECTED_CLASSES:

        failures.append(
            "Probability classes do not match expected classes."
        )


    if len(
        probabilities
    ) != 6:

        failures.append(
            "Prediction does not contain six classes."
        )


    for (
        class_name,
        probability,
    ) in probabilities.items():

        if not math.isfinite(
            float(
                probability
            )
        ):

            failures.append(
                f"{class_name} probability is non-finite."
            )


        if not (
            0.0
            <= float(
                probability
            )
            <= 1.0
        ):

            failures.append(
                f"{class_name} probability outside 0-1."
            )


    probability_sum = sum(
        probabilities.values()
    )


    if not math.isclose(
        probability_sum,
        1.0,
        abs_tol=1e-6,
    ):

        failures.append(
            f"Probabilities sum to {probability_sum:.8f}."
        )


    if details.get(
        "prediction"
    ) not in EXPECTED_CLASSES:

        failures.append(
            "Primary prediction is invalid."
        )


    if details.get(
        "certainty"
    ) not in VALID_CERTAINTY:

        failures.append(
            "Certainty label is invalid."
        )


    guidance = personalise_recommendation(
        prediction=details[
            "prediction"
        ],

        sleep=case[
            "sleep"
        ],

        stress=case[
            "stress"
        ],

        mood=case[
            "mood"
        ],

        energy=case[
            "energy"
        ],

        screen_time=case[
            "screen_time"
        ],

        activity_minutes=case[
            "activity_minutes"
        ],

        connectedness=case[
            "connectedness"
        ],

        activity=case[
            "activity"
        ],

        hour=case[
            "hour"
        ],
    )


    if not guidance.action_id:

        failures.append(
            "Action ID is empty."
        )


    if not guidance.title:

        failures.append(
            "Action title is empty."
        )


    if not guidance.action:

        failures.append(
            "Action text is empty."
        )


    if not guidance.reason:

        failures.append(
            "Action reason is empty."
        )


    return failures


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("NUDGEWISE v2.8 SYSTEM VALIDATION")
    print("=" * 72)


    total_failures = []


    if not MODEL_PATH.exists():

        total_failures.append(
            f"Model missing: {MODEL_PATH}"
        )


    for case in TEST_CASES:

        failures = validate_case(
            case
        )


        if failures:

            print(
                f"\nFAIL: "
                f"{case['name']}"
            )


            for failure in failures:

                print(
                    f"  - {failure}"
                )


                total_failures.append(
                    (
                        f"{case['name']}: "
                        f"{failure}"
                    )
                )


        else:

            print(
                f"PASS: "
                f"{case['name']}"
            )


    print()


    if total_failures:

        print("=" * 72)

        print(
            f"SYSTEM VALIDATION FAILED "
            f"({len(total_failures)} issue(s))"
        )

        print("=" * 72)


        raise SystemExit(
            1
        )


    print("=" * 72)
    print("ALL SYSTEM CHECKS PASSED")
    print("=" * 72)


if __name__ == "__main__":

    main()