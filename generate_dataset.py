"""
generate_dataset.py

NudgeWise AI v2.4
Evidence-informed synthetic behavioural dataset generator.

PURPOSE
-------
Generate synthetic behavioural observations for development and
controlled evaluation of the NudgeWise recommendation model.

IMPORTANT
---------
This dataset is synthetic.

It does NOT represent:
- real adolescent prevalence
- clinical ground truth
- validated intervention outcomes

The generator uses continuous, overlapping behavioural relationships
informed by adolescent sleep, activity, screen-use, social connection
and wellbeing literature.

Exact numerical coefficients are MODELLING ASSUMPTIONS, not published
effect sizes.

PRODUCTION SCHEMA
-----------------
sleep          float, hours
stress         integer, 1-5
mood           integer, 1-5
energy         integer, 1-5
screen_time    float, recreational hours
activity       Phone / Studying / Working / Relaxing / Exercise
social         Low / Medium / High
hour           integer, 0-23
day_type       Weekday / Weekend
best_nudge     one of six recommendation classes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42
N_SAMPLES = 20_000

OUTPUT_PATH = Path("data/training.csv")

RNG = np.random.default_rng(RANDOM_SEED)


# ============================================================
# Production vocabulary
# ============================================================

ACTIVITIES = [
    "Phone",
    "Studying",
    "Working",
    "Relaxing",
    "Exercise",
]

SOCIAL_LEVELS = [
    "Low",
    "Medium",
    "High",
]

DAY_TYPES = [
    "Weekday",
    "Weekend",
]

NUDGES = [
    "Connect socially",
    "Maintain habits",
    "Prepare for bed",
    "Reduce screen time",
    "Stay active",
    "Take a short break",
]


# ============================================================
# Mathematical helpers
# ============================================================

def sigmoid(value):
    """Numerically stable logistic transform."""

    value = np.clip(
        value,
        -30,
        30,
    )

    return 1.0 / (
        1.0 + np.exp(-value)
    )


def softmax(
    values: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Convert intervention utilities into probabilities.

    Lower temperature:
        clearer preference between interventions.

    Higher temperature:
        greater ambiguity.

    Temperature is a modelling parameter, not a clinical quantity.
    """

    scaled = values / temperature

    scaled -= np.max(
        scaled,
        axis=1,
        keepdims=True,
    )

    exponentials = np.exp(
        scaled
    )

    return (
        exponentials
        / exponentials.sum(
            axis=1,
            keepdims=True,
        )
    )


# ============================================================
# Context generation
# ============================================================

def generate_context(
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate weekday/weekend context and time of day."""

    day_type = RNG.choice(
        DAY_TYPES,
        size=n_samples,
        p=[
            5 / 7,
            2 / 7,
        ],
    )

    hours = np.arange(
        6,
        24,
    )

    # App engagement is more likely later in the day.
    hour_weights = np.array(
        [
            0.020,
            0.025,
            0.035,
            0.045,
            0.050,
            0.050,
            0.050,
            0.055,
            0.060,
            0.070,
            0.080,
            0.090,
            0.095,
            0.090,
            0.075,
            0.060,
            0.045,
            0.035,
        ]
    )

    hour_weights /= (
        hour_weights.sum()
    )

    hour = RNG.choice(
        hours,
        size=n_samples,
        p=hour_weights,
    )

    return (
        day_type,
        hour,
    )


# ============================================================
# Core behaviour generation
# ============================================================

def generate_behaviour(
    day_type: np.ndarray,
    hour: np.ndarray,
    n_samples: int,
):
    """
    Generate correlated behavioural observations.

    Relationships are deliberately moderate rather than extreme.
    """

    weekend = (
        day_type
        == "Weekend"
    ).astype(float)

    # --------------------------------------------------------
    # Sleep
    # --------------------------------------------------------

    sleep_mean = (
        7.25
        + 0.70 * weekend
    )

    sleep = RNG.normal(
        loc=sleep_mean,
        scale=1.05,
        size=n_samples,
    )

    sleep = np.clip(
        sleep,
        3.5,
        11.0,
    )


    # --------------------------------------------------------
    # Activity context
    # --------------------------------------------------------

    activity = RNG.choice(
        ACTIVITIES,
        size=n_samples,
        p=[
            0.28,  # Phone
            0.27,  # Studying
            0.09,  # Working
            0.21,  # Relaxing
            0.15,  # Exercise
        ],
    )


    # --------------------------------------------------------
    # Social interaction
    # --------------------------------------------------------

    social = RNG.choice(
        SOCIAL_LEVELS,
        size=n_samples,
        p=[
            0.25,
            0.52,
            0.23,
        ],
    )


    # --------------------------------------------------------
    # Context encodings used ONLY by generator equations
    # --------------------------------------------------------

    activity_effect = np.select(
        [
            activity == "Exercise",
            activity == "Studying",
            activity == "Working",
            activity == "Relaxing",
            activity == "Phone",
        ],
        [
            0.75,
            0.10,
            0.00,
            0.05,
            -0.20,
        ],
        default=0.0,
    )

    social_effect = np.select(
        [
            social == "Low",
            social == "Medium",
            social == "High",
        ],
        [
            -0.65,
            0.00,
            0.55,
        ],
    )


    # --------------------------------------------------------
    # Recreational screen time
    # --------------------------------------------------------

    screen_mean = (
        4.0
        - 0.18
        * (
            sleep
            - 7.5
        )
        - 0.25
        * activity_effect
        + 0.35
        * weekend
    )

    phone_context = (
        activity
        == "Phone"
    ).astype(float)

    evening = sigmoid(
        (
            hour
            - 18
        )
        / 2.5
    )

    screen_mean += (
        1.20
        * phone_context
        + 0.35
        * evening
    )

    screen_time = RNG.normal(
        screen_mean,
        1.25,
        n_samples,
    )

    screen_time = np.clip(
        screen_time,
        0.2,
        11.0,
    )


    return (
        sleep,
        screen_time,
        activity,
        social,
        activity_effect,
        social_effect,
    )


# ============================================================
# Correlated wellbeing states
# ============================================================

def generate_wellbeing(
    sleep,
    screen_time,
    activity_effect,
    social_effect,
    n_samples,
):
    """
    Generate stress, mood and energy on the SAME 1-5 scales used
    by the NudgeWise application.

    These are noisy correlated states, not clinical measures.
    """

    sleep_centered = (
        sleep
        - 7.5
    )

    screen_centered = (
        screen_time
        - 4.0
    )


    # --------------------------------------------------------
    # Stress
    # --------------------------------------------------------

    stress_latent = (
        3.00
        - 0.20
        * sleep_centered
        + 0.13
        * screen_centered
        - 0.22
        * activity_effect
        - 0.28
        * social_effect
        + RNG.normal(
            0,
            0.70,
            n_samples,
        )
    )


    # --------------------------------------------------------
    # Mood
    # --------------------------------------------------------

    mood_latent = (
        3.15
        - 0.34
        * (
            stress_latent
            - 3.0
        )
        + 0.16
        * sleep_centered
        + 0.24
        * social_effect
        + 0.18
        * activity_effect
        + RNG.normal(
            0,
            0.65,
            n_samples,
        )
    )


    # --------------------------------------------------------
    # Energy
    # --------------------------------------------------------

    energy_latent = (
        3.10
        + 0.27
        * sleep_centered
        - 0.23
        * (
            stress_latent
            - 3.0
        )
        - 0.08
        * screen_centered
        + 0.24
        * activity_effect
        + RNG.normal(
            0,
            0.60,
            n_samples,
        )
    )


    # Match Streamlit slider scale exactly.
    stress = np.clip(
        np.rint(
            stress_latent
        ),
        1,
        5,
    ).astype(int)

    mood = np.clip(
        np.rint(
            mood_latent
        ),
        1,
        5,
    ).astype(int)

    energy = np.clip(
        np.rint(
            energy_latent
        ),
        1,
        5,
    ).astype(int)


    return (
        stress,
        mood,
        energy,
    )


# ============================================================
# Continuous behavioural demand features
# ============================================================

def calculate_demands(
    sleep,
    stress,
    mood,
    energy,
    screen_time,
    activity,
    social,
    hour,
):
    """
    Compute smooth behavioural demand signals.

    No demand is a diagnosis.
    These are internal synthetic modelling constructs.
    """

    sleep_need = sigmoid(
        (
            7.8
            - sleep
        )
        / 0.90
    )

    evening_pressure = sigmoid(
        (
            hour
            - 20.0
        )
        / 1.45
    )

    screen_burden = sigmoid(
        (
            screen_time
            - 5.0
        )
        / 1.40
    )

    stress_pressure = sigmoid(
        (
            stress
            - 3.0
        )
        / 0.65
    )

    mood_pressure = sigmoid(
        (
            3.0
            - mood
        )
        / 0.65
    )

    energy_pressure = sigmoid(
        (
            3.0
            - energy
        )
        / 0.65
    )

    social_deficit = np.select(
        [
            social == "Low",
            social == "Medium",
            social == "High",
        ],
        [
            1.0,
            0.45,
            0.05,
        ],
    )

    phone_context = (
        activity
        == "Phone"
    ).astype(float)

    studying_context = (
        activity
        == "Studying"
    ).astype(float)

    working_context = (
        activity
        == "Working"
    ).astype(float)

    exercise_context = (
        activity
        == "Exercise"
    ).astype(float)


    return {
        "sleep_need":
            sleep_need,

        "evening_pressure":
            evening_pressure,

        "screen_burden":
            screen_burden,

        "stress_pressure":
            stress_pressure,

        "mood_pressure":
            mood_pressure,

        "energy_pressure":
            energy_pressure,

        "social_deficit":
            social_deficit,

        "phone_context":
            phone_context,

        "study_work_context":
            (
                studying_context
                + working_context
            ),

        "exercise_context":
            exercise_context,
    }


# ============================================================
# Intervention utilities
# ============================================================

def calculate_utilities(
    demands,
) -> np.ndarray:
    """
    Calculate continuous utilities for all six interventions.

    IMPORTANT:
    Coefficients below are transparent modelling parameters.
    They are NOT published medical effect sizes.
    """

    sleep_need = (
        demands[
            "sleep_need"
        ]
    )

    evening = (
        demands[
            "evening_pressure"
        ]
    )

    screen = (
        demands[
            "screen_burden"
        ]
    )

    stress = (
        demands[
            "stress_pressure"
        ]
    )

    mood = (
        demands[
            "mood_pressure"
        ]
    )

    energy = (
        demands[
            "energy_pressure"
        ]
    )

    social = (
        demands[
            "social_deficit"
        ]
    )

    phone = (
        demands[
            "phone_context"
        ]
    )

    study_work = (
        demands[
            "study_work_context"
        ]
    )

    exercise = (
        demands[
            "exercise_context"
        ]
    )


    utilities = np.zeros(
        (
            len(sleep_need),
            len(NUDGES),
        ),
        dtype=float,
    )


    # --------------------------------------------------------
    # Connect socially
    # --------------------------------------------------------

    utilities[:, 0] = (
        1.75 * social
        + 1.00 * mood
        + 0.30 * stress
        + 0.20 * social * mood
    )


    # --------------------------------------------------------
    # Maintain habits
    # --------------------------------------------------------

    overall_stability = (
        (
            1.0 - sleep_need
        )
        + (
            1.0 - screen
        )
        + (
            1.0 - stress
        )
        + (
            1.0 - mood
        )
        + (
            1.0 - energy
        )
    ) / 5.0

    utilities[:, 1] = (
        1.80
        * overall_stability
    )


    # --------------------------------------------------------
    # Prepare for bed
    # --------------------------------------------------------

    utilities[:, 2] = (
        1.55 * sleep_need
        + 1.35 * evening
        + 0.35 * energy
        + 0.30 * screen * evening
    )


    # --------------------------------------------------------
    # Reduce screen time
    # --------------------------------------------------------

    utilities[:, 3] = (
        1.70 * screen
        + 0.50 * phone
        + 0.30 * screen * evening
        + 0.15 * screen * sleep_need
    )


    # --------------------------------------------------------
    # Stay active
    # --------------------------------------------------------

    utilities[:, 4] = (
        0.95
        * (
            1.0
            - exercise
        )
        + 0.45
        * (
            1.0
            - energy
        )
        + 0.20
        * (
            1.0
            - mood
        )
    )


    # --------------------------------------------------------
    # Take a short break
    # --------------------------------------------------------

    utilities[:, 5] = (
        1.45 * stress
        + 1.05 * energy
        + 0.45 * mood
        + 0.35 * study_work
        + 0.20 * stress * energy
    )


    return utilities


# ============================================================
# Probabilistic label generation
# ============================================================

def generate_labels(
    utilities: np.ndarray,
):
    """
    Generate labels from intervention probabilities.

    v2.4 intentionally uses ONE stochastic label step.

    There is no additional random noise added directly to utilities.
    Behavioural uncertainty already exists through:
    - population variability
    - noisy wellbeing generation
    - probabilistic softmax sampling
    """

    probabilities = softmax(
        utilities,
        temperature=0.58,
    )


    labels = []

    confidence = []

    entropy = []


    for probability_row in probabilities:

        selected = RNG.choice(
            NUDGES,
            p=probability_row,
        )

        labels.append(
            selected
        )

        confidence.append(
            float(
                probability_row.max()
            )
        )

        entropy.append(
            float(
                -np.sum(
                    probability_row
                    * np.log(
                        probability_row
                        + 1e-12
                    )
                )
            )
        )


    return (
        np.array(
            labels
        ),
        np.array(
            confidence
        ),
        np.array(
            entropy
        ),
        probabilities,
    )


# ============================================================
# Main dataset generation
# ============================================================

def generate_dataset(
    n_samples: int = N_SAMPLES,
):

    (
        day_type,
        hour,
    ) = generate_context(
        n_samples
    )


    (
        sleep,
        screen_time,
        activity,
        social,
        activity_effect,
        social_effect,
    ) = generate_behaviour(
        day_type,
        hour,
        n_samples,
    )


    (
        stress,
        mood,
        energy,
    ) = generate_wellbeing(
        sleep,
        screen_time,
        activity_effect,
        social_effect,
        n_samples,
    )


    demands = calculate_demands(
        sleep,
        stress,
        mood,
        energy,
        screen_time,
        activity,
        social,
        hour,
    )


    utilities = calculate_utilities(
        demands
    )


    (
        labels,
        generator_confidence,
        generator_entropy,
        probabilities,
    ) = generate_labels(
        utilities
    )


    dataframe = pd.DataFrame(
        {
            "sleep":
                np.round(
                    sleep,
                    2,
                ),

            "stress":
                stress,

            "mood":
                mood,

            "energy":
                energy,

            "screen_time":
                np.round(
                    screen_time,
                    2,
                ),

            "activity":
                activity,

            "social":
                social,

            "hour":
                hour,

            "day_type":
                day_type,

            "best_nudge":
                labels,
        }
    )


    diagnostics = pd.DataFrame(
        {
            "generator_confidence":
                generator_confidence,

            "generator_entropy":
                generator_entropy,
        }
    )


    probability_dataframe = pd.DataFrame(
        probabilities,
        columns=[
            "p_"
            + nudge.lower()
            .replace(
                " ",
                "_",
            )
            for nudge in NUDGES
        ],
    )


    return (
        dataframe,
        diagnostics,
        probability_dataframe,
    )


# ============================================================
# Diagnostics
# ============================================================

def print_report(
    dataframe,
    diagnostics,
):

    print()
    print("=" * 72)
    print(
        "NUDGEWISE AI V2.4 SYNTHETIC DATASET"
    )
    print("=" * 72)


    print()
    print(
        f"Rows:     "
        f"{len(dataframe):,}"
    )

    print(
        f"Features: "
        f"{len(dataframe.columns) - 1}"
    )


    print()
    print(
        "Class distribution:"
    )

    distribution = (
        dataframe[
            "best_nudge"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        distribution
    )


    print()
    print(
        "Class percentages:"
    )

    print(
        (
            distribution
            / len(
                dataframe
            )
            * 100
        ).round(
            1
        )
    )


    print()
    print(
        "Numeric feature summary:"
    )

    print(
        dataframe[
            [
                "sleep",
                "stress",
                "mood",
                "energy",
                "screen_time",
                "hour",
            ]
        ]
        .describe()
        .round(
            2
        )
    )


    print()
    print(
        "Correlation matrix:"
    )

    print(
        dataframe[
            [
                "sleep",
                "stress",
                "mood",
                "energy",
                "screen_time",
                "hour",
            ]
        ]
        .corr()
        .round(
            2
        )
    )


    print()
    print(
        "Generator ambiguity:"
    )

    print(
        diagnostics
        .describe()
        .round(
            3
        )
    )


    print()
    print(
        "Activity distribution:"
    )

    print(
        dataframe[
            "activity"
        ]
        .value_counts()
    )


    print()
    print(
        "Social distribution:"
    )

    print(
        dataframe[
            "social"
        ]
        .value_counts()
    )


    print()
    print(
        "Day type distribution:"
    )

    print(
        dataframe[
            "day_type"
        ]
        .value_counts()
    )


# ============================================================
# Run
# ============================================================

def main():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        dataframe,
        diagnostics,
        probability_dataframe,
    ) = generate_dataset()


    # Production training dataset.
    dataframe.to_csv(
        OUTPUT_PATH,
        index=False,
    )


    # Research-only diagnostics.
    diagnostics_path = Path(
        "data/generator_diagnostics.csv"
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )


    probabilities_path = Path(
        "data/generator_probabilities.csv"
    )

    probability_dataframe.to_csv(
        probabilities_path,
        index=False,
    )


    print_report(
        dataframe,
        diagnostics,
    )


    print()
    print(
        f"Training data saved to: "
        f"{OUTPUT_PATH}"
    )

    print(
        f"Diagnostics saved to: "
        f"{diagnostics_path}"
    )

    print(
        f"Latent probabilities saved to: "
        f"{probabilities_path}"
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()