"""
generate_dataset.py

NudgeWise AI v2.6
Evidence-grounded synthetic behavioural dataset generator.

PURPOSE
-------
Generate synthetic behavioural scenarios for training and evaluating
the NudgeWise soft-probability recommendation model.

The generator converts clearly defined behavioural measurements into:

    measurements
        ->
    interpretable behavioural demand signals
        ->
    six intervention utilities
        ->
    soft intervention probabilities

IMPORTANT
---------
This dataset is synthetic.

It does NOT represent:
- adolescent prevalence
- clinical ground truth
- diagnosis
- validated treatment effects
- probabilities that an intervention will improve wellbeing

SCIENTIFIC DESIGN
-----------------
NudgeWise explicitly separates:

1. Evidence-backed reference points
2. Evidence-informed directional relationships
3. Modelling assumptions

Evidence-backed reference points include:
- adolescent sleep recommendation: 8-10 hours
- adolescent MVPA reference: average >=60 minutes/day across the week

The exact mathematical:
- coefficients
- transition widths
- synthetic population distributions
- softmax temperature

remain MODELLING ASSUMPTIONS.

They are not published clinical effect sizes.

PRODUCTION FEATURES
-------------------
sleep
    Hours during the last main sleep period.

stress
    Current stress.
    1 = very low
    5 = very high

mood
    Current mood.
    1 = very low
    5 = very good

energy
    Current energy.
    1 = very low
    5 = very high

screen_time
    Recreational screen exposure in hours.

activity_minutes
    Approximate moderate-to-vigorous physical activity minutes.

connectedness
    Perceived social connectedness.
    1 = not connected at all
    5 = very connected

activity
    Current behavioural context:
    Phone / Studying / Working / Relaxing / Exercise

hour
    Local hour of check-in.

day_type
    Weekday / Weekend

best_nudge
    A stochastic sampled label retained for secondary diagnostics.

PRIMARY TARGET
--------------
The six latent intervention probabilities stored in:

    data/generator_probabilities.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED = 42

N_SAMPLES = 20_000

RNG = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# Output paths
# ============================================================

DATA_FOLDER = Path(
    "data"
)

TRAINING_PATH = (
    DATA_FOLDER
    / "training.csv"
)

PROBABILITY_PATH = (
    DATA_FOLDER
    / "generator_probabilities.csv"
)

DIAGNOSTICS_PATH = (
    DATA_FOLDER
    / "generator_diagnostics.csv"
)

DEMAND_PATH = (
    DATA_FOLDER
    / "generator_demands.csv"
)


# ============================================================
# Evidence-backed reference points
# ============================================================

# ------------------------------------------------------------
# Sleep
#
# Research anchor:
# Adolescents aged 13-18 are recommended to regularly obtain
# 8-10 hours of sleep.
#
# We use 8 hours as the LOWER reference around which sleep need
# increases smoothly.
# ------------------------------------------------------------

ADOLESCENT_SLEEP_LOWER_HOURS = 8.0

ADOLESCENT_SLEEP_UPPER_HOURS = 10.0


# ------------------------------------------------------------
# Physical activity
#
# Research anchor:
# Children/adolescents should average at least approximately
# 60 minutes/day of moderate-to-vigorous physical activity
# across the week.
#
# IMPORTANT:
# This is NOT treated as a binary daily pass/fail threshold.
# ------------------------------------------------------------

ACTIVITY_REFERENCE_MINUTES = 60.0


# ============================================================
# Explicit modelling assumptions
# ============================================================

# ------------------------------------------------------------
# Transition widths
#
# These determine how gradually a demand changes around a
# reference point.
#
# They are modelling parameters, NOT published effect sizes.
# ------------------------------------------------------------

SLEEP_TRANSITION_WIDTH = 0.85

ACTIVITY_TRANSITION_WIDTH = 22.0

EVENING_REFERENCE_HOUR = 20.5

EVENING_TRANSITION_WIDTH = 1.35


# ------------------------------------------------------------
# Screen exposure scale
#
# There is deliberately NO universal harmful screen threshold.
#
# This scale controls a smooth saturation curve only.
# ------------------------------------------------------------

SCREEN_EXPOSURE_SCALE = 4.0


# ------------------------------------------------------------
# Probability ambiguity
#
# Higher temperature creates more overlapping intervention
# probabilities.
# ------------------------------------------------------------

SOFTMAX_TEMPERATURE = 0.62


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


PROBABILITY_COLUMNS = [
    "p_connect_socially",
    "p_maintain_habits",
    "p_prepare_for_bed",
    "p_reduce_screen_time",
    "p_stay_active",
    "p_take_a_short_break",
]


# ============================================================
# Mathematical helpers
# ============================================================

def sigmoid(
    value,
):
    """
    Numerically stable logistic transformation.
    """

    value = np.clip(
        value,
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -value
            )
        )
    )


def softmax(
    values: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Convert intervention utilities into probabilities.

    Temperature is explicitly a modelling assumption.
    """

    scaled = (
        values
        / temperature
    )

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
# Day/time context
# ============================================================

def generate_context(
    n_samples: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Generate synthetic weekday/weekend and hour context.

    These frequencies are synthetic and should NOT be
    interpreted as real adolescent app-use prevalence.
    """

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
        ],
        dtype=float,
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
# Current behavioural context
# ============================================================

def generate_activity_context(
    n_samples: int,
) -> np.ndarray:
    """
    Generate what the participant is doing at check-in.

    IMPORTANT:
    Current context is NOT treated as a measure of health.

    It modifies whether an intervention makes sense right now.
    """

    return RNG.choice(
        ACTIVITIES,
        size=n_samples,
        p=[
            0.27,  # Phone
            0.28,  # Studying
            0.08,  # Working
            0.22,  # Relaxing
            0.15,  # Exercise
        ],
    )


# ============================================================
# Perceived connectedness
# ============================================================

def generate_connectedness(
    n_samples: int,
) -> np.ndarray:
    """
    Generate perceived connectedness on the exact 1-5
    scale used by the application.

    This synthetic distribution is NOT a population estimate.
    """

    return RNG.choice(
        [
            1,
            2,
            3,
            4,
            5,
        ],
        size=n_samples,
        p=[
            0.08,
            0.17,
            0.38,
            0.25,
            0.12,
        ],
    )


# ============================================================
# Physical activity minutes
# ============================================================

def generate_activity_minutes(
    activity: np.ndarray,
    day_type: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Generate broad physical-activity exposure.

    Current exercise context moderately correlates with greater
    accumulated activity, but the two remain separate variables.

    The distribution spans substantially below and above the
    60-minute evidence reference.

    It does NOT estimate real adolescent prevalence.
    """

    weekend = (
        day_type
        == "Weekend"
    ).astype(float)


    currently_exercising = (
        activity
        == "Exercise"
    ).astype(float)


    currently_on_phone = (
        activity
        == "Phone"
    ).astype(float)


    base_activity = RNG.gamma(
        shape=2.15,
        scale=24.0,
        size=n_samples,
    )


    activity_minutes = (
        base_activity
        + 24.0
        * currently_exercising
        - 5.0
        * currently_on_phone
        + 4.0
        * weekend
        + RNG.normal(
            0.0,
            8.0,
            n_samples,
        )
    )


    activity_minutes = np.clip(
        activity_minutes,
        0.0,
        240.0,
    )


    return np.rint(
        activity_minutes
    ).astype(int)


# ============================================================
# Sleep
# ============================================================

def generate_sleep(
    day_type: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Generate synthetic sleep duration.

    The mean and variance are modelling choices.

    The evidence-backed 8-10 hour range is NOT being used to
    force the synthetic population to look healthy.
    """

    weekend = (
        day_type
        == "Weekend"
    ).astype(float)


    sleep_mean = (
        7.35
        + 0.55
        * weekend
    )


    sleep = RNG.normal(
        loc=sleep_mean,
        scale=1.05,
        size=n_samples,
    )


    return np.clip(
        sleep,
        3.5,
        11.5,
    )


# ============================================================
# Recreational screen exposure
# ============================================================

def generate_screen_time(
    sleep: np.ndarray,
    activity_minutes: np.ndarray,
    activity: np.ndarray,
    day_type: np.ndarray,
    hour: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """
    Generate recreational screen exposure.

    There is deliberately no universal harmful-hour cutoff.

    Several weak/moderate correlations are introduced to
    create realistic overlapping behavioural scenarios.

    These correlations are modelling assumptions and should
    not be interpreted causally.
    """

    weekend = (
        day_type
        == "Weekend"
    ).astype(float)


    phone_context = (
        activity
        == "Phone"
    ).astype(float)


    exercise_context = (
        activity
        == "Exercise"
    ).astype(float)


    later_day = sigmoid(
        (
            hour
            - 18.0
        )
        / 2.5
    )


    sleep_centered = (
        sleep
        - 8.0
    )


    activity_centered = (
        activity_minutes
        - ACTIVITY_REFERENCE_MINUTES
    ) / 60.0


    screen_mean = (
        3.80
        - 0.12
        * sleep_centered
        - 0.18
        * activity_centered
        + 0.95
        * phone_context
        - 0.22
        * exercise_context
        + 0.30
        * weekend
        + 0.25
        * later_day
    )


    screen_time = RNG.normal(
        loc=screen_mean,
        scale=1.35,
        size=n_samples,
    )


    return np.clip(
        screen_time,
        0.0,
        12.0,
    )


# ============================================================
# Wellbeing state generation
# ============================================================

def generate_wellbeing(
    sleep: np.ndarray,
    screen_time: np.ndarray,
    activity_minutes: np.ndarray,
    connectedness: np.ndarray,
    n_samples: int,
):
    """
    Generate correlated stress, mood and energy states.

    Directions are evidence-informed.

    Exact coefficients are modelling assumptions.

    No causal interpretation should be made.
    """

    sleep_centered = (
        sleep
        - ADOLESCENT_SLEEP_LOWER_HOURS
    )


    screen_centered = (
        screen_time
        - 4.0
    )


    activity_centered = (
        activity_minutes
        - ACTIVITY_REFERENCE_MINUTES
    ) / 60.0


    # Convert 1-5 connectedness into approximately -1 to +1.

    connectedness_centered = (
        connectedness
        - 3.0
    ) / 2.0


    # --------------------------------------------------------
    # Stress
    #
    # 1 = very low
    # 5 = very high
    # --------------------------------------------------------

    stress_latent = (
        3.00
        - 0.18
        * sleep_centered
        + 0.09
        * screen_centered
        - 0.14
        * activity_centered
        - 0.22
        * connectedness_centered
        + RNG.normal(
            0.0,
            0.72,
            n_samples,
        )
    )


    # --------------------------------------------------------
    # Mood
    #
    # 1 = very low
    # 5 = very good
    # --------------------------------------------------------

    mood_latent = (
        3.15
        - 0.34
        * (
            stress_latent
            - 3.0
        )
        + 0.14
        * sleep_centered
        + 0.12
        * activity_centered
        + 0.24
        * connectedness_centered
        + RNG.normal(
            0.0,
            0.65,
            n_samples,
        )
    )


    # --------------------------------------------------------
    # Energy
    #
    # 1 = very low
    # 5 = very high
    # --------------------------------------------------------

    energy_latent = (
        3.10
        + 0.27
        * sleep_centered
        - 0.22
        * (
            stress_latent
            - 3.0
        )
        + 0.10
        * activity_centered
        - 0.06
        * screen_centered
        + RNG.normal(
            0.0,
            0.62,
            n_samples,
        )
    )


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
# Behavioural demands
# ============================================================

def calculate_demands(
    sleep: np.ndarray,
    stress: np.ndarray,
    mood: np.ndarray,
    energy: np.ndarray,
    screen_time: np.ndarray,
    activity_minutes: np.ndarray,
    connectedness: np.ndarray,
    activity: np.ndarray,
    hour: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Convert raw measurements into interpretable demand signals.

    Demands range approximately between 0 and 1.

    None are clinical scores.
    """

    # --------------------------------------------------------
    # Sleep need
    #
    # Evidence-backed reference:
    # lower boundary of adolescent 8-10 hour recommendation.
    #
    # Smoothness remains a modelling assumption.
    # --------------------------------------------------------

    sleep_need = sigmoid(
        (
            ADOLESCENT_SLEEP_LOWER_HOURS
            - sleep
        )
        / SLEEP_TRANSITION_WIDTH
    )


    # --------------------------------------------------------
    # Evening relevance
    #
    # Time is a contextual modifier.
    #
    # Exact timing curve is a modelling assumption.
    # --------------------------------------------------------

    evening_relevance = sigmoid(
        (
            hour
            - EVENING_REFERENCE_HOUR
        )
        / EVENING_TRANSITION_WIDTH
    )


    # --------------------------------------------------------
    # Screen exposure
    #
    # No artificial cutoff.
    #
    # Greater recreational exposure increases the signal
    # gradually and saturates.
    # --------------------------------------------------------

    screen_load = (
        1.0
        - np.exp(
            -screen_time
            / SCREEN_EXPOSURE_SCALE
        )
    )


    screen_load = np.clip(
        screen_load,
        0.0,
        1.0,
    )


    # --------------------------------------------------------
    # Current stress
    #
    # Exact app scale:
    # 1 = very low
    # 5 = very high
    # --------------------------------------------------------

    stress_load = (
        stress
        - 1.0
    ) / 4.0


    # --------------------------------------------------------
    # Mood need
    #
    # 1 = very low mood
    # 5 = very good mood
    # --------------------------------------------------------

    mood_need = (
        5.0
        - mood
    ) / 4.0


    # --------------------------------------------------------
    # Low-energy need
    #
    # 1 = very low energy
    # 5 = very high energy
    # --------------------------------------------------------

    energy_need = (
        5.0
        - energy
    ) / 4.0


    # --------------------------------------------------------
    # Physical activity need
    #
    # 60 minutes is an evidence-backed reference from the
    # weekly-average guideline.
    #
    # It is NOT a pass/fail threshold.
    # --------------------------------------------------------

    activity_need = sigmoid(
        (
            ACTIVITY_REFERENCE_MINUTES
            - activity_minutes
        )
        / ACTIVITY_TRANSITION_WIDTH
    )


    # --------------------------------------------------------
    # Connection need
    #
    # Connectedness:
    # 1 = not connected at all
    # 5 = very connected
    #
    # This mapping follows the exact direction of the measure.
    # --------------------------------------------------------

    connection_need = (
        5.0
        - connectedness
    ) / 4.0


    # --------------------------------------------------------
    # Current context
    # --------------------------------------------------------

    phone_context = (
        activity
        == "Phone"
    ).astype(float)


    study_context = (
        activity
        == "Studying"
    ).astype(float)


    work_context = (
        activity
        == "Working"
    ).astype(float)


    relaxation_context = (
        activity
        == "Relaxing"
    ).astype(float)


    exercise_context = (
        activity
        == "Exercise"
    ).astype(float)


    cognitive_context = np.clip(
        study_context
        + work_context,
        0.0,
        1.0,
    )


    return {
        "sleep_need":
            sleep_need,

        "evening_relevance":
            evening_relevance,

        "screen_load":
            screen_load,

        "stress_load":
            stress_load,

        "mood_need":
            mood_need,

        "energy_need":
            energy_need,

        "activity_need":
            activity_need,

        "connection_need":
            connection_need,

        "phone_context":
            phone_context,

        "cognitive_context":
            cognitive_context,

        "relaxation_context":
            relaxation_context,

        "exercise_context":
            exercise_context,
    }


# ============================================================
# Intervention utilities
# ============================================================

def calculate_utilities(
    demands: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Convert behavioural demands into intervention utilities.

    CRITICAL INTERPRETATION
    -----------------------
    All coefficients in this function are transparent modelling
    parameters.

    They are NOT published effect sizes.

    Their purpose is to encode evidence-informed directional
    relationships while allowing controlled sensitivity testing.
    """

    sleep = demands[
        "sleep_need"
    ]

    evening = demands[
        "evening_relevance"
    ]

    screen = demands[
        "screen_load"
    ]

    stress = demands[
        "stress_load"
    ]

    mood = demands[
        "mood_need"
    ]

    energy = demands[
        "energy_need"
    ]

    activity_need = demands[
        "activity_need"
    ]

    connection = demands[
        "connection_need"
    ]

    phone = demands[
        "phone_context"
    ]

    cognitive = demands[
        "cognitive_context"
    ]

    relaxing = demands[
        "relaxation_context"
    ]

    exercising = demands[
        "exercise_context"
    ]


    utilities = np.zeros(
        (
            len(sleep),
            len(NUDGES),
        ),
        dtype=float,
    )


    # ========================================================
    # CONNECT SOCIALLY
    #
    # Primary signal:
    # perceived lack of connectedness.
    #
    # Mood/stress act only as secondary modifiers.
    # ========================================================

    utilities[:, 0] = (
        0.25
        + 1.55
        * connection
        + 0.38
        * mood
        + 0.14
        * stress
        + 0.28
        * connection
        * mood
    )


    # ========================================================
    # MAINTAIN HABITS
    #
    # Intended for genuinely stable states.
    #
    # It considers both:
    # - average need
    # - the strongest individual need
    #
    # This fixes v2.5's mathematical disadvantage for
    # Maintain habits.
    # ========================================================

    core_needs = np.column_stack(
        [
            sleep,
            screen,
            stress,
            mood,
            energy,
            activity_need,
            connection,
        ]
    )


    mean_need = np.mean(
        core_needs,
        axis=1,
    )


    strongest_need = np.max(
        core_needs,
        axis=1,
    )


    overall_stability = (
        1.0
        - mean_need
    )


    absence_of_strong_need = (
        1.0
        - strongest_need
    )


    utilities[:, 1] = (
        0.85
        + 1.35
        * overall_stability
        + 0.70
        * absence_of_strong_need
    )


    # ========================================================
    # PREPARE FOR BED
    #
    # Short sleep matters, but this intervention becomes much
    # more actionable later in the day.
    # ========================================================

    utilities[:, 2] = (
        0.20
        + 1.05
        * sleep
        + 1.05
        * evening
        + 0.62
        * sleep
        * evening
        + 0.20
        * screen
        * evening
        + 0.10
        * energy
    )


    # ========================================================
    # REDUCE SCREEN TIME
    #
    # Screen exposure is continuous.
    #
    # Phone context and evening context increase immediate
    # relevance.
    #
    # There is deliberately no universal hour threshold.
    # ========================================================

    utilities[:, 3] = (
        0.18
        + 1.30
        * screen
        + 0.45
        * phone
        + 0.30
        * screen
        * evening
        + 0.12
        * screen
        * sleep
    )


    # ========================================================
    # STAY ACTIVE
    #
    # Primary signal:
    # accumulated activity exposure.
    #
    # Current exercise context strongly suppresses an immediate
    # "be active" recommendation.
    # ========================================================

    immediate_activity_factor = (
        1.0
        - 0.78
        * exercising
    )


    utilities[:, 4] = (
        0.30
        + immediate_activity_factor
        * (
            1.55
            * activity_need
            + 0.16
            * mood
            + 0.12
            * relaxing
        )
    )


    # ========================================================
    # TAKE A SHORT BREAK
    #
    # Primary drivers:
    # stress and low energy.
    #
    # Studying/working increases contextual relevance.
    #
    # Mood is deliberately only a small modifier.
    # ========================================================

    utilities[:, 5] = (
        0.22
        + 1.18
        * stress
        + 0.68
        * energy
        + 0.42
        * cognitive
        + 0.25
        * stress
        * cognitive
        + 0.14
        * mood
    )


    return utilities


# ============================================================
# Probability generation
# ============================================================

def generate_probabilities(
    utilities: np.ndarray,
):
    """
    Convert utilities into a soft intervention distribution.

    The sampled best_nudge remains a secondary diagnostic only.
    """

    probabilities = softmax(
        utilities,
        temperature=(
            SOFTMAX_TEMPERATURE
        ),
    )


    generator_mode_indices = np.argmax(
        probabilities,
        axis=1,
    )


    generator_mode = np.array(
        [
            NUDGES[index]
            for index
            in generator_mode_indices
        ]
    )


    sampled_labels = np.array(
        [
            RNG.choice(
                NUDGES,
                p=row,
            )
            for row
            in probabilities
        ]
    )


    generator_confidence = np.max(
        probabilities,
        axis=1,
    )


    generator_entropy = (
        -np.sum(
            probabilities
            * np.log(
                probabilities
                + 1e-12
            ),
            axis=1,
        )
    )


    return (
        sampled_labels,
        generator_mode,
        generator_confidence,
        generator_entropy,
        probabilities,
    )


# ============================================================
# Dataset generation
# ============================================================

def generate_dataset(
    n_samples: int = N_SAMPLES,
):
    """
    Generate complete v2.6 data.
    """

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    (
        day_type,
        hour,
    ) = generate_context(
        n_samples
    )


    activity = (
        generate_activity_context(
            n_samples
        )
    )


    connectedness = (
        generate_connectedness(
            n_samples
        )
    )


    # --------------------------------------------------------
    # Behaviour
    # --------------------------------------------------------

    activity_minutes = (
        generate_activity_minutes(
            activity=activity,
            day_type=day_type,
            n_samples=n_samples,
        )
    )


    sleep = generate_sleep(
        day_type=day_type,
        n_samples=n_samples,
    )


    screen_time = (
        generate_screen_time(
            sleep=sleep,
            activity_minutes=activity_minutes,
            activity=activity,
            day_type=day_type,
            hour=hour,
            n_samples=n_samples,
        )
    )


    # --------------------------------------------------------
    # Current wellbeing
    # --------------------------------------------------------

    (
        stress,
        mood,
        energy,
    ) = generate_wellbeing(
        sleep=sleep,
        screen_time=screen_time,
        activity_minutes=activity_minutes,
        connectedness=connectedness,
        n_samples=n_samples,
    )


    # --------------------------------------------------------
    # Demand layer
    # --------------------------------------------------------

    demands = calculate_demands(
        sleep=sleep,
        stress=stress,
        mood=mood,
        energy=energy,
        screen_time=screen_time,
        activity_minutes=activity_minutes,
        connectedness=connectedness,
        activity=activity,
        hour=hour,
    )


    # --------------------------------------------------------
    # Utility layer
    # --------------------------------------------------------

    utilities = calculate_utilities(
        demands
    )


    # --------------------------------------------------------
    # Probability layer
    # --------------------------------------------------------

    (
        sampled_labels,
        generator_mode,
        generator_confidence,
        generator_entropy,
        probabilities,
    ) = generate_probabilities(
        utilities
    )


    # --------------------------------------------------------
    # Production training dataframe
    # --------------------------------------------------------

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

            "activity_minutes":
                activity_minutes,

            "connectedness":
                connectedness,

            "activity":
                activity,

            "hour":
                hour,

            "day_type":
                day_type,

            "best_nudge":
                sampled_labels,
        }
    )


    # --------------------------------------------------------
    # Probability targets
    # --------------------------------------------------------

    probability_dataframe = pd.DataFrame(
        probabilities,
        columns=PROBABILITY_COLUMNS,
    )


    # --------------------------------------------------------
    # Research diagnostics
    # --------------------------------------------------------

    diagnostics = pd.DataFrame(
        {
            "generator_mode":
                generator_mode,

            "generator_confidence":
                generator_confidence,

            "generator_entropy":
                generator_entropy,
        }
    )


    # --------------------------------------------------------
    # Demand audit
    # --------------------------------------------------------

    demand_dataframe = pd.DataFrame(
        demands
    )


    return (
        dataframe,
        probability_dataframe,
        diagnostics,
        demand_dataframe,
    )


# ============================================================
# Scenario diagnostics
# ============================================================

def scenario_diagnostics(
    dataframe: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    """
    Test whether the synthetic decision structure behaves
    sensibly in interpretable scenarios.

    These are sanity checks, not evidence of intervention
    effectiveness.
    """

    print()
    print("=" * 76)
    print(
        "SCENARIO SANITY CHECKS"
    )
    print("=" * 76)


    # --------------------------------------------------------
    # Stable state
    # --------------------------------------------------------

    stable = (
        (dataframe["sleep"] >= 8.0)
        & (dataframe["sleep"] <= 10.0)
        & (dataframe["stress"] <= 2)
        & (dataframe["mood"] >= 4)
        & (dataframe["energy"] >= 4)
        & (dataframe["activity_minutes"] >= 60)
        & (dataframe["connectedness"] >= 4)
    )


    stable_modes = diagnostics.loc[
        stable,
        "generator_mode",
    ]


    print()
    print(
        f"Stable scenarios: "
        f"{len(stable_modes):,}"
    )


    if len(
        stable_modes
    ) > 0:

        print(
            stable_modes
            .value_counts(
                normalize=True
            )
            .mul(
                100
            )
            .round(
                1
            )
            .head(
                6
            )
        )


    # --------------------------------------------------------
    # Low connectedness
    # --------------------------------------------------------

    disconnected = (
        dataframe[
            "connectedness"
        ]
        <= 2
    )


    social_rate = np.mean(
        diagnostics.loc[
            disconnected,
            "generator_mode",
        ]
        == "Connect socially"
    )


    print()
    print(
        "Connect-socially mode when connectedness <=2:"
    )

    print(
        f"  {social_rate:.1%}"
    )


    # --------------------------------------------------------
    # Very low activity
    # --------------------------------------------------------

    low_activity = (
        dataframe[
            "activity_minutes"
        ]
        < 30
    )


    active_rate = np.mean(
        diagnostics.loc[
            low_activity,
            "generator_mode",
        ]
        == "Stay active"
    )


    print()
    print(
        "Stay-active mode when activity <30 min:"
    )

    print(
        f"  {active_rate:.1%}"
    )


    # --------------------------------------------------------
    # Already exercising
    # --------------------------------------------------------

    exercising = (
        dataframe[
            "activity"
        ]
        == "Exercise"
    )


    redundant_active_rate = np.mean(
        diagnostics.loc[
            exercising,
            "generator_mode",
        ]
        == "Stay active"
    )


    print()
    print(
        "Stay-active mode while already exercising:"
    )

    print(
        f"  {redundant_active_rate:.1%}"
    )


    # --------------------------------------------------------
    # Late + inadequate sleep
    # --------------------------------------------------------

    late_short_sleep = (
        (
            dataframe[
                "hour"
            ]
            >= 21
        )
        & (
            dataframe[
                "sleep"
            ]
            < 7.0
        )
    )


    bedtime_rate = np.mean(
        diagnostics.loc[
            late_short_sleep,
            "generator_mode",
        ]
        == "Prepare for bed"
    )


    print()
    print(
        "Prepare-for-bed mode for >=21:00 and <7 h sleep:"
    )

    print(
        f"  {bedtime_rate:.1%}"
    )


    # --------------------------------------------------------
    # High stress during cognitive work
    # --------------------------------------------------------

    stressed_cognitive = (
        (
            dataframe[
                "stress"
            ]
            >= 4
        )
        & (
            dataframe[
                "activity"
            ].isin(
                [
                    "Studying",
                    "Working",
                ]
            )
        )
    )


    break_rate = np.mean(
        diagnostics.loc[
            stressed_cognitive,
            "generator_mode",
        ]
        == "Take a short break"
    )


    print()
    print(
        "Short-break mode for high stress while studying/working:"
    )

    print(
        f"  {break_rate:.1%}"
    )


# ============================================================
# Main report
# ============================================================

def print_report(
    dataframe: pd.DataFrame,
    probability_dataframe: pd.DataFrame,
    diagnostics: pd.DataFrame,
    demand_dataframe: pd.DataFrame,
) -> None:

    print()
    print("=" * 76)
    print(
        "NUDGEWISE AI V2.6"
    )
    print(
        "EVIDENCE-GROUNDED SYNTHETIC GENERATOR"
    )
    print("=" * 76)


    print()
    print(
        f"Rows:     {len(dataframe):,}"
    )

    print(
        f"Features: {len(dataframe.columns) - 1}"
    )


    # --------------------------------------------------------
    # Generator-mode distribution
    # --------------------------------------------------------

    print()
    print(
        "GENERATOR MODE DISTRIBUTION"
    )


    mode_counts = (
        diagnostics[
            "generator_mode"
        ]
        .value_counts()
        .reindex(
            NUDGES,
            fill_value=0,
        )
    )


    for nudge in NUDGES:

        percentage = (
            mode_counts[nudge]
            / len(
                diagnostics
            )
        )


        print(
            f"{nudge:<24}"
            f"{mode_counts[nudge]:>7,d}"
            f"   "
            f"{percentage:>6.1%}"
        )


    # --------------------------------------------------------
    # Sampled-label distribution
    # --------------------------------------------------------

    print()
    print(
        "SAMPLED LABEL DISTRIBUTION"
    )


    sampled_counts = (
        dataframe[
            "best_nudge"
        ]
        .value_counts()
        .reindex(
            NUDGES,
            fill_value=0,
        )
    )


    for nudge in NUDGES:

        percentage = (
            sampled_counts[nudge]
            / len(
                dataframe
            )
        )


        print(
            f"{nudge:<24}"
            f"{sampled_counts[nudge]:>7,d}"
            f"   "
            f"{percentage:>6.1%}"
        )


    # --------------------------------------------------------
    # Input summary
    # --------------------------------------------------------

    print()
    print(
        "NUMERIC INPUT SUMMARY"
    )


    numeric_columns = [
        "sleep",
        "stress",
        "mood",
        "energy",
        "screen_time",
        "activity_minutes",
        "connectedness",
        "hour",
    ]


    print(
        dataframe[
            numeric_columns
        ]
        .describe()
        .round(
            2
        )
    )


    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print()
    print(
        "NUMERIC CORRELATIONS"
    )


    print(
        dataframe[
            numeric_columns
        ]
        .corr()
        .round(
            2
        )
    )


    # --------------------------------------------------------
    # Ambiguity
    # --------------------------------------------------------

    print()
    print(
        "GENERATOR AMBIGUITY"
    )


    print(
        diagnostics[
            [
                "generator_confidence",
                "generator_entropy",
            ]
        ]
        .describe()
        .round(
            3
        )
    )


    # --------------------------------------------------------
    # Demand audit
    # --------------------------------------------------------

    print()
    print(
        "BEHAVIOURAL DEMAND SUMMARY"
    )


    print(
        demand_dataframe
        .describe()
        .loc[
            [
                "mean",
                "std",
                "min",
                "50%",
                "max",
            ]
        ]
        .round(
            3
        )
    )


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    print()
    print(
        "CURRENT CONTEXT DISTRIBUTION"
    )

    print(
        dataframe[
            "activity"
        ]
        .value_counts()
    )


    print()
    print(
        "CONNECTEDNESS DISTRIBUTION"
    )

    print(
        dataframe[
            "connectedness"
        ]
        .value_counts()
        .sort_index()
    )


    print()
    print(
        "DAY TYPE DISTRIBUTION"
    )

    print(
        dataframe[
            "day_type"
        ]
        .value_counts()
    )


    scenario_diagnostics(
        dataframe=dataframe,
        diagnostics=diagnostics,
    )


# ============================================================
# Run
# ============================================================

def main():

    DATA_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        dataframe,
        probability_dataframe,
        diagnostics,
        demand_dataframe,
    ) = generate_dataset()


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    dataframe.to_csv(
        TRAINING_PATH,
        index=False,
    )


    probability_dataframe.to_csv(
        PROBABILITY_PATH,
        index=False,
    )


    diagnostics.to_csv(
        DIAGNOSTICS_PATH,
        index=False,
    )


    demand_dataframe.to_csv(
        DEMAND_PATH,
        index=False,
    )


    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print_report(
        dataframe=dataframe,
        probability_dataframe=probability_dataframe,
        diagnostics=diagnostics,
        demand_dataframe=demand_dataframe,
    )


    print()
    print("=" * 76)

    print(
        "FILES SAVED"
    )

    print("=" * 76)

    print(
        f"Training data: "
        f"{TRAINING_PATH}"
    )

    print(
        f"Probability targets: "
        f"{PROBABILITY_PATH}"
    )

    print(
        f"Diagnostics: "
        f"{DIAGNOSTICS_PATH}"
    )

    print(
        f"Demand audit: "
        f"{DEMAND_PATH}"
    )


    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Generator probabilities describe the synthetic "
        "decision structure only."
    )

    print(
        "They are NOT probabilities that an intervention "
        "will work for a real adolescent."
    )

    print("=" * 76)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()