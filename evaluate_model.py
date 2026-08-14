"""
evaluate_model.py

NudgeWise AI v2.5
Production model behaviour and uncertainty evaluation.

Evaluates:
- recommendation frequency
- probability distribution quality
- certainty distribution
- top-probability distribution
- top-two margin
- entropy
- class dominance
- per-class prediction behaviour
- agreement with generator latent probabilities
- uncertainty vs prediction error

This script does NOT retrain the model.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

MODEL_PATH = Path(
    "models/nudge_model.pkl"
)

TRAINING_PATH = Path(
    "data/training.csv"
)

PROBABILITY_PATH = Path(
    "data/generator_probabilities.csv"
)

RESULTS_FOLDER = Path(
    "results"
)


# ============================================================
# Helpers
# ============================================================

def normalise_probabilities(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.clip(
        values,
        0.0,
        None,
    )

    values += 1e-12

    return (
        values
        / values.sum(
            axis=1,
            keepdims=True,
        )
    )


def calculate_entropy(
    probabilities,
):

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    entropy = (
        -np.sum(
            probabilities
            * np.log(
                probabilities
                + 1e-12
            ),
            axis=1,
        )
    )

    maximum_entropy = (
        np.log(
            probabilities.shape[1]
        )
    )

    return (
        entropy,
        entropy / maximum_entropy,
    )


def certainty_score(
    top_probability,
    margin,
    normalised_entropy,
):

    return (
        0.45 * top_probability
        + 0.35 * np.minimum(
            margin * 2.0,
            1.0,
        )
        + 0.20 * (
            1.0
            - normalised_entropy
        )
    )


def certainty_label(
    score,
):

    if score >= 0.58:
        return "High"

    if score >= 0.38:
        return "Moderate"

    return "Low"


# ============================================================
# Load model
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# Load evaluation data
# ============================================================

def load_data(
    bundle,
):

    if not TRAINING_PATH.exists():

        raise FileNotFoundError(
            f"Missing file: {TRAINING_PATH}"
        )

    if not PROBABILITY_PATH.exists():

        raise FileNotFoundError(
            f"Missing file: {PROBABILITY_PATH}"
        )


    training = pd.read_csv(
        TRAINING_PATH
    )

    generator = pd.read_csv(
        PROBABILITY_PATH
    )


    if len(training) != len(generator):

        raise ValueError(
            "Training and probability files "
            "contain different row counts."
        )


    features = bundle[
        "features"
    ]

    probability_columns = bundle[
        "probability_columns"
    ]


    X = training[
        features
    ].copy()


    true_probabilities = (
        generator[
            probability_columns
        ]
        .to_numpy(
            dtype=float
        )
    )


    true_probabilities = (
        normalise_probabilities(
            true_probabilities
        )
    )


    return (
        X,
        true_probabilities,
    )


# ============================================================
# Main evaluation
# ============================================================

def evaluate():

    RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )


    bundle = load_model()

    model = bundle[
        "model"
    ]

    classes = list(
        bundle[
            "classes"
        ]
    )


    (
        X,
        true_probabilities,
    ) = load_data(
        bundle
    )


    # --------------------------------------------------------
    # Model probabilities
    # --------------------------------------------------------

    predicted_probabilities = (
        model.predict(
            X
        )
    )


    predicted_probabilities = (
        normalise_probabilities(
            predicted_probabilities
        )
    )


    # --------------------------------------------------------
    # Top predictions
    # --------------------------------------------------------

    predicted_top = np.argmax(
        predicted_probabilities,
        axis=1,
    )

    true_top = np.argmax(
        true_probabilities,
        axis=1,
    )


    predicted_labels = np.array(
        [
            classes[index]
            for index
            in predicted_top
        ]
    )


    # --------------------------------------------------------
    # Top probability
    # --------------------------------------------------------

    top_probability = np.max(
        predicted_probabilities,
        axis=1,
    )


    # --------------------------------------------------------
    # Top-two margin
    # --------------------------------------------------------

    ordered_probabilities = np.sort(
        predicted_probabilities,
        axis=1,
    )


    second_probability = (
        ordered_probabilities[
            :,
            -2
        ]
    )


    probability_margin = (
        top_probability
        - second_probability
    )


    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    (
        entropy,
        normalised_entropy,
    ) = calculate_entropy(
        predicted_probabilities
    )


    # --------------------------------------------------------
    # Certainty score
    # --------------------------------------------------------

    certainty_scores = (
        certainty_score(
            top_probability,
            probability_margin,
            normalised_entropy,
        )
    )


    certainty_labels = np.array(
        [
            certainty_label(
                score
            )
            for score
            in certainty_scores
        ]
    )


    # --------------------------------------------------------
    # Prediction error
    # --------------------------------------------------------

    row_mae = np.mean(
        np.abs(
            predicted_probabilities
            - true_probabilities
        ),
        axis=1,
    )


    top1_correct = (
        predicted_top
        == true_top
    )


    # --------------------------------------------------------
    # Main evaluation table
    # --------------------------------------------------------

    evaluation = X.copy()


    evaluation[
        "prediction"
    ] = predicted_labels


    evaluation[
        "top_probability"
    ] = top_probability


    evaluation[
        "second_probability"
    ] = second_probability


    evaluation[
        "probability_margin"
    ] = probability_margin


    evaluation[
        "entropy"
    ] = entropy


    evaluation[
        "normalised_entropy"
    ] = normalised_entropy


    evaluation[
        "certainty_score"
    ] = certainty_scores


    evaluation[
        "certainty"
    ] = certainty_labels


    evaluation[
        "probability_mae"
    ] = row_mae


    evaluation[
        "top1_correct"
    ] = top1_correct


    # --------------------------------------------------------
    # Recommendation frequencies
    # --------------------------------------------------------

    recommendation_counts = (
        evaluation[
            "prediction"
        ]
        .value_counts()
        .reindex(
            classes,
            fill_value=0,
        )
    )


    recommendation_percentages = (
        recommendation_counts
        / len(
            evaluation
        )
    )


    recommendation_summary = (
        pd.DataFrame(
            {
                "recommendation":
                    classes,

                "count":
                    recommendation_counts.values,

                "percentage":
                    recommendation_percentages.values,
            }
        )
    )


    # --------------------------------------------------------
    # Certainty distribution
    # --------------------------------------------------------

    certainty_summary = (
        evaluation[
            "certainty"
        ]
        .value_counts()
        .reindex(
            [
                "Low",
                "Moderate",
                "High",
            ],
            fill_value=0,
        )
        .rename_axis(
            "certainty"
        )
        .reset_index(
            name="count"
        )
    )


    certainty_summary[
        "percentage"
    ] = (
        certainty_summary[
            "count"
        ]
        / len(
            evaluation
        )
    )


    # --------------------------------------------------------
    # Performance by certainty
    # --------------------------------------------------------

    certainty_performance = (
        evaluation
        .groupby(
            "certainty",
            observed=False,
        )
        .agg(
            rows=(
                "prediction",
                "size",
            ),

            top1_agreement=(
                "top1_correct",
                "mean",
            ),

            mean_probability_error=(
                "probability_mae",
                "mean",
            ),

            mean_top_probability=(
                "top_probability",
                "mean",
            ),

            mean_margin=(
                "probability_margin",
                "mean",
            ),
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Per-class behaviour
    # --------------------------------------------------------

    class_behaviour = (
        evaluation
        .groupby(
            "prediction"
        )
        .agg(
            count=(
                "prediction",
                "size",
            ),

            mean_top_probability=(
                "top_probability",
                "mean",
            ),

            median_top_probability=(
                "top_probability",
                "median",
            ),

            mean_margin=(
                "probability_margin",
                "mean",
            ),

            mean_probability_error=(
                "probability_mae",
                "mean",
            ),

            top1_agreement=(
                "top1_correct",
                "mean",
            ),
        )
        .reindex(
            classes
        )
        .reset_index()
    )


    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    overall_summary = {
        "rows":
            int(
                len(
                    evaluation
                )
            ),

        "mean_top_probability":
            float(
                np.mean(
                    top_probability
                )
            ),

        "median_top_probability":
            float(
                np.median(
                    top_probability
                )
            ),

        "minimum_top_probability":
            float(
                np.min(
                    top_probability
                )
            ),

        "maximum_top_probability":
            float(
                np.max(
                    top_probability
                )
            ),

        "mean_probability_margin":
            float(
                np.mean(
                    probability_margin
                )
            ),

        "mean_normalised_entropy":
            float(
                np.mean(
                    normalised_entropy
                )
            ),

        "top1_agreement":
            float(
                np.mean(
                    top1_correct
                )
            ),

        "mean_probability_mae":
            float(
                np.mean(
                    row_mae
                )
            ),

        "dominant_class":
            str(
                recommendation_summary
                .sort_values(
                    "count",
                    ascending=False,
                )
                .iloc[0][
                    "recommendation"
                ]
            ),

        "dominant_class_percentage":
            float(
                recommendation_summary[
                    "percentage"
                ].max()
            ),

        "low_certainty_percentage":
            float(
                np.mean(
                    certainty_labels
                    == "Low"
                )
            ),

        "moderate_certainty_percentage":
            float(
                np.mean(
                    certainty_labels
                    == "Moderate"
                )
            ),

        "high_certainty_percentage":
            float(
                np.mean(
                    certainty_labels
                    == "High"
                )
            ),
    }


    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    evaluation.to_csv(
        RESULTS_FOLDER
        / "v25_model_behaviour.csv",
        index=False,
    )


    recommendation_summary.to_csv(
        RESULTS_FOLDER
        / "v25_recommendation_distribution.csv",
        index=False,
    )


    certainty_summary.to_csv(
        RESULTS_FOLDER
        / "v25_certainty_distribution.csv",
        index=False,
    )


    certainty_performance.to_csv(
        RESULTS_FOLDER
        / "v25_certainty_performance.csv",
        index=False,
    )


    class_behaviour.to_csv(
        RESULTS_FOLDER
        / "v25_class_behaviour.csv",
        index=False,
    )


    with open(
        RESULTS_FOLDER
        / "v25_behaviour_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            overall_summary,
            file,
            indent=2,
        )


    # --------------------------------------------------------
    # Terminal report
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "NUDGEWISE V2.5 MODEL BEHAVIOUR EVALUATION"
    )
    print("=" * 76)


    print()
    print(
        f"Rows evaluated: "
        f"{overall_summary['rows']:,}"
    )


    print()
    print(
        "TOP PROBABILITY"
    )

    print(
        f"Mean:    "
        f"{overall_summary['mean_top_probability']:.3f}"
    )

    print(
        f"Median:  "
        f"{overall_summary['median_top_probability']:.3f}"
    )

    print(
        f"Minimum: "
        f"{overall_summary['minimum_top_probability']:.3f}"
    )

    print(
        f"Maximum: "
        f"{overall_summary['maximum_top_probability']:.3f}"
    )


    print()
    print(
        "UNCERTAINTY"
    )

    print(
        f"Mean top-two margin:       "
        f"{overall_summary['mean_probability_margin']:.3f}"
    )

    print(
        f"Mean normalised entropy:   "
        f"{overall_summary['mean_normalised_entropy']:.3f}"
    )


    print()
    print(
        "CERTAINTY DISTRIBUTION"
    )

    for _, row in certainty_summary.iterrows():

        print(
            f"{row['certainty']:<10}"
            f"{row['percentage']:>8.1%}"
        )


    print()
    print(
        "RECOMMENDATION DISTRIBUTION"
    )

    for _, row in recommendation_summary.iterrows():

        print(
            f"{row['recommendation']:<24}"
            f"{row['percentage']:>8.1%}"
        )


    print()
    print(
        "MODEL RECOVERY"
    )

    print(
        f"Top-1 generator agreement: "
        f"{overall_summary['top1_agreement']:.3f}"
    )

    print(
        f"Mean probability MAE:      "
        f"{overall_summary['mean_probability_mae']:.4f}"
    )


    print()
    print(
        "PERFORMANCE BY CERTAINTY"
    )

    print(
        certainty_performance.to_string(
            index=False
        )
    )


    print()
    print(
        "PER-RECOMMENDATION BEHAVIOUR"
    )

    print(
        class_behaviour.to_string(
            index=False
        )
    )


    print()
    print(
        "Saved evaluation files to:"
    )

    print(
        RESULTS_FOLDER
    )

    print("=" * 76)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    evaluate()