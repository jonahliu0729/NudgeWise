"""
train_model.py

NudgeWise AI v2.6
Evidence-grounded soft-label probability learning.

PURPOSE
-------
Train an AI model to reconstruct the probability distribution
produced by the NudgeWise v2.6 evidence-grounded synthetic
behavioural generator.

The model learns:

    behavioural measurements
        ->
    six intervention probabilities

rather than learning a single randomly sampled hard label.

IMPORTANT
---------
Synthetic performance measures how accurately the machine-learning
model reconstructs the synthetic decision structure.

It does NOT demonstrate:
- clinical validity
- intervention effectiveness
- improved adolescent wellbeing
- causal relationships

PRIMARY EVALUATION
------------------
1. Multiclass Brier score
2. Mean absolute probability error
3. Jensen-Shannon divergence
4. Top-1 agreement with generator mode
5. Top-2 coverage
6. Per-intervention probability MAE

SECONDARY EVALUATION
--------------------
Agreement with the randomly sampled best_nudge label is retained
for diagnostic purposes only.
"""

from __future__ import annotations

import json

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone

from sklearn.compose import (
    ColumnTransformer,
)

from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import (
    Ridge,
)

from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from sklearn.neural_network import (
    MLPRegressor,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42


TRAINING_PATH = Path(
    "data/training.csv"
)


PROBABILITY_PATH = Path(
    "data/generator_probabilities.csv"
)


MODEL_PATH = Path(
    "models/nudge_model.pkl"
)


RESULTS_FOLDER = Path(
    "results"
)


# ============================================================
# v2.6 production feature schema
# ============================================================

NUMERIC_FEATURES = [
    "sleep",
    "stress",
    "mood",
    "energy",
    "screen_time",
    "activity_minutes",
    "connectedness",
    "hour",
]


CATEGORICAL_FEATURES = [
    "activity",
    "day_type",
]


ALL_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)


# ============================================================
# Recommendation schema
# ============================================================

CLASSES = [
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
# Probability normalisation
# ============================================================

def normalise_probabilities(
    values: np.ndarray,
) -> np.ndarray:
    """
    Convert model regression outputs into valid probability
    distributions.

    Multi-output regression models can predict:
    - slightly negative values
    - rows that do not sum exactly to 1

    We therefore:
    1. clip negatives
    2. add a tiny epsilon
    3. normalise each row
    """

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


    row_sums = values.sum(
        axis=1,
        keepdims=True,
    )


    return (
        values
        / row_sums
    )


# ============================================================
# Jensen-Shannon divergence
# ============================================================

def jensen_shannon_divergence(
    true_probabilities: np.ndarray,
    predicted_probabilities: np.ndarray,
) -> float:
    """
    Calculate mean Jensen-Shannon divergence.

    Lower is better.

    JS divergence compares entire probability distributions and
    is symmetric and bounded compared with raw KL divergence.
    """

    epsilon = 1e-12


    p = np.clip(
        true_probabilities,
        epsilon,
        1.0,
    )


    q = np.clip(
        predicted_probabilities,
        epsilon,
        1.0,
    )


    midpoint = (
        p + q
    ) / 2.0


    kl_p = np.sum(
        p
        * np.log(
            p / midpoint
        ),
        axis=1,
    )


    kl_q = np.sum(
        q
        * np.log(
            q / midpoint
        ),
        axis=1,
    )


    js = (
        0.5
        * kl_p

        + 0.5
        * kl_q
    )


    return float(
        np.mean(
            js
        )
    )


# ============================================================
# Probability evaluation
# ============================================================

def evaluate_probabilities(
    true_probabilities: np.ndarray,
    predicted_probabilities: np.ndarray,
) -> dict:
    """
    Evaluate reconstruction of the generator's latent
    intervention probability distribution.
    """

    true_probabilities = (
        normalise_probabilities(
            true_probabilities
        )
    )


    predicted_probabilities = (
        normalise_probabilities(
            predicted_probabilities
        )
    )


    # --------------------------------------------------------
    # Multiclass Brier score
    # --------------------------------------------------------

    brier_score = float(
        np.mean(
            np.sum(
                (
                    predicted_probabilities
                    - true_probabilities
                )
                ** 2,
                axis=1,
            )
        )
    )


    # --------------------------------------------------------
    # Mean absolute probability error
    # --------------------------------------------------------

    probability_mae = float(
        np.mean(
            np.abs(
                predicted_probabilities
                - true_probabilities
            )
        )
    )


    # --------------------------------------------------------
    # Jensen-Shannon divergence
    # --------------------------------------------------------

    js_divergence = (
        jensen_shannon_divergence(
            true_probabilities,
            predicted_probabilities,
        )
    )


    # --------------------------------------------------------
    # Generator mode vs model mode
    # --------------------------------------------------------

    true_top1 = np.argmax(
        true_probabilities,
        axis=1,
    )


    predicted_top1 = np.argmax(
        predicted_probabilities,
        axis=1,
    )


    top1_agreement = float(
        np.mean(
            true_top1
            == predicted_top1
        )
    )


    # --------------------------------------------------------
    # Top-2 coverage
    #
    # Is the generator's strongest intervention among the
    # model's two strongest interventions?
    # --------------------------------------------------------

    predicted_top2 = np.argsort(
        predicted_probabilities,
        axis=1,
    )[:, -2:]


    top2_hits = [
        true_class
        in predicted_classes

        for (
            true_class,
            predicted_classes,
        )
        in zip(
            true_top1,
            predicted_top2,
        )
    ]


    top2_coverage = float(
        np.mean(
            top2_hits
        )
    )


    return {
        "brier_score":
            brier_score,

        "probability_mae":
            probability_mae,

        "js_divergence":
            js_divergence,

        "top1_agreement":
            top1_agreement,

        "top2_coverage":
            top2_coverage,
    }


# ============================================================
# Dataset loading
# ============================================================

def load_data():
    """
    Load and validate the exact v2.6 production schema.
    """

    if not TRAINING_PATH.exists():

        raise FileNotFoundError(
            f"Missing training data: "
            f"{TRAINING_PATH}"
        )


    if not PROBABILITY_PATH.exists():

        raise FileNotFoundError(
            f"Missing generator probabilities: "
            f"{PROBABILITY_PATH}"
        )


    training = pd.read_csv(
        TRAINING_PATH
    )


    probabilities = pd.read_csv(
        PROBABILITY_PATH
    )


    # --------------------------------------------------------
    # Alignment validation
    # --------------------------------------------------------

    if len(training) != len(
        probabilities
    ):

        raise ValueError(
            "training.csv and generator_probabilities.csv "
            "contain different numbers of rows."
        )


    # --------------------------------------------------------
    # Feature validation
    # --------------------------------------------------------

    missing_features = [
        feature

        for feature
        in ALL_FEATURES

        if feature
        not in training.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing v2.6 production features: "
            + ", ".join(
                missing_features
            )
        )


    # Legacy social must NOT accidentally remain in the AI
    # feature schema.

    if "social" in ALL_FEATURES:

        raise ValueError(
            "Legacy social variable must not be used "
            "by the v2.6 model."
        )


    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    missing_probability_columns = [
        column

        for column
        in PROBABILITY_COLUMNS

        if column
        not in probabilities.columns
    ]


    if missing_probability_columns:

        raise ValueError(
            "Missing latent probability columns: "
            + ", ".join(
                missing_probability_columns
            )
        )


    X = training[
        ALL_FEATURES
    ].copy()


    y_soft = (
        probabilities[
            PROBABILITY_COLUMNS
        ]
        .to_numpy(
            dtype=float
        )
    )


    y_soft = (
        normalise_probabilities(
            y_soft
        )
    )


    # --------------------------------------------------------
    # Generator mode
    #
    # Used for stratification and top-1 evaluation.
    # --------------------------------------------------------

    y_mode = np.argmax(
        y_soft,
        axis=1,
    )


    # --------------------------------------------------------
    # Sampled synthetic label
    #
    # Secondary diagnostic only.
    # --------------------------------------------------------

    sampled_labels = training[
        "best_nudge"
    ].copy()


    return (
        X,
        y_soft,
        y_mode,
        sampled_labels,
    )


# ============================================================
# Preprocessing
# ============================================================

def build_preprocessor():
    """
    Exact preprocessing used in production.

    Numeric:
        standardised

    Categorical:
        one-hot encoded
    """

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),

            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


# ============================================================
# Candidate models
# ============================================================

def build_models():
    """
    Candidate soft-probability regression models.

    All models predict six continuous outputs.
    """

    return {

        # ----------------------------------------------------
        # Linear baseline
        # ----------------------------------------------------

        "Ridge":
            Ridge(
                alpha=1.0,
            ),


        # ----------------------------------------------------
        # Random Forest
        # ----------------------------------------------------

        "Random Forest":
            RandomForestRegressor(
                n_estimators=500,

                min_samples_leaf=3,

                max_features=0.80,

                random_state=RANDOM_STATE,

                n_jobs=-1,
            ),


        # ----------------------------------------------------
        # Extra Trees
        # ----------------------------------------------------

        "Extra Trees":
            ExtraTreesRegressor(
                n_estimators=500,

                min_samples_leaf=3,

                max_features=0.85,

                random_state=RANDOM_STATE,

                n_jobs=-1,
            ),


        # ----------------------------------------------------
        # Neural probability approximator
        # ----------------------------------------------------

        "MLP":
            MLPRegressor(
                hidden_layer_sizes=(
                    96,
                    48,
                ),

                activation="relu",

                solver="adam",

                alpha=0.0005,

                batch_size=128,

                learning_rate_init=0.001,

                max_iter=500,

                early_stopping=True,

                validation_fraction=0.12,

                n_iter_no_change=25,

                random_state=RANDOM_STATE,
            ),
    }


# ============================================================
# Pipeline
# ============================================================

def make_pipeline(
    estimator,
):

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),

            (
                "model",
                estimator,
            ),
        ]
    )


# ============================================================
# Per-class probability error
# ============================================================

def per_class_mae(
    true_probabilities,
    predicted_probabilities,
):

    predicted_probabilities = (
        normalise_probabilities(
            predicted_probabilities
        )
    )


    true_probabilities = (
        normalise_probabilities(
            true_probabilities
        )
    )


    errors = np.mean(
        np.abs(
            true_probabilities
            - predicted_probabilities
        ),
        axis=0,
    )


    return {
        class_name:
            float(
                error
            )

        for (
            class_name,
            error,
        )
        in zip(
            CLASSES,
            errors,
        )
    }


# ============================================================
# Cross-validation
# ============================================================

def cross_validate_models(
    models,
    X_train,
    y_train,
    mode_train,
):
    """
    Compare candidate learners using the training partition only.

    The independent holdout is not touched during model
    selection.
    """

    cross_validator = (
        StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    )


    results = []


    print()
    print("=" * 78)
    print(
        "NUDGEWISE V2.6"
    )
    print(
        "SOFT-LABEL CROSS-VALIDATION"
    )
    print("=" * 78)


    for (
        model_name,
        estimator,
    ) in models.items():

        print()
        print(
            f"Testing: {model_name}"
        )


        fold_metrics = []


        for (
            fold,
            (
                train_indices,
                validation_indices,
            ),
        ) in enumerate(
            cross_validator.split(
                X_train,
                mode_train,
            ),
            start=1,
        ):

            X_fold_train = (
                X_train.iloc[
                    train_indices
                ]
            )


            X_fold_validation = (
                X_train.iloc[
                    validation_indices
                ]
            )


            y_fold_train = (
                y_train[
                    train_indices
                ]
            )


            y_fold_validation = (
                y_train[
                    validation_indices
                ]
            )


            pipeline = make_pipeline(
                clone(
                    estimator
                )
            )


            pipeline.fit(
                X_fold_train,
                y_fold_train,
            )


            predicted = pipeline.predict(
                X_fold_validation
            )


            metrics = evaluate_probabilities(
                y_fold_validation,
                predicted,
            )


            fold_metrics.append(
                metrics
            )


            print(
                f"  Fold {fold}: "
                f"Top1="
                f"{metrics['top1_agreement']:.3f}  "
                f"Top2="
                f"{metrics['top2_coverage']:.3f}  "
                f"Brier="
                f"{metrics['brier_score']:.4f}  "
                f"MAE="
                f"{metrics['probability_mae']:.4f}  "
                f"JS="
                f"{metrics['js_divergence']:.4f}"
            )


        # ----------------------------------------------------
        # Mean CV performance
        # ----------------------------------------------------

        average_metrics = {

            key:
                float(
                    np.mean(
                        [
                            fold_result[
                                key
                            ]

                            for fold_result
                            in fold_metrics
                        ]
                    )
                )

            for key
            in fold_metrics[0]
        }


        # ----------------------------------------------------
        # Model-selection score
        #
        # This combines:
        # decision recovery + probability reconstruction.
        #
        # It is a model-selection heuristic, not a scientific
        # outcome metric.
        # ----------------------------------------------------

        selection_score = (
            0.45
            * average_metrics[
                "top1_agreement"
            ]

            + 0.15
            * average_metrics[
                "top2_coverage"
            ]

            - 0.25
            * average_metrics[
                "brier_score"
            ]

            - 0.15
            * average_metrics[
                "js_divergence"
            ]
        )


        result = {
            "model":
                model_name,

            **average_metrics,

            "selection_score":
                float(
                    selection_score
                ),
        }


        results.append(
            result
        )


        print()
        print(
            f"{model_name} mean:"
        )

        print(
            f"  Top-1 agreement: "
            f"{average_metrics['top1_agreement']:.3f}"
        )

        print(
            f"  Top-2 coverage:  "
            f"{average_metrics['top2_coverage']:.3f}"
        )

        print(
            f"  Brier score:     "
            f"{average_metrics['brier_score']:.4f}"
        )

        print(
            f"  Probability MAE: "
            f"{average_metrics['probability_mae']:.4f}"
        )

        print(
            f"  JS divergence:   "
            f"{average_metrics['js_divergence']:.4f}"
        )


    comparison = (
        pd.DataFrame(
            results
        )
        .sort_values(
            "selection_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    return comparison


# ============================================================
# Holdout evaluation
# ============================================================

def evaluate_holdout(
    model,
    X_test,
    y_test,
    sampled_test,
):
    """
    Evaluate final selected model on the untouched 20% holdout.
    """

    predicted = model.predict(
        X_test
    )


    predicted = (
        normalise_probabilities(
            predicted
        )
    )


    metrics = evaluate_probabilities(
        y_test,
        predicted,
    )


    class_errors = per_class_mae(
        y_test,
        predicted,
    )


    # --------------------------------------------------------
    # Secondary sampled-label agreement
    # --------------------------------------------------------

    predicted_indices = np.argmax(
        predicted,
        axis=1,
    )


    predicted_labels = np.array(
        [
            CLASSES[index]

            for index
            in predicted_indices
        ]
    )


    sampled_label_accuracy = float(
        np.mean(
            predicted_labels
            == sampled_test.to_numpy()
        )
    )


    metrics[
        "sampled_label_accuracy"
    ] = sampled_label_accuracy


    return (
        metrics,
        class_errors,
        predicted,
    )


# ============================================================
# Holdout class distribution
# ============================================================

def holdout_mode_distribution(
    true_probabilities,
    predicted_probabilities,
):
    """
    Compare generator-mode and model-mode distributions on the
    untouched holdout set.
    """

    true_indices = np.argmax(
        true_probabilities,
        axis=1,
    )


    predicted_indices = np.argmax(
        predicted_probabilities,
        axis=1,
    )


    records = []


    for (
        index,
        class_name,
    ) in enumerate(
        CLASSES
    ):

        true_rate = float(
            np.mean(
                true_indices
                == index
            )
        )


        predicted_rate = float(
            np.mean(
                predicted_indices
                == index
            )
        )


        records.append(
            {
                "class":
                    class_name,

                "generator_mode_rate":
                    true_rate,

                "model_mode_rate":
                    predicted_rate,

                "absolute_difference":
                    abs(
                        true_rate
                        - predicted_rate
                    ),
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# Main
# ============================================================

def main():

    RESULTS_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )


    (
        X,
        y_soft,
        y_mode,
        sampled_labels,
    ) = load_data()


    print()
    print("=" * 78)
    print(
        "NUDGEWISE AI V2.6"
    )
    print(
        "EVIDENCE-GROUNDED SOFT-PROBABILITY LEARNING"
    )
    print("=" * 78)


    print()
    print(
        f"Rows:     {len(X):,}"
    )

    print(
        f"Features: {len(ALL_FEATURES)}"
    )

    print(
        f"Outputs:  {len(CLASSES)} probabilities"
    )


    print()
    print(
        "Production features:"
    )


    for feature in ALL_FEATURES:

        print(
            f"  - {feature}"
        )


    # --------------------------------------------------------
    # Generator ambiguity
    # --------------------------------------------------------

    generator_mean_max_probability = float(
        np.mean(
            np.max(
                y_soft,
                axis=1,
            )
        )
    )


    print()
    print(
        "Average generator maximum probability:"
    )

    print(
        f"  "
        f"{generator_mean_max_probability:.3f}"
    )


    print()
    print(
        "This describes synthetic decision ambiguity."
    )

    print(
        "It is NOT intervention effectiveness."
    )


    # ========================================================
    # Independent 80 / 20 split
    # ========================================================

    indices = np.arange(
        len(X)
    )


    (
        train_indices,
        test_indices,
    ) = train_test_split(
        indices,
        test_size=0.20,
        stratify=y_mode,
        random_state=RANDOM_STATE,
    )


    X_train = (
        X.iloc[
            train_indices
        ]
        .reset_index(
            drop=True
        )
    )


    X_test = (
        X.iloc[
            test_indices
        ]
        .reset_index(
            drop=True
        )
    )


    y_train = (
        y_soft[
            train_indices
        ]
    )


    y_test = (
        y_soft[
            test_indices
        ]
    )


    mode_train = (
        y_mode[
            train_indices
        ]
    )


    sampled_test = (
        sampled_labels.iloc[
            test_indices
        ]
        .reset_index(
            drop=True
        )
    )


    print()
    print(
        f"Training partition: "
        f"{len(X_train):,}"
    )

    print(
        f"Untouched holdout:  "
        f"{len(X_test):,}"
    )


    # ========================================================
    # Candidate comparison
    # ========================================================

    models = build_models()


    comparison = cross_validate_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        mode_train=mode_train,
    )


    print()
    print("=" * 78)
    print(
        "MODEL RANKING"
    )
    print("=" * 78)


    print(
        comparison.to_string(
            index=False
        )
    )


    comparison.to_csv(
        RESULTS_FOLDER
        / "v26_soft_model_comparison.csv",
        index=False,
    )


    # ========================================================
    # Select model using CV only
    # ========================================================

    best_name = str(
        comparison.iloc[0][
            "model"
        ]
    )


    print()
    print(
        f"Selected model: "
        f"{best_name}"
    )


    # ========================================================
    # Fit selected model on full 80% training partition
    # ========================================================

    final_model = make_pipeline(
        clone(
            models[
                best_name
            ]
        )
    )


    final_model.fit(
        X_train,
        y_train,
    )


    # ========================================================
    # ONE independent holdout evaluation
    # ========================================================

    (
        holdout_metrics,
        class_errors,
        holdout_predictions,
    ) = evaluate_holdout(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        sampled_test=sampled_test,
    )


    print()
    print("=" * 78)
    print(
        "INDEPENDENT V2.6 HOLDOUT TEST"
    )
    print("=" * 78)


    print(
        f"Top-1 generator agreement: "
        f"{holdout_metrics['top1_agreement']:.3f}"
    )

    print(
        f"Top-2 coverage:             "
        f"{holdout_metrics['top2_coverage']:.3f}"
    )

    print(
        f"Multiclass Brier score:     "
        f"{holdout_metrics['brier_score']:.4f}"
    )

    print(
        f"Probability MAE:            "
        f"{holdout_metrics['probability_mae']:.4f}"
    )

    print(
        f"Jensen-Shannon divergence:  "
        f"{holdout_metrics['js_divergence']:.4f}"
    )


    print()
    print(
        "Secondary sampled-label agreement:"
    )

    print(
        f"  "
        f"{holdout_metrics['sampled_label_accuracy']:.3f}"
    )


    # ========================================================
    # Per-class error
    # ========================================================

    print()
    print(
        "PER-INTERVENTION PROBABILITY MAE"
    )


    for (
        class_name,
        error,
    ) in class_errors.items():

        print(
            f"{class_name:<24}"
            f"{error:.4f}"
        )


    # ========================================================
    # Mode-distribution reconstruction
    # ========================================================

    mode_comparison = (
        holdout_mode_distribution(
            true_probabilities=y_test,
            predicted_probabilities=holdout_predictions,
        )
    )


    print()
    print(
        "HOLDOUT MODE DISTRIBUTION"
    )


    print(
        mode_comparison.to_string(
            index=False
        )
    )


    mode_comparison.to_csv(
        RESULTS_FOLDER
        / "v26_holdout_mode_distribution.csv",
        index=False,
    )


    # ========================================================
    # Save holdout probabilities
    # ========================================================

    true_dataframe = pd.DataFrame(
        y_test,
        columns=[
            f"true_{column}"

            for column
            in PROBABILITY_COLUMNS
        ],
    )


    predicted_dataframe = pd.DataFrame(
        holdout_predictions,
        columns=[
            f"pred_{column}"

            for column
            in PROBABILITY_COLUMNS
        ],
    )


    holdout_dataframe = pd.concat(
        [
            true_dataframe,
            predicted_dataframe,
        ],
        axis=1,
    )


    holdout_dataframe.to_csv(
        RESULTS_FOLDER
        / "v26_holdout_probabilities.csv",
        index=False,
    )


    # ========================================================
    # Experiment summary
    # ========================================================

    summary = {
        "version":
            "NudgeWise AI v2.6",

        "learning_problem":
            "evidence-grounded soft-label probability regression",

        "dataset_type":
            "synthetic behavioural scenarios",

        "rows":
            int(
                len(X)
            ),

        "training_rows":
            int(
                len(X_train)
            ),

        "holdout_rows":
            int(
                len(X_test)
            ),

        "features":
            ALL_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "classes":
            CLASSES,

        "selected_model":
            best_name,

        "generator_mean_max_probability":
            generator_mean_max_probability,

        "holdout_metrics":
            holdout_metrics,

        "per_class_probability_mae":
            class_errors,

        "random_state":
            RANDOM_STATE,

        "interpretation":
            (
                "Synthetic performance measures reconstruction "
                "of the evidence-grounded synthetic decision "
                "structure and does not establish real-world "
                "intervention effectiveness."
            ),
    }


    with open(
        RESULTS_FOLDER
        / "v26_experiment_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


    # ========================================================
    # Save production bundle
    # ========================================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    bundle = {
        "model":
            final_model,

        "model_name":
            best_name,

        "model_type":
            "soft_probability_regression",

        "version":
            "2.6",

        "features":
            ALL_FEATURES,

        "numeric_features":
            NUMERIC_FEATURES,

        "categorical_features":
            CATEGORICAL_FEATURES,

        "classes":
            CLASSES,

        "probability_columns":
            PROBABILITY_COLUMNS,

        "metrics":
            holdout_metrics,

        "per_class_probability_mae":
            class_errors,

        "random_state":
            RANDOM_STATE,
    }


    joblib.dump(
        bundle,
        MODEL_PATH,
    )


    # ========================================================
    # Completion
    # ========================================================

    print()
    print("=" * 78)
    print(
        "NUDGEWISE AI V2.6 MODEL SAVED"
    )
    print("=" * 78)


    print(
        f"Selected model: "
        f"{best_name}"
    )

    print(
        f"Production bundle: "
        f"{MODEL_PATH}"
    )

    print(
        f"Research results: "
        f"{RESULTS_FOLDER}/"
    )


    print()
    print(
        "INTERPRETATION"
    )

    print(
        "The model predicts the latent intervention probability "
        "distribution generated by the v2.6 synthetic behavioural "
        "decision system."
    )


    print()
    print(
        "The randomly sampled best_nudge is NOT treated "
        "as perfect ground truth."
    )


    print()
    print(
        "Synthetic model performance does NOT establish "
        "real-world effectiveness or clinical validity."
    )


    print()
    print(
        "Do not modify the v2.6 generator in response to this "
        "holdout result without creating a new experimental version."
    )


    print("=" * 78)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()