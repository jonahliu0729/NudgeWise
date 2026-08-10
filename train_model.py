"""
train_model.py

NudgeWise AI v2.5
Soft-label probability learning.

RESEARCH IDEA
-------------
Previous versions trained a multiclass classifier against
"best_nudge", which is a RANDOM SAMPLE from the evidence-informed
synthetic intervention probability distribution.

That creates an artificial hard-label accuracy ceiling.

v2.5 instead learns the COMPLETE probability distribution:

    behavioural state
        ->
    six latent intervention probabilities
        ->
    learned probability model

The sampled best_nudge label is retained only as a secondary
diagnostic.

PRIMARY EVALUATION
------------------
1. Multiclass Brier score
2. Mean absolute probability error
3. Jensen-Shannon divergence
4. Top-1 agreement with generator's most likely intervention
5. Top-2 coverage
6. Per-nudge probability error

IMPORTANT
---------
Performance on this synthetic task measures how well the AI
recovers the synthetic evidence-informed decision structure.

It does NOT demonstrate real-world intervention effectiveness.
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
# Exact production feature schema
# ============================================================

NUMERIC_FEATURES = [
    "sleep",
    "stress",
    "mood",
    "energy",
    "screen_time",
    "hour",
]

CATEGORICAL_FEATURES = [
    "activity",
    "social",
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
# Probability helpers
# ============================================================

def normalise_probabilities(
    values: np.ndarray,
) -> np.ndarray:
    """
    Convert arbitrary multi-output regression predictions into
    valid probability distributions.

    Regression models may predict slightly negative values or
    values that do not sum exactly to one.

    We:
        1. clip negatives
        2. add epsilon
        3. normalise each row to sum to one
    """

    values = np.asarray(
        values,
        dtype=float,
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

    JS divergence is symmetric and numerically safer for this
    probability-comparison task than using raw KL divergence alone.
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
        0.5 * kl_p
        + 0.5 * kl_q
    )

    return float(
        np.mean(js)
    )


# ============================================================
# Evaluation metrics
# ============================================================

def evaluate_probabilities(
    true_probabilities: np.ndarray,
    predicted_probabilities: np.ndarray,
) -> dict:
    """
    Evaluate how closely the model reconstructs the generator's
    underlying probability distribution.
    """

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
    # Generator's actual most-likely intervention
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
    # Is the generator's best intervention among the model's
    # two strongest suggestions?
    # --------------------------------------------------------

    predicted_top2 = np.argsort(
        predicted_probabilities,
        axis=1,
    )[:, -2:]


    top2_coverage = float(
        np.mean(
            [
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
    # Row alignment check
    # --------------------------------------------------------

    if (
        len(training)
        != len(probabilities)
    ):

        raise ValueError(
            "training.csv and generator_probabilities.csv "
            "contain different numbers of rows."
        )


    # --------------------------------------------------------
    # Feature schema validation
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in ALL_FEATURES
        if feature
        not in training.columns
    ]


    if missing_features:

        raise ValueError(
            "Missing production features: "
            + ", ".join(
                missing_features
            )
        )


    missing_probability_columns = [
        column
        for column in PROBABILITY_COLUMNS
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


    y_soft = probabilities[
        PROBABILITY_COLUMNS
    ].to_numpy(
        dtype=float
    )


    y_soft = normalise_probabilities(
        y_soft
    )


    # Generator mode is used ONLY for stratification
    # and top-1 evaluation.

    y_mode = np.argmax(
        y_soft,
        axis=1,
    )


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
# Preprocessor
# ============================================================

def build_preprocessor():

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
# Candidate probability models
# ============================================================

def build_models():
    """
    Candidate multi-output regression models.

    The targets are six continuous probabilities rather than
    six hard classes.
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
        # Nonlinear ensemble
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
        # Highly randomised ensemble
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


    errors = np.mean(
        np.abs(
            true_probabilities
            - predicted_probabilities
        ),
        axis=0,
    )


    return {
        class_name:
            float(error)

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

    cross_validator = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


    results = []


    print()
    print("=" * 76)
    print(
        "NUDGEWISE V2.5 SOFT-LABEL CROSS VALIDATION"
    )
    print("=" * 76)


    for (
        model_name,
        estimator,
    ) in models.items():

        fold_metrics = []


        print()
        print(
            f"Testing: {model_name}"
        )


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
                f"Top1={metrics['top1_agreement']:.3f}  "
                f"Brier={metrics['brier_score']:.4f}  "
                f"JS={metrics['js_divergence']:.4f}"
            )


        # ----------------------------------------------------
        # Average folds
        # ----------------------------------------------------

        average_metrics = {

            key:
                float(
                    np.mean(
                        [
                            fold[
                                key
                            ]
                            for fold
                            in fold_metrics
                        ]
                    )
                )

            for key
            in fold_metrics[0]
        }


        # ----------------------------------------------------
        # Selection score
        #
        # Higher = better.
        #
        # Top-1 and top-2 capture decision recovery.
        # Brier and JS capture probability quality.
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


    return (
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


# ============================================================
# Independent holdout evaluation
# ============================================================

def evaluate_holdout(
    model,
    X_test,
    y_test,
    sampled_test,
):

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


    probability_errors = (
        per_class_mae(
            y_test,
            predicted,
        )
    )


    # --------------------------------------------------------
    # Secondary metric:
    # agreement with RANDOMLY SAMPLED synthetic label.
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
        probability_errors,
        predicted,
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
    print("=" * 76)
    print(
        "NUDGEWISE AI V2.5"
    )
    print(
        "SOFT-LABEL PROBABILITY LEARNING"
    )
    print("=" * 76)


    print()
    print(
        f"Rows:     "
        f"{len(X):,}"
    )

    print(
        f"Features: "
        f"{len(ALL_FEATURES)}"
    )

    print(
        f"Outputs:  "
        f"{len(CLASSES)} probabilities"
    )


    # --------------------------------------------------------
    # Generator theoretical ambiguity
    # --------------------------------------------------------

    generator_mode_confidence = float(
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
        f"  {generator_mode_confidence:.3f}"
    )


    print()
    print(
        "This is the expected hard-label accuracy of an "
        "oracle that always chooses the generator's most "
        "likely intervention against its randomly sampled label."
    )


    # --------------------------------------------------------
    # Independent holdout split
    # --------------------------------------------------------

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


    X_train = X.iloc[
        train_indices
    ].reset_index(
        drop=True
    )

    X_test = X.iloc[
        test_indices
    ].reset_index(
        drop=True
    )


    y_train = y_soft[
        train_indices
    ]

    y_test = y_soft[
        test_indices
    ]


    mode_train = y_mode[
        train_indices
    ]


    sampled_test = (
        sampled_labels.iloc[
            test_indices
        ]
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # Compare candidate models
    # --------------------------------------------------------

    models = build_models()


    comparison = cross_validate_models(
        models,
        X_train,
        y_train,
        mode_train,
    )


    print()
    print("=" * 76)
    print(
        "MODEL RANKING"
    )
    print("=" * 76)


    print(
        comparison.to_string(
            index=False
        )
    )


    comparison.to_csv(
        RESULTS_FOLDER
        / "v25_soft_model_comparison.csv",
        index=False,
    )


    best_name = (
        comparison.iloc[0][
            "model"
        ]
    )


    print()
    print(
        f"Selected model: "
        f"{best_name}"
    )


    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    final_model = make_pipeline(
        models[
            best_name
        ]
    )


    final_model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Independent holdout
    # --------------------------------------------------------

    (
        holdout_metrics,
        class_errors,
        holdout_predictions,
    ) = evaluate_holdout(
        final_model,
        X_test,
        y_test,
        sampled_test,
    )


    print()
    print("=" * 76)
    print(
        "INDEPENDENT SOFT-LABEL HOLDOUT TEST"
    )
    print("=" * 76)


    print(
        f"Top-1 agreement with generator mode: "
        f"{holdout_metrics['top1_agreement']:.3f}"
    )

    print(
        f"Top-2 coverage:                       "
        f"{holdout_metrics['top2_coverage']:.3f}"
    )

    print(
        f"Multiclass Brier score:               "
        f"{holdout_metrics['brier_score']:.4f}"
    )

    print(
        f"Probability MAE:                      "
        f"{holdout_metrics['probability_mae']:.4f}"
    )

    print(
        f"Jensen-Shannon divergence:            "
        f"{holdout_metrics['js_divergence']:.4f}"
    )


    print()
    print(
        "Secondary sampled-label accuracy:"
    )

    print(
        f"  "
        f"{holdout_metrics['sampled_label_accuracy']:.3f}"
    )


    print()
    print(
        "Per-nudge probability MAE:"
    )


    for (
        class_name,
        error,
    ) in class_errors.items():

        print(
            f"  {class_name:<24}"
            f"{error:.4f}"
        )


    # --------------------------------------------------------
    # Save holdout comparison
    # --------------------------------------------------------

    true_holdout_dataframe = pd.DataFrame(
        y_test,
        columns=[
            f"true_{column}"
            for column
            in PROBABILITY_COLUMNS
        ],
    )


    predicted_holdout_dataframe = pd.DataFrame(
        holdout_predictions,
        columns=[
            f"pred_{column}"
            for column
            in PROBABILITY_COLUMNS
        ],
    )


    holdout_comparison = pd.concat(
        [
            true_holdout_dataframe,
            predicted_holdout_dataframe,
        ],
        axis=1,
    )


    holdout_comparison.to_csv(
        RESULTS_FOLDER
        / "v25_holdout_probabilities.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Save experiment summary
    # --------------------------------------------------------

    summary = {
        "version":
            "NudgeWise AI v2.5",

        "learning_problem":
            "soft-label probability regression",

        "rows":
            int(
                len(X)
            ),

        "features":
            ALL_FEATURES,

        "classes":
            CLASSES,

        "selected_model":
            best_name,

        "generator_mean_max_probability":
            generator_mode_confidence,

        "holdout_metrics":
            holdout_metrics,

        "per_class_probability_mae":
            class_errors,

        "random_state":
            RANDOM_STATE,
    }


    with open(
        RESULTS_FOLDER
        / "v25_experiment_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )


    # --------------------------------------------------------
    # Save production model bundle
    # --------------------------------------------------------

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
            "2.5",

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


    print()
    print("=" * 76)
    print(
        "NUDGEWISE V2.5 MODEL SAVED"
    )
    print("=" * 76)


    print(
        f"Model: "
        f"{best_name}"
    )

    print(
        f"Path: "
        f"{MODEL_PATH}"
    )

    print(
        f"Results: "
        f"{RESULTS_FOLDER}/"
    )


    print()
    print(
        "PRIMARY interpretation:"
    )

    print(
        "The model predicts the evidence-informed latent "
        "intervention probability distribution."
    )

    print()
    print(
        "The sampled hard label is NOT treated as perfect "
        "ground truth."
    )

    print()
    print(
        "Synthetic performance does NOT establish "
        "real-world intervention effectiveness."
    )

    print("=" * 76)


if __name__ == "__main__":
    main()