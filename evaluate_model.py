"""
evaluate_model.py

NudgeWise v2.8
Independent synthetic model evaluation.

PURPOSE
-------
Evaluate how faithfully the trained NudgeWise model reproduces
the behaviour of the evidence-informed synthetic decision generator.

Two complementary evaluations are reported:

1. SAMPLED-LABEL AGREEMENT

   The generator produces a probability distribution across six
   interventions and then probabilistically samples one label.

   Agreement with that single sampled label is useful, but the sampled
   label is not deterministic ground truth.

2. LATENT DISTRIBUTION FIDELITY

   The trained model's full six-class probability distribution is
   compared directly with the generator's underlying probability
   distribution.

   This is the stronger fidelity measure for NudgeWise because the
   synthetic generator deliberately represents behavioural ambiguity.

IMPORTANT
---------
These results evaluate fidelity to the NudgeWise synthetic behavioural
model.

They DO NOT establish:
- real-world intervention effectiveness
- clinical effectiveness
- causal wellbeing effects
- validated medical recommendations
"""

from __future__ import annotations

from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "models/nudge_model.pkl"
)

OUTPUT_DIRECTORY = Path(
    "evaluation"
)

EVALUATION_ROWS = 5_000

# Deliberately different from the synthetic training seed.
EVALUATION_SEED = 20260816


# ============================================================
# Expected production interventions
# ============================================================

EXPECTED_CLASSES = [
    "Connect socially",
    "Maintain habits",
    "Prepare for bed",
    "Reduce screen time",
    "Stay active",
    "Take a short break",
]


# ============================================================
# Probability helpers
# ============================================================

def normalise_probabilities(
    values,
) -> np.ndarray:
    """
    Convert an array into valid row-wise probability distributions.
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


    if values.ndim != 2:

        raise ValueError(
            "Probability output must be a two-dimensional array."
        )


    if not np.all(
        np.isfinite(
            values
        )
    ):

        raise ValueError(
            "Probability output contains NaN or infinite values."
        )


    values = np.clip(
        values,
        0.0,
        None,
    )


    # Prevent zero-total rows.
    values += 1e-12


    row_totals = values.sum(
        axis=1,
        keepdims=True,
    )


    return (
        values
        / row_totals
    )


def calculate_normalised_entropy(
    probabilities: np.ndarray,
) -> np.ndarray:
    """
    Calculate row-wise entropy scaled to 0-1.

    0 = highly concentrated probability distribution
    1 = maximally ambiguous uniform distribution
    """

    probabilities = normalise_probabilities(
        probabilities
    )


    entropy = -np.sum(
        probabilities
        * np.log(
            probabilities
            + 1e-12
        ),
        axis=1,
    )


    maximum_entropy = np.log(
        probabilities.shape[
            1
        ]
    )


    if maximum_entropy <= 0:

        return np.zeros(
            probabilities.shape[
                0
            ]
        )


    return (
        entropy
        / maximum_entropy
    )


def calculate_jensen_shannon_divergence(
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    """
    Calculate row-wise Jensen-Shannon divergence.

    Lower is better.

    0 means the two probability distributions are identical.
    """

    first = normalise_probabilities(
        first
    )

    second = normalise_probabilities(
        second
    )


    midpoint = (
        first
        + second
    ) / 2.0


    first_divergence = np.sum(
        first
        * np.log(
            (
                first
                + 1e-12
            )
            /
            (
                midpoint
                + 1e-12
            )
        ),
        axis=1,
    )


    second_divergence = np.sum(
        second
        * np.log(
            (
                second
                + 1e-12
            )
            /
            (
                midpoint
                + 1e-12
            )
        ),
        axis=1,
    )


    return (
        0.5
        * first_divergence
        + 0.5
        * second_divergence
    )


# ============================================================
# Column-name normalisation
# ============================================================

def normalise_probability_column_name(
    value: str,
) -> str:
    """
    Reduce different probability-column naming conventions
    to a comparable intervention key.

    Examples:
        p_connect_socially
        probability_connect_socially
        Connect socially

    all become:
        connect_socially
    """

    cleaned = str(
        value
    ).strip().lower()


    cleaned = re.sub(
        r"[^a-z0-9]+",
        "_",
        cleaned,
    )


    cleaned = cleaned.strip(
        "_"
    )


    known_prefixes = [
        "generator_probabilities_",
        "generator_probability_",
        "generator_probs_",
        "generator_prob_",
        "generator_p_",
        "probabilities_",
        "probability_",
        "probs_",
        "prob_",
        "p_",
    ]


    changed = True


    while changed:

        changed = False


        for prefix in known_prefixes:

            if cleaned.startswith(
                prefix
            ):

                cleaned = cleaned[
                    len(
                        prefix
                    ):
                ]

                changed = True

                break


    return cleaned


# ============================================================
# Fresh synthetic evaluation data
# ============================================================

def generate_fresh_evaluation_data():
    """
    Generate an independent synthetic evaluation sample.

    Automatically identifies:
    - the main evaluation dataframe
    - the generator diagnostics dataframe
    - the six-class latent probability dataframe

    This avoids depending on the exact tuple order returned by
    generate_dataset().
    """

    import generate_dataset as generator


    generator.RNG = (
        np.random.default_rng(
            EVALUATION_SEED
        )
    )


    generated = generator.generate_dataset(
        n_samples=EVALUATION_ROWS
    )


    if not isinstance(
        generated,
        tuple,
    ):

        raise TypeError(
            "generate_dataset() must return a tuple."
        )


    # ========================================================
    # Find main evaluation dataframe
    # ========================================================

    evaluation_data = None


    for item in generated:

        if not isinstance(
            item,
            pd.DataFrame,
        ):

            continue


        if (
            "best_nudge"
            in item.columns
        ):

            evaluation_data = item

            break


    if evaluation_data is None:

        raise ValueError(
            "Could not find the generated behavioural dataset."
        )


    # ========================================================
    # Find diagnostics dataframe
    # ========================================================

    generator_diagnostics = None


    for item in generated:

        if not isinstance(
            item,
            pd.DataFrame,
        ):

            continue


        if (
            "generator_confidence"
            in item.columns
            or "generator_entropy"
            in item.columns
        ):

            generator_diagnostics = item

            break


    if generator_diagnostics is None:

        generator_diagnostics = pd.DataFrame()


    # ========================================================
    # Find six-class probability dataframe
    # ========================================================

    latent_probability_data = None


    for item in generated:

        if not isinstance(
            item,
            pd.DataFrame,
        ):

            continue


        # Do not accidentally choose the main dataset
        # or diagnostics table.

        if item is evaluation_data:

            continue


        if item is generator_diagnostics:

            continue


        numeric = item.apply(
            pd.to_numeric,
            errors="coerce",
        )


        valid_columns = []


        for column in numeric.columns:

            series = numeric[
                column
            ]


            if series.isna().any():

                continue


            if (
                series.min() >= -1e-9
                and series.max() <= 1.0 + 1e-9
            ):

                valid_columns.append(
                    column
                )


        if len(
            valid_columns
        ) < 6:

            continue


        # Try likely six-column probability groups.

        from itertools import combinations


        for candidate_columns in combinations(
            valid_columns,
            6,
        ):

            candidate_values = (
                numeric[
                    list(
                        candidate_columns
                    )
                ]
                .to_numpy(
                    dtype=float
                )
            )


            row_sums = candidate_values.sum(
                axis=1
            )


            mean_sum_error = float(
                np.mean(
                    np.abs(
                        row_sums
                        - 1.0
                    )
                )
            )


            if mean_sum_error < 1e-5:

                latent_probability_data = (
                    item[
                        list(
                            candidate_columns
                        )
                    ].copy()
                )

                break


        if latent_probability_data is not None:

            break


    # ========================================================
    # Final validation
    # ========================================================

    if latent_probability_data is None:

        print()
        print(
            "DEBUG: generate_dataset() returned:"
        )


        for index, item in enumerate(
            generated
        ):

            if isinstance(
                item,
                pd.DataFrame,
            ):

                print(
                    f"{index}: DataFrame "
                    f"{list(item.columns)}"
                )

            else:

                print(
                    f"{index}: "
                    f"{type(item).__name__}"
                )


        raise ValueError(
            "Could not automatically locate the "
            "six-class generator probability dataframe."
        )


    if len(
        evaluation_data
    ) != len(
        latent_probability_data
    ):

        raise ValueError(
            "Evaluation rows and probability rows do not match."
        )


    return (
        evaluation_data,
        generator_diagnostics,
        latent_probability_data,
    )
    """
    Generate an independent synthetic evaluation sample.

    The existing generator is reused, but its RNG is replaced with
    a deterministic seed not used for production training data.
    """

    import generate_dataset as generator


    generator.RNG = (
        np.random.default_rng(
            EVALUATION_SEED
        )
    )


    generated = generator.generate_dataset(
        n_samples=EVALUATION_ROWS
    )


    if not isinstance(
        generated,
        tuple,
    ):

        raise TypeError(
            "generate_dataset() must return a tuple."
        )


    if len(
        generated
    ) < 3:

        raise ValueError(
            "generate_dataset() returned fewer than three outputs. "
            "The evaluator requires the dataset, diagnostics and "
            "latent probability output."
        )


    evaluation_data = (
        generated[
            0
        ]
    )


    generator_diagnostics = (
        generated[
            1
        ]
    )


    latent_probability_data = (
        generated[
            2
        ]
    )


    if not isinstance(
        evaluation_data,
        pd.DataFrame,
    ):

        evaluation_data = pd.DataFrame(
            evaluation_data
        )


    if not isinstance(
        generator_diagnostics,
        pd.DataFrame,
    ):

        generator_diagnostics = pd.DataFrame(
            generator_diagnostics
        )


    if not isinstance(
        latent_probability_data,
        pd.DataFrame,
    ):

        latent_probability_data = pd.DataFrame(
            latent_probability_data
        )


    return (
        evaluation_data,
        generator_diagnostics,
        latent_probability_data,
    )


# ============================================================
# Generator probability alignment
# ============================================================

def extract_generator_probabilities(
    probability_dataframe: pd.DataFrame,
    classes: list[str],
) -> tuple[np.ndarray, list[str]]:
    """
    Align generator probability columns with model class order.

    The current generator may use a naming convention different from
    the production prediction database.

    This function first performs semantic column matching.

    If semantic matching is impossible but the dataframe contains
    exactly six probability columns, positional alignment is used as
    a controlled backwards-compatible fallback.
    """

    if probability_dataframe.empty:

        raise ValueError(
            "Generator probability dataframe is empty."
        )


    # --------------------------------------------------------
    # Build normalised lookup
    # --------------------------------------------------------

    normalised_lookup: dict[str, str] = {}


    for column in probability_dataframe.columns:

        normalised = (
            normalise_probability_column_name(
                column
            )
        )


        normalised_lookup[
            normalised
        ] = column


    selected_columns = []

    missing_classes = []


    # --------------------------------------------------------
    # Try semantic matching first
    # --------------------------------------------------------

    for class_name in classes:

        target = (
            normalise_probability_column_name(
                class_name
            )
        )


        matching_column = (
            normalised_lookup.get(
                target
            )
        )


        if matching_column is None:

            missing_classes.append(
                class_name
            )

        else:

            selected_columns.append(
                matching_column
            )


    # --------------------------------------------------------
    # Exact semantic match succeeded
    # --------------------------------------------------------

    if not missing_classes:

        values = (
            probability_dataframe[
                selected_columns
            ]
            .to_numpy(
                dtype=float
            )
        )


        return (
            normalise_probabilities(
                values
            ),
            selected_columns,
        )


    # --------------------------------------------------------
    # Try identifying numeric probability columns
    # --------------------------------------------------------

    numeric_columns = []


    for column in probability_dataframe.columns:

        series = pd.to_numeric(
            probability_dataframe[
                column
            ],
            errors="coerce",
        )


        if series.notna().all():

            minimum = float(
                series.min()
            )


            maximum = float(
                series.max()
            )


            if (
                minimum >= -1e-9
                and maximum <= 1.0 + 1e-9
            ):

                numeric_columns.append(
                    column
                )


    # --------------------------------------------------------
    # Controlled six-column fallback
    # --------------------------------------------------------

    if len(
        numeric_columns
    ) == len(
        classes
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "Generator probability columns could not be matched "
            "semantically."
        )

        print(
            "Using six-column positional alignment."
        )

        print(
            f"Generator columns: {numeric_columns}"
        )

        print(
            f"Model classes:     {classes}"
        )


        values = (
            probability_dataframe[
                numeric_columns
            ]
            .to_numpy(
                dtype=float
            )
        )


        return (
            normalise_probabilities(
                values
            ),
            numeric_columns,
        )


    # --------------------------------------------------------
    # Final diagnostic failure
    # --------------------------------------------------------

    raise ValueError(
        "\nCould not align generator probability columns.\n"
        f"Missing classes: {missing_classes}\n"
        f"Available columns: "
        f"{list(probability_dataframe.columns)}\n"
        f"Probability-like numeric columns: {numeric_columns}"
    )


# ============================================================
# Model version
# ============================================================

def resolve_model_version(
    bundle: dict,
) -> str:
    """
    Retrieve a reproducible model-version label.
    """

    candidates = [
        bundle.get(
            "model_version"
        ),
        bundle.get(
            "version"
        ),
    ]


    for candidate in candidates:

        if candidate:

            return str(
                candidate
            )


    # Production predict.py currently contains MODEL_VERSION.
    try:

        from predict import MODEL_VERSION

        return str(
            MODEL_VERSION
        )


    except (
        ImportError,
        AttributeError,
    ):

        return "unversioned-model"


# ============================================================
# Validate classes
# ============================================================

def validate_classes(
    classes: list[str],
) -> None:
    """
    Verify that the deployed model uses the six production classes.
    """

    if len(
        classes
    ) != 6:

        raise ValueError(
            f"Expected six classes, found {len(classes)}."
        )


    if set(
        classes
    ) != set(
        EXPECTED_CLASSES
    ):

        raise ValueError(
            "Model intervention classes do not match the "
            "NudgeWise production vocabulary.\n"
            f"Found: {classes}"
        )


# ============================================================
# Evaluation
# ============================================================

def main() -> None:

    print()
    print("=" * 76)
    print(
        "NUDGEWISE v2.8 MODEL EVALUATION"
    )
    print("=" * 76)


    # ========================================================
    # Load model
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )


    bundle = joblib.load(
        MODEL_PATH
    )


    model = bundle[
        "model"
    ]


    classes = list(
        bundle[
            "classes"
        ]
    )


    features = list(
        bundle[
            "features"
        ]
    )


    model_version = (
        resolve_model_version(
            bundle
        )
    )


    validate_classes(
        classes
    )


    # ========================================================
    # Generate independent evaluation data
    # ========================================================

    (
        evaluation_data,
        generator_diagnostics,
        latent_probability_dataframe,
    ) = generate_fresh_evaluation_data()


    if len(
        evaluation_data
    ) != len(
        latent_probability_dataframe
    ):

        raise ValueError(
            "Evaluation dataset and generator probability output "
            "contain different numbers of rows."
        )


    # ========================================================
    # Validate model features
    # ========================================================

    missing_features = [
        feature
        for feature in features
        if feature
        not in evaluation_data.columns
    ]


    if missing_features:

        raise ValueError(
            "Evaluation data is missing model features: "
            + ", ".join(
                missing_features
            )
        )


    if (
        "best_nudge"
        not in evaluation_data.columns
    ):

        raise ValueError(
            "Evaluation data is missing the best_nudge label."
        )


    X = evaluation_data[
        features
    ]


    y_sampled = (
        evaluation_data[
            "best_nudge"
        ]
        .astype(
            str
        )
    )


    # ========================================================
    # Model probability distribution
    # ========================================================

    raw_model_output = model.predict(
        X
    )


    model_probabilities = (
        normalise_probabilities(
            raw_model_output
        )
    )


    if model_probabilities.shape[
        1
    ] != len(
        classes
    ):

        raise ValueError(
            "Model output dimension does not match "
            "the number of intervention classes."
        )


    # ========================================================
    # Generator latent probabilities
    # ========================================================

    (
        generator_probabilities,
        matched_generator_columns,
    ) = extract_generator_probabilities(
        probability_dataframe=(
            latent_probability_dataframe
        ),
        classes=classes,
    )


    if generator_probabilities.shape != (
        model_probabilities.shape
    ):

        raise ValueError(
            "Generator and model probability matrices "
            "have incompatible shapes."
        )


    # ========================================================
    # Hard choices
    # ========================================================

    model_indices = np.argmax(
        model_probabilities,
        axis=1,
    )


    generator_indices = np.argmax(
        generator_probabilities,
        axis=1,
    )


    model_predictions = np.asarray(
        [
            classes[
                index
            ]
            for index in model_indices
        ]
    )


    generator_preferred_classes = np.asarray(
        [
            classes[
                index
            ]
            for index in generator_indices
        ]
    )


    # ========================================================
    # Sampled-label metrics
    # ========================================================

    sampled_label_accuracy = accuracy_score(
        y_sampled,
        model_predictions,
    )


    macro_precision = precision_score(
        y_sampled,
        model_predictions,
        labels=classes,
        average="macro",
        zero_division=0,
    )


    macro_recall = recall_score(
        y_sampled,
        model_predictions,
        labels=classes,
        average="macro",
        zero_division=0,
    )


    macro_f1 = f1_score(
        y_sampled,
        model_predictions,
        labels=classes,
        average="macro",
        zero_division=0,
    )


    sampled_label_log_loss = log_loss(
        y_sampled,
        model_probabilities,
        labels=classes,
    )


    # ========================================================
    # Latent-generator fidelity
    # ========================================================

    preferred_choice_agreement = (
        accuracy_score(
            generator_preferred_classes,
            model_predictions,
        )
    )


    probability_difference = (
        model_probabilities
        - generator_probabilities
    )


    mean_probability_mae = float(
        np.mean(
            np.abs(
                probability_difference
            )
        )
    )


    probability_rmse = float(
        np.sqrt(
            np.mean(
                probability_difference
                ** 2
            )
        )
    )


    js_divergences = (
        calculate_jensen_shannon_divergence(
            generator_probabilities,
            model_probabilities,
        )
    )


    mean_js_divergence = float(
        np.mean(
            js_divergences
        )
    )


    median_js_divergence = float(
        np.median(
            js_divergences
        )
    )


    # ========================================================
    # Distribution ambiguity
    # ========================================================

    generator_top_probabilities = np.max(
        generator_probabilities,
        axis=1,
    )


    model_top_probabilities = np.max(
        model_probabilities,
        axis=1,
    )


    generator_entropy = (
        calculate_normalised_entropy(
            generator_probabilities
        )
    )


    model_entropy = (
        calculate_normalised_entropy(
            model_probabilities
        )
    )


    mean_generator_top_probability = float(
        np.mean(
            generator_top_probabilities
        )
    )


    mean_model_top_probability = float(
        np.mean(
            model_top_probabilities
        )
    )


    mean_generator_entropy = float(
        np.mean(
            generator_entropy
        )
    )


    mean_model_entropy = float(
        np.mean(
            model_entropy
        )
    )


    # ========================================================
    # Classification report
    # ========================================================

    classification_dictionary = (
        classification_report(
            y_sampled,
            model_predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    )


    classification_dataframe = (
        pd.DataFrame(
            classification_dictionary
        )
        .transpose()
    )


    # ========================================================
    # Sampled-label confusion matrix
    # ========================================================

    sampled_matrix = confusion_matrix(
        y_sampled,
        model_predictions,
        labels=classes,
    )


    sampled_confusion_dataframe = (
        pd.DataFrame(
            sampled_matrix,
            index=[
                f"actual_{class_name}"
                for class_name in classes
            ],
            columns=[
                f"predicted_{class_name}"
                for class_name in classes
            ],
        )
    )


    # ========================================================
    # Generator-preferred confusion matrix
    # ========================================================

    preferred_matrix = confusion_matrix(
        generator_preferred_classes,
        model_predictions,
        labels=classes,
    )


    preferred_confusion_dataframe = (
        pd.DataFrame(
            preferred_matrix,
            index=[
                f"generator_{class_name}"
                for class_name in classes
            ],
            columns=[
                f"model_{class_name}"
                for class_name in classes
            ],
        )
    )


    # ========================================================
    # Per-row evaluation output
    # ========================================================

    results = evaluation_data.copy()


    results[
        "sampled_label"
    ] = y_sampled.to_numpy()


    results[
        "generator_preferred_class"
    ] = generator_preferred_classes


    results[
        "model_prediction"
    ] = model_predictions


    results[
        "sampled_label_agreement"
    ] = (
        y_sampled.to_numpy()
        == model_predictions
    )


    results[
        "generator_preferred_agreement"
    ] = (
        generator_preferred_classes
        == model_predictions
    )


    results[
        "generator_top_probability"
    ] = generator_top_probabilities


    results[
        "model_top_probability"
    ] = model_top_probabilities


    results[
        "generator_normalised_entropy"
    ] = generator_entropy


    results[
        "model_normalised_entropy"
    ] = model_entropy


    results[
        "jensen_shannon_divergence"
    ] = js_divergences


    for index, class_name in enumerate(
        classes
    ):

        clean_name = (
            normalise_probability_column_name(
                class_name
            )
        )


        results[
            f"generator_p_{clean_name}"
        ] = generator_probabilities[
            :,
            index
        ]


        results[
            f"model_p_{clean_name}"
        ] = model_probabilities[
            :,
            index
        ]


    # ========================================================
    # Summary
    # ========================================================

    summary = pd.DataFrame(
        [
            {
                "model_version":
                    model_version,

                "evaluation_seed":
                    EVALUATION_SEED,

                "evaluation_rows":
                    len(
                        evaluation_data
                    ),

                "sampled_label_accuracy":
                    sampled_label_accuracy,

                "sampled_label_macro_precision":
                    macro_precision,

                "sampled_label_macro_recall":
                    macro_recall,

                "sampled_label_macro_f1":
                    macro_f1,

                "sampled_label_log_loss":
                    sampled_label_log_loss,

                "generator_preferred_choice_agreement":
                    preferred_choice_agreement,

                "mean_probability_mae":
                    mean_probability_mae,

                "probability_rmse":
                    probability_rmse,

                "mean_js_divergence":
                    mean_js_divergence,

                "median_js_divergence":
                    median_js_divergence,

                "mean_generator_top_probability":
                    mean_generator_top_probability,

                "mean_model_top_probability":
                    mean_model_top_probability,

                "mean_generator_normalised_entropy":
                    mean_generator_entropy,

                "mean_model_normalised_entropy":
                    mean_model_entropy,
            }
        ]
    )


    # ========================================================
    # Save outputs
    # ========================================================

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    summary.to_csv(
        OUTPUT_DIRECTORY
        / "evaluation_summary.csv",
        index=False,
    )


    classification_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "classification_report.csv"
    )


    sampled_confusion_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "sampled_label_confusion_matrix.csv"
    )


    preferred_confusion_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "generator_preferred_confusion_matrix.csv"
    )


    results.to_csv(
        OUTPUT_DIRECTORY
        / "evaluation_predictions.csv",
        index=False,
    )


    generator_diagnostics.to_csv(
        OUTPUT_DIRECTORY
        / "evaluation_generator_diagnostics.csv",
        index=False,
    )


    latent_probability_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "evaluation_generator_probabilities.csv",
        index=False,
    )


    # ========================================================
    # Console report
    # ========================================================

    print()
    print(
        f"Model version:                  "
        f"{model_version}"
    )

    print(
        f"Evaluation observations:        "
        f"{len(evaluation_data):,}"
    )

    print(
        f"Independent evaluation seed:    "
        f"{EVALUATION_SEED}"
    )

    print(
        f"Generator probability columns:  "
        f"{matched_generator_columns}"
    )


    # --------------------------------------------------------
    # Hard-label agreement
    # --------------------------------------------------------

    print()
    print("-" * 76)
    print(
        "SAMPLED-LABEL AGREEMENT"
    )
    print("-" * 76)

    print(
        "The synthetic generator probabilistically samples "
        "one recommendation from its latent distribution."
    )

    print()

    print(
        f"Sampled-label agreement:         "
        f"{sampled_label_accuracy:.3f} "
        f"({sampled_label_accuracy:.1%})"
    )

    print(
        f"Macro precision:                 "
        f"{macro_precision:.3f}"
    )

    print(
        f"Macro recall:                    "
        f"{macro_recall:.3f}"
    )

    print(
        f"Macro F1:                        "
        f"{macro_f1:.3f}"
    )

    print(
        f"Sampled-label log loss:          "
        f"{sampled_label_log_loss:.3f}"
    )


    # --------------------------------------------------------
    # Distribution fidelity
    # --------------------------------------------------------

    print()
    print("-" * 76)
    print(
        "LATENT DISTRIBUTION FIDELITY"
    )
    print("-" * 76)

    print(
        "This directly compares the model's recommendation "
        "distribution with the generator's latent distribution."
    )

    print()

    print(
        f"Preferred-choice agreement:      "
        f"{preferred_choice_agreement:.3f} "
        f"({preferred_choice_agreement:.1%})"
    )

    print(
        f"Mean probability MAE:            "
        f"{mean_probability_mae:.4f}"
    )

    print(
        f"Probability RMSE:                "
        f"{probability_rmse:.4f}"
    )

    print(
        f"Mean Jensen-Shannon divergence:  "
        f"{mean_js_divergence:.4f}"
    )

    print(
        f"Median JS divergence:            "
        f"{median_js_divergence:.4f}"
    )


    # --------------------------------------------------------
    # Ambiguity
    # --------------------------------------------------------

    print()
    print("-" * 76)
    print(
        "PROBABILITY DISTRIBUTION AMBIGUITY"
    )
    print("-" * 76)

    print(
        f"Generator mean top probability:  "
        f"{mean_generator_top_probability:.3f}"
    )

    print(
        f"Model mean top probability:      "
        f"{mean_model_top_probability:.3f}"
    )

    print(
        f"Generator mean norm. entropy:    "
        f"{mean_generator_entropy:.3f}"
    )

    print(
        f"Model mean norm. entropy:        "
        f"{mean_model_entropy:.3f}"
    )


    # --------------------------------------------------------
    # Per-class
    # --------------------------------------------------------

    print()
    print("-" * 76)
    print(
        "PER-CLASS SAMPLED-LABEL PERFORMANCE"
    )
    print("-" * 76)

    print(
        classification_dataframe[
            [
                "precision",
                "recall",
                "f1-score",
                "support",
            ]
        ].round(
            3
        )
    )


    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    print()
    print("=" * 76)
    print(
        "INTERPRETATION"
    )
    print("=" * 76)

    print(
        "Sampled-label agreement measures agreement with one "
        "stochastically selected synthetic recommendation."
    )

    print(
        "Preferred-choice agreement compares the model's top "
        "recommendation with the generator's highest-probability "
        "recommendation."
    )

    print(
        "Probability MAE, RMSE and Jensen-Shannon divergence "
        "measure similarity between the full distributions."
    )

    print(
        "Lower MAE, RMSE and Jensen-Shannon divergence indicate "
        "closer reproduction of the synthetic decision model."
    )

    print(
        "These metrics do not demonstrate that recommendations "
        "improve real adolescent wellbeing."
    )

    print()

    print(
        f"Outputs saved to: "
        f"{OUTPUT_DIRECTORY.resolve()}"
    )

    print("=" * 76)


if __name__ == "__main__":

    main()