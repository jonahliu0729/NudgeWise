"""
export_research_data.py

NudgeWise v2.8
De-identified research dataset exporter.

Combines:
- participant research ID
- derived age at check-in
- check-in variables
- prediction snapshot
- uncertainty
- contextual action
- feedback

Excluded from research export:
- participant display name
- authentication hash
- exact date of birth

IMPORTANT
---------
This script reads from Supabase but does not modify the database.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from database import (
    get_supabase,
)


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIRECTORY = Path(
    "research_exports"
)


OUTPUT_PATH = (
    OUTPUT_DIRECTORY
    / "nudgewise_research_export.csv"
)


# ============================================================
# Helpers
# ============================================================

def calculate_age_at_date(
    dob_value,
    target_value,
):
    """
    Calculate participant age on a historical check-in date.

    Returns None where DOB is unavailable.
    """

    if not dob_value:

        return None


    try:

        birth_date = date.fromisoformat(
            str(
                dob_value
            )
        )


        target_date = date.fromisoformat(
            str(
                target_value
            )
        )


    except (
        TypeError,
        ValueError,
    ):

        return None


    return (
        target_date.year
        - birth_date.year
        - (
            (
                target_date.month,
                target_date.day,
            )
            <
            (
                birth_date.month,
                birth_date.day,
            )
        )
    )


def fetch_table(
    table_name: str,
) -> pd.DataFrame:
    """
    Fetch one complete Supabase table.
    """

    supabase = get_supabase()


    response = (
        supabase
        .table(
            table_name
        )
        .select(
            "*"
        )
        .execute()
    )


    return pd.DataFrame(
        response.data
        or []
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 72)
    print("NUDGEWISE RESEARCH EXPORT")
    print("=" * 72)


    users = fetch_table(
        "users"
    )


    checkins = fetch_table(
        "checkins"
    )


    predictions = fetch_table(
        "predictions"
    )


    feedback = fetch_table(
        "feedback"
    )


    if checkins.empty:

        print(
            "No check-ins found."
        )

        return


    # ========================================================
    # Participant research information
    # ========================================================

    if users.empty:

        participant_data = pd.DataFrame(
            columns=[
                "user_id",
                "participant_code",
                "profile_age",
                "date_of_birth",
            ]
        )


    else:

        participant_data = users[
            [
                "id",
                "participant_code",
                "age",
                "date_of_birth",
            ]
        ].copy()


        participant_data = (
            participant_data.rename(
                columns={
                    "id":
                        "user_id",

                    "age":
                        "profile_age",
                }
            )
        )


    # ========================================================
    # Check-ins
    # ========================================================

    dataset = checkins.copy()


    dataset = dataset.merge(
        participant_data,
        on="user_id",
        how="left",
    )


    # ========================================================
    # Historical age
    # ========================================================

    dataset[
        "age_at_checkin"
    ] = dataset.apply(
        lambda row:
            calculate_age_at_date(
                row.get(
                    "date_of_birth"
                ),
                row.get(
                    "checkin_date"
                ),
            ),
        axis=1,
    )


    # Fall back to legacy profile age where no DOB exists.

    dataset[
        "age_at_checkin"
    ] = dataset[
        "age_at_checkin"
    ].fillna(
        dataset[
            "profile_age"
        ]
    )


    # ========================================================
    # Prediction data
    # ========================================================

    if not predictions.empty:

        prediction_data = (
            predictions.rename(
                columns={
                    "id":
                        "prediction_id",

                    "created_at":
                        "prediction_created_at",
                }
            )
        )


        dataset = dataset.merge(
            prediction_data,
            left_on="id",
            right_on="checkin_id",
            how="left",
            suffixes=(
                "",
                "_prediction",
            ),
        )


    # ========================================================
    # Feedback data
    # ========================================================

    if (
        not feedback.empty
        and "prediction_id"
        in dataset.columns
    ):

        feedback_data = (
            feedback.rename(
                columns={
                    "id":
                        "feedback_id",

                    "action_id":
                        "feedback_action_id",

                    "created_at":
                        "feedback_created_at",
                }
            )
        )


        dataset = dataset.merge(
            feedback_data,
            on="prediction_id",
            how="left",
            suffixes=(
                "",
                "_feedback",
            ),
        )


    # ========================================================
    # De-identification
    # ========================================================

    columns_to_remove = [
        "date_of_birth",
        "profile_age",
        "user_id",
        "social",
    ]


    dataset = dataset.drop(
        columns=[
            column

            for column
            in columns_to_remove

            if column
            in dataset.columns
        ]
    )


    # Rename check-in ID clearly.

    if "id" in dataset.columns:

        dataset = dataset.rename(
            columns={
                "id":
                    "checkin_id"
            }
        )


    # ========================================================
    # Useful derived fields
    # ========================================================

    if "rating" in dataset.columns:

        dataset[
            "feedback_available"
        ] = dataset[
            "rating"
        ].notna()


    if "completed" in dataset.columns:

        dataset[
            "followed_recommendation"
        ] = dataset[
            "completed"
        ].map(
            {
                0:
                    "No",

                1:
                    "Partly",

                2:
                    "Yes",
            }
        )


    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    dataset.to_csv(
        OUTPUT_PATH,
        index=False,
    )


    print()
    print(
        f"Rows exported: "
        f"{len(dataset):,}"
    )


    print(
        f"Participants:  "
        f"{dataset['participant_code'].nunique()}"
        if "participant_code"
        in dataset.columns
        else "Participants: unknown"
    )


    print(
        f"Output:        "
        f"{OUTPUT_PATH.resolve()}"
    )


    print()
    print(
        "Excluded from export:"
    )

    print(
        "- display names"
    )

    print(
        "- authentication hashes"
    )

    print(
        "- exact dates of birth"
    )


    print()
    print("=" * 72)


if __name__ == "__main__":

    main()