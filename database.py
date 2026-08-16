"""
database.py

Persistent Supabase/PostgreSQL storage for NudgeWise.

NudgeWise v2.8 research logging architecture.

Stores:
- authenticated participants
- live and retrospective check-ins
- behavioural measurements
- complete AI prediction snapshots
- model version
- six-class probability distributions
- uncertainty metrics
- contextual action shown to the participant
- recommendation feedback
- usability feedback

IMPORTANT
---------
The Supabase secret key must remain in Streamlit Secrets and
must never be committed to Git.

Existing prediction records created before v2.8 remain valid.
Their newer research metadata fields may simply be NULL.
"""

from __future__ import annotations

import secrets

from typing import Any

import streamlit as st

from supabase import (
    Client,
    create_client,
)


# ============================================================
# Errors
# ============================================================

class DuplicateCheckinError(Exception):
    """
    Raised when a participant already has a check-in
    for the selected date.
    """


# ============================================================
# Supabase client
# ============================================================

@st.cache_resource
def get_supabase() -> Client:
    """
    Create one cached Supabase client for the Streamlit process.
    """

    try:

        url = st.secrets[
            "supabase"
        ][
            "url"
        ]

        key = st.secrets[
            "supabase"
        ][
            "key"
        ]

    except Exception as error:

        raise RuntimeError(
            "Supabase credentials are missing. "
            "Add [supabase] url and key to Streamlit secrets."
        ) from error


    if not url or not key:

        raise RuntimeError(
            "Supabase credentials are empty."
        )


    return create_client(
        str(
            url
        ),
        str(
            key
        ),
    )


# ============================================================
# Compatibility setup
# ============================================================

def create_tables() -> None:
    """
    Retained for compatibility with the existing application.

    PostgreSQL schema migrations are performed through the
    Supabase SQL Editor.
    """

    get_supabase()


# ============================================================
# Generic helpers
# ============================================================

def _safe_probability(
    value,
) -> float | None:
    """
    Convert a probability-like value to a safe 0-1 float.

    None remains None so older prediction records can remain
    backwards compatible.
    """

    if value is None:

        return None


    try:

        probability = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


    return max(
        0.0,
        min(
            1.0,
            probability,
        ),
    )


def _probability_from_distribution(
    probabilities: dict[str, float] | None,
    class_name: str,
) -> float | None:
    """
    Safely extract one intervention probability.
    """

    if not probabilities:

        return None


    return _safe_probability(
        probabilities.get(
            class_name
        )
    )


# ============================================================
# Connectedness compatibility
# ============================================================

def connectedness_to_social(
    connectedness: int,
) -> str:
    """
    Preserve the legacy social field for compatibility.

    The production v2.6+ AI uses connectedness directly.
    """

    value = max(
        1,
        min(
            5,
            int(
                connectedness
            ),
        ),
    )


    if value <= 2:

        return "Low"


    if value == 3:

        return "Medium"


    return "High"


# ============================================================
# Participant codes
# ============================================================

def _generate_participant_code() -> str:

    alphabet = (
        "ABCDEFGHJKLMNPQRSTUVWXYZ"
        "23456789"
    )


    token = "".join(
        secrets.choice(
            alphabet
        )
        for _ in range(
            6
        )
    )


    return (
        f"NW-{token}"
    )


def _participant_code_exists(
    participant_code: str,
) -> bool:

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "users"
        )
        .select(
            "id"
        )
        .eq(
            "participant_code",
            participant_code,
        )
        .limit(
            1
        )
        .execute()
    )


    return bool(
        response.data
    )


# ============================================================
# Participant creation
# ============================================================

def create_authenticated_participant(
    auth_subject_hash: str,
    nickname: str,
    age: int,
) -> dict[str, Any]:

    existing = get_user_by_auth_hash(
        auth_subject_hash
    )


    if existing is not None:

        return existing


    while True:

        participant_code = (
            _generate_participant_code()
        )


        if not _participant_code_exists(
            participant_code
        ):

            break


    clean_nickname = (
        nickname.strip()
        or "Participant"
    )


    payload = {
        "name":
            clean_nickname,

        "nickname":
            clean_nickname,

        "age":
            int(
                age
            ),

        "participant_code":
            participant_code,

        "auth_subject_hash":
            auth_subject_hash,
    }


    supabase = get_supabase()


    response = (
        supabase
        .table(
            "users"
        )
        .insert(
            payload
        )
        .execute()
    )


    if not response.data:

        raise RuntimeError(
            "Supabase did not return the newly created participant."
        )


    return dict(
        response.data[
            0
        ]
    )


# ============================================================
# Participant retrieval
# ============================================================

def get_user(
    user_id: int,
) -> dict[str, Any] | None:

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "users"
        )
        .select(
            "*"
        )
        .eq(
            "id",
            int(
                user_id
            ),
        )
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return dict(
        response.data[
            0
        ]
    )


def get_user_by_auth_hash(
    auth_subject_hash: str,
) -> dict[str, Any] | None:

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "users"
        )
        .select(
            "*"
        )
        .eq(
            "auth_subject_hash",
            auth_subject_hash,
        )
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return dict(
        response.data[
            0
        ]
    )


# ============================================================
# Check-in retrieval
# ============================================================

def get_checkin_for_date(
    user_id: int,
    checkin_date: str,
) -> dict[str, Any] | None:

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "checkins"
        )
        .select(
            "*"
        )
        .eq(
            "user_id",
            int(
                user_id
            ),
        )
        .eq(
            "checkin_date",
            checkin_date,
        )
        .order(
            "id",
            desc=True,
        )
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return dict(
        response.data[
            0
        ]
    )


def get_checkin_by_id(
    checkin_id: int,
    user_id: int | None = None,
) -> dict[str, Any] | None:

    supabase = get_supabase()


    query = (
        supabase
        .table(
            "checkins"
        )
        .select(
            "*"
        )
        .eq(
            "id",
            int(
                checkin_id
            ),
        )
    )


    if user_id is not None:

        query = query.eq(
            "user_id",
            int(
                user_id
            ),
        )


    response = (
        query
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return dict(
        response.data[
            0
        ]
    )


# ============================================================
# Save check-in
# ============================================================

def save_checkin(
    user_id: int,
    sleep: float,
    stress: int,
    mood: int,
    energy: int,
    screen_time: float,
    activity_minutes: int,
    activity: str,
    connectedness: int,
    hour: int,
    checkin_date: str,
    entry_type: str = "live",
) -> int:

    existing = get_checkin_for_date(
        user_id=user_id,
        checkin_date=checkin_date,
    )


    if existing is not None:

        raise DuplicateCheckinError(
            "A check-in already exists for this date."
        )


    connectedness = max(
        1,
        min(
            5,
            int(
                connectedness
            ),
        ),
    )


    payload = {
        "user_id":
            int(
                user_id
            ),

        "sleep":
            float(
                sleep
            ),

        "stress":
            int(
                stress
            ),

        "mood":
            int(
                mood
            ),

        "energy":
            int(
                energy
            ),

        "screen_time":
            float(
                screen_time
            ),

        "activity_minutes":
            int(
                activity_minutes
            ),

        "activity":
            activity,

        "connectedness":
            connectedness,

        "social":
            connectedness_to_social(
                connectedness
            ),

        "hour":
            int(
                hour
            ),

        "checkin_date":
            checkin_date,

        "entry_type":
            entry_type,
    }


    supabase = get_supabase()


    try:

        response = (
            supabase
            .table(
                "checkins"
            )
            .insert(
                payload
            )
            .execute()
        )

    except Exception as error:

        error_text = str(
            error
        ).lower()


        if (
            "duplicate"
            in error_text
            or "unique"
            in error_text
        ):

            raise DuplicateCheckinError(
                "A check-in already exists for this date."
            ) from error


        raise


    if not response.data:

        raise RuntimeError(
            "Supabase did not return the newly created check-in."
        )


    return int(
        response.data[
            0
        ][
            "id"
        ]
    )


# ============================================================
# Update check-in
# ============================================================

def update_checkin(
    checkin_id: int,
    user_id: int,
    sleep: float,
    stress: int,
    mood: int,
    energy: int,
    screen_time: float,
    activity_minutes: int,
    activity: str,
    connectedness: int,
    hour: int,
) -> bool:

    connectedness = max(
        1,
        min(
            5,
            int(
                connectedness
            ),
        ),
    )


    payload = {
        "sleep":
            float(
                sleep
            ),

        "stress":
            int(
                stress
            ),

        "mood":
            int(
                mood
            ),

        "energy":
            int(
                energy
            ),

        "screen_time":
            float(
                screen_time
            ),

        "activity_minutes":
            int(
                activity_minutes
            ),

        "activity":
            activity,

        "connectedness":
            connectedness,

        "social":
            connectedness_to_social(
                connectedness
            ),

        "hour":
            int(
                hour
            ),
    }


    supabase = get_supabase()


    response = (
        supabase
        .table(
            "checkins"
        )
        .update(
            payload
        )
        .eq(
            "id",
            int(
                checkin_id
            ),
        )
        .eq(
            "user_id",
            int(
                user_id
            ),
        )
        .execute()
    )


    return bool(
        response.data
    )


# ============================================================
# Delete check-in
# ============================================================

def delete_checkin(
    checkin_id: int,
    user_id: int,
) -> bool:

    existing = get_checkin_by_id(
        checkin_id=checkin_id,
        user_id=user_id,
    )


    if existing is None:

        return False


    supabase = get_supabase()


    response = (
        supabase
        .table(
            "checkins"
        )
        .delete()
        .eq(
            "id",
            int(
                checkin_id
            ),
        )
        .eq(
            "user_id",
            int(
                user_id
            ),
        )
        .execute()
    )


    return bool(
        response.data
    )


# ============================================================
# Check-in history
# ============================================================

def get_recent_checkins(
    user_id: int,
    limit: int = 30,
):

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "checkins"
        )
        .select(
            "*"
        )
        .eq(
            "user_id",
            int(
                user_id
            ),
        )
        .order(
            "checkin_date",
            desc=True,
        )
        .order(
            "created_at",
            desc=True,
        )
        .limit(
            int(
                limit
            )
        )
        .execute()
    )


    return [
        dict(
            row
        )
        for row
        in (
            response.data
            or []
        )
    ]


# ============================================================
# Prediction payload builder
# ============================================================

def _build_prediction_payload(
    checkin_id: int,
    nudge: str,
    confidence: float,
    model_version: str | None = None,
    action_engine_version: str | None = None,
    probabilities: dict[str, float] | None = None,
    probability_margin: float | None = None,
    entropy: float | None = None,
    normalised_entropy: float | None = None,
    certainty: str | None = None,
    action_id: str | None = None,
    action_title: str | None = None,
    action_text: str | None = None,
    action_reason: str | None = None,
) -> dict[str, Any]:
    """
    Build one complete prediction snapshot.

    Optional values preserve compatibility with prediction rows
    created before research-grade logging was introduced.
    """

    payload = {
        "checkin_id":
            int(
                checkin_id
            ),

        "predicted_nudge":
            str(
                nudge
            ),

        "confidence":
            float(
                confidence
            ),

        "model_version":
            model_version,

        "action_engine_version":
            action_engine_version,

        "p_connect_socially":
            _probability_from_distribution(
                probabilities,
                "Connect socially",
            ),

        "p_maintain_habits":
            _probability_from_distribution(
                probabilities,
                "Maintain habits",
            ),

        "p_prepare_for_bed":
            _probability_from_distribution(
                probabilities,
                "Prepare for bed",
            ),

        "p_reduce_screen_time":
            _probability_from_distribution(
                probabilities,
                "Reduce screen time",
            ),

        "p_stay_active":
            _probability_from_distribution(
                probabilities,
                "Stay active",
            ),

        "p_take_a_short_break":
            _probability_from_distribution(
                probabilities,
                "Take a short break",
            ),

        "probability_margin":
            (
                float(
                    probability_margin
                )
                if probability_margin is not None
                else None
            ),

        "entropy":
            (
                float(
                    entropy
                )
                if entropy is not None
                else None
            ),

        "normalised_entropy":
            (
                float(
                    normalised_entropy
                )
                if normalised_entropy is not None
                else None
            ),

        "certainty":
            certainty,

        "action_id":
            action_id,

        "action_title":
            action_title,

        "action_text":
            action_text,

        "action_reason":
            action_reason,
    }


    return payload


# ============================================================
# Save prediction
# ============================================================

def save_prediction(
    checkin_id: int,
    nudge: str,
    confidence: float,
    model_version: str | None = None,
    action_engine_version: str | None = None,
    probabilities: dict[str, float] | None = None,
    probability_margin: float | None = None,
    entropy: float | None = None,
    normalised_entropy: float | None = None,
    certainty: str | None = None,
    action_id: str | None = None,
    action_title: str | None = None,
    action_text: str | None = None,
    action_reason: str | None = None,
) -> int:
    """
    Save a prediction.

    Newer code can store the complete research snapshot.

    Older code that only provides:
        checkin_id
        nudge
        confidence

    remains valid.
    """

    payload = _build_prediction_payload(
        checkin_id=checkin_id,
        nudge=nudge,
        confidence=confidence,
        model_version=model_version,
        action_engine_version=action_engine_version,
        probabilities=probabilities,
        probability_margin=probability_margin,
        entropy=entropy,
        normalised_entropy=normalised_entropy,
        certainty=certainty,
        action_id=action_id,
        action_title=action_title,
        action_text=action_text,
        action_reason=action_reason,
    )


    supabase = get_supabase()


    response = (
        supabase
        .table(
            "predictions"
        )
        .insert(
            payload
        )
        .execute()
    )


    if not response.data:

        raise RuntimeError(
            "Supabase did not return the created prediction."
        )


    return int(
        response.data[
            0
        ][
            "id"
        ]
    )


# ============================================================
# Replace prediction
# ============================================================

def replace_prediction(
    checkin_id: int,
    nudge: str,
    confidence: float,
    model_version: str | None = None,
    action_engine_version: str | None = None,
    probabilities: dict[str, float] | None = None,
    probability_margin: float | None = None,
    entropy: float | None = None,
    normalised_entropy: float | None = None,
    certainty: str | None = None,
    action_id: str | None = None,
    action_title: str | None = None,
    action_text: str | None = None,
    action_reason: str | None = None,
) -> int:
    """
    Replace the prediction associated with an edited check-in.

    Any old feedback attached to the replaced prediction is removed.
    """

    supabase = get_supabase()


    old_predictions = (
        supabase
        .table(
            "predictions"
        )
        .select(
            "id"
        )
        .eq(
            "checkin_id",
            int(
                checkin_id
            ),
        )
        .execute()
    )


    for prediction in (
        old_predictions.data
        or []
    ):

        (
            supabase
            .table(
                "feedback"
            )
            .delete()
            .eq(
                "prediction_id",
                int(
                    prediction[
                        "id"
                    ]
                ),
            )
            .execute()
        )


    (
        supabase
        .table(
            "predictions"
        )
        .delete()
        .eq(
            "checkin_id",
            int(
                checkin_id
            ),
        )
        .execute()
    )


    return save_prediction(
        checkin_id=checkin_id,
        nudge=nudge,
        confidence=confidence,
        model_version=model_version,
        action_engine_version=action_engine_version,
        probabilities=probabilities,
        probability_margin=probability_margin,
        entropy=entropy,
        normalised_entropy=normalised_entropy,
        certainty=certainty,
        action_id=action_id,
        action_title=action_title,
        action_text=action_text,
        action_reason=action_reason,
    )


# ============================================================
# Update prediction research metadata
# ============================================================

def update_prediction_metadata(
    prediction_id: int,
    model_version: str | None = None,
    action_engine_version: str | None = None,
    probabilities: dict[str, float] | None = None,
    probability_margin: float | None = None,
    entropy: float | None = None,
    normalised_entropy: float | None = None,
    certainty: str | None = None,
    action_id: str | None = None,
    action_title: str | None = None,
    action_text: str | None = None,
    action_reason: str | None = None,
) -> bool:
    """
    Update research metadata on an existing prediction.

    Useful during the transition from older prediction code.
    """

    payload: dict[str, Any] = {}


    if model_version is not None:

        payload[
            "model_version"
        ] = model_version


    if action_engine_version is not None:

        payload[
            "action_engine_version"
        ] = action_engine_version


    if probabilities is not None:

        payload.update(
            {
                "p_connect_socially":
                    _probability_from_distribution(
                        probabilities,
                        "Connect socially",
                    ),

                "p_maintain_habits":
                    _probability_from_distribution(
                        probabilities,
                        "Maintain habits",
                    ),

                "p_prepare_for_bed":
                    _probability_from_distribution(
                        probabilities,
                        "Prepare for bed",
                    ),

                "p_reduce_screen_time":
                    _probability_from_distribution(
                        probabilities,
                        "Reduce screen time",
                    ),

                "p_stay_active":
                    _probability_from_distribution(
                        probabilities,
                        "Stay active",
                    ),

                "p_take_a_short_break":
                    _probability_from_distribution(
                        probabilities,
                        "Take a short break",
                    ),
            }
        )


    optional_fields = {
        "probability_margin":
            probability_margin,

        "entropy":
            entropy,

        "normalised_entropy":
            normalised_entropy,

        "certainty":
            certainty,

        "action_id":
            action_id,

        "action_title":
            action_title,

        "action_text":
            action_text,

        "action_reason":
            action_reason,
    }


    for field, value in optional_fields.items():

        if value is not None:

            payload[
                field
            ] = value


    if not payload:

        return True


    supabase = get_supabase()


    response = (
        supabase
        .table(
            "predictions"
        )
        .update(
            payload
        )
        .eq(
            "id",
            int(
                prediction_id
            ),
        )
        .execute()
    )


    return bool(
        response.data
    )


# ============================================================
# Prediction retrieval
# ============================================================

def get_prediction_by_id(
    prediction_id: int,
) -> dict[str, Any] | None:

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "predictions"
        )
        .select(
            "*"
        )
        .eq(
            "id",
            int(
                prediction_id
            ),
        )
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return dict(
        response.data[
            0
        ]
    )


def get_prediction_for_checkin(
    checkin_id: int,
) -> dict[str, Any] | None:

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "predictions"
        )
        .select(
            "*"
        )
        .eq(
            "checkin_id",
            int(
                checkin_id
            ),
        )
        .order(
            "created_at",
            desc=True,
        )
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return dict(
        response.data[
            0
        ]
    )


def get_recent_predictions(
    user_id: int,
    limit: int = 30,
):

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "predictions"
        )
        .select(
            """
            id,
            checkin_id,
            predicted_nudge,
            confidence,
            model_version,
            action_engine_version,
            p_connect_socially,
            p_maintain_habits,
            p_prepare_for_bed,
            p_reduce_screen_time,
            p_stay_active,
            p_take_a_short_break,
            probability_margin,
            entropy,
            normalised_entropy,
            certainty,
            action_id,
            action_title,
            action_text,
            action_reason,
            created_at,
            checkins!inner(user_id)
            """
        )
        .eq(
            "checkins.user_id",
            int(
                user_id
            ),
        )
        .order(
            "created_at",
            desc=True,
        )
        .limit(
            int(
                limit
            )
        )
        .execute()
    )


    results = []


    for row in (
        response.data
        or []
    ):

        clean_row = dict(
            row
        )


        clean_row.pop(
            "checkins",
            None,
        )


        results.append(
            clean_row
        )


    return results


def get_latest_prediction(
    user_id: int,
):

    rows = get_recent_predictions(
        user_id=user_id,
        limit=1,
    )


    if not rows:

        return None


    return rows[
        0
    ]


# ============================================================
# Recommendation feedback
# ============================================================

def get_feedback_for_prediction(
    prediction_id: int,
) -> dict[str, Any] | None:

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "feedback"
        )
        .select(
            "*"
        )
        .eq(
            "prediction_id",
            int(
                prediction_id
            ),
        )
        .order(
            "id",
            desc=True,
        )
        .limit(
            1
        )
        .execute()
    )


    if not response.data:

        return None


    return dict(
        response.data[
            0
        ]
    )


def save_feedback(
    prediction_id: int,
    accepted: int,
    completed: int,
    rating: int,
    makes_sense: int | None = None,
    comment: str | None = None,
    action_id: str | None = None,
) -> None:
    """
    Save feedback for a prediction.

    If action_id is omitted, NudgeWise automatically retrieves the
    action linked to the prediction so old dashboard code remains
    compatible.
    """

    supabase = get_supabase()


    if action_id is None:

        prediction = get_prediction_by_id(
            prediction_id
        )


        if prediction:

            action_id = (
                prediction.get(
                    "action_id"
                )
            )


    payload = {
        "prediction_id":
            int(
                prediction_id
            ),

        "accepted":
            int(
                accepted
            ),

        "completed":
            int(
                completed
            ),

        "rating":
            int(
                rating
            ),

        "makes_sense":
            (
                int(
                    makes_sense
                )
                if makes_sense is not None
                else None
            ),

        "comment":
            comment,

        "action_id":
            action_id,
    }


    existing = get_feedback_for_prediction(
        prediction_id
    )


    if existing:

        (
            supabase
            .table(
                "feedback"
            )
            .update(
                payload
            )
            .eq(
                "prediction_id",
                int(
                    prediction_id
                ),
            )
            .execute()
        )

    else:

        (
            supabase
            .table(
                "feedback"
            )
            .insert(
                payload
            )
            .execute()
        )


# ============================================================
# Usability feedback
# ============================================================

def save_usability_feedback(
    user_id: int,
    ease_of_use: int,
    interface_clarity: int,
    trust: int,
    confusing: str | None,
    improvement: str | None,
) -> None:

    payload = {
        "user_id":
            int(
                user_id
            ),

        "ease_of_use":
            int(
                ease_of_use
            ),

        "interface_clarity":
            int(
                interface_clarity
            ),

        "trust":
            int(
                trust
            ),

        "confusing":
            confusing,

        "improvement":
            improvement,
    }


    supabase = get_supabase()


    (
        supabase
        .table(
            "usability_feedback"
        )
        .insert(
            payload
        )
        .execute()
    )


# ============================================================
# Connection test
# ============================================================

def test_database_connection() -> bool:
    """
    Small development diagnostic.
    """

    supabase = get_supabase()


    response = (
        supabase
        .table(
            "users"
        )
        .select(
            "id"
        )
        .limit(
            1
        )
        .execute()
    )


    return response.data is not None