"""
services/auth.py

Google authentication bridge for NudgeWise.

Google performs authentication through Streamlit OIDC.

NudgeWise does not store:
- Google passwords
- OAuth access tokens
- Google email addresses

Instead, Google's stable OIDC subject identifier is combined with
a private application pepper and hashed. The resulting pseudonymous
identifier is linked to the user's NudgeWise participant record.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from database import get_user_by_auth_hash

# ============================================================
# Participant restoration
# ============================================================

def ensure_current_participant():
    """
    Restore the NudgeWise participant associated with the
    currently authenticated Google identity.

    Returns:
        participant dictionary when profile exists
        None when first-time setup is required
    """

    if not is_logged_in():

        st.session_state.pop(
            "user_id",
            None,
        )

        return None

    auth_hash = get_auth_hash()

    if auth_hash is None:
        return None

    participant = (
        get_user_by_auth_hash(
            auth_hash
        )
    )

    if participant is None:

        st.session_state.pop(
            "user_id",
            None,
        )

        return None

    st.session_state.user_id = (
        participant["id"]
    )

    return participant


# ============================================================
# Authentication
# ============================================================

def is_logged_in() -> bool:
    """Return True when Streamlit has an authenticated user."""

    return bool(
        st.user.is_logged_in
    )


def login() -> None:
    """Start the Google OIDC login process."""

    st.login()


def logout() -> None:
    """
    Clear NudgeWise session information and log out.

    Database records are not deleted.
    """

    session_keys = [
        "user_id",
        "prediction",
        "confidence",
        "prediction_id",
        "reasons",
        "checkin_submitted",
        "feedback_submitted",
    ]

    for key in session_keys:
        st.session_state.pop(
            key,
            None,
        )

    st.logout()


# ============================================================
# Google identity
# ============================================================

def get_google_subject() -> str | None:
    """
    Return Google's stable OIDC subject identifier.

    We deliberately avoid using email as the database identity.
    """

    if not is_logged_in():
        return None

    subject = st.user.get(
        "sub"
    )

    if not subject:
        return None

    return str(subject)


def get_google_display_name() -> str:
    """Return Google's display name when available."""

    if not is_logged_in():
        return "Participant"

    name = st.user.get(
        "name"
    )

    if not name:
        return "Participant"

    return str(name)


# ============================================================
# Pseudonymous research identity
# ============================================================

def get_auth_hash() -> str | None:
    """
    Generate a stable pseudonymous identifier.

    Google subject ID
        +
    private identity pepper
        ->
    SHA-256 hash

    Neither the Google subject nor email is stored directly
    in the NudgeWise database.
    """

    subject = get_google_subject()

    if subject is None:
        return None

    try:
        pepper = st.secrets[
            "research"
        ][
            "identity_pepper"
        ]

    except Exception as error:
        raise RuntimeError(
            "NudgeWise could not find "
            "[research].identity_pepper in "
            ".streamlit/secrets.toml"
        ) from error

    value = (
        f"google:{subject}:{pepper}"
    )

    return hashlib.sha256(
        value.encode(
            "utf-8"
        )
    ).hexdigest()