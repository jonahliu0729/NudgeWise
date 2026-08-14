"""
components/navigation.py

Shared navigation for NudgeWise Version 2.

Important:
- No CSS hides the sidebar.
- No CSS manipulates Streamlit collapse controls.
- Default Streamlit page navigation is disabled in config.toml.
- NudgeWise renders its own links using st.page_link().
"""

from __future__ import annotations

import html

import streamlit as st

from database import get_user

from services.auth import (
    ensure_current_participant,
    is_logged_in,
    logout,
)


# ============================================================
# Participant identity
# ============================================================

def render_participant() -> bool:
    """Render the currently authenticated participant."""

    user_id = st.session_state.get(
        "user_id"
    )

    if user_id is None:
        return False

    participant = get_user(
        user_id
    )

    if participant is None:
        return False

    nickname = (
        participant.get("nickname")
        or participant.get("name")
        or "Participant"
    )

    safe_nickname = html.escape(
        str(nickname)
    )

    st.caption(
        "YOUR PROFILE"
    )

    st.markdown(
        f"""
<div style="
    font-size:0.96rem;
    font-weight:600;
    color:#20201F;
    margin-top:0.15rem;
    margin-bottom:0.15rem;
">
{safe_nickname}
</div>
""",
        unsafe_allow_html=True,
    )

    return True


# ============================================================
# Navigation links
# ============================================================

def render_page_links() -> None:
    """Render NudgeWise's custom navigation."""

    st.page_link(
        "app.py",
        label="Check-in",
    )

    st.page_link(
        "pages/dashboard.py",
        label="Your wellbeing",
    )

    st.page_link(
        "pages/feedback.py",
        label="Give feedback",
    )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar() -> None:
    """Render the shared NudgeWise sidebar."""

    logged_in = is_logged_in()

    if logged_in:
        ensure_current_participant()

    with st.sidebar:

        # ----------------------------------------------------
        # Brand
        # ----------------------------------------------------

        st.markdown(
            """
<div style="
    font-size:1.28rem;
    font-weight:650;
    letter-spacing:-0.035em;
    margin-bottom:0.12rem;
    color:#20201F;
">
NudgeWise
</div>

<div style="
    font-size:0.78rem;
    color:#85857F;
    line-height:1.45;
">
Digital wellbeing, made personal.
</div>
""",
            unsafe_allow_html=True,
        )

        st.divider()

        # ----------------------------------------------------
        # Logged-out state
        # ----------------------------------------------------

        if not logged_in:

            st.caption(
                "Sign in to begin."
            )

            return

        # ----------------------------------------------------
        # Participant
        # ----------------------------------------------------

        participant_exists = (
            render_participant()
        )

        if participant_exists:
            st.divider()

        # ----------------------------------------------------
        # Navigation
        # ----------------------------------------------------

        render_page_links()

        st.divider()

        # ----------------------------------------------------
        # Sign out
        # ----------------------------------------------------

        if st.button(
            "Sign out",
            type="secondary",
            width="stretch",
            key="nudgewise_sign_out",
        ):

            logout()