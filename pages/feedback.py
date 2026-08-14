"""
pages/feedback.py

NudgeWise Version 2
Independent usability feedback page.
"""

from __future__ import annotations

import streamlit as st

from components.theme import (
    apply_theme,
    configure_page,
)

configure_page()
apply_theme()

from components.navigation import (
    render_sidebar,
)

from database import (
    create_tables,
    get_user,
    save_usability_feedback,
)

from services.auth import (
    ensure_current_participant,
    is_logged_in,
)


# ============================================================
# Setup
# ============================================================

create_tables()
render_sidebar()


# ============================================================
# Authentication
# ============================================================

if not is_logged_in():

    st.switch_page(
        "app.py"
    )


participant = (
    ensure_current_participant()
)


if participant is None:

    st.switch_page(
        "app.py"
    )


user_id = st.session_state.get(
    "user_id"
)


if user_id is None:

    st.switch_page(
        "app.py"
    )


participant = get_user(
    user_id
)


if participant is None:

    st.switch_page(
        "app.py"
    )


nickname = (
    participant.get("nickname")
    or "Participant"
)


# ============================================================
# Page
# ============================================================

st.caption(
    "NUDGEWISE USER TESTING"
)

st.title(
    "Tell us about your experience"
)

st.write(
    f"Thanks for testing NudgeWise, {nickname}. "
    "This feedback is about the app itself rather than "
    "an individual AI recommendation."
)

st.divider()


# ============================================================
# Submission state
# ============================================================

if (
    "usability_feedback_submitted"
    not in st.session_state
):

    st.session_state[
        "usability_feedback_submitted"
    ] = False


if st.session_state[
    "usability_feedback_submitted"
]:

    st.success(
        "Thank you. Your feedback has been recorded."
    )

    st.write(
        "Your response will help evaluate and improve "
        "the NudgeWise user experience."
    )

    if st.button(
        "Submit another response",
        type="secondary",
    ):

        st.session_state[
            "usability_feedback_submitted"
        ] = False

        st.rerun()

    st.stop()


# ============================================================
# Feedback form
# ============================================================

with st.form(
    "usability_feedback_form"
):

    st.subheader(
        "Overall experience"
    )


    ease_of_use = st.slider(
        "How easy was NudgeWise to use?",
        min_value=1,
        max_value=5,
        value=3,
    )


    interface_clarity = st.slider(
        "How clear was the interface?",
        min_value=1,
        max_value=5,
        value=3,
    )


    trust = st.slider(
        "How trustworthy did NudgeWise feel?",
        min_value=1,
        max_value=5,
        value=3,
    )


    st.divider()


    st.subheader(
        "Your comments"
    )


    confusing = st.text_area(
        "Was anything confusing?",
        placeholder=(
            "Navigation, wording, graphs, login, "
            "recommendations..."
        ),
        max_chars=750,
    )


    improvement = st.text_area(
        "What would you improve?",
        placeholder=(
            "Tell us what would make NudgeWise "
            "more useful or easier to use."
        ),
        max_chars=750,
    )


    st.divider()


    submitted = (
        st.form_submit_button(
            "Submit app feedback",
            type="primary",
            width="stretch",
        )
    )


# ============================================================
# Save
# ============================================================

if submitted:

    save_usability_feedback(
        user_id=user_id,
        ease_of_use=int(
            ease_of_use
        ),
        interface_clarity=int(
            interface_clarity
        ),
        trust=int(
            trust
        ),
        confusing=(
            confusing.strip()
            or None
        ),
        improvement=(
            improvement.strip()
            or None
        ),
    )


    st.session_state[
        "usability_feedback_submitted"
    ] = True


    st.rerun()


# ============================================================
# Research note
# ============================================================

st.divider()


with st.expander(
    "Why NudgeWise collects this feedback"
):

    st.write(
        """
        Usability feedback is stored separately from feedback
        about individual AI recommendations.

        This allows the research to distinguish between
        perceptions of the AI model and perceptions of the
        user interface.
        """
    )