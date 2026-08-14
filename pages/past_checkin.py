"""
pages/past_checkin.py

NudgeWise Version 2
Retrospective missed-day check-in.

Allows participants to add a check-in for a previous date while
marking it explicitly as retrospective research data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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
    DuplicateCheckinError,
    create_tables,
    get_checkin_for_date,
    get_user,
    save_checkin,
    save_prediction,
)

from predict import (
    explain_prediction,
    predict_nudge,
)

from services.auth import (
    ensure_current_participant,
    is_logged_in,
)


# ============================================================
# Configuration
# ============================================================

NZ_TIMEZONE = ZoneInfo(
    "Pacific/Auckland"
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


participant = ensure_current_participant()

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


# ============================================================
# Date
# ============================================================

now = datetime.now(
    NZ_TIMEZONE
)

today = now.date()

yesterday = (
    today - timedelta(days=1)
)


# ============================================================
# Header
# ============================================================

st.caption(
    "RETROSPECTIVE CHECK-IN"
)

st.title(
    "Add a missed check-in"
)

st.write(
    "Use this if you missed a previous day. "
    "Retrospective entries are stored separately from "
    "same-day check-ins so they can be distinguished during research analysis."
)

st.info(
    "Try to answer based on what you remember from that day. "
    "Because this is recalled information, NudgeWise marks it as retrospective."
)

st.divider()


# ============================================================
# Form
# ============================================================

with st.form(
    "past_checkin_form"
):

    selected_date = st.date_input(
        "Which day did you miss?",
        value=yesterday,
        max_value=yesterday,
    )

    st.subheader(
        "How were you feeling?"
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    with left:

        sleep = st.number_input(
            "Sleep",
            min_value=0.0,
            max_value=16.0,
            value=7.5,
            step=0.5,
            format="%.1f",
        )

        mood = st.slider(
            "Mood",
            min_value=1,
            max_value=5,
            value=3,
        )

    with right:

        stress = st.slider(
            "Stress",
            min_value=1,
            max_value=5,
            value=3,
        )

        energy = st.slider(
            "Energy",
            min_value=1,
            max_value=5,
            value=3,
        )

    st.divider()

    st.subheader(
        "Your digital habits"
    )

    screen_time = st.number_input(
        "Recreational screen time",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.5,
        format="%.1f",
    )

    activity = st.selectbox(
        "Main activity",
        [
            "Phone",
            "Studying",
            "Working",
            "Relaxing",
            "Exercise",
        ],
    )

    social = st.selectbox(
        "Social interaction",
        [
            "Low",
            "Medium",
            "High",
        ],
        index=1,
    )

    approximate_hour = st.slider(
        "Approximate time this check-in represents",
        min_value=6,
        max_value=23,
        value=18,
        help=(
            "Choose roughly when during that day "
            "these answers best describe you."
        ),
    )

    st.divider()

    submitted = (
        st.form_submit_button(
            "Add missed check-in",
            type="primary",
            width="stretch",
        )
    )


# ============================================================
# Save
# ============================================================

if submitted:

    selected_string = (
        selected_date.isoformat()
    )

    existing = get_checkin_for_date(
        user_id=user_id,
        checkin_date=selected_string,
    )

    if existing is not None:

        st.warning(
            "A check-in already exists for that date."
        )

        st.stop()


    selected_day_type = (
        "Weekend"
        if selected_date.weekday() >= 5
        else "Weekday"
    )


    try:

        prediction, confidence = (
            predict_nudge(
                sleep=sleep,
                stress=stress,
                mood=mood,
                energy=energy,
                screen_time=screen_time,
                activity=activity,
                day_type=selected_day_type,
                social=social,
                hour=approximate_hour,
            )
        )


        reasons = explain_prediction(
            prediction=prediction,
            sleep=sleep,
            stress=stress,
            screen_time=screen_time,
            hour=approximate_hour,
            energy=energy,
            mood=mood,
            activity=activity,
            social=social,
            day_type=selected_day_type,
        )


        checkin_id = save_checkin(
            user_id=user_id,
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity=activity,
            social=social,
            hour=approximate_hour,
            checkin_date=selected_string,
            entry_type="retrospective",
        )


        prediction_id = save_prediction(
            checkin_id=checkin_id,
            nudge=prediction,
            confidence=confidence,
        )


        st.session_state.prediction = (
            prediction
        )

        st.session_state.confidence = (
            confidence
        )

        st.session_state.reasons = (
            reasons
        )

        st.session_state.prediction_id = (
            prediction_id
        )


        st.success(
            "Missed check-in added."
        )

        st.switch_page(
            "pages/dashboard.py"
        )


    except DuplicateCheckinError:

        st.warning(
            "A check-in already exists for that date."
        )


    except Exception as error:

        st.error(
            "NudgeWise could not save this retrospective check-in."
        )

        st.exception(
            error
        )