"""
pages/past_checkin.py

NudgeWise Version 2.6
Retrospective missed-day check-in.

Uses the same v2.6 production variables as the live check-in:
- sleep
- stress
- mood
- energy
- recreational screen time
- physical activity minutes
- perceived connectedness
- current context
- hour
- day type

Retrospective responses are marked separately so they can be
distinguished during research analysis.
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


# ============================================================
# Date
# ============================================================

now = datetime.now(
    NZ_TIMEZONE
)

today = now.date()

yesterday = (
    today
    - timedelta(
        days=1
    )
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
    "Use this for a previous day you forgot to record. "
    "Try to answer based on what you remember from that day."
)

st.info(
    "Retrospective responses rely on memory, so NudgeWise "
    "stores them separately from same-day check-ins for "
    "research analysis."
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


    # --------------------------------------------------------
    # Wellbeing
    # --------------------------------------------------------

    st.subheader(
        "How were you feeling?"
    )

    st.caption(
        "Use the same scale definitions as the normal daily check-in."
    )


    left, right = st.columns(
        2,
        gap="large",
    )


    with left:

        sleep = st.number_input(
            "Sleep in your main sleep period before that day (hours)",
            min_value=0.0,
            max_value=16.0,
            value=7.5,
            step=0.5,
            format="%.1f",
            help=(
                "Approximate total hours of sleep during "
                "your main sleep period before the selected day."
            ),
        )


        mood = st.slider(
            "Mood",
            min_value=1,
            max_value=5,
            value=3,
            help=(
                "1 = very low mood. "
                "5 = very good mood."
            ),
        )

        st.caption(
            "1 = very low mood · 5 = very good mood"
        )


    with right:

        stress = st.slider(
            "Stress",
            min_value=1,
            max_value=5,
            value=3,
            help=(
                "1 = very low stress. "
                "5 = very high stress."
            ),
        )

        st.caption(
            "1 = very low stress · 5 = very high stress"
        )


        energy = st.slider(
            "Energy",
            min_value=1,
            max_value=5,
            value=3,
            help=(
                "1 = very low energy. "
                "5 = very high energy."
            ),
        )

        st.caption(
            "1 = very low energy · 5 = very high energy"
        )


    st.divider()


    # --------------------------------------------------------
    # Daily behaviour
    # --------------------------------------------------------

    st.subheader(
        "Your daily habits"
    )


    screen_time = st.number_input(
        "Recreational screen time that day (hours)",
        min_value=0.0,
        max_value=24.0,
        value=3.0,
        step=0.5,
        format="%.1f",
        help=(
            "Include social media, gaming, streaming and "
            "recreational browsing. Do not include required "
            "schoolwork or work."
        ),
    )


    activity_minutes = st.number_input(
        "Moderate-to-vigorous physical activity that day (minutes)",
        min_value=0,
        max_value=300,
        value=30,
        step=10,
        help=(
            "Approximate minutes of activity that noticeably "
            "raised your breathing or heart rate, such as sport, "
            "running, brisk cycling or active training."
        ),
    )

    st.caption(
        "Enter minutes. For example, 60 = one hour."
    )


    connectedness = st.slider(
        "How connected to other people did you feel that day?",
        min_value=1,
        max_value=5,
        value=3,
        help=(
            "Think about meaningful connection rather than "
            "simply how many people you were around."
        ),
    )

    st.caption(
        "1 = not connected at all · 5 = very connected"
    )


    st.divider()


    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    st.subheader(
        "Context"
    )

    st.caption(
        "This describes the situation around the time "
        "represented by the retrospective check-in."
    )


    activity = st.selectbox(
        "What were you mainly doing around that time?",
        [
            "Phone",
            "Studying",
            "Working",
            "Relaxing",
            "Exercise",
        ],
    )


    approximate_hour = st.slider(
        "Approximate time this check-in represents",
        min_value=6,
        max_value=23,
        value=18,
        help=(
            "Choose roughly when during the selected day "
            "these responses best describe you."
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
# Process retrospective submission
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

        # ----------------------------------------------------
        # v2.6 prediction
        # ----------------------------------------------------

        prediction, confidence = (
            predict_nudge(
                sleep=sleep,
                stress=stress,
                mood=mood,
                energy=energy,
                screen_time=screen_time,
                activity_minutes=int(
                    activity_minutes
                ),
                connectedness=int(
                    connectedness
                ),
                activity=activity,
                day_type=selected_day_type,
                hour=approximate_hour,
            )
        )


        reasons = (
            explain_prediction(
                prediction=prediction,
                sleep=sleep,
                stress=stress,
                screen_time=screen_time,
                hour=approximate_hour,
                energy=energy,
                mood=mood,
                activity_minutes=int(
                    activity_minutes
                ),
                connectedness=int(
                    connectedness
                ),
                activity=activity,
                day_type=selected_day_type,
            )
        )


        # ----------------------------------------------------
        # Save retrospective check-in
        # ----------------------------------------------------

        checkin_id = save_checkin(
            user_id=user_id,
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity_minutes=int(
                activity_minutes
            ),
            activity=activity,
            connectedness=int(
                connectedness
            ),
            hour=approximate_hour,
            checkin_date=selected_string,
            entry_type="retrospective",
        )


        # ----------------------------------------------------
        # Save associated prediction
        # ----------------------------------------------------

        prediction_id = save_prediction(
            checkin_id=checkin_id,
            nudge=prediction,
            confidence=confidence,
        )


        # ----------------------------------------------------
        # Session
        # ----------------------------------------------------

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

        st.session_state.checkin_submitted = (
            True
        )

        st.session_state.feedback_submitted = (
            False
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
            "NudgeWise could not save this "
            "retrospective check-in."
        )

        st.exception(
            error
        )