"""
NudgeWise
Digital wellbeing, made personal.

Version 2 participant check-in application.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from components.navigation import (
    render_sidebar,
)

from database import (
    create_participant,
    create_tables,
    get_user,
    get_user_by_code,
    save_checkin,
    save_prediction,
)

from predict import (
    explain_prediction,
    predict_nudge,
)


# ============================================================
# Page
# ============================================================

st.set_page_config(
    page_title="NudgeWise",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:#FAFAF8;
        color:#20201F;
    }

    [data-testid="stHeader"] {
        background:transparent;
    }

    [data-testid="stSidebar"] {
        background:#F4F4F1;
        border-right:1px solid #E5E5E0;
    }

    .block-container {
        max-width:1080px;
        padding-top:3.4rem;
        padding-bottom:5rem;
    }

    h1 {
        font-size:2.7rem !important;
        font-weight:650 !important;
        letter-spacing:-0.05em !important;
        color:#171717 !important;
    }

    h2 {
        font-size:1.4rem !important;
        font-weight:600 !important;
        letter-spacing:-0.025em !important;
    }

    p {
        color:#696963;
        line-height:1.6;
    }

    hr {
        border:none;
        border-top:1px solid #E5E5E0;
        margin:2.3rem 0;
    }

    .stButton > button,
    .stFormSubmitButton > button {
        min-height:3rem;
        border-radius:11px !important;
        font-weight:600 !important;
    }

    [data-testid="stForm"] {
        border:none !important;
        padding:0 !important;
    }

    .participant-code {
        font-size:2rem;
        font-weight:650;
        letter-spacing:-0.04em;
        color:#171717;
        margin:0.5rem 0 0.6rem 0;
    }

    .quiet {
        font-size:0.86rem;
        color:#85857F;
        line-height:1.55;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Database
# ============================================================

create_tables()


# ============================================================
# Session defaults
# ============================================================

defaults = {
    "user_id": None,
    "prediction": None,
    "confidence": None,
    "prediction_id": None,
    "reasons": [],
    "checkin_submitted": False,
    "feedback_submitted": False,
    "new_participant": False,
}

for key, default in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# Sidebar
# ============================================================

render_sidebar()


# ============================================================
# PARTICIPANT ENTRY
# ============================================================

if st.session_state.user_id is None:

    st.caption(
        datetime.now().strftime(
            "%A, %d %B %Y"
        )
    )

    st.title(
        "Welcome to NudgeWise"
    )

    st.write(
        "Start a new participant session or resume "
        "an existing wellbeing history."
    )

    st.divider()

    new_tab, returning_tab = st.tabs(
        [
            "New participant",
            "Returning participant",
        ]
    )

    # --------------------------------------------------------
    # New participant
    # --------------------------------------------------------

    with new_tab:

        st.subheader(
            "Start a new profile"
        )

        st.write(
            "NudgeWise uses an anonymous participant code "
            "to keep each person's check-ins separate."
        )

        with st.form(
            "new_participant_form"
        ):

            nickname = st.text_input(
                "Nickname",
                placeholder="Optional display name",
            )

            age = st.number_input(
                "Age",
                min_value=10,
                max_value=18,
                value=15,
                step=1,
            )

            create_clicked = (
                st.form_submit_button(
                    "Create participant",
                    type="primary",
                    width="stretch",
                )
            )

        if create_clicked:

            participant = create_participant(
                nickname=(
                    nickname.strip()
                    or "Participant"
                ),
                age=int(age),
            )

            st.session_state.user_id = (
                participant["id"]
            )

            st.session_state.participant_code = (
                participant[
                    "participant_code"
                ]
            )

            st.session_state.new_participant = (
                True
            )

            st.rerun()

    # --------------------------------------------------------
    # Returning participant
    # --------------------------------------------------------

    with returning_tab:

        st.subheader(
            "Resume your profile"
        )

        st.write(
            "Enter the participant code you received "
            "when you first used NudgeWise."
        )

        with st.form(
            "returning_participant_form"
        ):

            participant_code = (
                st.text_input(
                    "Participant code",
                    placeholder="NW-ABC234",
                )
            )

            resume_clicked = (
                st.form_submit_button(
                    "Continue",
                    type="primary",
                    width="stretch",
                )
            )

        if resume_clicked:

            participant = (
                get_user_by_code(
                    participant_code
                )
            )

            if participant is None:

                st.error(
                    "That participant code "
                    "could not be found."
                )

            else:

                st.session_state.user_id = (
                    participant["id"]
                )

                st.session_state.participant_code = (
                    participant[
                        "participant_code"
                    ]
                )

                st.session_state.new_participant = (
                    False
                )

                st.rerun()

    st.stop()


# ============================================================
# Current participant
# ============================================================

participant = get_user(
    st.session_state.user_id
)

if participant is None:

    st.session_state.user_id = None
    st.rerun()


nickname = (
    participant.get("nickname")
    or participant.get("name")
    or "Participant"
)

participant_code = (
    participant.get("participant_code")
)


# ============================================================
# Newly created participant
# ============================================================

if st.session_state.new_participant:

    st.caption(
        "PARTICIPANT PROFILE CREATED"
    )

    st.title(
        f"Welcome, {nickname}"
    )

    st.write(
        "Your participant code keeps future check-ins "
        "connected to this wellbeing history."
    )

    st.markdown(
        f"""
        <div class="participant-code">
            {participant_code}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="quiet">
            Keep this code if you want to return to the same
            profile later. It is not a password and should not
            contain personal information.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    if st.button(
        "Continue to check-in",
        type="primary",
        width="stretch",
    ):

        st.session_state.new_participant = False
        st.rerun()

    st.stop()


# ============================================================
# Check-in header
# ============================================================

st.caption(
    datetime.now().strftime(
        "%A, %d %B %Y"
    )
)

st.title(
    "Today's check-in"
)

st.write(
    "A short check-in helps NudgeWise understand "
    "your current habits and context."
)

st.divider()


# ============================================================
# Check-in form
# ============================================================

with st.form(
    "daily_checkin"
):

    st.subheader(
        "How are you feeling?"
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
            help=(
                "Approximate hours "
                "of sleep last night."
            ),
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
        "Current activity",
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
    )

    st.divider()

    submitted = (
        st.form_submit_button(
            "Generate personalised guidance",
            type="primary",
            width="stretch",
        )
    )


# ============================================================
# Submit check-in
# ============================================================

if submitted:

    try:

        now = datetime.now()

        hour = now.hour

        day_type = (
            "Weekend"
            if now.weekday() >= 5
            else "Weekday"
        )

        # ----------------------------------------------------
        # NudgeWise AI v2.5
        # ----------------------------------------------------

        prediction, confidence = (
            predict_nudge(
                sleep=sleep,
                stress=stress,
                mood=mood,
                energy=energy,
                screen_time=screen_time,
                activity=activity,
                day_type=day_type,
                social=social,
                hour=hour,
            )
        )

        reasons = explain_prediction(
            prediction=prediction,
            sleep=sleep,
            stress=stress,
            screen_time=screen_time,
            hour=hour,
            energy=energy,
            mood=mood,
        )

        # ----------------------------------------------------
        # Save check-in
        # ----------------------------------------------------

        checkin_id = save_checkin(
            user_id=st.session_state.user_id,
            sleep=sleep,
            stress=stress,
            mood=mood,
            energy=energy,
            screen_time=screen_time,
            activity=activity,
            social=social,
            hour=hour,
        )

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

        # ----------------------------------------------------
        # Go straight to dashboard
        # ----------------------------------------------------

        st.switch_page(
            "pages/dashboard.py"
        )

    except Exception as error:

        st.error(
            "NudgeWise could not complete "
            "this check-in."
        )

        st.exception(
            error
        )