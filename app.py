"""
app.py

NudgeWise Version 2.6
Authenticated daily digital wellbeing check-in.

Features:
- Google authentication
- one check-in per participant per NZ calendar day
- edit today's check-in
- regenerate recommendation after edits
- persistent participant history
- physical activity minutes
- perceived connectedness
- fully integrated v2.6 AI prediction interface
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from components.navigation import (
    render_sidebar,
)

from database import (
    DuplicateCheckinError,
    create_authenticated_participant,
    create_tables,
    get_checkin_for_date,
    get_user,
    replace_prediction,
    save_checkin,
    save_prediction,
    update_checkin,
)

from predict import (
    explain_prediction,
    predict_nudge,
)

from services.auth import (
    ensure_current_participant,
    get_auth_hash,
    get_google_display_name,
    is_logged_in,
    login,
)


# ============================================================
# Configuration
# ============================================================

NZ_TIMEZONE = ZoneInfo(
    "Pacific/Auckland"
)


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="NudgeWise",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="locked",
)


# ============================================================
# Visual system
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

h3 {
    font-size:1.08rem !important;
    font-weight:600 !important;
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

[data-testid="stForm"] {
    border:none !important;
    padding:0 !important;
    background:transparent !important;
}

.stButton > button,
.stFormSubmitButton > button {
    min-height:3rem;
    border-radius:11px !important;
    font-weight:600 !important;
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
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[
            key
        ] = value


# ============================================================
# Sidebar
# ============================================================

render_sidebar()


# ============================================================
# Authentication gate
# ============================================================

if not is_logged_in():

    st.caption(
        "NUDGEWISE"
    )

    st.title(
        "Digital wellbeing, made personal."
    )

    st.write(
        "Sign in securely to keep your check-ins "
        "and wellbeing history connected across visits."
    )

    st.write("")

    if st.button(
        "Continue with Google",
        type="primary",
        width="stretch",
    ):

        login()

    st.caption(
        "NudgeWise does not store your Google password."
    )

    st.stop()


# ============================================================
# Restore participant
# ============================================================

participant = (
    ensure_current_participant()
)


# ============================================================
# First-time participant
# ============================================================

if participant is None:

    st.caption(
        "WELCOME TO NUDGEWISE"
    )

    st.title(
        "Set up your profile"
    )

    google_name = (
        get_google_display_name()
    )

    st.write(
        "Your Google account has been verified. "
        "NudgeWise only needs a small amount of "
        "information to create your profile."
    )

    with st.form(
        "profile_setup"
    ):

        nickname = st.text_input(
            "Display name",
            value=google_name,
            help=(
                "Used to personalise your "
                "NudgeWise interface."
            ),
        )

        age = st.number_input(
            "Age",
            min_value=10,
            max_value=18,
            value=15,
            step=1,
        )

        create_profile = (
            st.form_submit_button(
                "Create my NudgeWise profile",
                type="primary",
                width="stretch",
            )
        )

    if create_profile:

        auth_hash = (
            get_auth_hash()
        )

        if auth_hash is None:

            st.error(
                "NudgeWise could not verify "
                "your authenticated account."
            )

            st.stop()

        participant = (
            create_authenticated_participant(
                auth_subject_hash=auth_hash,
                nickname=(
                    nickname.strip()
                    or "Participant"
                ),
                age=int(age),
            )
        )

        st.session_state.user_id = (
            participant["id"]
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

    st.error(
        "Your NudgeWise profile could not be loaded."
    )

    st.stop()


nickname = (
    participant.get(
        "nickname"
    )
    or "Participant"
)


# ============================================================
# NZ date / time
# ============================================================

now = datetime.now(
    NZ_TIMEZONE
)

today_string = (
    now.date().isoformat()
)

current_hour = (
    now.hour
)

day_type = (
    "Weekend"
    if now.weekday() >= 5
    else "Weekday"
)


# ============================================================
# Existing daily check-in
# ============================================================

existing_checkin = (
    get_checkin_for_date(
        user_id=st.session_state.user_id,
        checkin_date=today_string,
    )
)

is_editing = (
    existing_checkin is not None
)


# ============================================================
# Header
# ============================================================

st.caption(
    now.strftime(
        "%A, %d %B %Y"
    )
)


if is_editing:

    st.title(
        "Edit today's check-in"
    )

    st.write(
        f"Welcome back, {nickname}. "
        "You have already completed today's check-in. "
        "You can correct or update your responses below."
    )

    with st.container(
        border=True
    ):

        st.markdown(
            "**Today's check-in is already recorded**"
        )

        st.write(
            "Saving changes will update today's record "
            "and regenerate your NudgeWise recommendation."
        )

else:

    st.title(
        "Today's check-in"
    )

    st.write(
        f"Welcome back, {nickname}. "
        "Answer based on your current state and today's behaviour."
    )


st.divider()


# ============================================================
# Prefill values
# ============================================================

if is_editing:

    default_sleep = float(
        existing_checkin[
            "sleep"
        ]
    )

    default_stress = int(
        existing_checkin[
            "stress"
        ]
    )

    default_mood = int(
        existing_checkin[
            "mood"
        ]
    )

    default_energy = int(
        existing_checkin[
            "energy"
        ]
    )

    default_screen_time = float(
        existing_checkin[
            "screen_time"
        ]
    )

    default_activity_minutes = int(
        existing_checkin.get(
            "activity_minutes",
            0,
        )
    )

    default_connectedness = int(
        existing_checkin.get(
            "connectedness",
            3,
        )
    )

    default_activity = (
        existing_checkin[
            "activity"
        ]
    )

else:

    default_sleep = 7.5
    default_stress = 3
    default_mood = 3
    default_energy = 3
    default_screen_time = 3.0
    default_activity_minutes = 30
    default_connectedness = 3
    default_activity = "Phone"


activity_options = [
    "Phone",
    "Studying",
    "Working",
    "Relaxing",
    "Exercise",
]


if default_activity not in activity_options:

    default_activity = "Phone"


activity_index = (
    activity_options.index(
        default_activity
    )
)


# ============================================================
# Daily check-in
# ============================================================

with st.form(
    "daily_checkin"
):

    # --------------------------------------------------------
    # Current wellbeing
    # --------------------------------------------------------

    st.subheader(
        "How are you feeling right now?"
    )

    st.caption(
        "Use the same scale definitions each day so your "
        "responses stay comparable."
    )


    left, right = st.columns(
        2,
        gap="large",
    )


    with left:

        sleep = st.number_input(
            "Sleep in your last main sleep period (hours)",
            min_value=0.0,
            max_value=16.0,
            value=default_sleep,
            step=0.5,
            format="%.1f",
            help=(
                "Approximate total hours of sleep during "
                "your most recent main sleep period."
            ),
        )


        mood = st.slider(
            "Mood",
            min_value=1,
            max_value=5,
            value=default_mood,
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
            value=default_stress,
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
            value=default_energy,
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
        "Recreational screen time today (hours)",
        min_value=0.0,
        max_value=24.0,
        value=default_screen_time,
        step=0.5,
        format="%.1f",
        help=(
            "Include social media, gaming, streaming and "
            "recreational browsing. Do not include screen use "
            "required for schoolwork or work."
        ),
    )


    activity_minutes = st.number_input(
        "Moderate-to-vigorous physical activity today (minutes)",
        min_value=0,
        max_value=300,
        value=default_activity_minutes,
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
        "How connected to other people have you felt today?",
        min_value=1,
        max_value=5,
        value=default_connectedness,
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
    # Immediate context
    # --------------------------------------------------------

    st.subheader(
        "Your current context"
    )

    st.caption(
        "This helps NudgeWise judge whether a suggestion "
        "makes sense right now. It is not a wellbeing score."
    )


    activity = st.selectbox(
        "What are you mainly doing right now?",
        activity_options,
        index=activity_index,
    )


    st.divider()


    submit_label = (
        "Save changes"
        if is_editing
        else "Generate personalised guidance"
    )


    submitted = (
        st.form_submit_button(
            submit_label,
            type="primary",
            width="stretch",
        )
    )


# ============================================================
# Process submission
# ============================================================

if submitted:

    try:

        # ----------------------------------------------------
        # v2.6 AI prediction
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
                day_type=day_type,
                hour=current_hour,
            )
        )


        reasons = (
            explain_prediction(
                prediction=prediction,
                sleep=sleep,
                stress=stress,
                screen_time=screen_time,
                hour=current_hour,
                energy=energy,
                mood=mood,
                activity_minutes=int(
                    activity_minutes
                ),
                connectedness=int(
                    connectedness
                ),
                activity=activity,
                day_type=day_type,
            )
        )


        # ----------------------------------------------------
        # Edit today's check-in
        # ----------------------------------------------------

        if is_editing:

            checkin_id = int(
                existing_checkin[
                    "id"
                ]
            )


            updated = update_checkin(
                checkin_id=checkin_id,
                user_id=st.session_state.user_id,
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
                hour=current_hour,
            )


            if not updated:

                st.error(
                    "NudgeWise could not update today's check-in."
                )

                st.stop()


            prediction_id = (
                replace_prediction(
                    checkin_id=checkin_id,
                    nudge=prediction,
                    confidence=confidence,
                )
            )


        # ----------------------------------------------------
        # New check-in
        # ----------------------------------------------------

        else:

            checkin_id = save_checkin(
                user_id=st.session_state.user_id,
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
                hour=current_hour,
                checkin_date=today_string,
                entry_type="live",
            )


            prediction_id = (
                save_prediction(
                    checkin_id=checkin_id,
                    nudge=prediction,
                    confidence=confidence,
                )
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


        st.switch_page(
            "pages/dashboard.py"
        )


    except DuplicateCheckinError:

        st.warning(
            "Today's check-in already exists. "
            "Reload the page to edit it."
        )


    except Exception as error:

        st.error(
            "NudgeWise could not save your check-in."
        )

        st.exception(
            error
        )