"""
NudgeWise
Digital wellbeing, made personal.

Version 2 application entry point.
Diagnostic build.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from database import (
    create_tables,
    create_user,
    save_checkin,
    save_feedback,
    save_prediction,
)

from predict import (
    explain_prediction,
    predict_nudge,
)


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="NudgeWise",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Visual system
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #FAFAF8;
        color: #20201F;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    [data-testid="stSidebar"] {
        background: #F4F4F1;
        border-right: 1px solid #E5E5E0;
    }

    [data-testid="stSidebar"] * {
        color: #30302E;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 3.5rem;
        padding-bottom: 4rem;
    }


    /* Typography */

    h1,
    h2,
    h3 {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "SF Pro Display",
            "Helvetica Neue",
            Arial,
            sans-serif;
    }

    h1 {
        font-size: 2.8rem !important;
        font-weight: 650 !important;
        letter-spacing: -0.045em !important;
        color: #171717 !important;
    }

    h2 {
        font-size: 1.45rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
        color: #20201F !important;
    }

    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #20201F !important;
    }

    p {
        color: #666660;
        line-height: 1.6;
    }


    /* Inputs */

    [data-baseweb="input"],
    [data-baseweb="select"],
    [data-baseweb="textarea"] {
        border-radius: 10px;
    }

    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        border-radius: 10px;
    }

    .stSlider {
        padding-top: 0.2rem;
        padding-bottom: 0.8rem;
    }


    /* Buttons */

    .stButton > button,
    .stFormSubmitButton > button {
        width: 100%;
        min-height: 3rem;

        border-radius: 12px !important;

        border: 1px solid #1D1D1F !important;
        background-color: #1D1D1F !important;
        color: #FFFFFF !important;

        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "SF Pro Text",
            "Helvetica Neue",
            Arial,
            sans-serif;

        font-size: 0.95rem !important;
        font-weight: 600 !important;

        padding: 0.6rem 1.2rem !important;

        opacity: 1 !important;
        cursor: pointer !important;

        box-shadow: none !important;

        transition:
            background-color 0.15s ease,
            border-color 0.15s ease,
            transform 0.15s ease;
    }

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {
        background-color: #343436 !important;
        border-color: #343436 !important;
        color: #FFFFFF !important;
    }

    .stButton > button:active,
    .stFormSubmitButton > button:active {
        background-color: #111111 !important;
        border-color: #111111 !important;
        transform: scale(0.99);
    }


    /* Form */

    [data-testid="stForm"] {
        border: none !important;
        padding: 0 !important;
        background: transparent !important;
    }


    /* Sidebar */

    .sidebar-brand {
        font-size: 1.25rem;
        font-weight: 650;
        letter-spacing: -0.025em;
        color: #20201F;
    }

    .sidebar-description {
        font-size: 0.78rem;
        line-height: 1.5;
        color: #777771;
        margin-top: 0.15rem;
    }


    /* Recommendation */

    .recommendation-title {
        font-size: 1.55rem;
        font-weight: 600;
        letter-spacing: -0.025em;
        color: #20201F;
        margin-top: 0.3rem;
        margin-bottom: 0.6rem;
    }

    .confidence {
        font-size: 0.8rem;
        color: #777771;
        margin-top: 0.8rem;
    }


    /* Dividers */

    hr {
        border: none;
        border-top: 1px solid #E5E5E0;
        margin: 2.5rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Database
# ============================================================

st.write("DEBUG 0: creating database tables")

create_tables()

st.write("DEBUG 0.1: database ready")


# ============================================================
# Session state
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = None

if "reasons" not in st.session_state:
    st.session_state.reasons = []

if "prediction_id" not in st.session_state:
    st.session_state.prediction_id = None

if "page_after_checkin" not in st.session_state:
    st.session_state.page_after_checkin = False


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">NudgeWise</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sidebar-description">
            Digital wellbeing, made personal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Today")

    selected_page = st.radio(
        "Navigate",
        [
            "Check-in",
            "Your wellbeing",
        ],
        label_visibility="collapsed",
    )


# ============================================================
# Header
# ============================================================

st.caption(
    datetime.now().strftime("%A, %d %B %Y")
)

st.title("NudgeWise")

st.write(
    "A personal view of your digital wellbeing "
    "and daily habits."
)


# ============================================================
# CHECK-IN
# ============================================================

if selected_page == "Check-in":

    st.divider()

    st.subheader("Today's check-in")

    st.write(
        "A short check-in helps NudgeWise understand "
        "your current habits and context."
    )

    st.write("")


    # ========================================================
    # Check-in form
    # ========================================================

    with st.form("daily_checkin"):

        # ----------------------------------------------------
        # Personal information
        # ----------------------------------------------------

        personal_col_1, personal_col_2 = st.columns(
            2,
            gap="large",
        )

        with personal_col_1:

            name = st.text_input(
                "Name",
                placeholder="Your name",
            )

        with personal_col_2:

            age = st.number_input(
                "Age",
                min_value=10,
                max_value=100,
                value=15,
                step=1,
            )


        st.divider()


        # ----------------------------------------------------
        # Wellbeing
        # ----------------------------------------------------

        st.subheader("How are you feeling?")

        wellbeing_col_1, wellbeing_col_2 = st.columns(
            2,
            gap="large",
        )

        with wellbeing_col_1:

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

        with wellbeing_col_2:

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


        # ----------------------------------------------------
        # Digital behaviour
        # ----------------------------------------------------

        st.subheader("Your digital habits")

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


        # ----------------------------------------------------
        # Submit
        # ----------------------------------------------------

        submitted = st.form_submit_button(
            "Continue",
            type="primary",
            width="stretch",
        )

        st.write(
            "DEBUG BUTTON:",
            submitted,
        )


    # ========================================================
    # Process submission
    # ========================================================

    if submitted:

        st.write("DEBUG 1: submission received")

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not name.strip():

            st.error(
                "Please enter your name before continuing."
            )

            st.write(
                "DEBUG STOP: name was empty"
            )

        else:

            st.write(
                "DEBUG 1.1: name validation passed"
            )

            try:

                # ------------------------------------------------
                # Current context
                # ------------------------------------------------

                current_datetime = datetime.now()

                hour = current_datetime.hour

                day_type = (
                    "Weekend"
                    if current_datetime.weekday() >= 5
                    else "Weekday"
                )

                st.write(
                    "DEBUG 1.2: context generated"
                )

                st.write(
                    "DEBUG hour:",
                    hour,
                )

                st.write(
                    "DEBUG day_type:",
                    day_type,
                )


                # ------------------------------------------------
                # User
                # ------------------------------------------------

                st.write(
                    "DEBUG 2: creating user"
                )

                user_id = create_user(
                    name.strip(),
                    age,
                )

                st.write(
                    "DEBUG 2: user created"
                )

                st.write(
                    "DEBUG user_id:",
                    user_id,
                )


                # ------------------------------------------------
                # AI prediction
                # ------------------------------------------------

                st.write(
                    "DEBUG 3: starting prediction"
                )

                prediction, confidence = predict_nudge(
                    sleep,
                    stress,
                    mood,
                    energy,
                    screen_time,
                    activity,
                    day_type,
                    social,
                    hour,
                )

                st.write(
                    "DEBUG 3: prediction generated"
                )

                st.write(
                    "DEBUG prediction:",
                    prediction,
                )

                st.write(
                    "DEBUG confidence:",
                    confidence,
                )


                # ------------------------------------------------
                # Explanation
                # ------------------------------------------------

                st.write(
                    "DEBUG 4: starting explanation"
                )

                reasons = explain_prediction(
                    prediction,
                    sleep,
                    stress,
                    screen_time,
                    hour,
                    energy,
                    mood,
                )

                st.write(
                    "DEBUG 4: explanation generated"
                )

                st.write(
                    "DEBUG reasons:",
                    reasons,
                )


                # ------------------------------------------------
                # Save check-in
                # ------------------------------------------------

                st.write(
                    "DEBUG 5: saving check-in"
                )

                checkin_id = save_checkin(
                    user_id,
                    sleep,
                    stress,
                    mood,
                    energy,
                    screen_time,
                    activity,
                    social,
                    hour,
                )

                st.write(
                    "DEBUG 5: check-in saved"
                )

                st.write(
                    "DEBUG checkin_id:",
                    checkin_id,
                )


                # ------------------------------------------------
                # Save prediction
                # ------------------------------------------------

                st.write(
                    "DEBUG 6: saving prediction"
                )

                prediction_id = save_prediction(
                    checkin_id,
                    prediction,
                    confidence,
                )

                st.write(
                    "DEBUG 6: prediction saved"
                )

                st.write(
                    "DEBUG prediction_id:",
                    prediction_id,
                )


                # ------------------------------------------------
                # Session state
                # ------------------------------------------------

                st.write(
                    "DEBUG 7: storing session state"
                )

                st.session_state.prediction = prediction

                st.session_state.confidence = confidence

                st.session_state.reasons = reasons

                st.session_state.prediction_id = prediction_id

                st.session_state.page_after_checkin = True

                st.write(
                    "DEBUG 7: session state stored"
                )


                # ------------------------------------------------
                # Success
                # ------------------------------------------------

                st.success(
                    "Your check-in has been saved."
                )

                st.write(
                    "DEBUG 8: COMPLETE"
                )

                st.info(
                    "Diagnostic mode: submission pipeline completed."
                )

                st.write(
                    "The result is stored in session state. "
                    "The next step will be automatically switching "
                    "the interface to the wellbeing dashboard."
                )


            except Exception as error:

                st.error(
                    "NudgeWise encountered an error while "
                    "processing your check-in."
                )

                st.write(
                    "DEBUG ERROR:"
                )

                st.exception(error)


# ============================================================
# WELLBEING
# ============================================================

else:

    st.divider()

    st.subheader("Your wellbeing")


    # ========================================================
    # No prediction
    # ========================================================

    if st.session_state.prediction is None:

        st.info(
            "Complete a check-in to see your "
            "personalised guidance."
        )

        st.stop()


    # ========================================================
    # Confidence
    # ========================================================

    confidence = st.session_state.confidence

    if confidence is None:

        confidence_display = "Not available"

    else:

        confidence_display = (
            f"{confidence:.0%}"
            if confidence <= 1
            else f"{confidence:.0f}%"
        )


    # ========================================================
    # Recommendation
    # ========================================================

    st.caption("Latest model prediction")

    st.markdown(
        f"""
        <div class="recommendation-title">
            {st.session_state.prediction}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        "NudgeWise selected this recommendation based "
        "on the information provided in your latest check-in."
    )

    st.markdown(
        f"""
        <div class="confidence">
            Model confidence: {confidence_display}
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.divider()


    # ========================================================
    # Explanation
    # ========================================================

    st.subheader("Why this was suggested")

    reasons = st.session_state.reasons

    if reasons:

        if isinstance(reasons, str):

            st.write(reasons)

        else:

            reason_text = " · ".join(
                str(reason)
                for reason in reasons
            )

            st.write(reason_text)

    else:

        st.caption(
            "No explanation was returned by the model."
        )


    st.divider()


    # ========================================================
    # Feedback
    # ========================================================

    st.subheader("Recommendation feedback")

    st.write(
        "Was this recommendation useful?"
    )

    feedback_col_1, feedback_col_2 = st.columns(
        2,
        gap="large",
    )

    with feedback_col_1:

        useful = st.button(
            "Yes, it was useful",
            width="stretch",
        )

    with feedback_col_2:

        not_useful = st.button(
            "Not really",
            width="stretch",
        )


    if useful:

        if st.session_state.prediction_id:

            save_feedback(
                st.session_state.prediction_id,
                1,
                1,
                5,
            )

        st.success(
            "Thanks. Your feedback has been recorded."
        )


    if not_useful:

        if st.session_state.prediction_id:

            save_feedback(
                st.session_state.prediction_id,
                0,
                0,
                1,
            )

        st.info(
            "Thanks. This helps evaluate how useful "
            "the recommendation was."
        )


    st.divider()


    # ========================================================
    # Research transparency
    # ========================================================

    with st.expander("About NudgeWise"):

        st.write(
            """
            NudgeWise is a research prototype exploring
            personalised digital wellbeing recommendations.

            Model confidence describes the confidence associated
            with the recommendation. It is not a clinical score
            and should not be interpreted as a diagnosis or
            measurement of mental health.

            Feedback can be used to evaluate how useful the
            recommendations are over time.
            """
        )