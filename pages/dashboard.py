"""
pages/dashboard.py

NudgeWise Version 2
Authenticated longitudinal wellbeing dashboard.

Features:
- participant-specific data
- latest wellbeing indicator
- latest AI recommendation
- research-quality recommendation feedback
- longitudinal trends
- recent check-in management
- edit today's check-in
- delete any check-in
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from components.navigation import (
    render_sidebar,
)

from components.cards import (
    divider,
    empty_state,
    metric_row,
    recommendation,
    section_heading,
    wellbeing_score,
)

from components.charts import (
    mood_energy_chart,
    screen_time_chart,
    wellbeing_history_chart,
)

from components.theme import (
    apply_theme,
    configure_page,
)

from database import (
    create_tables,
    delete_checkin,
    get_feedback_for_prediction,
    get_latest_prediction,
    get_recent_checkins,
    get_user,
    save_feedback,
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
# Page setup
# ============================================================

configure_page()
apply_theme()

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


# ============================================================
# Participant
# ============================================================

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

    st.session_state.pop(
        "user_id",
        None,
    )

    st.switch_page(
        "app.py"
    )


nickname = (
    participant.get(
        "nickname"
    )
    or participant.get(
        "name"
    )
    or "Participant"
)


# ============================================================
# Current date
# ============================================================

now = datetime.now(
    NZ_TIMEZONE
)

today_string = (
    now.date()
    .isoformat()
)


# ============================================================
# Data
# ============================================================

rows = get_recent_checkins(
    user_id=user_id,
    limit=60,
)

checkins = pd.DataFrame(
    rows
)


# ============================================================
# Helpers
# ============================================================

def safe_number(
    value,
    default: float = 0.0,
) -> float:

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def wellbeing_indicator(
    checkin: pd.Series,
) -> int:

    sleep = safe_number(
        checkin.get("sleep")
    )

    screen_time = safe_number(
        checkin.get("screen_time")
    )

    mood = safe_number(
        checkin.get("mood")
    )

    stress = safe_number(
        checkin.get("stress")
    )

    energy = safe_number(
        checkin.get("energy")
    )


    sleep_component = min(
        sleep / 8.0,
        1.0,
    )

    screen_component = max(
        0.0,
        1.0 - screen_time / 10.0,
    )

    mood_component = (
        mood / 5.0
    )

    stress_component = (
        1.0 - stress / 5.0
    )

    energy_component = (
        energy / 5.0
    )


    score = (
        sleep_component * 20
        + screen_component * 20
        + mood_component * 20
        + stress_component * 20
        + energy_component * 20
    )


    return int(
        max(
            0,
            min(
                100,
                round(score),
            ),
        )
    )


def wellbeing_description(
    score: int,
) -> str:

    if score >= 75:

        return (
            "Your latest check-in suggests that "
            "things are generally tracking well."
        )

    if score >= 50:

        return (
            "Your latest check-in looks fairly balanced, "
            "with a few areas worth keeping an eye on."
        )

    return (
        "Your latest check-in suggests there may be "
        "a few areas worth paying attention to."
    )


def build_history(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if dataframe.empty:
        return pd.DataFrame()

    history = dataframe.copy()

    history[
        "created_at"
    ] = pd.to_datetime(
        history["created_at"],
        errors="coerce",
    )

    if "checkin_date" in history.columns:

        parsed_dates = pd.to_datetime(
            history["checkin_date"],
            errors="coerce",
        )

        history[
            "date"
        ] = parsed_dates.dt.strftime(
            "%d %b"
        )

    else:

        history[
            "date"
        ] = history[
            "created_at"
        ].dt.strftime(
            "%d %b"
        )

    history[
        "score"
    ] = history.apply(
        wellbeing_indicator,
        axis=1,
    )

    return history


# ============================================================
# Empty dashboard
# ============================================================

if checkins.empty:

    st.caption(
        now.strftime(
            "%A, %d %B %Y"
        )
    )

    st.title(
        "Your wellbeing"
    )

    st.write(
        f"Welcome, {nickname}."
    )

    empty_state(
        "Your dashboard is ready.",
        (
            "Complete your first daily check-in "
            "to begin building your personal wellbeing history."
        ),
    )

    st.write("")

    if st.button(
        "Complete today's check-in",
        type="primary",
        width="stretch",
    ):

        st.switch_page(
            "app.py"
        )

    st.stop()


# ============================================================
# Prepare data
# ============================================================

checkins[
    "created_at"
] = pd.to_datetime(
    checkins["created_at"],
    errors="coerce",
)


if "checkin_date" not in checkins.columns:

    checkins[
        "checkin_date"
    ] = (
        checkins[
            "created_at"
        ]
        .dt.date
        .astype(str)
    )


checkins = (
    checkins
    .sort_values(
        [
            "checkin_date",
            "created_at",
        ]
    )
    .reset_index(
        drop=True
    )
)


latest = (
    checkins.iloc[-1]
)


# ============================================================
# Header
# ============================================================

latest_date = pd.to_datetime(
    latest.get(
        "checkin_date"
    ),
    errors="coerce",
)


if pd.isna(
    latest_date
):

    date_caption = (
        now.strftime(
            "%A, %d %B %Y"
        )
    )

else:

    date_caption = (
        latest_date.strftime(
            "%A, %d %B %Y"
        )
    )


st.caption(
    date_caption
)

st.title(
    "Your wellbeing"
)

st.write(
    f"A personal view of {nickname}'s recent "
    "habits, patterns and NudgeWise guidance."
)


# ============================================================
# Wellbeing indicator
# ============================================================

score = wellbeing_indicator(
    latest
)

description = wellbeing_description(
    score
)

wellbeing_score(
    score=score,
    description=description,
)


# ============================================================
# Current metrics
# ============================================================

sleep = safe_number(
    latest.get("sleep")
)

screen_time = safe_number(
    latest.get("screen_time")
)

mood = safe_number(
    latest.get("mood")
)

energy = safe_number(
    latest.get("energy")
)


metric_row(
    [
        (
            "Sleep",
            f"{sleep:.1f} h",
            "latest check-in",
        ),

        (
            "Screen time",
            f"{screen_time:.1f} h",
            "recreational",
        ),

        (
            "Mood",
            f"{mood:.0f} / 5",
            "self-reported",
        ),

        (
            "Energy",
            f"{energy:.0f} / 5",
            "self-reported",
        ),
    ]
)


divider()


# ============================================================
# Latest AI recommendation
# ============================================================

section_heading(
    "Personalised guidance"
)


prediction = get_latest_prediction(
    user_id
)


if prediction:

    confidence = safe_number(
        prediction.get(
            "confidence"
        )
    )

    confidence_display = (
        f"{confidence:.0%}"
        if confidence <= 1
        else f"{confidence:.0f}%"
    )

    recommendation(
        title=str(
            prediction.get(
                "predicted_nudge",
                "Personalised guidance",
            )
        ),
        explanation=(
            "Based on the behavioural pattern "
            "in your latest check-in."
        ),
        confidence=confidence_display,
    )

else:

    empty_state(
        "No recommendation yet.",
        (
            "Complete a daily check-in "
            "to generate personalised guidance."
        ),
    )


# ============================================================
# Recommendation feedback
# ============================================================

if prediction:

    st.write("")

    section_heading(
        "How was this recommendation?",
        (
            "Your feedback helps evaluate whether "
            "NudgeWise recommendations are useful "
            "and understandable."
        ),
    )


    prediction_id = int(
        prediction["id"]
    )


    existing_feedback = (
        get_feedback_for_prediction(
            prediction_id
        )
    )


    if existing_feedback:

        st.success(
            "You've already provided feedback "
            "for this recommendation."
        )

        summary_col_1, summary_col_2 = (
            st.columns(
                2,
                gap="large",
            )
        )

        with summary_col_1:

            st.metric(
                "Helpfulness",
                (
                    f"{existing_feedback['rating']} / 5"
                ),
            )

        with summary_col_2:

            makes_sense_value = (
                existing_feedback.get(
                    "makes_sense"
                )
            )

            if makes_sense_value is not None:

                st.metric(
                    "Made sense",
                    (
                        f"{makes_sense_value} / 5"
                    ),
                )

        completed_value = (
            existing_feedback.get(
                "completed"
            )
        )

        completed_labels = {
            0: "No",
            1: "Partly",
            2: "Yes",
        }

        if completed_value in completed_labels:

            st.caption(
                "Followed recommendation: "
                + completed_labels[
                    completed_value
                ]
            )


    else:

        with st.form(
            "recommendation_feedback"
        ):

            helpfulness = st.slider(
                "How helpful was this recommendation?",
                min_value=1,
                max_value=5,
                value=3,
            )


            makes_sense = st.slider(
                "How much did this recommendation make sense?",
                min_value=1,
                max_value=5,
                value=3,
            )


            followed = st.radio(
                "Did you follow the recommendation?",
                [
                    "No",
                    "Partly",
                    "Yes",
                ],
                horizontal=True,
            )


            comment = st.text_area(
                "Anything you would change? (optional)",
                placeholder=(
                    "For example: too generic, useful timing, "
                    "not relevant, easy to follow..."
                ),
                max_chars=500,
            )


            feedback_submit = (
                st.form_submit_button(
                    "Submit feedback",
                    type="primary",
                    width="stretch",
                )
            )


        if feedback_submit:

            completed_map = {
                "No": 0,
                "Partly": 1,
                "Yes": 2,
            }


            accepted = (
                1
                if helpfulness >= 3
                else 0
            )


            save_feedback(
                prediction_id=prediction_id,
                accepted=accepted,
                completed=(
                    completed_map[
                        followed
                    ]
                ),
                rating=int(
                    helpfulness
                ),
                makes_sense=int(
                    makes_sense
                ),
                comment=(
                    comment.strip()
                    or None
                ),
            )


            st.rerun()


divider()


# ============================================================
# Trends
# ============================================================

history = build_history(
    checkins
)


section_heading(
    "Your trends",
    (
        "Patterns become more informative "
        "as your personal history grows."
    ),
)


if not history.empty:

    wellbeing_history_chart(
        history,
        date_column="date",
        score_column="score",
    )

    mood_energy_chart(
        history,
        date_column="date",
    )

    screen_time_chart(
        history,
        date_column="date",
        screen_column="screen_time",
    )

else:

    empty_state(
        "No trend data yet.",
        (
            "Complete more daily check-ins "
            "to build your wellbeing history."
        ),
    )


divider()


# ============================================================
# Recent check-ins
# ============================================================

section_heading(
    "Recent check-ins",
    (
        "Review your recent entries. Today's entry "
        "can be edited, and any entry can be deleted."
    ),
)


recent = (
    checkins
    .sort_values(
        [
            "checkin_date",
            "created_at",
        ],
        ascending=False,
    )
    .head(7)
    .copy()
)


# ============================================================
# Check-in cards
# ============================================================

for _, row in recent.iterrows():

    checkin_id = int(
        row["id"]
    )

    checkin_date = str(
        row.get(
            "checkin_date",
            "",
        )
    )

    is_today = (
        checkin_date
        == today_string
    )


    parsed_date = pd.to_datetime(
        checkin_date,
        errors="coerce",
    )


    if pd.isna(
        parsed_date
    ):

        date_display = (
            checkin_date
            or "Recorded check-in"
        )

    else:

        date_display = (
            parsed_date.strftime(
                "%A, %d %B"
            )
        )


    with st.container(
        border=True
    ):

        heading_col, action_col = (
            st.columns(
                [3.2, 1],
                gap="large",
            )
        )


        with heading_col:

            st.markdown(
                f"### {date_display}"
            )

            st.caption(
                (
                    f"Sleep {safe_number(row.get('sleep')):.1f} h"
                    f"  ·  Screen {safe_number(row.get('screen_time')):.1f} h"
                    f"  ·  Mood {safe_number(row.get('mood')):.0f}/5"
                    f"  ·  Stress {safe_number(row.get('stress')):.0f}/5"
                    f"  ·  Energy {safe_number(row.get('energy')):.0f}/5"
                )
            )


            activity = str(
                row.get(
                    "activity",
                    "",
                )
            )

            social = str(
                row.get(
                    "social",
                    "",
                )
            )


            st.caption(
                f"{activity} · Social interaction: {social}"
            )


        # ----------------------------------------------------
        # Edit
        # ----------------------------------------------------

        with action_col:

            if is_today:

                if st.button(
                    "Edit",
                    key=f"edit_{checkin_id}",
                    width="stretch",
                ):

                    st.switch_page(
                        "app.py"
                    )

            else:

                st.caption(
                    "Past entry"
                )


        # ----------------------------------------------------
        # Delete
        # ----------------------------------------------------

        delete_state_key = (
            f"confirm_delete_{checkin_id}"
        )


        if not st.session_state.get(
            delete_state_key,
            False,
        ):

            if st.button(
                "Delete log",
                key=f"delete_{checkin_id}",
                type="secondary",
            ):

                st.session_state[
                    delete_state_key
                ] = True

                st.rerun()


        else:

            st.warning(
                "Delete this check-in permanently? "
                "Its recommendation and feedback "
                "will also be removed."
            )


            confirm_col, cancel_col = (
                st.columns(
                    2,
                    gap="medium",
                )
            )


            with confirm_col:

                if st.button(
                    "Yes, delete",
                    key=f"confirm_{checkin_id}",
                    type="primary",
                    width="stretch",
                ):

                    deleted = (
                        delete_checkin(
                            checkin_id=checkin_id,
                            user_id=user_id,
                        )
                    )


                    st.session_state.pop(
                        delete_state_key,
                        None,
                    )


                    if deleted:

                        st.session_state.prediction = None
                        st.session_state.confidence = None
                        st.session_state.prediction_id = None
                        st.session_state.reasons = []

                        st.rerun()

                    else:

                        st.error(
                            "NudgeWise could not delete "
                            "that check-in."
                        )


            with cancel_col:

                if st.button(
                    "Cancel",
                    key=f"cancel_{checkin_id}",
                    width="stretch",
                ):

                    st.session_state[
                        delete_state_key
                    ] = False

                    st.rerun()


divider()


# ============================================================
# Today's check-in status
# ============================================================

today_exists = (
    today_string
    in set(
        checkins[
            "checkin_date"
        ].astype(str)
    )
)


section_heading(
    "Today's check-in"
)


if today_exists:

    st.write(
        "You've already completed today's check-in. "
        "You can update it if something changes "
        "or if you entered something incorrectly."
    )

    if st.button(
        "Edit today's check-in",
        type="primary",
        width="stretch",
        key="edit_today_bottom",
    ):

        st.switch_page(
            "app.py"
        )


else:

    st.write(
        "You haven't completed today's check-in yet."
    )

    if st.button(
        "Complete today's check-in",
        type="primary",
        width="stretch",
        key="complete_today_bottom",
    ):

        st.switch_page(
            "app.py"
        )


# ============================================================
# Research transparency
# ============================================================

divider()


with st.expander(
    "About this dashboard"
):

    st.write(
        """
        NudgeWise is a research prototype exploring
        personalised digital wellbeing recommendations.

        The wellbeing indicator is a product-level research
        measure derived from self-reported check-in data.
        It is not a medical or clinical assessment.

        Recommendation probability represents the model's
        estimated preference among available NudgeWise
        interventions. It is not model accuracy and does not
        establish that an intervention will be effective.

        Recommendation feedback is collected to evaluate
        perceived usefulness, relevance, and whether users
        followed the suggested behaviour.

        Editing today's check-in regenerates the recommendation
        using the updated information.

        Deleting a check-in also removes its associated AI
        prediction and recommendation feedback.
        """
    )