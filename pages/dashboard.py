"""
dashboard.py

NudgeWise Version 2
Personal longitudinal wellbeing dashboard.
"""

from __future__ import annotations

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
    get_latest_prediction,
    get_recent_checkins,
    get_user,
)


# ============================================================
# Setup
# ============================================================

configure_page()
apply_theme()

create_tables()

render_sidebar()


# ============================================================
# Current participant
# ============================================================

user_id = st.session_state.get(
    "user_id"
)

if user_id is None:

    st.warning(
        "Start or resume a participant "
        "session before opening the dashboard."
    )

    if st.button(
        "Go to NudgeWise",
        type="primary",
    ):
        st.switch_page(
            "app.py"
        )

    st.stop()


participant = get_user(
    user_id
)

if participant is None:

    st.session_state.user_id = None

    st.switch_page(
        "app.py"
    )


nickname = (
    participant.get("nickname")
    or participant.get("name")
    or "Participant"
)


# ============================================================
# Data
# ============================================================

rows = get_recent_checkins(
    user_id=user_id,
    limit=30,
)

checkins = pd.DataFrame(
    rows
)


# ============================================================
# Empty state
# ============================================================

if checkins.empty:

    st.title(
        "Your wellbeing"
    )

    empty_state(
        "Your dashboard is ready.",
        (
            "Complete your first check-in "
            "to begin building your personal "
            "wellbeing history."
        ),
    )

    st.stop()


checkins["created_at"] = pd.to_datetime(
    checkins["created_at"],
    errors="coerce",
)

checkins = (
    checkins
    .sort_values(
        "created_at"
    )
    .reset_index(
        drop=True
    )
)

latest = checkins.iloc[-1]


# ============================================================
# Utilities
# ============================================================

def safe_number(
    value,
    default: float = 0,
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
    """
    Product-level research indicator.

    Not a clinical wellbeing score.
    """

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

    mood_component = mood / 5.0

    stress_component = (
        1.0 - stress / 5.0
    )

    energy_component = energy / 5.0

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
            "Your latest check-in suggests "
            "that things are generally tracking well."
        )

    if score >= 50:

        return (
            "Your latest check-in looks fairly "
            "balanced, with a few areas worth "
            "keeping an eye on."
        )

    return (
        "Your latest check-in suggests there "
        "may be a few areas worth paying attention to."
    )


def build_history(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    history = dataframe.copy()

    history["date"] = (
        history[
            "created_at"
        ]
        .dt.strftime(
            "%d %b"
        )
    )

    history["score"] = (
        history.apply(
            wellbeing_indicator,
            axis=1,
        )
    )

    return history


# ============================================================
# Header
# ============================================================

latest_date = latest[
    "created_at"
]

st.caption(
    latest_date.strftime(
        "%A, %d %B %Y"
    )
)

st.title(
    "Your wellbeing"
)

st.write(
    f"A personal view of {nickname}'s recent "
    "habits, patterns and NudgeWise guidance."
)


# ============================================================
# Indicator
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
# Recommendation
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
            "Complete another check-in "
            "to generate personalised guidance."
        ),
    )


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


divider()


# ============================================================
# Recent check-ins
# ============================================================

section_heading(
    "Recent check-ins",
    "Your latest recorded wellbeing information.",
)

recent = (
    checkins
    .tail(7)
    .copy()
)


display_columns = [
    column
    for column in [
        "created_at",
        "sleep",
        "screen_time",
        "mood",
        "stress",
        "energy",
        "activity",
    ]
    if column
    in recent.columns
]


recent = recent[
    display_columns
].copy()


if "created_at" in recent.columns:

    recent[
        "created_at"
    ] = (
        recent[
            "created_at"
        ]
        .dt.strftime(
            "%d %b %Y"
        )
    )


if "sleep" in recent.columns:

    recent[
        "sleep"
    ] = recent[
        "sleep"
    ].map(
        lambda value:
            f"{safe_number(value):.1f} h"
    )


if "screen_time" in recent.columns:

    recent[
        "screen_time"
    ] = recent[
        "screen_time"
    ].map(
        lambda value:
            f"{safe_number(value):.1f} h"
    )


recent = recent.rename(
    columns={
        "created_at": "Date",
        "sleep": "Sleep",
        "screen_time": "Screen time",
        "mood": "Mood",
        "stress": "Stress",
        "energy": "Energy",
        "activity": "Activity",
    }
)


st.dataframe(
    recent,
    width="stretch",
    hide_index=True,
)


# ============================================================
# Transparency
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
        estimated preference among the available NudgeWise
        interventions. It is not model accuracy and does not
        establish that an intervention will be effective.
        """
    )