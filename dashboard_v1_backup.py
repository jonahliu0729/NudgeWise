"""
SmartScreen Version 2
Personal wellbeing dashboard.

Presentation layer only:
- Reads real check-in data from SQLite
- Reads AI predictions linked to check-ins
- Presents the recommendation clearly
- Avoids clinical claims
- Handles insufficient longitudinal data gracefully
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Configuration
# ============================================================

DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "database"
    / "nudge.db"
)

# Temporary development user.
# This will eventually come from authentication/session state.
USER_ID = 3


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="SmartScreen",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Global
    ------------------------------------------------------- */

    .stApp {
        background: #f7f7f5;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 3.5rem;
        padding-bottom: 5rem;
    }

    h1, h2, h3, p {
        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "SF Pro Display",
            "SF Pro Text",
            "Inter",
            sans-serif;
    }

    /* -------------------------------------------------------
       Header
    ------------------------------------------------------- */

    .page-title {
        font-size: 2.45rem;
        font-weight: 650;
        letter-spacing: -0.045em;
        line-height: 1.05;
        color: #171717;
        margin-bottom: 0.45rem;
    }

    .page-subtitle {
        font-size: 1rem;
        line-height: 1.55;
        color: #777771;
        margin-bottom: 2.8rem;
    }

    .date-label {
        font-size: 0.82rem;
        color: #8a8a84;
        margin-bottom: 0.55rem;
    }

    /* -------------------------------------------------------
       Main wellbeing panel
    ------------------------------------------------------- */

    .wellbeing-panel {
        background: #ffffff;
        border: 1px solid #e8e8e3;
        border-radius: 24px;
        padding: 2.2rem 2.4rem;
        min-height: 270px;
    }

    .panel-eyebrow {
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #999992;
        margin-bottom: 1.2rem;
    }

    .score-number {
        font-size: 5.2rem;
        line-height: 0.95;
        font-weight: 650;
        letter-spacing: -0.075em;
        color: #171717;
    }

    .score-denominator {
        font-size: 1rem;
        color: #9a9a94;
        margin-left: 0.25rem;
    }

    .score-description {
        margin-top: 1.1rem;
        font-size: 0.96rem;
        line-height: 1.55;
        color: #70706a;
        max-width: 420px;
    }

    /* -------------------------------------------------------
       Metrics
    ------------------------------------------------------- */

    .metric-panel {
        background: #ffffff;
        border: 1px solid #e8e8e3;
        border-radius: 20px;
        padding: 1.5rem 1.55rem;
        min-height: 125px;
    }

    .metric-label {
        font-size: 0.78rem;
        color: #85857f;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        font-size: 1.75rem;
        font-weight: 620;
        letter-spacing: -0.035em;
        color: #20201f;
    }

    .metric-note {
        margin-top: 0.25rem;
        font-size: 0.78rem;
        color: #a0a09a;
    }

    /* -------------------------------------------------------
       Recommendation
    ------------------------------------------------------- */

    .recommendation {
        background: #ffffff;
        border: 1px solid #e8e8e3;
        border-radius: 24px;
        padding: 2.25rem 2.4rem;
        margin-top: 1.5rem;
    }

    .recommendation-title {
        font-size: 2rem;
        font-weight: 630;
        letter-spacing: -0.045em;
        color: #171717;
        margin-bottom: 0.8rem;
    }

    .recommendation-description {
        font-size: 1rem;
        line-height: 1.65;
        color: #666660;
        max-width: 680px;
    }

    .confidence {
        margin-top: 1.5rem;
        font-size: 0.82rem;
        color: #888882;
    }

    .confidence strong {
        color: #444440;
        font-weight: 600;
    }

    /* -------------------------------------------------------
       Explanation
    ------------------------------------------------------- */

    .explanation-panel {
        background: #f0f0ec;
        border-radius: 18px;
        padding: 1.25rem 1.45rem;
        margin-top: 1.2rem;
    }

    .explanation-title {
        font-size: 0.82rem;
        font-weight: 620;
        color: #4d4d48;
        margin-bottom: 0.5rem;
    }

    .explanation-text {
        font-size: 0.9rem;
        line-height: 1.6;
        color: #6d6d67;
    }

    /* -------------------------------------------------------
       Trend panel
    ------------------------------------------------------- */

    .trend-panel {
        background: #ffffff;
        border: 1px solid #e8e8e3;
        border-radius: 24px;
        padding: 1.7rem 1.8rem 1.1rem;
        margin-top: 1.5rem;
    }

    .trend-title {
        font-size: 1.25rem;
        font-weight: 620;
        letter-spacing: -0.025em;
        color: #20201f;
        margin-bottom: 0.3rem;
    }

    .trend-description {
        font-size: 0.86rem;
        color: #85857f;
        margin-bottom: 1rem;
    }

    /* -------------------------------------------------------
       Empty state
    ------------------------------------------------------- */

    .empty-state {
        background: #ffffff;
        border: 1px solid #e8e8e3;
        border-radius: 24px;
        padding: 3rem;
        text-align: center;
    }

    .empty-title {
        font-size: 1.35rem;
        font-weight: 620;
        color: #20201f;
        margin-bottom: 0.5rem;
    }

    .empty-description {
        max-width: 520px;
        margin: auto;
        font-size: 0.92rem;
        line-height: 1.6;
        color: #81817b;
    }

    /* -------------------------------------------------------
       Check-in table
    ------------------------------------------------------- */

    .section-title {
        margin-top: 2.8rem;
        margin-bottom: 0.25rem;
        font-size: 1.25rem;
        font-weight: 620;
        letter-spacing: -0.025em;
        color: #20201f;
    }

    .section-description {
        font-size: 0.86rem;
        color: #85857f;
        margin-bottom: 1rem;
    }

    /* -------------------------------------------------------
       Streamlit cleanup
    ------------------------------------------------------- */

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Database
# ============================================================

@st.cache_data(ttl=30)
def load_checkins(user_id: int) -> pd.DataFrame:
    """Load check-ins belonging to the selected user."""

    connection = sqlite3.connect(DATABASE_PATH)

    query = """
        SELECT
            id,
            user_id,
            sleep,
            stress,
            mood,
            energy,
            screen_time,
            activity,
            social,
            hour,
            created_at
        FROM checkins
        WHERE user_id = ?
        ORDER BY datetime(created_at) ASC
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
        params=(user_id,),
    )

    connection.close()

    if not dataframe.empty:
        dataframe["created_at"] = pd.to_datetime(
            dataframe["created_at"],
            errors="coerce",
        )

    return dataframe


@st.cache_data(ttl=30)
def load_predictions(user_id: int) -> pd.DataFrame:
    """
    Load predictions through the check-in relationship.

    Predictions use checkin_id rather than user_id.
    """

    connection = sqlite3.connect(DATABASE_PATH)

    query = """
        SELECT
            p.id,
            p.checkin_id,
            p.predicted_nudge,
            p.confidence,
            p.created_at
        FROM predictions AS p
        INNER JOIN checkins AS c
            ON p.checkin_id = c.id
        WHERE c.user_id = ?
        ORDER BY datetime(p.created_at) ASC
    """

    dataframe = pd.read_sql_query(
        query,
        connection,
        params=(user_id,),
    )

    connection.close()

    if not dataframe.empty:
        dataframe["created_at"] = pd.to_datetime(
            dataframe["created_at"],
            errors="coerce",
        )

    return dataframe


# ============================================================
# Wellbeing calculations
# ============================================================

def calculate_wellbeing_indicator(
    checkin: pd.Series,
) -> int:
    """
    Calculate a product-level wellbeing indicator.

    This is NOT a validated clinical score.
    It is intentionally labelled as an indicator in the UI.
    """

    sleep = float(checkin["sleep"])
    stress = float(checkin["stress"])
    mood = float(checkin["mood"])
    energy = float(checkin["energy"])
    screen_time = float(checkin["screen_time"])

    sleep_component = min(sleep / 8.0, 1.0) * 25
    stress_component = max(0.0, 1.0 - stress / 5.0) * 20
    mood_component = (mood / 5.0) * 20
    energy_component = (energy / 5.0) * 20
    screen_component = max(
        0.0,
        1.0 - screen_time / 10.0,
    ) * 15

    score = (
        sleep_component
        + stress_component
        + mood_component
        + energy_component
        + screen_component
    )

    return max(0, min(100, round(score)))


def wellbeing_description(score: int) -> str:
    """Provide neutral language for the indicator."""

    if score >= 75:
        return (
            "Your current check-in suggests a relatively positive "
            "wellbeing state."
        )

    if score >= 55:
        return (
            "Your current check-in suggests a fairly balanced "
            "wellbeing state."
        )

    return (
        "Your current check-in suggests there may be a few areas "
        "worth paying attention to."
    )


# ============================================================
# Recommendation explanation
# ============================================================

def build_explanation(
    checkin: pd.Series,
    recommendation: str,
) -> str:
    """
    Produce a concise natural-language explanation.

    This is presentation logic, not the prediction model itself.
    """

    signals: list[str] = []

    if float(checkin["energy"]) <= 2:
        signals.append("lower energy")

    if float(checkin["mood"]) <= 2:
        signals.append("lower mood")

    if float(checkin["stress"]) >= 4:
        signals.append("higher stress")

    if float(checkin["screen_time"]) >= 6:
        signals.append("higher recreational screen time")

    if float(checkin["sleep"]) < 7:
        signals.append("less sleep")

    if int(checkin["hour"]) >= 21:
        signals.append("the late time of day")

    if not signals:
        return (
            f"The model selected {recommendation.lower()} based on "
            "the overall pattern in your latest check-in."
        )

    if len(signals) == 1:
        signal_text = signals[0]

    elif len(signals) == 2:
        signal_text = f"{signals[0]} and {signals[1]}"

    else:
        signal_text = (
            ", ".join(signals[:-1])
            + f", and {signals[-1]}"
        )

    return (
        f"The model selected {recommendation.lower()} after considering "
        f"{signal_text} in your latest check-in."
    )


# ============================================================
# Formatting helpers
# ============================================================

def format_hours(value: float) -> str:
    """Format a number of hours."""

    if value.is_integer():
        return f"{int(value)} h"

    return f"{value:.1f} h"


def format_confidence(value: float) -> str:
    """
    Convert model confidence to a percentage.

    Supports both:
    - 0.43
    - 43
    """

    confidence = float(value)

    if confidence <= 1:
        confidence *= 100

    return f"{round(confidence)}%"


# ============================================================
# Load data
# ============================================================

checkins = load_checkins(USER_ID)
predictions = load_predictions(USER_ID)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="date-label">Monday, 10 August</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="page-title">Good evening</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-subtitle">
        Here's a snapshot of how you're doing today.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Empty state
# ============================================================

if checkins.empty:

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-title">
                Start your first check-in
            </div>

            <div class="empty-description">
                A check-in gives SmartScreen the information it needs
                to understand your current wellbeing and provide
                personalised guidance.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# ============================================================
# Latest check-in
# ============================================================

latest = checkins.iloc[-1]

wellbeing_score = calculate_wellbeing_indicator(latest)
description = wellbeing_description(wellbeing_score)


# ============================================================
# Wellbeing snapshot
# ============================================================

left, right = st.columns(
    [1.35, 1],
    gap="large",
)

with left:

    st.markdown(
        f"""
        <div class="wellbeing-panel">

            <div class="panel-eyebrow">
                Wellbeing indicator
            </div>

            <div>
                <span class="score-number">
                    {wellbeing_score}
                </span>

                <span class="score-denominator">
                    / 100
                </span>
            </div>

            <div class="score-description">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with right:

    metric_1, metric_2 = st.columns(2)

    with metric_1:

        st.markdown(
            f"""
            <div class="metric-panel">

                <div class="metric-label">
                    Sleep
                </div>

                <div class="metric-value">
                    {format_hours(float(latest["sleep"]))}
                </div>

                <div class="metric-note">
                    Last check-in
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_2:

        st.markdown(
            f"""
            <div class="metric-panel">

                <div class="metric-label">
                    Screen time
                </div>

                <div class="metric-value">
                    {format_hours(float(latest["screen_time"]))}
                </div>

                <div class="metric-note">
                    Recreational
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    metric_3, metric_4 = st.columns(2)

    with metric_3:

        st.markdown(
            f"""
            <div class="metric-panel">

                <div class="metric-label">
                    Mood
                </div>

                <div class="metric-value">
                    {int(latest["mood"])} / 5
                </div>

                <div class="metric-note">
                    Self-reported
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric_4:

        st.markdown(
            f"""
            <div class="metric-panel">

                <div class="metric-label">
                    Energy
                </div>

                <div class="metric-value">
                    {int(latest["energy"])} / 5
                </div>

                <div class="metric-note">
                    Self-reported
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Recommendation
# ============================================================

if not predictions.empty:

    latest_prediction = predictions.iloc[-1]

    recommendation = str(
        latest_prediction["predicted_nudge"]
    )

    confidence = format_confidence(
        latest_prediction["confidence"]
    )

    explanation = build_explanation(
        latest,
        recommendation,
    )

    st.markdown(
        f"""
        <div class="recommendation">

            <div class="panel-eyebrow">
                A suggestion for this evening
            </div>

            <div class="recommendation-title">
                {recommendation}
            </div>

            <div class="recommendation-description">
                {explanation}
            </div>

            <div class="confidence">
                Model confidence
                <strong>{confidence}</strong>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why this suggestion?"):

        st.markdown(
            f"""
            <div class="explanation-panel">

                <div class="explanation-title">
                    Signals considered
                </div>

                <div class="explanation-text">
                    The recommendation was generated from your
                    latest check-in, including sleep, stress, mood,
                    energy, recreational screen time, activity,
                    social interaction and time of day.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Progress / trends
# ============================================================

st.markdown(
    '<div class="section-title">Your progress</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Your wellbeing history will become more useful as you complete
        more check-ins.
    </div>
    """,
    unsafe_allow_html=True,
)


if len(checkins) < 5:

    remaining = 5 - len(checkins)

    st.markdown(
        f"""
        <div class="empty-state">

            <div class="empty-title">
                {remaining} more check-in{"s" if remaining != 1 else ""}
                to unlock your trend
            </div>

            <div class="empty-description">
                SmartScreen waits until there is enough personal
                history to show a meaningful trend rather than
                presenting an unreliable chart.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    trend_data = checkins.copy()

    trend_data["wellbeing"] = trend_data.apply(
        calculate_wellbeing_indicator,
        axis=1,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=trend_data["created_at"],
            y=trend_data["wellbeing"],
            mode="lines+markers",
            line={
                "color": "#20201f",
                "width": 2.5,
            },
            marker={
                "size": 7,
                "color": "#20201f",
            },
            hovertemplate=(
                "%{x|%d %b}<br>"
                "Indicator: %{y}/100"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        height=300,
        margin={
            "l": 10,
            "r": 10,
            "t": 10,
            "b": 10,
        },
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={
            "family": (
                "-apple-system, BlinkMacSystemFont, "
                "'SF Pro Text', Inter, sans-serif"
            ),
            "color": "#777771",
        },
        xaxis={
            "showgrid": False,
            "showline": False,
            "zeroline": False,
        },
        yaxis={
            "range": [0, 100],
            "showgrid": True,
            "gridcolor": "#eeeeea",
            "showline": False,
            "zeroline": False,
            "title": None,
        },
        showlegend=False,
    )

    st.markdown(
        '<div class="trend-panel">',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="trend-title">Wellbeing over time</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="trend-description">
            Your personal wellbeing indicator across recent check-ins.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": False,
        },
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# Recent check-ins
# ============================================================

st.markdown(
    '<div class="section-title">Recent check-ins</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Your latest recorded wellbeing information.
    </div>
    """,
    unsafe_allow_html=True,
)

recent = checkins.tail(5).copy()

recent["created_at"] = recent["created_at"].dt.strftime(
    "%d %b %Y"
)

recent["sleep"] = recent["sleep"].map(
    lambda value: f"{value:.1f} h"
)

recent["screen_time"] = recent["screen_time"].map(
    lambda value: f"{value:.1f} h"
)

recent = recent[
    [
        "created_at",
        "sleep",
        "screen_time",
        "mood",
        "stress",
        "energy",
        "activity",
    ]
]

recent.columns = [
    "Date",
    "Sleep",
    "Screen time",
    "Mood",
    "Stress",
    "Energy",
    "Activity",
]

st.dataframe(
    recent,
    width="stretch",
    hide_index=True,
)


# ============================================================
# Transparency
# ============================================================

with st.expander("About the wellbeing indicator"):

    st.markdown(
        """
        The wellbeing indicator is a product-level measure derived
        from the information provided during your check-in. It is
        not a medical or clinical assessment.

        Model confidence represents the probability associated with
        the model's selected recommendation. A higher confidence
        does not mean the recommendation is guaranteed to be useful.

        SmartScreen is designed to provide supportive wellbeing
        guidance rather than diagnose or treat health conditions.
        """
    )