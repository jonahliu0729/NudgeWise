"""
pages/dashboard.py

NudgeWise Version 2.8
Authenticated personal wellbeing dashboard.

Features:
- participant-specific longitudinal data
- NudgeWise wellbeing indicator
- stored AI prediction snapshots
- full six-class probabilities
- model uncertainty
- contextual actions
- recommendation feedback
- longitudinal trends
- collapsible recent history
- retrospective entries
- check-in editing and deletion

IMPORTANT
---------
Research-grade prediction records display the prediction snapshot
stored when the check-in was processed.

Historical recommendations are not silently replaced by newer
model outputs.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from components.theme import (
    apply_theme,
    configure_page,
)


# ============================================================
# Page configuration
# ============================================================

configure_page()
apply_theme()


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

from database import (
    create_tables,
    delete_checkin,
    get_feedback_for_prediction,
    get_prediction_for_checkin,
    get_recent_checkins,
    get_user,
    save_feedback,
)

from predict import (
    get_prediction_details,
)

from services.auth import (
    ensure_current_participant,
    is_logged_in,
)

from services.recommendations import (
    personalise_recommendation,
)


# ============================================================
# Configuration
# ============================================================

NZ_TIMEZONE = ZoneInfo(
    "Pacific/Auckland"
)


PROBABILITY_FIELDS = {
    "Connect socially":
        "p_connect_socially",

    "Maintain habits":
        "p_maintain_habits",

    "Prepare for bed":
        "p_prepare_for_bed",

    "Reduce screen time":
        "p_reduce_screen_time",

    "Stay active":
        "p_stay_active",

    "Take a short break":
        "p_take_a_short_break",
}


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
    now.date().isoformat()
)


# ============================================================
# Load data
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

        if pd.isna(
            value
        ):

            return default


        return float(
            value
        )


    except (
        TypeError,
        ValueError,
    ):

        return default


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:

    return max(
        minimum,
        min(
            maximum,
            value,
        ),
    )


def get_day_type_for_date(
    date_value,
) -> str:

    parsed = pd.to_datetime(
        date_value,
        errors="coerce",
    )


    if pd.isna(
        parsed
    ):

        return (
            "Weekend"
            if now.weekday() >= 5
            else "Weekday"
        )


    return (
        "Weekend"
        if parsed.weekday() >= 5
        else "Weekday"
    )


def stored_probability_distribution(
    prediction_record: dict | None,
) -> dict[str, float]:
    """
    Read the stored six-class distribution.

    Returns an empty dictionary for legacy records.
    """

    if not prediction_record:

        return {}


    probabilities = {}


    for (
        class_name,
        field_name,
    ) in PROBABILITY_FIELDS.items():

        value = prediction_record.get(
            field_name
        )


        if value is None:

            return {}


        try:

            probability = float(
                value
            )


        except (
            TypeError,
            ValueError,
        ):

            return {}


        probabilities[
            class_name
        ] = clamp(
            probability
        )


    total = sum(
        probabilities.values()
    )


    if total <= 0:

        return {}


    return {
        class_name:
            probability / total

        for (
            class_name,
            probability,
        )
        in probabilities.items()
    }


def has_research_snapshot(
    prediction_record: dict | None,
) -> bool:

    if not prediction_record:

        return False


    probabilities = (
        stored_probability_distribution(
            prediction_record
        )
    )


    return bool(
        probabilities
        and prediction_record.get(
            "action_id"
        )
        and prediction_record.get(
            "action_title"
        )
        and prediction_record.get(
            "action_text"
        )
    )


# ============================================================
# Wellbeing indicator
# ============================================================

def wellbeing_indicator(
    checkin: pd.Series,
) -> int:
    """
    Product-level NudgeWise wellbeing indicator.

    NOT a medical or clinical assessment.
    """

    sleep = safe_number(
        checkin.get(
            "sleep"
        )
    )


    stress = safe_number(
        checkin.get(
            "stress"
        ),
        3.0,
    )


    mood = safe_number(
        checkin.get(
            "mood"
        ),
        3.0,
    )


    energy = safe_number(
        checkin.get(
            "energy"
        ),
        3.0,
    )


    activity_minutes = safe_number(
        checkin.get(
            "activity_minutes"
        )
    )


    connectedness = safe_number(
        checkin.get(
            "connectedness"
        ),
        3.0,
    )


    sleep_component = clamp(
        sleep / 8.0
    )


    stress_component = clamp(
        (
            5.0
            - stress
        )
        / 4.0
    )


    mood_component = clamp(
        (
            mood
            - 1.0
        )
        / 4.0
    )


    energy_component = clamp(
        (
            energy
            - 1.0
        )
        / 4.0
    )


    activity_component = clamp(
        activity_minutes
        / 60.0
    )


    connection_component = clamp(
        (
            connectedness
            - 1.0
        )
        / 4.0
    )


    components = [
        sleep_component,
        stress_component,
        mood_component,
        energy_component,
        activity_component,
        connection_component,
    ]


    score = (
        sum(
            components
        )
        / len(
            components
        )
        * 100.0
    )


    return int(
        round(
            clamp(
                score,
                0.0,
                100.0,
            )
        )
    )


def wellbeing_description(
    score: int,
) -> str:

    if score >= 75:

        return (
            "Your latest self-reported check-in is generally "
            "tracking well across the measures included in "
            "the NudgeWise wellbeing indicator."
        )


    if score >= 50:

        return (
            "Your latest check-in looks mixed overall, with "
            "some measures tracking well and others potentially "
            "worth paying attention to."
        )


    return (
        "Several measures in your latest check-in are currently "
        "below the stronger end of your NudgeWise wellbeing indicator."
    )


# ============================================================
# History preparation
# ============================================================

def build_history(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    if dataframe.empty:

        return pd.DataFrame()


    history = dataframe.copy()


    history[
        "created_at"
    ] = pd.to_datetime(
        history[
            "created_at"
        ],
        errors="coerce",
    )


    parsed_dates = pd.to_datetime(
        history[
            "checkin_date"
        ],
        errors="coerce",
    )


    history[
        "date"
    ] = parsed_dates.dt.strftime(
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


    st.write("")


    if st.button(
        "Add a missed check-in",
        width="stretch",
        key="empty_add_missed",
    ):

        st.switch_page(
            "pages/past_checkin.py"
        )


    st.stop()


# ============================================================
# Prepare data
# ============================================================

checkins[
    "created_at"
] = pd.to_datetime(
    checkins[
        "created_at"
    ],
    errors="coerce",
)


if "entry_type" not in checkins.columns:

    checkins[
        "entry_type"
    ] = "live"


if "activity_minutes" not in checkins.columns:

    checkins[
        "activity_minutes"
    ] = 0


if "connectedness" not in checkins.columns:

    checkins[
        "connectedness"
    ] = 3


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
    checkins.iloc[
        -1
    ]
)


latest_checkin_id = int(
    latest[
        "id"
    ]
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


wellbeing_score(
    score=score,
    description=(
        wellbeing_description(
            score
        )
    ),
)


st.caption(
    "Research indicator only · not a medical or clinical score"
)


# ============================================================
# Metrics
# ============================================================

metric_row(
    [
        (
            "Sleep",
            f"{safe_number(latest.get('sleep')):.1f} h",
            "last main sleep period",
        ),

        (
            "Physical activity",
            f"{safe_number(latest.get('activity_minutes')):.0f} min",
            "reported for this day",
        ),

        (
            "Connectedness",
            f"{safe_number(latest.get('connectedness'), 3):.0f} / 5",
            "self-reported",
        ),

        (
            "Screen time",
            f"{safe_number(latest.get('screen_time')):.1f} h",
            "recreational",
        ),
    ]
)


st.write("")


metric_row(
    [
        (
            "Mood",
            f"{safe_number(latest.get('mood')):.0f} / 5",
            "higher = better",
        ),

        (
            "Stress",
            f"{safe_number(latest.get('stress')):.0f} / 5",
            "higher = more stress",
        ),

        (
            "Energy",
            f"{safe_number(latest.get('energy')):.0f} / 5",
            "higher = more energy",
        ),

        (
            "Context",
            str(
                latest.get(
                    "activity",
                    "—",
                )
            ),
            "at check-in",
        ),
    ]
)


divider()


# ============================================================
# Stored prediction
# ============================================================

section_heading(
    "Personalised guidance"
)


prediction_record = (
    get_prediction_for_checkin(
        latest_checkin_id
    )
)


latest_day_type = (
    get_day_type_for_date(
        latest.get(
            "checkin_date"
        )
    )
)


# ============================================================
# Current explainability calculation
# ============================================================

explanation_details = get_prediction_details(
    sleep=safe_number(
        latest.get(
            "sleep"
        )
    ),

    stress=int(
        safe_number(
            latest.get(
                "stress"
            ),
            3,
        )
    ),

    mood=int(
        safe_number(
            latest.get(
                "mood"
            ),
            3,
        )
    ),

    energy=int(
        safe_number(
            latest.get(
                "energy"
            ),
            3,
        )
    ),

    screen_time=safe_number(
        latest.get(
            "screen_time"
        )
    ),

    activity_minutes=int(
        safe_number(
            latest.get(
                "activity_minutes"
            ),
            0,
        )
    ),

    connectedness=int(
        safe_number(
            latest.get(
                "connectedness"
            ),
            3,
        )
    ),

    activity=str(
        latest.get(
            "activity",
            "Relaxing",
        )
    ),

    day_type=latest_day_type,

    hour=int(
        safe_number(
            latest.get(
                "hour"
            ),
            now.hour,
        )
    ),

    include_explanation=True,
)


# ============================================================
# Stored v2.8 snapshot
# ============================================================

snapshot_available = (
    has_research_snapshot(
        prediction_record
    )
)


if snapshot_available:

    displayed_prediction = str(
        prediction_record[
            "predicted_nudge"
        ]
    )


    displayed_confidence = safe_number(
        prediction_record.get(
            "confidence"
        )
    )


    displayed_probabilities = (
        stored_probability_distribution(
            prediction_record
        )
    )


    ordered_probabilities = sorted(
        displayed_probabilities.items(),
        key=lambda item:
            item[
                1
            ],
        reverse=True,
    )


    displayed_certainty = (
        prediction_record.get(
            "certainty"
        )
        or "Unknown"
    )


    displayed_margin = safe_number(
        prediction_record.get(
            "probability_margin"
        )
    )


    displayed_entropy = safe_number(
        prediction_record.get(
            "entropy"
        )
    )


    displayed_normalised_entropy = safe_number(
        prediction_record.get(
            "normalised_entropy"
        )
    )


    displayed_action_id = str(
        prediction_record.get(
            "action_id"
        )
        or ""
    )


    displayed_action_title = str(
        prediction_record.get(
            "action_title"
        )
        or displayed_prediction
    )


    displayed_action_text = str(
        prediction_record.get(
            "action_text"
        )
        or ""
    )


    displayed_action_reason = str(
        prediction_record.get(
            "action_reason"
        )
        or (
            "This action was selected from the context "
            "available when the recommendation was generated."
        )
    )


    displayed_model_version = (
        prediction_record.get(
            "model_version"
        )
        or "Unknown"
    )


    displayed_action_engine_version = (
        prediction_record.get(
            "action_engine_version"
        )
        or "Unknown"
    )


# ============================================================
# Legacy fallback
# ============================================================

else:

    displayed_prediction = (
        explanation_details[
            "prediction"
        ]
    )


    displayed_confidence = float(
        explanation_details[
            "confidence"
        ]
    )


    displayed_probabilities = (
        explanation_details[
            "probabilities"
        ]
    )


    ordered_probabilities = (
        explanation_details[
            "ordered_probabilities"
        ]
    )


    displayed_certainty = (
        explanation_details[
            "certainty"
        ]
    )


    displayed_margin = safe_number(
        explanation_details.get(
            "probability_margin"
        )
    )


    displayed_entropy = safe_number(
        explanation_details.get(
            "entropy"
        )
    )


    displayed_normalised_entropy = safe_number(
        explanation_details.get(
            "normalised_entropy"
        )
    )


    legacy_guidance = (
        personalise_recommendation(
            prediction=displayed_prediction,

            sleep=safe_number(
                latest.get(
                    "sleep"
                )
            ),

            stress=int(
                safe_number(
                    latest.get(
                        "stress"
                    ),
                    3,
                )
            ),

            mood=int(
                safe_number(
                    latest.get(
                        "mood"
                    ),
                    3,
                )
            ),

            energy=int(
                safe_number(
                    latest.get(
                        "energy"
                    ),
                    3,
                )
            ),

            screen_time=safe_number(
                latest.get(
                    "screen_time"
                )
            ),

            activity_minutes=int(
                safe_number(
                    latest.get(
                        "activity_minutes"
                    ),
                    0,
                )
            ),

            connectedness=int(
                safe_number(
                    latest.get(
                        "connectedness"
                    ),
                    3,
                )
            ),

            activity=str(
                latest.get(
                    "activity",
                    "Relaxing",
                )
            ),

            hour=int(
                safe_number(
                    latest.get(
                        "hour"
                    ),
                    now.hour,
                )
            ),
        )
    )


    displayed_action_id = (
        legacy_guidance.action_id
    )


    displayed_action_title = (
        legacy_guidance.title
    )


    displayed_action_text = (
        legacy_guidance.action
    )


    displayed_action_reason = (
        legacy_guidance.reason
    )


    displayed_model_version = (
        "Legacy record"
    )


    displayed_action_engine_version = (
        "Legacy reconstruction"
    )


# ============================================================
# Recommendation card
# ============================================================

recommendation(
    title=displayed_action_title,
    explanation=displayed_action_text,
    confidence=(
        f"{displayed_confidence:.0%}"
    ),
    certainty=displayed_certainty,
)


st.caption(
    (
        f"AI intervention: "
        f"{displayed_prediction}"
        f" · Action ID: "
        f"{displayed_action_id}"
    )
)


if not snapshot_available:

    st.caption(
        "Legacy record · recommendation metadata was reconstructed."
    )


# ============================================================
# Certainty and alternative
# ============================================================

certainty_col, alternative_col = (
    st.columns(
        2,
        gap="large",
    )
)


with certainty_col:

    st.caption(
        "MODEL CERTAINTY"
    )


    st.markdown(
        f"### {displayed_certainty}"
    )


    st.caption(
        "How strongly the model separated its preferred "
        "intervention from alternatives. This is not a "
        "probability that the intervention will work."
    )


with alternative_col:

    st.caption(
        "NEXT MOST LIKELY"
    )


    if len(
        ordered_probabilities
    ) >= 2:

        alternative = (
            ordered_probabilities[
                1
            ]
        )


        st.markdown(
            f"### {alternative[0]}"
        )


        st.caption(
            f"Model probability: "
            f"{alternative[1]:.0%}"
        )


    else:

        st.markdown(
            "### No alternative"
        )


# ============================================================
# Explanation
# ============================================================

st.write("")


section_heading(
    "Why NudgeWise suggested this",
    (
        "The AI selects a broad intervention and the "
        "contextual action engine turns that intervention "
        "into a specific action."
    ),
)


st.markdown(
    "**Why this specific action**"
)


st.write(
    displayed_action_reason
)


st.write("")


st.markdown(
    "**Why the AI selected this intervention**"
)


st.caption(
    "These explanations describe local model sensitivity "
    "and should not be interpreted as causal effects."
)


for reason in explanation_details.get(
    "reasons",
    [],
):

    st.write(
        f"• {reason}"
    )


# ============================================================
# Technical explainability
# ============================================================

with st.expander(
    "Model sensitivity details",
    expanded=False,
):

    positive_sensitivities = [
        item

        for item
        in explanation_details.get(
            "sensitivities",
            []
        )

        if item[
            "effect"
        ] > 0
    ]


    if positive_sensitivities:

        for item in positive_sensitivities[
            :5
        ]:

            st.write(
                f"**{item['display_name'].title()}**"
            )


            st.caption(
                f"Local probability influence: "
                f"{item['effect']:+.1%}"
            )


    else:

        st.write(
            "No single input had a strong positive local effect. "
            "The recommendation emerged from the combined pattern."
        )


    st.caption(
        "Local sensitivity is an explainability method, "
        "not a causal-effect estimate."
    )


# ============================================================
# Probability distribution
# ============================================================

with st.expander(
    "Recommendation probabilities",
    expanded=False,
):

    for (
        recommendation_name,
        probability,
    ) in ordered_probabilities:

        st.write(
            recommendation_name
        )


        st.progress(
            float(
                clamp(
                    probability
                )
            )
        )


        st.caption(
            f"{probability:.1%}"
        )


    st.caption(
        "These values describe relative model preference, "
        "not probabilities that an intervention will improve wellbeing."
    )


# ============================================================
# Research metadata
# ============================================================

with st.expander(
    "Research metadata",
    expanded=False,
):

    metadata_left, metadata_right = (
        st.columns(
            2,
            gap="large",
        )
    )


    with metadata_left:

        st.caption(
            "MODEL VERSION"
        )

        st.write(
            str(
                displayed_model_version
            )
        )


        st.caption(
            "ACTION ENGINE VERSION"
        )

        st.write(
            str(
                displayed_action_engine_version
            )
        )


    with metadata_right:

        st.caption(
            "TOP-TWO MARGIN"
        )

        st.write(
            f"{displayed_margin:.3f}"
        )


        st.caption(
            "NORMALISED ENTROPY"
        )

        st.write(
            f"{displayed_normalised_entropy:.3f}"
        )


    st.caption(
        f"Raw entropy: {displayed_entropy:.3f}"
    )


divider()


# ============================================================
# Recommendation feedback
# ============================================================

if prediction_record:

    section_heading(
        "How was this recommendation?",
        (
            "Your feedback helps evaluate whether specific "
            "NudgeWise actions are useful and understandable."
        ),
    )


    prediction_id = int(
        prediction_record[
            "id"
        ]
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


        feedback_col_1, feedback_col_2 = (
            st.columns(
                2,
                gap="large",
            )
        )


        with feedback_col_1:

            st.metric(
                "Helpfulness",
                f"{existing_feedback['rating']} / 5",
            )


        with feedback_col_2:

            sense_value = (
                existing_feedback.get(
                    "makes_sense"
                )
            )


            if sense_value is not None:

                st.metric(
                    "Made sense",
                    f"{sense_value} / 5",
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


            save_feedback(
                prediction_id=prediction_id,

                accepted=(
                    1
                    if helpfulness >= 3
                    else 0
                ),

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

                action_id=(
                    displayed_action_id
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
        "Patterns become more informative as your "
        "personal history grows."
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
# Collapsible recent check-ins
# ============================================================

section_heading(
    "Recent check-ins",
    (
        "Your history stays collapsed until you need "
        "to review, edit or delete an entry."
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
    .head(
        14
    )
)


with st.expander(
    (
        f"View and manage recent check-ins "
        f"({len(recent)})"
    ),
    expanded=False,
):

    for _, row in recent.iterrows():

        checkin_id = int(
            row[
                "id"
            ]
        )


        checkin_date = str(
            row.get(
                "checkin_date",
                "",
            )
        )


        entry_type = str(
            row.get(
                "entry_type",
                "live",
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


        date_display = (
            parsed_date.strftime(
                "%A, %d %B"
            )

            if not pd.isna(
                parsed_date
            )

            else checkin_date
        )


        with st.container(
            border=True
        ):

            information_col, action_col = (
                st.columns(
                    [
                        4,
                        1,
                    ],
                    gap="large",
                )
            )


            with information_col:

                st.markdown(
                    f"### {date_display}"
                )


                st.caption(
                    (
                        f"Sleep "
                        f"{safe_number(row.get('sleep')):.1f} h"
                        f" · Mood "
                        f"{safe_number(row.get('mood')):.0f}/5"
                        f" · Stress "
                        f"{safe_number(row.get('stress')):.0f}/5"
                        f" · Energy "
                        f"{safe_number(row.get('energy')):.0f}/5"
                    )
                )


                st.caption(
                    (
                        f"Activity "
                        f"{safe_number(row.get('activity_minutes')):.0f} min"
                        f" · Connectedness "
                        f"{safe_number(row.get('connectedness'), 3):.0f}/5"
                        f" · Screen "
                        f"{safe_number(row.get('screen_time')):.1f} h"
                    )
                )


                st.caption(
                    f"Context: "
                    f"{row.get('activity', '')}"
                )


                if entry_type == "retrospective":

                    st.caption(
                        "Retrospective entry"
                    )


            with action_col:

                if is_today:

                    if st.button(
                        "Edit",
                        key=(
                            f"edit_{checkin_id}"
                        ),
                        width="stretch",
                    ):

                        st.switch_page(
                            "app.py"
                        )


                else:

                    st.caption(
                        "Past entry"
                    )


            delete_key = (
                f"delete_confirm_{checkin_id}"
            )


            if not st.session_state.get(
                delete_key,
                False,
            ):

                if st.button(
                    "Delete",
                    key=(
                        f"delete_{checkin_id}"
                    ),
                    type="secondary",
                ):

                    st.session_state[
                        delete_key
                    ] = True

                    st.rerun()


            else:

                st.warning(
                    "Delete this check-in permanently? "
                    "Its prediction and feedback will also be removed."
                )


                confirm_col, cancel_col = (
                    st.columns(
                        2
                    )
                )


                with confirm_col:

                    if st.button(
                        "Yes, delete",
                        key=(
                            f"yes_{checkin_id}"
                        ),
                        type="primary",
                        width="stretch",
                    ):

                        delete_checkin(
                            checkin_id=checkin_id,
                            user_id=user_id,
                        )


                        st.session_state.pop(
                            delete_key,
                            None,
                        )


                        st.rerun()


                with cancel_col:

                    if st.button(
                        "Cancel",
                        key=(
                            f"cancel_{checkin_id}"
                        ),
                        width="stretch",
                    ):

                        st.session_state.pop(
                            delete_key,
                            None,
                        )


                        st.rerun()


divider()


# ============================================================
# Quick actions
# ============================================================

section_heading(
    "Check-in options"
)


today_exists = (
    today_string
    in set(
        checkins[
            "checkin_date"
        ].astype(
            str
        )
    )
)


today_col, missed_col = (
    st.columns(
        2,
        gap="large",
    )
)


with today_col:

    with st.container(
        border=True
    ):

        st.markdown(
            "### Today"
        )


        if today_exists:

            st.write(
                "Today's check-in is complete."
            )


            if st.button(
                "Edit today's check-in",
                type="primary",
                width="stretch",
                key="edit_today",
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
                key="complete_today",
            ):

                st.switch_page(
                    "app.py"
                )


with missed_col:

    with st.container(
        border=True
    ):

        st.markdown(
            "### Missed a day?"
        )


        st.write(
            "Add a retrospective check-in for a previous date."
        )


        if st.button(
            "Add a missed check-in",
            width="stretch",
            key="add_missed_checkin",
        ):

            st.switch_page(
                "pages/past_checkin.py"
            )


# ============================================================
# Research transparency
# ============================================================

divider()


with st.expander(
    "About this dashboard",
    expanded=False,
):

    st.write(
        """
        NudgeWise is a research prototype exploring personalised
        digital wellbeing recommendations.

        The wellbeing indicator summarises sleep, stress, mood,
        energy, physical activity and perceived connectedness.
        It is a product-level research measure and is not a
        medical or clinical assessment.

        Recreational screen time is used by the recommendation
        system but is not directly converted into a universal
        wellbeing-score penalty.

        The AI model selects one of six broad intervention classes.
        The contextual action engine then translates that intervention
        into a specific action using the participant's current context.

        Research-grade prediction records store the model version,
        full six-class probability distribution, uncertainty measures
        and exact contextual action shown to the participant.

        Model probabilities represent relative model preference.
        They are not probabilities that an intervention will improve
        wellbeing.

        Local sensitivity explanations describe model behaviour and
        should not be interpreted as causal effects.

        Retrospective check-ins are stored separately because they rely
        on recalled rather than same-day information.

        Editing a check-in regenerates its recommendation because the
        underlying input data have changed.

        Deleting a check-in also removes its linked prediction and
        recommendation feedback.
        """
    )