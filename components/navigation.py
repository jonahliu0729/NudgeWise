"""
components/navigation.py

Shared navigation for every NudgeWise page.
"""

from __future__ import annotations

import streamlit as st

from database import get_user


# ============================================================
# Session reset
# ============================================================

def switch_participant() -> None:
    """
    End the current participant session.

    Database records are NOT deleted.
    """

    keys_to_clear = [
        "user_id",
        "participant_code",
        "prediction",
        "confidence",
        "prediction_id",
        "reasons",
        "checkin_submitted",
        "feedback_submitted",
        "new_participant",
    ]

    for key in keys_to_clear:
        st.session_state.pop(
            key,
            None,
        )

    st.switch_page(
        "app.py"
    )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar() -> None:
    """Render the same sidebar everywhere in NudgeWise."""

    with st.sidebar:

        st.markdown(
            """
            <div style="
                font-size:1.28rem;
                font-weight:650;
                letter-spacing:-0.035em;
                margin-bottom:0.12rem;
            ">
                NudgeWise
            </div>

            <div style="
                font-size:0.78rem;
                color:#85857F;
                line-height:1.45;
            ">
                Digital wellbeing, made personal.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        user_id = st.session_state.get(
            "user_id"
        )

        # ----------------------------------------------------
        # Participant signed in
        # ----------------------------------------------------

        if user_id is not None:

            participant = get_user(
                user_id
            )

            if participant:

                nickname = (
                    participant.get("nickname")
                    or participant.get("name")
                    or "Participant"
                )

                participant_code = (
                    participant.get(
                        "participant_code"
                    )
                    or "Legacy participant"
                )

                st.caption(
                    "CURRENT PARTICIPANT"
                )

                st.markdown(
                    f"""
                    <div style="
                        font-size:0.95rem;
                        font-weight:600;
                        color:#20201F;
                    ">
                        {nickname}
                    </div>

                    <div style="
                        font-size:0.76rem;
                        color:#85857F;
                        margin-top:0.15rem;
                    ">
                        {participant_code}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.divider()

                st.page_link(
                    "app.py",
                    label="Check-in",
                )

                st.page_link(
                    "pages/dashboard.py",
                    label="Your wellbeing",
                )

                st.divider()

                if st.button(
                    "Switch participant",
                    type="secondary",
                    width="stretch",
                ):

                    switch_participant()

        # ----------------------------------------------------
        # No participant
        # ----------------------------------------------------

        else:

            st.caption(
                "Start or resume a participant session "
                "to use NudgeWise."
            )