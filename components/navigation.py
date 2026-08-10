"""
NudgeWise navigation components.
"""

from __future__ import annotations

import streamlit as st


PAGES = {
    "Dashboard": "dashboard",
    "Daily Check-in": "checkin",
    "AI Coach": "ai_coach",
    "Insights": "insights",
    "Progress": "progress",
    "Settings": "settings",
}


def render_brand() -> None:
    """Display the NudgeWise sidebar brand."""

    st.sidebar.markdown(
        """
        <div style="
            padding: 0.5rem 0.25rem 1.5rem 0.25rem;
        ">
            <div style="
                font-size:1.45rem;
                font-weight:800;
                letter-spacing:-0.04em;
                color:#FFFFFF;
            ">
                NudgeWise
            </div>

            <div style="
                margin-top:0.3rem;
                font-size:0.78rem;
                color:#CBD5E1;
                line-height:1.4;
            ">
                Your habits.<br>
                Your data.<br>
                Your next step.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the main NudgeWise sidebar."""

    render_brand()

    st.sidebar.markdown(
        """
        <div style="
            color:#94A3B8;
            font-size:0.7rem;
            font-weight:750;
            letter-spacing:0.1em;
            text-transform:uppercase;
            margin-bottom:0.55rem;
        ">
            Navigation
        </div>
        """,
        unsafe_allow_html=True,
    )

    for label, page in PAGES.items():
        st.sidebar.page_link(
            f"pages/{page}.py",
            label=label,
        )

    st.sidebar.markdown(
        """
        <div style="
            margin-top:2rem;
            padding:0.9rem;
            border-radius:12px;
            background:rgba(255,255,255,0.06);
            border:1px solid rgba(255,255,255,0.08);
        ">
            <div style="
                font-size:0.72rem;
                color:#94A3B8;
                text-transform:uppercase;
                letter-spacing:0.08em;
                font-weight:700;
            ">
                NudgeWise AI
            </div>

            <div style="
                margin-top:0.35rem;
                font-size:0.82rem;
                color:#E2E8F0;
                line-height:1.4;
            ">
                Personalised digital wellbeing insights.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar(
    greeting: str = "Welcome back",
    user_name: str = "",
) -> None:
    """Render a clean page header."""

    name_html = ""

    if user_name:
        name_html = f"""
        <span style="
            color:#2563EB;
        ">
            {user_name}
        </span>
        """

    st.markdown(
        f"""
        <div style="
            display:flex;
            justify-content:space-between;
            align-items:flex-end;
            margin-bottom:1.5rem;
        ">

            <div>

                <div style="
                    color:#718096;
                    font-size:0.9rem;
                    font-weight:600;
                    margin-bottom:0.25rem;
                ">
                    {greeting} {name_html}
                </div>

                <h1 style="
                    margin:0;
                    color:#172033;
                ">
                    Your wellbeing
                </h1>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )