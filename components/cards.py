"""
Reusable NudgeWise dashboard cards.
"""

import streamlit as st


def metric_card(
    label: str,
    value: str,
    icon: str = "",
    subtitle: str = "",
) -> None:
    """Display a reusable metric card."""

    st.markdown(
        f"""
        <div class="nw-metric">

            <div class="nw-metric-label">
                {icon} {label}
            </div>

            <div class="nw-metric-value">
                {value}
            </div>

            {
                f'<div style="color:#718096;font-size:0.8rem;margin-top:0.35rem;">'
                f'{subtitle}</div>'
                if subtitle
                else ""
            }

        </div>
        """,
        unsafe_allow_html=True,
    )


def wellbeing_score_card(
    score: int,
    status: str,
) -> None:
    """Display the main wellbeing score."""

    score = max(0, min(100, int(score)))

    st.markdown(
        f"""
        <div class="nw-score">

            <div class="nw-score-label">
                OVERALL WELLBEING
            </div>

            <div class="nw-score-value">
                {score}
                <span style="
                    font-size:1rem;
                    font-weight:600;
                    color:#526077;
                ">
                    / 100
                </span>
            </div>

            <div style="
                color:#526077;
                font-weight:650;
                margin-bottom:0.8rem;
            ">
                {status}
            </div>

            <div style="
                width:100%;
                height:8px;
                background:#DBEAFE;
                border-radius:999px;
                overflow:hidden;
            ">

                <div style="
                    width:{score}%;
                    height:100%;
                    background:#2563EB;
                    border-radius:999px;
                "></div>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


def ai_recommendation_card(
    recommendation: str,
    confidence: float,
    explanation: str = "",
) -> None:
    """Display the AI recommendation."""

    confidence_percent = round(confidence * 100)

    st.markdown(
        f"""
        <div class="nw-ai">

            <div class="nw-ai-title">
                AI COACH
            </div>

            <div class="nw-ai-nudge">
                {recommendation}
            </div>

            <div style="
                margin-top:0.8rem;
                color:#526077;
                font-size:0.9rem;
            ">
                AI confidence:
                <strong>{confidence_percent}%</strong>
            </div>

            {
                f'''
                <div style="
                    margin-top:0.7rem;
                    color:#526077;
                    font-size:0.88rem;
                    line-height:1.5;
                ">
                    {explanation}
                </div>
                '''
                if explanation
                else ""
            }

        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(
    title: str,
    subtitle: str = "",
) -> None:
    """Display a consistent section heading."""

    subtitle_html = ""

    if subtitle:
        subtitle_html = f"""
        <p style="
            margin-top:-0.5rem;
            margin-bottom:1rem;
            color:#718096;
        ">
            {subtitle}
        </p>
        """

    st.markdown(
        f"""
        <div style="margin-bottom:1rem;">

            <h2 style="margin-bottom:0.2rem;">
                {title}
            </h2>

            {subtitle_html}

        </div>
        """,
        unsafe_allow_html=True,
    )