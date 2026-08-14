"""
components/cards.py

Reusable presentation components for NudgeWise.

This version avoids large nested HTML blocks because Streamlit can
sometimes display them as literal text when formatting becomes complex.

Recommendation cards are uncertainty-aware:
- Low certainty
- Moderate certainty
- High certainty

The underlying model probability is preserved exactly as produced
by the model.
"""

from __future__ import annotations

from html import escape

import streamlit as st


# ============================================================
# Helpers
# ============================================================

def _safe(value) -> str:
    if value is None:
        return ""
    return escape(str(value))


# ============================================================
# Divider
# ============================================================

def divider() -> None:
    st.divider()


# ============================================================
# Section heading
# ============================================================

def section_heading(
    title: str,
    description: str | None = None,
) -> None:

    st.markdown(
        f"""
<div style="font-size:1.22rem;font-weight:600;letter-spacing:-0.025em;color:#20201F;margin-bottom:0.35rem;">
{_safe(title)}
</div>
""",
        unsafe_allow_html=True,
    )

    if description:
        st.markdown(
            f"""
<div style="font-size:0.90rem;color:#85857F;line-height:1.55;max-width:680px;margin-bottom:1.35rem;">
{_safe(description)}
</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# Wellbeing score
# ============================================================

def wellbeing_score(
    score: int | float,
    description: str,
) -> None:

    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 0

    score = max(0, min(100, score))

    st.caption("Wellbeing indicator")

    score_col, text_col = st.columns(
        [1, 2.4],
        gap="large",
    )

    with score_col:

        st.caption("Current indicator")

        st.markdown(
            f"""
<div style="display:flex;align-items:baseline;gap:0.4rem;">
    <span style="font-size:3.5rem;font-weight:650;letter-spacing:-0.06em;line-height:1;color:#20201F;">
        {score}
    </span>
    <span style="font-size:1rem;color:#A0A09A;">
        / 100
    </span>
</div>
""",
            unsafe_allow_html=True,
        )

    with text_col:

        st.markdown(
            f"""
<div style="font-size:1rem;color:#696963;line-height:1.6;padding-top:1.45rem;max-width:560px;">
{_safe(description)}
</div>
""",
            unsafe_allow_html=True,
        )


# ============================================================
# Metric
# ============================================================

def metric(
    label: str,
    value: str,
    subtitle: str = "",
) -> None:

    st.caption(
        _safe(label)
    )

    st.markdown(
        f"""
<div style="font-size:2.7rem;font-weight:620;letter-spacing:-0.055em;line-height:1;color:#20201F;margin:0.15rem 0 0.55rem 0;">
{_safe(value)}
</div>
""",
        unsafe_allow_html=True,
    )

    if subtitle:
        st.caption(
            _safe(subtitle)
        )


# ============================================================
# Metric row
# ============================================================

def metric_row(
    metrics: list[tuple[str, str, str]],
) -> None:

    if not metrics:
        return

    columns = st.columns(
        len(metrics),
        gap="large",
    )

    for column, item in zip(
        columns,
        metrics,
    ):

        if len(item) == 2:
            label, value = item
            subtitle = ""
        else:
            label, value, subtitle = item

        with column:
            metric(
                label=label,
                value=value,
                subtitle=subtitle,
            )


# ============================================================
# Recommendation certainty language
# ============================================================

def _certainty_message(
    certainty: str | None,
) -> str | None:
    """
    Convert a model certainty category into restrained,
    user-facing language.

    This does not change or inflate the model probability.
    """

    if not certainty:
        return None

    normalised = (
        str(certainty)
        .strip()
        .lower()
    )

    if normalised == "low":
        return (
            "This is NudgeWise's leading suggestion, "
            "although several options were plausible for this check-in."
        )

    if normalised == "moderate":
        return (
            "NudgeWise found a moderate preference for this suggestion "
            "over the available alternatives."
        )

    if normalised == "high":
        return (
            "NudgeWise identified a relatively clear preference "
            "for this suggestion."
        )

    return None


# ============================================================
# Recommendation
# ============================================================

def recommendation(
    title: str,
    explanation: str,
    confidence: str | None = None,
    certainty: str | None = None,
) -> None:

    with st.container(
        border=True
    ):

        st.markdown(
            f"""
<div style="font-size:1.3rem;font-weight:620;letter-spacing:-0.03em;color:#20201F;margin-bottom:0.45rem;">
{_safe(title)}
</div>
""",
            unsafe_allow_html=True,
        )

        st.write(
            explanation
        )

        certainty_text = (
            _certainty_message(
                certainty
            )
        )

        if certainty_text:

            st.write(
                certainty_text
            )

        if confidence:

            st.caption(
                f"Model probability: {confidence}"
            )

       


# ============================================================
# Empty state
# ============================================================

def empty_state(
    title: str,
    description: str,
) -> None:

    with st.container(
        border=True
    ):

        st.markdown(
            f"""
<div style="font-size:1rem;font-weight:600;color:#30302E;margin-bottom:0.35rem;">
{_safe(title)}
</div>
""",
            unsafe_allow_html=True,
        )

        st.write(
            description
        )