"""
SmartScreen presentation components.

Clean, native Streamlit components for the Version 2 dashboard.
No inline HTML is used for dashboard content.
"""

from __future__ import annotations

from typing import Optional

import streamlit as st


def section_heading(
    title: str,
    description: Optional[str] = None,
) -> None:
    """Render a clean section heading."""

    st.subheader(title)

    if description:
        st.caption(description)


def wellbeing_score(
    score: int,
    description: str,
) -> None:
    """Render the main wellbeing indicator."""

    st.caption("Wellbeing indicator")

    score_column, description_column = st.columns(
        [1, 3],
        gap="large",
    )

    with score_column:
        st.metric(
            label="Current indicator",
            value=f"{score} / 100",
        )

    with description_column:
        st.write("")
        st.write("")
        st.write(description)


def metric_row(
    metrics: list[tuple[str, str, str]],
) -> None:
    """
    Render a row of wellbeing metrics.

    Each metric should be:

        (label, value, note)
    """

    columns = st.columns(
        len(metrics),
        gap="large",
    )

    for column, metric in zip(columns, metrics):
        label, value, note = metric

        with column:
            st.metric(
                label=label,
                value=value,
            )

            if note:
                st.caption(note)


def recommendation(
    title: str,
    explanation: str,
    confidence: str,
) -> None:
    """Render the model's current recommendation."""

    st.caption("Personalised guidance")

    st.markdown(
        f"### {title}"
    )

    st.write(explanation)

    st.caption(
        f"Model confidence: {confidence}"
    )


def empty_state(
    title: str,
    description: str,
) -> None:
    """Render a simple empty state."""

    st.subheader(title)
    st.caption(description)


def divider() -> None:
    """Render a subtle section divider."""

    st.divider()