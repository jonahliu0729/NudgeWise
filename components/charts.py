"""
SmartScreen chart components.

Charts are intentionally restrained:
- no rainbow palettes
- no decorative backgrounds
- minimal gridlines
- meaningful labels
- longitudinal data is prioritised
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Chart constants
# ============================================================

TEXT = "#20201F"
SECONDARY = "#777771"
GRID = "#ECECE8"
ACCENT = "#2F6F68"


BASE_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": (
            "-apple-system, BlinkMacSystemFont, "
            "'SF Pro Text', Inter, sans-serif"
        ),
        "color": SECONDARY,
    },
    "margin": {
        "l": 8,
        "r": 8,
        "t": 20,
        "b": 8,
    },
    "hovermode": "x unified",
}


# ============================================================
# Helpers
# ============================================================

def _style_axes(
    figure: go.Figure,
    y_range: Optional[list[float]] = None,
    y_title: Optional[str] = None,
) -> None:
    """Apply consistent axis styling."""

    figure.update_xaxes(
        showgrid=False,
        showline=False,
        zeroline=False,
        title=None,
        tickfont={
            "size": 11,
            "color": SECONDARY,
        },
    )

    figure.update_yaxes(
        showgrid=True,
        gridcolor=GRID,
        gridwidth=1,
        showline=False,
        zeroline=False,
        title=y_title,
        tickfont={
            "size": 11,
            "color": SECONDARY,
        },
    )

    if y_range is not None:
        figure.update_yaxes(
            range=y_range,
        )


def _display(
    figure: go.Figure,
    height: int = 310,
) -> None:
    """Display a chart with consistent settings."""

    figure.update_layout(
        **BASE_LAYOUT,
        height=height,
        showlegend=False,
    )

    st.plotly_chart(
        figure,
        width="stretch",
        config={
            "displayModeBar": False,
            "responsive": True,
        },
    )


# ============================================================
# Empty state
# ============================================================

def empty_chart(
    message: str = (
        "Complete a few more check-ins to reveal "
        "your personal trend."
    ),
) -> None:
    """Display a quiet chart empty state."""

    st.markdown(
        f"""
        <div style="
            padding:2.5rem 0;
            border-top:1px solid #ECECE8;
            border-bottom:1px solid #ECECE8;
        ">

            <div style="
                font-size:0.95rem;
                font-weight:600;
                color:#20201F;
                margin-bottom:0.35rem;
            ">
                Not enough data yet
            </div>

            <div style="
                font-size:0.84rem;
                color:#85857F;
                line-height:1.55;
                max-width:520px;
            ">
                {message}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Generic trend chart
# ============================================================

def trend_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: Optional[str] = None,
    y_title: Optional[str] = None,
    y_range: Optional[list[float]] = None,
) -> None:
    """Display a single longitudinal metric."""

    if (
        data.empty
        or x not in data.columns
        or y not in data.columns
    ):
        empty_chart()
        return

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data[x],
            y=data[y],
            mode="lines+markers",
            line={
                "color": ACCENT,
                "width": 2.5,
            },
            marker={
                "color": ACCENT,
                "size": 6,
            },
            hovertemplate=(
                "%{x}<br>"
                "%{y}"
                "<extra></extra>"
            ),
        )
    )

    _style_axes(
        figure,
        y_range=y_range,
        y_title=y_title,
    )

    if title:
        figure.update_layout(
            title={
                "text": title,
                "font": {
                    "size": 16,
                    "color": TEXT,
                },
                "x": 0,
                "xanchor": "left",
            },
        )

    _display(figure)


# ============================================================
# Wellbeing history
# ============================================================

def wellbeing_history_chart(
    data: pd.DataFrame,
    date_column: str = "date",
    score_column: str = "score",
) -> None:
    """Display wellbeing indicator over time."""

    if (
        data.empty
        or date_column not in data.columns
        or score_column not in data.columns
    ):
        empty_chart(
            "Your wellbeing trend will appear after "
            "you have enough personal history."
        )
        return

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data[date_column],
            y=data[score_column],
            mode="lines+markers",
            line={
                "color": ACCENT,
                "width": 2.8,
            },
            marker={
                "color": ACCENT,
                "size": 7,
            },
            hovertemplate=(
                "%{x}<br>"
                "Indicator: %{y}/100"
                "<extra></extra>"
            ),
        )
    )

    _style_axes(
        figure,
        y_range=[0, 100],
        y_title=None,
    )

    figure.update_layout(
        title={
            "text": "Wellbeing over time",
            "font": {
                "size": 16,
                "color": TEXT,
            },
            "x": 0,
            "xanchor": "left",
        },
    )

    _display(figure)


# ============================================================
# Mood and energy
# ============================================================

def mood_energy_chart(
    data: pd.DataFrame,
    date_column: str = "date",
) -> None:
    """Display mood and energy on a shared 1–5 scale."""

    required = {
        date_column,
        "mood",
        "energy",
    }

    if data.empty or not required.issubset(data.columns):
        empty_chart(
            "Mood and energy trends will appear after "
            "more check-ins."
        )
        return

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data[date_column],
            y=data["mood"],
            mode="lines+markers",
            name="Mood",
            line={
                "color": ACCENT,
                "width": 2.5,
            },
            marker={
                "size": 6,
            },
            hovertemplate=(
                "%{x}<br>"
                "Mood: %{y}/5"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=data[date_column],
            y=data["energy"],
            mode="lines+markers",
            name="Energy",
            line={
                "color": "#777771",
                "width": 2.5,
                "dash": "dot",
            },
            marker={
                "size": 6,
            },
            hovertemplate=(
                "%{x}<br>"
                "Energy: %{y}/5"
                "<extra></extra>"
            ),
        )
    )

    _style_axes(
        figure,
        y_range=[0, 5],
    )

    figure.update_layout(
        title={
            "text": "Mood and energy",
            "font": {
                "size": 16,
                "color": TEXT,
            },
            "x": 0,
            "xanchor": "left",
        },
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {
                "size": 11,
            },
        },
    )

    _display(figure)


# ============================================================
# Screen time
# ============================================================

def screen_time_chart(
    data: pd.DataFrame,
    date_column: str = "date",
    screen_column: str = "screen_time",
) -> None:
    """Display recreational screen time over time."""

    if (
        data.empty
        or date_column not in data.columns
        or screen_column not in data.columns
    ):
        empty_chart(
            "Screen-time trends will appear after "
            "more check-ins."
        )
        return

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=data[date_column],
            y=data[screen_column],
            mode="lines+markers",
            line={
                "color": ACCENT,
                "width": 2.5,
            },
            marker={
                "size": 6,
            },
            fill="tozeroy",
            fillcolor="rgba(47,111,104,0.08)",
            hovertemplate=(
                "%{x}<br>"
                "Screen time: %{y:.1f} h"
                "<extra></extra>"
            ),
        )
    )

    _style_axes(
        figure,
        y_title="Hours",
    )

    figure.update_layout(
        title={
            "text": "Recreational screen time",
            "font": {
                "size": 16,
                "color": TEXT,
            },
            "x": 0,
            "xanchor": "left",
        },
    )

    _display(figure)


# ============================================================
# Multi metric trend
# ============================================================

def multi_metric_chart(
    data: pd.DataFrame,
    x: str,
    metrics: list[str],
    title: str,
) -> None:
    """
    Display multiple metrics.

    This remains available for compatibility with the
    existing dashboard architecture.
    """

    if data.empty or x not in data.columns:
        empty_chart()
        return

    available = [
        metric
        for metric in metrics
        if metric in data.columns
    ]

    if not available:
        empty_chart()
        return

    figure = go.Figure()

    line_styles = [
        ACCENT,
        "#777771",
        "#A6A69F",
    ]

    for index, metric in enumerate(available):

        colour = line_styles[
            min(index, len(line_styles) - 1)
        ]

        figure.add_trace(
            go.Scatter(
                x=data[x],
                y=data[metric],
                mode="lines+markers",
                name=metric.replace("_", " ").title(),
                line={
                    "color": colour,
                    "width": 2.3,
                },
                marker={
                    "size": 5,
                },
            )
        )

    _style_axes(figure)

    figure.update_layout(
        title={
            "text": title,
            "font": {
                "size": 16,
                "color": TEXT,
            },
            "x": 0,
            "xanchor": "left",
        },
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "font": {
                "size": 11,
            },
        },
    )

    _display(figure)