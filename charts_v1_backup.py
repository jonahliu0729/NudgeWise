"""
Reusable NudgeWise charts.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Chart Theme
# ============================================================

CHART_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {
        "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
        "color": "#526077",
    },
    "margin": {
        "l": 10,
        "r": 10,
        "t": 45,
        "b": 10,
    },
}


# ============================================================
# Empty State
# ============================================================

def empty_chart(
    message: str = "Not enough data yet.",
) -> None:
    """Display a clean empty state when there is insufficient data."""

    st.markdown(
        f"""
        <div style="
            background:#FFFFFF;
            border:1px solid #E2E8F0;
            border-radius:18px;
            padding:2.5rem 1.5rem;
            text-align:center;
            color:#718096;
        ">
            <div style="
                font-size:1.8rem;
                margin-bottom:0.5rem;
            ">
                📊
            </div>

            <div style="
                font-weight:650;
                color:#526077;
            ">
                {message}
            </div>

            <div style="
                font-size:0.82rem;
                margin-top:0.35rem;
            ">
                Complete more daily check-ins to unlock your trends.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Trend Chart
# ============================================================

def trend_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    y_title: Optional[str] = None,
) -> None:
    """Display a clean line chart for longitudinal wellbeing data."""

    if data.empty or x not in data.columns or y not in data.columns:
        empty_chart()
        return

    chart = px.line(
        data,
        x=x,
        y=y,
        markers=True,
        title=title,
    )

    chart.update_layout(
        **CHART_LAYOUT,
        xaxis_title=None,
        yaxis_title=y_title,
        hovermode="x unified",
    )

    chart.update_traces(
        line=dict(
            width=3,
        ),
        marker=dict(
            size=7,
        ),
    )

    st.plotly_chart(
        chart,
        width="stretch",
    )


# ============================================================
# Multi Metric Trend
# ============================================================

def multi_metric_chart(
    data: pd.DataFrame,
    x: str,
    metrics: list[str],
    title: str,
) -> None:
    """Display multiple wellbeing variables on one chart."""

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

    melted = data[[x] + available].melt(
        id_vars=x,
        var_name="Metric",
        value_name="Value",
    )

    chart = px.line(
        melted,
        x=x,
        y="Value",
        color="Metric",
        markers=True,
        title=title,
    )

    chart.update_layout(
        **CHART_LAYOUT,
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
        legend_title=None,
    )

    chart.update_traces(
        line=dict(
            width=2.5,
        ),
        marker=dict(
            size=6,
        ),
    )

    st.plotly_chart(
        chart,
        width="stretch",
    )


# ============================================================
# Wellbeing Score Chart
# ============================================================

def wellbeing_history_chart(
    data: pd.DataFrame,
    date_column: str = "date",
    score_column: str = "score",
) -> None:
    """Display wellbeing score over time."""

    if (
        data.empty
        or date_column not in data.columns
        or score_column not in data.columns
    ):
        empty_chart(
            "Your wellbeing trend will appear here."
        )
        return

    chart = go.Figure()

    chart.add_trace(
        go.Scatter(
            x=data[date_column],
            y=data[score_column],
            mode="lines+markers",
            name="Wellbeing",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=7,
            ),
        )
    )

    chart.update_layout(
        **CHART_LAYOUT,
        title="Wellbeing Over Time",
        xaxis_title=None,
        yaxis_title="Score",
        yaxis=dict(
            range=[0, 100],
        ),
        showlegend=False,
        hovermode="x unified",
    )

    st.plotly_chart(
        chart,
        width="stretch",
    )