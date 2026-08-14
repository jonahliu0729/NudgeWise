"""
components/theme.py

Global NudgeWise visual system.

Design goals:
- calm, modern health-tech aesthetic
- professional neutral palette
- restrained accent colour
- strong readability
- consistent cards, inputs, buttons, and sidebar
- minimal dependence on default Streamlit styling
"""

from __future__ import annotations

import streamlit as st


# ============================================================
# Page configuration
# ============================================================

def configure_page() -> None:
    """
    Apply consistent page configuration across every
    NudgeWise page.
    """

    st.set_page_config(
        page_title="NudgeWise",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="locked",
    )


# ============================================================
# Global theme
# ============================================================

def apply_theme() -> None:
    """Apply the shared NudgeWise visual design."""

    st.markdown(
        """
        <style>

        /* ====================================================
           Design tokens
           ==================================================== */

        :root {
            --nw-bg: #F7F8FA;
            --nw-surface: #FFFFFF;
            --nw-surface-soft: #F2F4F7;
            --nw-sidebar: #F3F5F7;

            --nw-text: #171A1F;
            --nw-text-secondary: #5F6670;
            --nw-text-muted: #8A919B;

            --nw-border: #E3E7EC;
            --nw-border-strong: #D5DAE1;

            --nw-accent: #4F6BFF;
            --nw-accent-hover: #435DE6;
            --nw-accent-soft: #EEF1FF;

            --nw-success: #2E7D5A;
            --nw-warning: #A56A13;
            --nw-danger: #C54A4A;

            --nw-radius-sm: 10px;
            --nw-radius-md: 14px;
            --nw-radius-lg: 18px;

            --nw-shadow:
                0 1px 2px rgba(16, 24, 40, 0.04),
                0 4px 14px rgba(16, 24, 40, 0.04);
        }


        /* ====================================================
           Application
           ==================================================== */

        html,
        body,
        [class*="css"] {
            color: var(--nw-text);
        }

        .stApp {
            background: var(--nw-bg);
            color: var(--nw-text);
        }


        /* ====================================================
           Header
           ==================================================== */

        [data-testid="stHeader"] {
            background: rgba(247, 248, 250, 0.92);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(227, 231, 236, 0.7);
        }


        /* ====================================================
           Sidebar
           ==================================================== */

        [data-testid="stSidebar"] {
            background: var(--nw-sidebar);
            border-right: 1px solid var(--nw-border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span {
            color: var(--nw-text-secondary);
        }


        /* ====================================================
           Main content
           ==================================================== */

        .block-container {
            max-width: 1120px;
            padding-top: 3.6rem;
            padding-bottom: 5rem;
        }


        /* ====================================================
           Typography
           ==================================================== */

        h1 {
            font-size: 2.65rem !important;
            font-weight: 680 !important;
            letter-spacing: -0.045em !important;
            line-height: 1.05 !important;
            color: var(--nw-text) !important;
            margin-bottom: 0.6rem !important;
        }

        h2 {
            font-size: 1.42rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.025em !important;
            color: var(--nw-text) !important;
        }

        h3 {
            font-size: 1.08rem !important;
            font-weight: 640 !important;
            letter-spacing: -0.012em !important;
            color: var(--nw-text) !important;
        }

        p {
            color: var(--nw-text-secondary);
            line-height: 1.65;
        }

        small,
        [data-testid="stCaptionContainer"],
        .stCaption {
            color: var(--nw-text-muted) !important;
        }


        /* ====================================================
           Dividers
           ==================================================== */

        hr {
            border: none;
            border-top: 1px solid var(--nw-border);
            margin: 2.2rem 0;
        }


        /* ====================================================
           Generic containers / cards
           ==================================================== */

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--nw-surface);
            border: 1px solid var(--nw-border) !important;
            border-radius: var(--nw-radius-md) !important;
            box-shadow: var(--nw-shadow);
        }


        /* ====================================================
           Forms
           ==================================================== */

        [data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
            background: transparent !important;
        }


        /* ====================================================
           Text inputs
           ==================================================== */

        [data-testid="stTextInput"] input {
            background: var(--nw-surface) !important;
            color: var(--nw-text) !important;
            border: 1px solid var(--nw-border-strong) !important;
            border-radius: var(--nw-radius-sm) !important;
        }

        [data-testid="stTextInput"] input:focus {
            border-color: var(--nw-accent) !important;
            box-shadow: 0 0 0 3px rgba(79, 107, 255, 0.12) !important;
        }

        [data-testid="stTextInput"] input::placeholder {
            color: var(--nw-text-muted) !important;
        }


        /* ====================================================
           Number inputs
           ==================================================== */

        [data-testid="stNumberInput"] input {
            background: var(--nw-surface) !important;
            color: var(--nw-text) !important;
            border-radius: var(--nw-radius-sm) !important;
        }

        [data-testid="stNumberInput"] button {
            background: var(--nw-surface-soft) !important;
            color: var(--nw-text-secondary) !important;
            border-color: var(--nw-border) !important;
        }


        /* ====================================================
           Text areas
           ==================================================== */

        [data-testid="stTextArea"] textarea {
            background: #20242C !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            caret-color: #FFFFFF !important;

            border: 1px solid #343A46 !important;
            border-radius: var(--nw-radius-sm) !important;

            line-height: 1.55 !important;
        }

        [data-testid="stTextArea"] textarea:focus {
            border-color: #7086FF !important;
            box-shadow: 0 0 0 3px rgba(79, 107, 255, 0.18) !important;
        }

        [data-testid="stTextArea"] textarea::placeholder {
            color: rgba(255, 255, 255, 0.48) !important;
            -webkit-text-fill-color: rgba(255, 255, 255, 0.48) !important;
        }


        /* ====================================================
           Select boxes
           ==================================================== */

        [data-baseweb="select"] > div {
            background: var(--nw-surface) !important;
            border-color: var(--nw-border-strong) !important;
            border-radius: var(--nw-radius-sm) !important;
        }

        [data-baseweb="select"] span {
            color: var(--nw-text) !important;
        }


        /* ====================================================
           Radio controls
           ==================================================== */

        [data-testid="stRadio"] label {
            color: var(--nw-text-secondary) !important;
        }


        /* ====================================================
           Sliders
           ==================================================== */

        [data-testid="stSlider"] [role="slider"] {
            background: var(--nw-accent) !important;
        }

        [data-testid="stSlider"] [data-testid="stTickBar"] {
            color: var(--nw-text-muted);
        }


        /* ====================================================
           Progress bars
           ==================================================== */

        [data-testid="stProgress"] > div > div {
            background: var(--nw-accent) !important;
        }

        [data-testid="stProgress"] > div {
            background: var(--nw-accent-soft) !important;
            border-radius: 999px;
        }


        /* ====================================================
           Buttons
           ==================================================== */

        .stButton > button,
        .stFormSubmitButton > button {
            min-height: 3rem;
            border-radius: var(--nw-radius-sm) !important;
            font-weight: 620 !important;
            transition:
                background 0.16s ease,
                border-color 0.16s ease,
                transform 0.10s ease;
        }

        .stButton > button:hover,
        .stFormSubmitButton > button:hover {
            transform: translateY(-1px);
        }


        /* Primary buttons */

        button[kind="primary"] {
            background: var(--nw-accent) !important;
            color: #FFFFFF !important;
            border: 1px solid var(--nw-accent) !important;
        }

        button[kind="primary"]:hover {
            background: var(--nw-accent-hover) !important;
            border-color: var(--nw-accent-hover) !important;
        }


        /* Secondary buttons */

        button[kind="secondary"] {
            background: var(--nw-surface) !important;
            color: var(--nw-text) !important;
            border: 1px solid var(--nw-border-strong) !important;
        }

        button[kind="secondary"]:hover {
            background: var(--nw-surface-soft) !important;
        }


        /* ====================================================
           Metrics
           ==================================================== */

        [data-testid="stMetric"] {
            background: var(--nw-surface);
            border: 1px solid var(--nw-border);
            border-radius: var(--nw-radius-md);
            padding: 1rem 1.05rem;
            box-shadow: var(--nw-shadow);
        }

        [data-testid="stMetricLabel"] {
            color: var(--nw-text-muted) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--nw-text) !important;
            font-weight: 650 !important;
        }


        /* ====================================================
           Expanders
           ==================================================== */

        [data-testid="stExpander"] {
            background: var(--nw-surface);
            border: 1px solid var(--nw-border) !important;
            border-radius: var(--nw-radius-md) !important;
            overflow: hidden;
        }

        [data-testid="stExpander"] summary {
            color: var(--nw-text) !important;
            font-weight: 600 !important;
        }


        /* ====================================================
           Alerts
           ==================================================== */

        [data-testid="stAlert"] {
            border-radius: var(--nw-radius-md) !important;
            border: 1px solid var(--nw-border) !important;
        }


        /* ====================================================
           Tooltips / help
           ==================================================== */

        [data-testid="stTooltipContent"] {
            border-radius: var(--nw-radius-sm) !important;
        }


        /* ====================================================
           Links
           ==================================================== */

        a {
            color: var(--nw-accent);
        }

        a:hover {
            color: var(--nw-accent-hover);
        }


        /* ====================================================
           Focus accessibility
           ==================================================== */

        button:focus-visible,
        input:focus-visible,
        textarea:focus-visible,
        [role="slider"]:focus-visible {
            outline: 2px solid var(--nw-accent) !important;
            outline-offset: 2px !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )