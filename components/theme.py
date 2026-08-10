"""
components/theme.py

Shared visual system for NudgeWise Version 2.

Provides:
- Streamlit page configuration
- global typography
- sidebar styling
- buttons
- inputs
- cards
- dataframe styling
- chart spacing
- responsive layout

The sidebar remains visible and expanded on every page.
"""

from __future__ import annotations

import streamlit as st


# ============================================================
# Page configuration
# ============================================================

def configure_page() -> None:
    """
    Configure a NudgeWise Streamlit page.

    IMPORTANT:
    The sidebar remains expanded so navigation is always
    available from dashboard and other pages.
    """

    st.set_page_config(
        page_title="NudgeWise",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ============================================================
# Theme
# ============================================================

def apply_theme() -> None:
    """Apply the shared NudgeWise visual system."""

    st.markdown(
        """
        <style>

        /* ===================================================
           Global application
        =================================================== */

        .stApp {
            background: #FAFAF8;
            color: #20201F;
        }


        html,
        body,
        [class*="css"] {
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "SF Pro Text",
                "Helvetica Neue",
                Arial,
                sans-serif;
        }


        [data-testid="stHeader"] {
            background: transparent;
        }


        [data-testid="stToolbar"] {
            visibility: hidden;
        }


        [data-testid="stDecoration"] {
            display: none;
        }


        /* ===================================================
           Main content
        =================================================== */

        .block-container {
            max-width: 1180px;
            padding-top: 3.4rem;
            padding-bottom: 5rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }


        /* ===================================================
           Sidebar
        =================================================== */

        section[data-testid="stSidebar"] {
            background: #F4F4F1;
            border-right: 1px solid #E5E5E0;

            visibility: visible !important;
            display: block !important;
        }


        section[data-testid="stSidebar"] > div {
            padding-top: 1.8rem;
        }


        section[data-testid="stSidebar"] * {
            color: #30302E;
        }


        section[data-testid="stSidebar"] hr {
            border: none;
            border-top: 1px solid #E0E0DA;
            margin: 1.4rem 0;
        }


        /* ===================================================
           Sidebar page links
        =================================================== */

        [data-testid="stSidebar"] [data-testid="stPageLink"] {
            border-radius: 9px;
            transition:
                background-color 0.15s ease,
                transform 0.15s ease;
        }


        [data-testid="stSidebar"] [data-testid="stPageLink"] a {
            padding:
                0.55rem
                0.65rem;
        }


        [data-testid="stSidebar"] [data-testid="stPageLink"]:hover {
            background: #EAEAE5;
        }


        /* ===================================================
           Typography
        =================================================== */

        h1 {
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "SF Pro Display",
                "Helvetica Neue",
                Arial,
                sans-serif;

            font-size: 2.75rem !important;
            line-height: 1.08 !important;

            font-weight: 650 !important;

            letter-spacing:
                -0.05em !important;

            color: #171717 !important;

            margin-bottom: 0.65rem !important;
        }


        h2 {
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "SF Pro Display",
                "Helvetica Neue",
                Arial,
                sans-serif;

            font-size: 1.45rem !important;

            line-height: 1.2 !important;

            font-weight: 600 !important;

            letter-spacing:
                -0.028em !important;

            color: #20201F !important;
        }


        h3 {
            font-size: 1.08rem !important;

            font-weight: 600 !important;

            color: #20201F !important;
        }


        p {
            color: #696963;

            line-height: 1.62;
        }


        /* ===================================================
           Captions
        =================================================== */

        [data-testid="stCaptionContainer"] {
            color: #85857F;

            font-size: 0.78rem;
        }


        /* ===================================================
           Dividers
        =================================================== */

        hr {
            border: none;

            border-top:
                1px solid #E5E5E0;

            margin:
                2.4rem 0;
        }


        /* ===================================================
           Forms
        =================================================== */

        [data-testid="stForm"] {
            border: none !important;

            padding: 0 !important;

            background:
                transparent !important;
        }


        /* ===================================================
           Text input
        =================================================== */

        [data-testid="stTextInput"] input,
        [data-testid="stNumberInput"] input {
            border-radius:
                10px !important;

            background:
                #FFFFFF !important;
        }


        [data-baseweb="input"] {
            border-radius:
                10px !important;
        }


        /* ===================================================
           Select boxes
        =================================================== */

        [data-baseweb="select"] > div {
            border-radius:
                10px !important;

            background:
                #FFFFFF !important;
        }


        /* ===================================================
           Sliders
        =================================================== */

        .stSlider {
            padding-top:
                0.2rem;

            padding-bottom:
                0.75rem;
        }


        /* ===================================================
           Buttons
        =================================================== */

        .stButton > button,
        .stFormSubmitButton > button {

            min-height:
                3rem;

            border-radius:
                11px !important;

            border:
                1px solid
                #1D1D1F !important;

            background:
                #1D1D1F !important;

            color:
                #FFFFFF !important;

            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "SF Pro Text",
                "Helvetica Neue",
                Arial,
                sans-serif;

            font-size:
                0.92rem !important;

            font-weight:
                600 !important;

            padding:
                0.62rem
                1.15rem !important;

            box-shadow:
                none !important;

            transition:
                background-color
                0.15s ease,
                border-color
                0.15s ease,
                transform
                0.15s ease;
        }


        .stButton > button:hover,
        .stFormSubmitButton > button:hover {

            background:
                #343436 !important;

            border-color:
                #343436 !important;

            color:
                #FFFFFF !important;
        }


        .stButton > button:active,
        .stFormSubmitButton > button:active {

            transform:
                scale(0.99);
        }


        /* ===================================================
           Secondary buttons
        =================================================== */

        .stButton > button[kind="secondary"] {

            background:
                transparent !important;

            color:
                #30302E !important;

            border:
                1px solid
                #D8D8D2 !important;
        }


        .stButton > button[kind="secondary"]:hover {

            background:
                #ECECE7 !important;

            color:
                #20201F !important;

            border-color:
                #CCCCCC !important;
        }


        /* ===================================================
           Metrics
        =================================================== */

        [data-testid="stMetric"] {

            background:
                transparent;

            border:
                none;

            padding:
                0;
        }


        [data-testid="stMetricLabel"] {

            color:
                #85857F;

            font-size:
                0.78rem;
        }


        [data-testid="stMetricValue"] {

            color:
                #20201F;

            font-weight:
                600;

            letter-spacing:
                -0.025em;
        }


        /* ===================================================
           Expanders
        =================================================== */

        [data-testid="stExpander"] {

            background:
                transparent;

            border:
                1px solid
                #E5E5E0;

            border-radius:
                12px;
        }


        /* ===================================================
           Dataframes
        =================================================== */

        [data-testid="stDataFrame"] {

            border:
                1px solid
                #E5E5E0;

            border-radius:
                12px;

            overflow:
                hidden;
        }


        /* ===================================================
           Alerts
        =================================================== */

        [data-testid="stAlert"] {

            border-radius:
                11px;

            border:
                none;
        }


        /* ===================================================
           Tabs
        =================================================== */

        [data-baseweb="tab-list"] {

            gap:
                0.4rem;

            border-bottom:
                1px solid
                #E5E5E0;
        }


        [data-baseweb="tab"] {

            border-radius:
                8px 8px 0 0;

            font-weight:
                500;
        }


        /* ===================================================
           Plotly charts
        =================================================== */

        [data-testid="stPlotlyChart"] {

            margin-top:
                0.5rem;

            margin-bottom:
                1.4rem;
        }


        /* ===================================================
           App navigation menu icon
        =================================================== */

        [data-testid="collapsedControl"] {

            visibility:
                visible !important;
        }


        /* ===================================================
           Responsive layout
        =================================================== */

        @media (
            max-width: 900px
        ) {

            .block-container {

                padding-left:
                    1.4rem;

                padding-right:
                    1.4rem;

                padding-top:
                    2rem;
            }


            h1 {

                font-size:
                    2.25rem !important;
            }

        }


        </style>
        """,
        unsafe_allow_html=True,
    )