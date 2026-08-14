"""
components/theme.py

Global NudgeWise visual system.

Ensures every NudgeWise page uses the same:
- layout
- typography
- background
- permanently available sidebar
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

        # Keep navigation permanently visible on desktop.
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
           Application
           ==================================================== */

        .stApp {
            background: #FAFAF8;
            color: #20201F;
        }


        /* ====================================================
           Header
           ==================================================== */

        [data-testid="stHeader"] {
            background: transparent;
        }


        /* ====================================================
           Sidebar
           ==================================================== */

        [data-testid="stSidebar"] {
            background: #F4F4F1;
            border-right: 1px solid #E5E5E0;
        }


        /* ====================================================
           Main content
           ==================================================== */

        .block-container {
            max-width: 1080px;
            padding-top: 3.4rem;
            padding-bottom: 5rem;
        }


        /* ====================================================
           Typography
           ==================================================== */

        h1 {
            font-size: 2.7rem !important;
            font-weight: 650 !important;
            letter-spacing: -0.05em !important;
            color: #171717 !important;
        }

        h2 {
            font-size: 1.4rem !important;
            font-weight: 600 !important;
            letter-spacing: -0.025em !important;
        }

        h3 {
            font-size: 1.08rem !important;
            font-weight: 600 !important;
        }

        p {
            color: #696963;
            line-height: 1.6;
        }


        /* ====================================================
           Dividers
           ==================================================== */

        hr {
            border: none;
            border-top: 1px solid #E5E5E0;
            margin: 2.3rem 0;
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
           Buttons
           ==================================================== */

        .stButton > button,
        .stFormSubmitButton > button {
            min-height: 3rem;
            border-radius: 11px !important;
            font-weight: 600 !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )