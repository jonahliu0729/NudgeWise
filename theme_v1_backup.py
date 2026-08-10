"""
NudgeWise Design System
Centralised styling for the Version 2 application.
"""

import streamlit as st


# ============================================================
# DESIGN TOKENS
# ============================================================

COLORS = {
    "background": "#F4F7FB",
    "surface": "#FFFFFF",
    "surface_soft": "#F8FAFC",
    "primary": "#2563EB",
    "primary_dark": "#1D4ED8",
    "secondary": "#0F766E",
    "secondary_soft": "#CCFBF1",
    "success": "#16A34A",
    "success_soft": "#DCFCE7",
    "warning": "#D97706",
    "warning_soft": "#FEF3C7",
    "danger": "#DC2626",
    "danger_soft": "#FEE2E2",
    "text": "#172033",
    "text_secondary": "#526077",
    "text_muted": "#718096",
    "border": "#E2E8F0",
    "sidebar": "#172033",
    "sidebar_text": "#F8FAFC",
}


# ============================================================
# GLOBAL CSS
# ============================================================

def apply_theme() -> None:
    """Apply the NudgeWise global visual theme."""

    st.markdown(
        f"""
        <style>

        /* ----------------------------------------------------
           GLOBAL
        ---------------------------------------------------- */

        .stApp {{
            background: {COLORS["background"]};
        }}

        .main {{
            background: {COLORS["background"]};
        }}

        html,
        body,
        [class*="css"] {{
            font-family:
                -apple-system,
                BlinkMacSystemFont,
                "Segoe UI",
                sans-serif;
            color: {COLORS["text"]};
        }}


        /* ----------------------------------------------------
           HEADINGS
        ---------------------------------------------------- */

        h1 {{
            color: {COLORS["text"]} !important;
            font-size: 2.2rem !important;
            font-weight: 750 !important;
            letter-spacing: -0.03em;
        }}

        h2 {{
            color: {COLORS["text"]} !important;
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }}

        h3 {{
            color: {COLORS["text"]} !important;
            font-weight: 650 !important;
        }}

        p {{
            color: {COLORS["text_secondary"]};
        }}


        /* ----------------------------------------------------
           SIDEBAR
        ---------------------------------------------------- */

        section[data-testid="stSidebar"] {{
            background: {COLORS["sidebar"]};
            border-right: 1px solid rgba(255,255,255,0.08);
        }}

        section[data-testid="stSidebar"] * {{
            color: {COLORS["sidebar_text"]};
        }}

        section[data-testid="stSidebar"] p {{
            color: #CBD5E1;
        }}


        /* ----------------------------------------------------
           INPUT LABELS
        ---------------------------------------------------- */

        label {{
            color: {COLORS["text"]} !important;
            font-weight: 600 !important;
        }}

        [data-testid="stWidgetLabel"] p {{
            color: {COLORS["text"]} !important;
            font-weight: 600 !important;
        }}


        /* ----------------------------------------------------
           INPUTS
        ---------------------------------------------------- */

        div[data-baseweb="select"] > div {{
            background: {COLORS["surface"]};
            border-color: {COLORS["border"]};
            color: {COLORS["text"]};
            border-radius: 10px;
        }}

        div[data-baseweb="input"] {{
            background: {COLORS["surface"]};
            border-color: {COLORS["border"]};
            border-radius: 10px;
        }}

        input {{
            color: {COLORS["text"]} !important;
            background: {COLORS["surface"]} !important;
        }}


        /* ----------------------------------------------------
           BUTTONS
        ---------------------------------------------------- */

        .stButton > button {{
            background: {COLORS["primary"]};
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 650;
            transition: all 0.15s ease;
        }}

        .stButton > button:hover {{
            background: {COLORS["primary_dark"]};
            color: white;
            transform: translateY(-1px);
        }}


        /* ----------------------------------------------------
           CARDS
        ---------------------------------------------------- */

        .nw-card {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 18px;
            padding: 1.35rem;
            box-shadow:
                0 4px 14px rgba(15, 23, 42, 0.05);
        }}

        .nw-card-soft {{
            background: {COLORS["surface_soft"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 18px;
            padding: 1.35rem;
        }}


        /* ----------------------------------------------------
           METRIC CARDS
        ---------------------------------------------------- */

        .nw-metric {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 16px;
            padding: 1.1rem;
            min-height: 120px;
        }}

        .nw-metric-label {{
            color: {COLORS["text_secondary"]};
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 0.4rem;
        }}

        .nw-metric-value {{
            color: {COLORS["text"]};
            font-size: 1.65rem;
            font-weight: 750;
        }}


        /* ----------------------------------------------------
           SCORE
        ---------------------------------------------------- */

        .nw-score {{
            background:
                linear-gradient(
                    135deg,
                    #EFF6FF 0%,
                    #ECFEFF 100%
                );
            border: 1px solid #BFDBFE;
            border-radius: 20px;
            padding: 1.6rem;
        }}

        .nw-score-label {{
            color: {COLORS["text_secondary"]};
            font-size: 0.9rem;
            font-weight: 650;
        }}

        .nw-score-value {{
            color: {COLORS["primary_dark"]};
            font-size: 3rem;
            line-height: 1;
            font-weight: 800;
            margin: 0.45rem 0;
        }}


        /* ----------------------------------------------------
           AI RECOMMENDATION
        ---------------------------------------------------- */

        .nw-ai {{
            background: #F0FDFA;
            border: 1px solid #99F6E4;
            border-radius: 18px;
            padding: 1.4rem;
        }}

        .nw-ai-title {{
            color: #115E59;
            font-size: 0.8rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }}

        .nw-ai-nudge {{
            color: {COLORS["text"]};
            font-size: 1.35rem;
            font-weight: 750;
            margin-top: 0.45rem;
        }}


        /* ----------------------------------------------------
           DIVIDERS
        ---------------------------------------------------- */

        hr {{
            border-color: {COLORS["border"]};
        }}


        /* ----------------------------------------------------
           HIDE STREAMLIT BRANDING
        ---------------------------------------------------- */

        #MainMenu {{
            visibility: hidden;
        }}

        footer {{
            visibility: hidden;
        }}

        header {{
            background: transparent !important;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )