"""
components/navigation.py

Shared authenticated navigation for NudgeWise Version 2.

Provides:
- NudgeWise branding
- participant identity
- Check-in navigation
- Wellbeing dashboard navigation
- usability feedback navigation
- sign out
- hides Streamlit's automatic page list
"""

from __future__ import annotations

import streamlit as st

from database import (
    get_user,
)

from services.auth import (
    ensure_current_participant,
    is_logged_in,
    logout,
)


# ============================================================
# Hide Streamlit default page navigation
# ============================================================

def hide_streamlit_navigation() -> None:
    """
    Hide Streamlit's automatic multipage navigation.

    NudgeWise renders its own branded navigation instead.
    """

    st.markdown(
        """
        <style>

        [data-testid="stSidebarNav"] {
            display: none !important;
        }

        [data-testid="stSidebarNavItems"] {
            display: none !important;
        }

        section[data-testid="stSidebar"] nav {
            display: none !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Sidebar
# ============================================================

def render_sidebar() -> None:
    """Render the shared NudgeWise sidebar."""

    hide_streamlit_navigation()


    # --------------------------------------------------------
    # Restore participant from Google authentication
    # --------------------------------------------------------

    if is_logged_in():

        ensure_current_participant()


    # --------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------

    with st.sidebar:

        # ----------------------------------------------------
        # Brand
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                font-size:1.28rem;
                font-weight:650;
                letter-spacing:-0.035em;
                margin-bottom:0.12rem;
                color:#20201F;
            ">
                NudgeWise
            </div>

            <div style="
                font-size:0.78rem;
                color:#85857F;
                line-height:1.45;
            ">
                Digital wellbeing, made personal.
            </div>
            """,
            unsafe_allow_html=True,
        )


        st.divider()


        # ----------------------------------------------------
        # Logged out
        # ----------------------------------------------------

        if not is_logged_in():

            st.caption(
                "Sign in to begin."
            )

            return


        # ----------------------------------------------------
        # Participant
        # ----------------------------------------------------

        user_id = st.session_state.get(
            "user_id"
        )


        if user_id is not None:

            participant = get_user(
                user_id
            )


            if participant:

                nickname = (
                    participant.get(
                        "nickname"
                    )
                    or participant.get(
                        "name"
                    )
                    or "Participant"
                )


                st.caption(
                    "YOUR PROFILE"
                )


                st.markdown(
                    f"""
                    <div style="
                        font-size:0.96rem;
                        font-weight:600;
                        color:#20201F;
                        margin-top:0.15rem;
                    ">
                        {nickname}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


                st.divider()


                # =================================================
                # Navigation
                # =================================================

                st.page_link(
                    "app.py",
                    label="Check-in",
                )


                st.page_link(
                    "pages/dashboard.py",
                    label="Your wellbeing",
                )


                st.page_link(
                    "pages/feedback.py",
                    label="Give feedback",
                )


                st.divider()


        # ----------------------------------------------------
        # Sign out
        # ----------------------------------------------------

        if st.button(
            "Sign out",
            type="secondary",
            width="stretch",
        ):

            logout()