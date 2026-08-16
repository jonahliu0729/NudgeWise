"""
pages/profile.py

NudgeWise Version 2.8
Participant profile management.

Allows participants to:
- view participant code
- edit display name
- add or update date of birth
- automatically calculate current age

Age is calculated using New Zealand local date.

The existing age database field is retained for backwards
compatibility, while date_of_birth becomes the preferred source
for current age.
"""

from __future__ import annotations

from datetime import (
    date,
    datetime,
)

from zoneinfo import ZoneInfo

import streamlit as st

from components.theme import (
    apply_theme,
    configure_page,
)


# ============================================================
# Page setup
# ============================================================

configure_page()
apply_theme()


from components.navigation import (
    render_sidebar,
)

from database import (
    create_tables,
    get_user,
    update_user_profile,
)

from services.auth import (
    ensure_current_participant,
    is_logged_in,
)


# ============================================================
# Configuration
# ============================================================

NZ_TIMEZONE = ZoneInfo(
    "Pacific/Auckland"
)


# ============================================================
# Age calculation
# ============================================================

def calculate_age(
    date_of_birth: date,
    today: date | None = None,
) -> int:
    """
    Calculate age correctly around the participant's birthday.
    """

    if today is None:

        today = datetime.now(
            NZ_TIMEZONE
        ).date()


    return (
        today.year
        - date_of_birth.year
        - (
            (
                today.month,
                today.day,
            )
            <
            (
                date_of_birth.month,
                date_of_birth.day,
            )
        )
    )


# ============================================================
# Setup
# ============================================================

create_tables()

render_sidebar()


# ============================================================
# Authentication guard
# ============================================================

if not is_logged_in():

    st.switch_page(
        "app.py"
    )


participant = (
    ensure_current_participant()
)


if participant is None:

    st.switch_page(
        "app.py"
    )


user_id = st.session_state.get(
    "user_id"
)


if user_id is None:

    st.switch_page(
        "app.py"
    )


participant = get_user(
    user_id
)


if participant is None:

    st.session_state.pop(
        "user_id",
        None,
    )

    st.switch_page(
        "app.py"
    )


# ============================================================
# Existing participant values
# ============================================================

nickname = (
    participant.get(
        "nickname"
    )
    or participant.get(
        "name"
    )
    or "Participant"
)


participant_code = (
    participant.get(
        "participant_code"
    )
    or "—"
)


stored_age = (
    participant.get(
        "age"
    )
)


stored_dob = (
    participant.get(
        "date_of_birth"
    )
)


parsed_dob = None


if stored_dob:

    try:

        parsed_dob = date.fromisoformat(
            str(
                stored_dob
            )
        )

    except ValueError:

        parsed_dob = None


today = datetime.now(
    NZ_TIMEZONE
).date()


# ============================================================
# Determine current age
# ============================================================

if parsed_dob is not None:

    current_age = calculate_age(
        parsed_dob,
        today,
    )


elif stored_age is not None:

    current_age = int(
        stored_age
    )


else:

    current_age = 15


# ============================================================
# Header
# ============================================================

st.caption(
    "PROFILE"
)


st.title(
    "Your profile"
)


st.write(
    "Manage the basic information connected to your "
    "NudgeWise participant account."
)


st.divider()


# ============================================================
# Profile overview
# ============================================================

left, right = st.columns(
    2,
    gap="large",
)


with left:

    st.caption(
        "PARTICIPANT CODE"
    )


    st.markdown(
        f"### {participant_code}"
    )


    st.caption(
        "Your participant code identifies your research "
        "records without relying on your display name."
    )


with right:

    st.caption(
        "CURRENT AGE"
    )


    st.markdown(
        f"### {current_age}"
    )


    if parsed_dob is not None:

        st.caption(
            "Calculated automatically from your date of birth."
        )

    else:

        st.caption(
            "Based on the age entered when your account was created."
        )


st.divider()


# ============================================================
# Edit profile
# ============================================================

st.subheader(
    "Edit profile"
)


st.write(
    "Once you add your date of birth, NudgeWise can update "
    "your age automatically when you have a birthday."
)


# ============================================================
# Default birth date
# ============================================================

if parsed_dob is not None:

    default_birth_date = (
        parsed_dob
    )


else:

    # Use July 1 rather than January 1 so an unknown birthday
    # does not strongly imply an exact historical age.
    default_year = (
        today.year
        - current_age
    )


    default_birth_date = date(
        default_year,
        7,
        1,
    )


# ============================================================
# Form
# ============================================================

with st.form(
    "edit_profile"
):

    updated_nickname = st.text_input(
        "Display name",
        value=str(
            nickname
        ),
        help=(
            "This is the name shown inside NudgeWise."
        ),
    )


    updated_birth_date = st.date_input(
        "Date of birth",
        value=default_birth_date,
        min_value=date(
            today.year - 19,
            1,
            1,
        ),
        max_value=today,
        help=(
            "Used to calculate your current age automatically."
        ),
    )


    calculated_age = calculate_age(
        updated_birth_date,
        today,
    )


    st.caption(
        f"Calculated age: {calculated_age}"
    )


    valid_age = (
        10
        <= calculated_age
        <= 18
    )


    if not valid_age:

        st.warning(
            "NudgeWise currently supports participant "
            "profiles aged 10 to 18."
        )


    submitted = (
        st.form_submit_button(
            "Save profile changes",
            type="primary",
            width="stretch",
            disabled=(
                not valid_age
            ),
        )
    )


# ============================================================
# Save changes
# ============================================================

if submitted:

    clean_nickname = (
        updated_nickname.strip()
        or "Participant"
    )


    try:

        updated = update_user_profile(
            user_id=user_id,
            nickname=clean_nickname,
            date_of_birth=(
                updated_birth_date.isoformat()
            ),
            age=calculated_age,
        )


        if updated:

            st.success(
                "Profile updated."
            )

            st.rerun()


        else:

            st.error(
                "NudgeWise could not update your profile."
            )


    except Exception as error:

        st.error(
            "NudgeWise could not save your profile changes."
        )

        st.exception(
            error
        )


# ============================================================
# Privacy
# ============================================================

st.divider()


with st.expander(
    "How age information is used"
):

    st.write(
        """
        NudgeWise stores your date of birth in your participant
        profile so your current age can be calculated automatically.

        Your birthday does not need to be entered again when your
        age changes.

        For research analysis, age or age groups should be used
        wherever possible instead of exposing exact dates of birth.

        Your participant code remains the preferred identifier for
        research records.
        """
    )