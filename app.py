# ============================================================
# AI Nudge Dashboard
# Professional UI Version
# ============================================================


import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime


from predict import (
    predict_nudge,
    explain_prediction
)


from database import (
    create_tables,
    create_user,
    save_checkin,
    save_prediction,
    save_feedback
)



# ============================================================
# Page Configuration
# ============================================================


st.set_page_config(

    page_title="AI Nudge",

    page_icon="🌱",

    layout="wide",

    initial_sidebar_state="expanded"

)



# ============================================================
# Database Setup
# ============================================================


create_tables()



# ============================================================
# Professional Theme
# ============================================================


st.markdown(

"""
<style>


/* =====================
GLOBAL
===================== */


.stApp {

    background-color:#edf2f7;

}


html, body, [class*="css"] {

    font-family:"Inter", Arial, sans-serif;

    color:#1e293b !important;

}



/* =====================
HEADINGS
===================== */


h1 {

    color:#0f172a !important;

    font-weight:800;

}


h2 {

    color:#1e3a8a !important;

}


h3 {

    color:#334155 !important;

}


p {

    color:#334155 !important;

}



/* =====================
SIDEBAR
===================== */


section[data-testid="stSidebar"] {


    background-color:#1e293b;


}


section[data-testid="stSidebar"] * {


    color:#f8fafc !important;


}



/* =====================
CARDS
===================== */


.card {


    background-color:#f8fafc;

    padding:25px;

    border-radius:18px;

    border:1px solid #dbe4ee;

    box-shadow:
    0px 6px 18px rgba(15,23,42,0.08);

    margin-bottom:20px;


}



/* =====================
INPUTS
===================== */


label {


    color:#334155 !important;

    font-weight:600;


}


[data-testid="stSlider"] * {


    color:#334155 !important;


}


[data-testid="stRadio"] * {


    color:#334155 !important;


}


[data-testid="stSelectbox"] * {


    color:#334155 !important;


}



input {


    background:white !important;

    color:#111827 !important;


}



/* =====================
BUTTONS
===================== */


button {


    background-color:#2563eb !important;

    color:white !important;

    border-radius:12px !important;

    font-weight:700 !important;


}


button:hover {


    background-color:#1d4ed8 !important;


}



/* =====================
AI RESULT
===================== */


.recommendation {


    background-color:#dbeafe;


    padding:30px;


    border-radius:18px;


    border-left:
    8px solid #2563eb;


}



</style>
""",

unsafe_allow_html=True

)



# ============================================================
# Sidebar
# ============================================================


with st.sidebar:


    st.title("🌱 AI Nudge")


    st.write(

        """
        Your AI-powered
        digital wellbeing coach.
        """

    )


    st.divider()


    st.subheader("About")


    st.write(

        """
        AI Nudge analyses your daily
        habits and recommends small
        behaviour changes.
        """

    )



# ============================================================
# Header
# ============================================================


st.title("🌱 AI Nudge Dashboard")


st.write(

    "Personalised wellbeing recommendations powered by AI"

)



st.divider()
# ============================================================
# User Profile
# ============================================================


st.markdown(

"""
<div class="card">

<h2>👤 User Profile</h2>

</div>
""",

unsafe_allow_html=True

)



col1, col2 = st.columns(2)



with col1:


    name = st.text_input(

        "Name",

        value="Jonah"

    )



with col2:


    age = st.number_input(

        "Age",

        min_value=10,

        max_value=100,

        value=16

    )



# ============================================================
# Daily Metrics Cards
# ============================================================


st.markdown(

"""
<div class="card">

<h2>📊 Daily Check-in</h2>

</div>
""",

unsafe_allow_html=True

)



col1, col2, col3 = st.columns(3)



with col1:


    sleep = st.select_slider(

        "😴 Sleep hours",

        options=[

            3.0,

            3.5,

            4.0,

            4.5,

            5.0,

            5.5,

            6.0,

            6.5,

            7.0,

            7.5,

            8.0,

            8.5,

            9.0,

            9.5,

            10.0

        ],

        value=7.5

    )



    stress = st.radio(

        "😰 Stress",

        [1,2,3,4,5],

        horizontal=True

    )



    mood = st.radio(

        "🙂 Mood",

        [1,2,3,4,5],

        horizontal=True

    )



with col2:


    energy = st.radio(

        "⚡ Energy",

        [1,2,3,4,5],

        horizontal=True

    )



    screen_time = st.select_slider(

        "📱 Recreational screen time",

        options=[

            0,

            0.5,

            1,

            1.5,

            2,

            2.5,

            3,

            4,

            5,

            6,

            7,

            8,

            9,

            10,

            12

        ],

        value=3

    )



    hour = st.selectbox(

        "🕒 Current hour",

        list(range(24)),

        index=datetime.now().hour

    )



with col3:


    activity = st.selectbox(

        "🏃 Current activity",

        [

            "Studying",

            "Working",

            "Exercise",

            "Relaxing",

            "Phone"

        ]

    )


    social = st.selectbox(

        "👥 Social interaction",

        [

            "Low",

            "Medium",

            "High"

        ]

    )



st.divider()



# ============================================================
# AI Recommendation
# ============================================================



st.markdown(

"""
<div class="card">

<h2>🤖 AI Recommendation</h2>

</div>
""",

unsafe_allow_html=True

)



if st.button(

    "Generate My AI Nudge"

):


    user_id = create_user(

        name,

        age

    )



    prediction, confidence = predict_nudge(

        sleep=sleep,

        stress=stress,

        mood=mood,

        energy=energy,

        screen_time=screen_time,

        activity=activity,

        social=social,

        hour=hour

    )



    reasons = explain_prediction(

        sleep,

        stress,

        mood,

        energy,

        screen_time,

        hour

    )



    checkin_id = save_checkin(

        user_id,

        sleep,

        stress,

        mood,

        energy,

        screen_time,

        activity,

        social,

        hour

    )



    prediction_id = save_prediction(

        checkin_id,

        prediction,

        confidence

    )



    st.session_state.prediction = prediction

    st.session_state.confidence = confidence

    st.session_state.reasons = reasons

    st.session_state.prediction_id = prediction_id
# ============================================================
# Display AI Result
# ============================================================


if "prediction" in st.session_state:


    st.markdown(

    f"""

    <div class="recommendation">


    <h2>
    🌱 {st.session_state.prediction}
    </h2>


    <h3>
    AI Confidence:
    {st.session_state.confidence:.0%}
    </h3>


    <p>
    <b>Why this recommendation?</b>
    </p>


    """,

    unsafe_allow_html=True

    )


    for reason in st.session_state.reasons:


        st.write(

            "✓ " + reason

        )



    st.markdown(

    """

    <p>

    Small actions create long-term behaviour change.

    </p>


    </div>

    """,

    unsafe_allow_html=True

    )



# ============================================================
# Dashboard Metrics
# ============================================================


st.divider()


st.markdown(

"""
<div class="card">

<h2>📈 Wellbeing Overview</h2>

</div>
""",

unsafe_allow_html=True

)



metric1, metric2, metric3, metric4 = st.columns(4)



with metric1:


    st.metric(

        "Sleep",

        f"{sleep} hrs"

    )


with metric2:


    st.metric(

        "Screen Time",

        f"{screen_time} hrs"

    )


with metric3:


    st.metric(

        "Mood",

        f"{mood}/5"

    )


with metric4:


    st.metric(

        "Energy",

        f"{energy}/5"

    )



# ============================================================
# Example Trend Graphs
# ============================================================


st.divider()



st.subheader(

    "📊 Weekly Habit Trends"

)



chart_data = pd.DataFrame(

    {

        "Day":[

            "Mon",

            "Tue",

            "Wed",

            "Thu",

            "Fri",

            "Sat",

            "Sun"

        ],

        "Sleep":[

            7,

            7.5,

            6.5,

            8,

            7,

            8.5,

            sleep

        ],

        "Screen Time":[

            5,

            6,

            7,

            4,

            8,

            3,

            screen_time

        ]

    }

)



fig1 = px.line(

    chart_data,

    x="Day",

    y="Sleep",

    markers=True,

    title="Sleep Trend"

)


st.plotly_chart(

    fig1,

    use_container_width=True

)



fig2 = px.line(

    chart_data,

    x="Day",

    y="Screen Time",

    markers=True,

    title="Screen Time Trend"

)


st.plotly_chart(

    fig2,

    use_container_width=True

)



# ============================================================
# Feedback
# ============================================================


st.divider()



st.subheader(

    "💬 AI Feedback"

)



if "prediction_id" in st.session_state:


    completed = st.radio(

        "Did you complete the recommendation?",

        [

            "Yes",

            "No"

        ]

    )


    rating = st.slider(

        "Rate this recommendation",

        1,

        5,

        4

    )



    if st.button(

        "Submit Feedback"

    ):


        save_feedback(

            st.session_state.prediction_id,

            1,

            1 if completed=="Yes" else 0,

            rating

        )


        st.success(

            "Feedback saved. Thank you!"

        )



# ============================================================
# Footer
# ============================================================


st.divider()


st.caption(

    "AI Nudge | Personalised Digital Wellbeing Coach"

)