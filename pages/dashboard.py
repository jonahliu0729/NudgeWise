import streamlit as st
import sqlite3
import pandas as pd


DATABASE = "database/nudge.db"


def load_data(table):

    conn = sqlite3.connect(DATABASE)

    df = pd.read_sql_query(
        f"SELECT * FROM {table}",
        conn
    )

    conn.close()

    return df



# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="AI Nudge Dashboard",
    page_icon="📊"
)


st.title("📊 AI Nudge Analytics Dashboard")

st.write(
    "Monitoring AI recommendations and user responses."
)


# -----------------------------
# Load data
# -----------------------------

checkins = load_data("checkins")

predictions = load_data("predictions")

feedback = load_data("feedback")



# -----------------------------
# Key metrics
# -----------------------------

st.header("System Performance")


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Nudges",
        len(predictions)
    )


with col2:

    acceptance = (
        feedback["accepted"].mean() * 100
        if len(feedback) > 0
        else 0
    )

    st.metric(
        "Acceptance Rate",
        f"{acceptance:.0f}%"
    )


with col3:

    confidence = (
        predictions["confidence"].mean()*100
        if len(predictions)>0
        else 0
    )

    st.metric(
        "Average Confidence",
        f"{confidence:.0f}%"
    )



# -----------------------------
# Prediction analysis
# -----------------------------

st.divider()

st.header("Most Common AI Recommendations")


if len(predictions)>0:

    counts = predictions[
        "predicted_nudge"
    ].value_counts()


    st.bar_chart(counts)



# -----------------------------
# Feedback analysis
# -----------------------------

st.divider()

st.header("User Response Data")


if len(feedback)>0:

    st.dataframe(
        feedback
    )

else:

    st.info(
        "No feedback collected yet."
    )



# -----------------------------
# Raw data
# -----------------------------

st.divider()

st.header("Database Records")


with st.expander("Check-ins"):

    st.dataframe(checkins)


with st.expander("Predictions"):

    st.dataframe(predictions)