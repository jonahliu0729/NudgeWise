"""
database.py

Handles SQLite storage for AI Nudge.
"""

import sqlite3
from pathlib import Path


DATABASE_FOLDER = Path("database")
DATABASE_FOLDER.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_FOLDER / "nudge.db"



def connect_db():

    return sqlite3.connect(DATABASE_PATH)



def create_tables():

    conn = connect_db()
    cursor = conn.cursor()


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (

        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS checkins (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        user_id INTEGER,

        sleep REAL,
        stress INTEGER,
        mood INTEGER,
        energy INTEGER,

        screen_time REAL,

        activity TEXT,

        social TEXT,

        hour INTEGER,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        checkin_id INTEGER,

        predicted_nudge TEXT,

        confidence REAL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)


    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        prediction_id INTEGER,

        accepted INTEGER,

        completed INTEGER,

        rating INTEGER,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    )
    """)


    conn.commit()
    conn.close()



def create_user(name, age):

    conn = connect_db()

    cursor = conn.cursor()


    cursor.execute(
        """
        INSERT INTO users(name, age)

        VALUES (?, ?)
        """,

        (
            name,
            age
        )

    )


    user_id = cursor.lastrowid


    conn.commit()

    conn.close()


    return user_id




def save_checkin(

    user_id,

    sleep,

    stress,

    mood,

    energy,

    screen_time,

    activity,

    social,

    hour

):

    conn = connect_db()

    cursor = conn.cursor()


    cursor.execute(
        """

        INSERT INTO checkins

        (
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


        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)

        """,

        (
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

    )


    checkin_id = cursor.lastrowid


    conn.commit()

    conn.close()


    return checkin_id




def save_prediction(

    checkin_id,

    nudge,

    confidence

):

    conn = connect_db()

    cursor = conn.cursor()


    cursor.execute(

        """

        INSERT INTO predictions

        (
        checkin_id,
        predicted_nudge,
        confidence
        )


        VALUES (?, ?, ?)

        """,

        (
            checkin_id,
            nudge,
            confidence
        )

    )


    prediction_id = cursor.lastrowid


    conn.commit()

    conn.close()


    return prediction_id




def save_feedback(

    prediction_id,

    accepted,

    completed,

    rating

):

    conn = connect_db()

    cursor = conn.cursor()


    cursor.execute(

        """

        INSERT INTO feedback

        (

        prediction_id,
        accepted,
        completed,
        rating

        )


        VALUES (?, ?, ?, ?)

        """,

        (

        prediction_id,
        accepted,
        completed,
        rating

        )

    )


    conn.commit()

    conn.close()

def get_recent_checkins(user_id, limit=7):
    """Return the most recent check-ins for a user."""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            user_id,
            sleep,
            stress,
            mood,
            energy,
            screen_time,
            activity,
            social,
            hour,
            created_at
        FROM checkins
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_recent_predictions(user_id, limit=7):
    """Return the most recent AI predictions for a user."""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_recent_feedback(user_id, limit=7):
    """Return the most recent user feedback."""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM feedback
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )

    rows = cursor.fetchall()

    conn.close()

    return rows

def get_recent_predictions(user_id, limit=7):
    """Return the most recent AI predictions for a user."""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM predictions
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )

    rows = cursor.fetchall()

    conn.close()

    return rows


def get_recent_feedback(user_id, limit=7):
    """Return the most recent user feedback."""

    conn = connect_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM feedback
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (user_id, limit),
    )

    rows = cursor.fetchall()

    conn.close()

    return rows