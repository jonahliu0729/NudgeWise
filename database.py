"""
database.py

Persistent SQLite storage for NudgeWise Version 2.

Relationships:
    participant -> check-ins -> predictions -> feedback
"""

from __future__ import annotations

import secrets
import sqlite3
import string
from pathlib import Path
from typing import Any


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_FOLDER = BASE_DIR / "database"
DATABASE_FOLDER.mkdir(exist_ok=True)

DATABASE_PATH = DATABASE_FOLDER / "nudge.db"


# ============================================================
# Connection
# ============================================================

def connect_db() -> sqlite3.Connection:
    """Return a SQLite connection."""

    connection = sqlite3.connect(DATABASE_PATH)

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# Schema helpers
# ============================================================

def _column_exists(
    cursor: sqlite3.Cursor,
    table: str,
    column: str,
) -> bool:
    """Check whether a table already contains a column."""

    cursor.execute(
        f"PRAGMA table_info({table})"
    )

    columns = {
        row["name"]
        for row in cursor.fetchall()
    }

    return column in columns


# ============================================================
# Schema
# ============================================================

def create_tables() -> None:
    """
    Create database tables and migrate older Version 1/2 schemas.

    Existing data is preserved.
    """

    connection = connect_db()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # Users / participants
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # Add V2 participant fields without deleting old data.

    if not _column_exists(
        cursor,
        "users",
        "participant_code",
    ):
        cursor.execute(
            """
            ALTER TABLE users
            ADD COLUMN participant_code TEXT
            """
        )

    if not _column_exists(
        cursor,
        "users",
        "nickname",
    ):
        cursor.execute(
            """
            ALTER TABLE users
            ADD COLUMN nickname TEXT
            """
        )

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_users_participant_code
        ON users(participant_code)
        """
    )

    # --------------------------------------------------------
    # Check-ins
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER NOT NULL,

            sleep REAL,
            stress INTEGER,
            mood INTEGER,
            energy INTEGER,

            screen_time REAL,

            activity TEXT,
            social TEXT,

            hour INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (user_id)
                REFERENCES users(id)
        )
        """
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            checkin_id INTEGER NOT NULL,

            predicted_nudge TEXT,
            confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (checkin_id)
                REFERENCES checkins(id)
        )
        """
    )

    # --------------------------------------------------------
    # Feedback
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prediction_id INTEGER NOT NULL,

            accepted INTEGER,
            completed INTEGER,
            rating INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

            FOREIGN KEY (prediction_id)
                REFERENCES predictions(id)
        )
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# Participant codes
# ============================================================

def _generate_participant_code() -> str:
    """
    Generate an anonymous participant code.

    Confusing characters such as O/0 and I/1 are excluded.
    """

    alphabet = (
        "ABCDEFGHJKLMNPQRSTUVWXYZ"
        "23456789"
    )

    token = "".join(
        secrets.choice(alphabet)
        for _ in range(6)
    )

    return f"NW-{token}"


def _participant_code_exists(
    participant_code: str,
) -> bool:
    """Return True if a participant code already exists."""

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT 1
        FROM users
        WHERE participant_code = ?
        LIMIT 1
        """,
        (participant_code,),
    )

    exists = cursor.fetchone() is not None

    connection.close()

    return exists


# ============================================================
# Participants
# ============================================================

def create_participant(
    nickname: str,
    age: int,
) -> dict[str, Any]:
    """
    Create a new anonymous participant.

    Returns:
        {
            id,
            participant_code,
            nickname,
            age
        }
    """

    while True:

        participant_code = (
            _generate_participant_code()
        )

        if not _participant_code_exists(
            participant_code
        ):
            break

    clean_nickname = nickname.strip()

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO users (
            name,
            nickname,
            age,
            participant_code
        )
        VALUES (?, ?, ?, ?)
        """,
        (
            clean_nickname,
            clean_nickname,
            int(age),
            participant_code,
        ),
    )

    user_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return {
        "id": int(user_id),
        "participant_code": participant_code,
        "nickname": clean_nickname,
        "age": int(age),
    }


def get_user(
    user_id: int,
) -> dict[str, Any] | None:
    """Return one participant by internal ID."""

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            participant_code,
            nickname,
            name,
            age,
            created_at
        FROM users
        WHERE id = ?
        """,
        (user_id,),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def get_user_by_code(
    participant_code: str,
) -> dict[str, Any] | None:
    """Find a participant using their anonymous code."""

    clean_code = (
        participant_code
        .strip()
        .upper()
    )

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            participant_code,
            nickname,
            name,
            age,
            created_at
        FROM users
        WHERE UPPER(participant_code) = ?
        """,
        (clean_code,),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


# ============================================================
# Check-ins
# ============================================================

def save_checkin(
    user_id: int,
    sleep: float,
    stress: int,
    mood: int,
    energy: int,
    screen_time: float,
    activity: str,
    social: str,
    hour: int,
) -> int:
    """Save a participant check-in."""

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO checkins (
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
            hour,
        ),
    )

    checkin_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return int(checkin_id)


def get_recent_checkins(
    user_id: int,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Return recent check-ins for one participant."""

    connection = connect_db()
    cursor = connection.cursor()

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
        (
            user_id,
            limit,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


# ============================================================
# Predictions
# ============================================================

def save_prediction(
    checkin_id: int,
    nudge: str,
    confidence: float,
) -> int:
    """Save one AI recommendation."""

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO predictions (
            checkin_id,
            predicted_nudge,
            confidence
        )
        VALUES (?, ?, ?)
        """,
        (
            checkin_id,
            nudge,
            confidence,
        ),
    )

    prediction_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return int(prediction_id)


def get_recent_predictions(
    user_id: int,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Return predictions associated with one participant."""

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            predictions.id,
            predictions.checkin_id,
            predictions.predicted_nudge,
            predictions.confidence,
            predictions.created_at
        FROM predictions

        INNER JOIN checkins
            ON predictions.checkin_id = checkins.id

        WHERE checkins.user_id = ?

        ORDER BY predictions.created_at DESC

        LIMIT ?
        """,
        (
            user_id,
            limit,
        ),
    )

    rows = cursor.fetchall()

    connection.close()

    return [
        dict(row)
        for row in rows
    ]


def get_latest_prediction(
    user_id: int,
) -> dict[str, Any] | None:
    """Return the most recent AI prediction."""

    predictions = get_recent_predictions(
        user_id=user_id,
        limit=1,
    )

    if not predictions:
        return None

    return predictions[0]


# ============================================================
# Feedback
# ============================================================

def save_feedback(
    prediction_id: int,
    accepted: int,
    completed: int,
    rating: int,
) -> None:
    """Save recommendation feedback."""

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO feedback (
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
            rating,
        ),
    )

    connection.commit()
    connection.close()