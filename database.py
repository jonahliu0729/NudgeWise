"""
database.py

Persistent SQLite storage for NudgeWise Version 2.6.

Supports:
- authenticated participants
- one check-in per participant per calendar day
- live and retrospective check-ins
- physical activity minutes
- perceived connectedness
- backward compatibility with the legacy social variable
- editing and deleting check-ins
- AI predictions
- recommendation feedback
- usability feedback

Migration strategy:
- existing databases are preserved
- activity_minutes is added if missing
- connectedness is added if missing
- historical social categories are mapped approximately to
  connectedness values for backward compatibility
"""

from __future__ import annotations

import secrets
import sqlite3

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
# Errors
# ============================================================

class DuplicateCheckinError(Exception):
    """Raised when a participant already has a log for a date."""


# ============================================================
# Connection
# ============================================================

def connect_db() -> sqlite3.Connection:

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


# ============================================================
# Schema helpers
# ============================================================

def _column_exists(
    cursor: sqlite3.Cursor,
    table: str,
    column: str,
) -> bool:

    cursor.execute(
        f"PRAGMA table_info({table})"
    )

    columns = {
        row["name"]
        for row in cursor.fetchall()
    }

    return column in columns


# ============================================================
# Connectedness compatibility
# ============================================================

def connectedness_to_social(
    connectedness: int,
) -> str:
    """
    Convert the new 1-5 connectedness measure into the old
    Low / Medium / High representation.

    This exists only to keep the v2.5 model operational during
    the v2.6 migration.
    """

    value = max(
        1,
        min(
            5,
            int(connectedness),
        ),
    )

    if value <= 2:
        return "Low"

    if value == 3:
        return "Medium"

    return "High"


# ============================================================
# Database setup
# ============================================================

def create_tables() -> None:

    connection = connect_db()
    cursor = connection.cursor()

    # --------------------------------------------------------
    # Users
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
        "auth_subject_hash",
    ):
        cursor.execute(
            """
            ALTER TABLE users
            ADD COLUMN auth_subject_hash TEXT
            """
        )

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS
        idx_users_auth_subject_hash
        ON users(auth_subject_hash)
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

            user_id INTEGER,

            sleep REAL,
            stress INTEGER,
            mood INTEGER,
            energy INTEGER,

            screen_time REAL,
            activity_minutes INTEGER,

            activity TEXT,

            connectedness INTEGER,
            social TEXT,

            hour INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    if not _column_exists(
        cursor,
        "checkins",
        "checkin_date",
    ):
        cursor.execute(
            """
            ALTER TABLE checkins
            ADD COLUMN checkin_date TEXT
            """
        )

        cursor.execute(
            """
            UPDATE checkins
            SET checkin_date = DATE(created_at)
            WHERE checkin_date IS NULL
            """
        )

    # --------------------------------------------------------
    # Entry type
    # --------------------------------------------------------

    if not _column_exists(
        cursor,
        "checkins",
        "entry_type",
    ):
        cursor.execute(
            """
            ALTER TABLE checkins
            ADD COLUMN entry_type TEXT
            DEFAULT 'live'
            """
        )

        cursor.execute(
            """
            UPDATE checkins
            SET entry_type = 'live'
            WHERE entry_type IS NULL
            """
        )

    # --------------------------------------------------------
    # Physical activity migration
    # --------------------------------------------------------

    if not _column_exists(
        cursor,
        "checkins",
        "activity_minutes",
    ):
        cursor.execute(
            """
            ALTER TABLE checkins
            ADD COLUMN activity_minutes INTEGER
            DEFAULT 0
            """
        )

        cursor.execute(
            """
            UPDATE checkins
            SET activity_minutes = 0
            WHERE activity_minutes IS NULL
            """
        )

    # --------------------------------------------------------
    # Connectedness migration
    # --------------------------------------------------------

    if not _column_exists(
        cursor,
        "checkins",
        "connectedness",
    ):
        cursor.execute(
            """
            ALTER TABLE checkins
            ADD COLUMN connectedness INTEGER
            """
        )

    # Historical approximation only.
    #
    # Low    -> 2
    # Medium -> 3
    # High   -> 4

    cursor.execute(
        """
        UPDATE checkins

        SET connectedness =
            CASE social
                WHEN 'Low' THEN 2
                WHEN 'Medium' THEN 3
                WHEN 'High' THEN 4
                ELSE 3
            END

        WHERE connectedness IS NULL
        """
    )

    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS
        idx_checkins_user_date
        ON checkins(user_id, checkin_date)
        """
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            checkin_id INTEGER,

            predicted_nudge TEXT,
            confidence REAL,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    # --------------------------------------------------------
    # Recommendation feedback
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prediction_id INTEGER,

            accepted INTEGER,
            completed INTEGER,
            rating INTEGER,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    if not _column_exists(
        cursor,
        "feedback",
        "makes_sense",
    ):
        cursor.execute(
            """
            ALTER TABLE feedback
            ADD COLUMN makes_sense INTEGER
            """
        )

    if not _column_exists(
        cursor,
        "feedback",
        "comment",
    ):
        cursor.execute(
            """
            ALTER TABLE feedback
            ADD COLUMN comment TEXT
            """
        )

    # --------------------------------------------------------
    # Usability feedback
    # --------------------------------------------------------

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS usability_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            user_id INTEGER,

            ease_of_use INTEGER,
            interface_clarity INTEGER,
            trust INTEGER,

            confusing TEXT,
            improvement TEXT,

            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    connection.commit()
    connection.close()


# ============================================================
# Participant code
# ============================================================

def _generate_participant_code() -> str:

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

    exists = (
        cursor.fetchone()
        is not None
    )

    connection.close()

    return exists


# ============================================================
# Participant creation / retrieval
# ============================================================

def create_authenticated_participant(
    auth_subject_hash: str,
    nickname: str,
    age: int,
) -> dict[str, Any]:

    existing = get_user_by_auth_hash(
        auth_subject_hash
    )

    if existing is not None:
        return existing

    while True:

        participant_code = (
            _generate_participant_code()
        )

        if not _participant_code_exists(
            participant_code
        ):
            break

    clean_nickname = (
        nickname.strip()
        or "Participant"
    )

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO users (
            name,
            nickname,
            age,
            participant_code,
            auth_subject_hash
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            clean_nickname,
            clean_nickname,
            int(age),
            participant_code,
            auth_subject_hash,
        ),
    )

    user_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return {
        "id": int(user_id),
        "nickname": clean_nickname,
        "age": int(age),
        "participant_code": participant_code,
        "auth_subject_hash": auth_subject_hash,
    }


def get_user(
    user_id: int,
) -> dict[str, Any] | None:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            name,
            nickname,
            age,
            participant_code,
            auth_subject_hash,
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


def get_user_by_auth_hash(
    auth_subject_hash: str,
) -> dict[str, Any] | None:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT
            id,
            name,
            nickname,
            age,
            participant_code,
            auth_subject_hash,
            created_at
        FROM users
        WHERE auth_subject_hash = ?
        LIMIT 1
        """,
        (auth_subject_hash,),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


# ============================================================
# Check-in retrieval
# ============================================================

def get_checkin_for_date(
    user_id: int,
    checkin_date: str,
) -> dict[str, Any] | None:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM checkins
        WHERE user_id = ?
          AND checkin_date = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (
            user_id,
            checkin_date,
        ),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def get_checkin_by_id(
    checkin_id: int,
    user_id: int | None = None,
) -> dict[str, Any] | None:

    connection = connect_db()
    cursor = connection.cursor()

    if user_id is None:

        cursor.execute(
            """
            SELECT *
            FROM checkins
            WHERE id = ?
            """,
            (checkin_id,),
        )

    else:

        cursor.execute(
            """
            SELECT *
            FROM checkins
            WHERE id = ?
              AND user_id = ?
            """,
            (
                checkin_id,
                user_id,
            ),
        )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


# ============================================================
# Save check-in
# ============================================================

def save_checkin(
    user_id: int,
    sleep: float,
    stress: int,
    mood: int,
    energy: int,
    screen_time: float,
    activity_minutes: int,
    activity: str,
    connectedness: int,
    hour: int,
    checkin_date: str,
    entry_type: str = "live",
) -> int:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM checkins
        WHERE user_id = ?
          AND checkin_date = ?
        LIMIT 1
        """,
        (
            user_id,
            checkin_date,
        ),
    )

    if cursor.fetchone() is not None:

        connection.close()

        raise DuplicateCheckinError(
            "A check-in already exists for this date."
        )

    connectedness = max(
        1,
        min(
            5,
            int(connectedness),
        ),
    )

    legacy_social = (
        connectedness_to_social(
            connectedness
        )
    )

    cursor.execute(
        """
        INSERT INTO checkins (
            user_id,
            sleep,
            stress,
            mood,
            energy,
            screen_time,
            activity_minutes,
            activity,
            connectedness,
            social,
            hour,
            checkin_date,
            entry_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            sleep,
            stress,
            mood,
            energy,
            screen_time,
            int(activity_minutes),
            activity,
            connectedness,
            legacy_social,
            hour,
            checkin_date,
            entry_type,
        ),
    )

    checkin_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return int(checkin_id)


# ============================================================
# Update check-in
# ============================================================

def update_checkin(
    checkin_id: int,
    user_id: int,
    sleep: float,
    stress: int,
    mood: int,
    energy: int,
    screen_time: float,
    activity_minutes: int,
    activity: str,
    connectedness: int,
    hour: int,
) -> bool:

    connectedness = max(
        1,
        min(
            5,
            int(connectedness),
        ),
    )

    legacy_social = (
        connectedness_to_social(
            connectedness
        )
    )

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        UPDATE checkins

        SET
            sleep = ?,
            stress = ?,
            mood = ?,
            energy = ?,
            screen_time = ?,
            activity_minutes = ?,
            activity = ?,
            connectedness = ?,
            social = ?,
            hour = ?

        WHERE id = ?
          AND user_id = ?
        """,
        (
            sleep,
            stress,
            mood,
            energy,
            screen_time,
            int(activity_minutes),
            activity,
            connectedness,
            legacy_social,
            hour,
            checkin_id,
            user_id,
        ),
    )

    updated = (
        cursor.rowcount > 0
    )

    connection.commit()
    connection.close()

    return updated


# ============================================================
# Delete check-in
# ============================================================

def delete_checkin(
    checkin_id: int,
    user_id: int,
) -> bool:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT id
        FROM checkins
        WHERE id = ?
          AND user_id = ?
        """,
        (
            checkin_id,
            user_id,
        ),
    )

    if cursor.fetchone() is None:

        connection.close()
        return False

    cursor.execute(
        """
        DELETE FROM feedback
        WHERE prediction_id IN (
            SELECT id
            FROM predictions
            WHERE checkin_id = ?
        )
        """,
        (checkin_id,),
    )

    cursor.execute(
        """
        DELETE FROM predictions
        WHERE checkin_id = ?
        """,
        (checkin_id,),
    )

    cursor.execute(
        """
        DELETE FROM checkins
        WHERE id = ?
          AND user_id = ?
        """,
        (
            checkin_id,
            user_id,
        ),
    )

    deleted = (
        cursor.rowcount > 0
    )

    connection.commit()
    connection.close()

    return deleted


# ============================================================
# Check-in history
# ============================================================

def get_recent_checkins(
    user_id: int,
    limit: int = 30,
):

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
            activity_minutes,
            activity,
            connectedness,
            social,
            hour,
            checkin_date,
            entry_type,
            created_at

        FROM checkins

        WHERE user_id = ?

        ORDER BY
            checkin_date DESC,
            created_at DESC

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


def replace_prediction(
    checkin_id: int,
    nudge: str,
    confidence: float,
) -> int:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM feedback
        WHERE prediction_id IN (
            SELECT id
            FROM predictions
            WHERE checkin_id = ?
        )
        """,
        (checkin_id,),
    )

    cursor.execute(
        """
        DELETE FROM predictions
        WHERE checkin_id = ?
        """,
        (checkin_id,),
    )

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
):

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
):

    rows = get_recent_predictions(
        user_id=user_id,
        limit=1,
    )

    if not rows:
        return None

    return rows[0]


# ============================================================
# Recommendation feedback
# ============================================================

def get_feedback_for_prediction(
    prediction_id: int,
) -> dict[str, Any] | None:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT *
        FROM feedback
        WHERE prediction_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (prediction_id,),
    )

    row = cursor.fetchone()

    connection.close()

    if row is None:
        return None

    return dict(row)


def save_feedback(
    prediction_id: int,
    accepted: int,
    completed: int,
    rating: int,
    makes_sense: int | None = None,
    comment: str | None = None,
) -> None:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        DELETE FROM feedback
        WHERE prediction_id = ?
        """,
        (prediction_id,),
    )

    cursor.execute(
        """
        INSERT INTO feedback (
            prediction_id,
            accepted,
            completed,
            rating,
            makes_sense,
            comment
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            accepted,
            completed,
            rating,
            makes_sense,
            comment,
        ),
    )

    connection.commit()
    connection.close()


# ============================================================
# Usability feedback
# ============================================================

def save_usability_feedback(
    user_id: int,
    ease_of_use: int,
    interface_clarity: int,
    trust: int,
    confusing: str | None,
    improvement: str | None,
) -> None:

    connection = connect_db()
    cursor = connection.cursor()

    cursor.execute(
        """
        INSERT INTO usability_feedback (
            user_id,
            ease_of_use,
            interface_clarity,
            trust,
            confusing,
            improvement
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            ease_of_use,
            interface_clarity,
            trust,
            confusing,
            improvement,
        ),
    )

    connection.commit()
    connection.close()