import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "btc_terminal.db"


def get_connection():
    """
    Returns a SQLite connection to the project database.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def initialize_database():
    """
    Creates required tables if they do not exist.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Macro Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS macro_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_name TEXT NOT NULL,
        event_type TEXT,
        country TEXT,
        datetime_utc TEXT NOT NULL,
        forecast REAL,
        actual REAL,
        previous REAL,
        impact_level TEXT,
        source TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # BTC Hourly Price Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS btc_price_1h (
        timestamp_utc TEXT PRIMARY KEY,
        open REAL NOT NULL,
        high REAL NOT NULL,
        low REAL NOT NULL,
        close REAL NOT NULL,
        volume REAL
    );
    """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")