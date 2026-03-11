import sqlite3
import pandas as pd
from datetime import datetime, timezone

DB_PATH = "database/btc_terminal.db"


def _load_recent_price_data(limit: int = 48) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        f"""
        SELECT timestamp, open, high, low, close, volume
        FROM btc_price_1h
        ORDER BY timestamp DESC
        LIMIT {limit}
        """,
        conn,
    )

    conn.close()

    if df.empty:
        raise ValueError("No BTC price data found in btc_price_1h")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _get_current_session(now_utc: datetime) -> str:
    hour = now_utc.hour

    if 0 <= hour < 8:
        return "ASIA"
    elif 8 <= hour < 13:
        return "LONDON"
    elif 13 <= hour < 22:
        return "NEW_YORK"
    else:
        return "LATE_HOURS"


def _session_schedule(current_session: str) -> list[dict]:
    sessions = [
        {"name": "ASIA", "start_utc": "00:00", "end_utc": "08:00", "active": False},
        {"name": "LONDON", "start_utc": "08:00", "end_utc": "13:00", "active": False},
        {"name": "NEW_YORK", "start_utc": "13:00", "end_utc": "22:00", "active": False},
        {"name": "LATE_HOURS", "start_utc": "22:00", "end_utc": "00:00", "active": False},
    ]

    for s in sessions:
        if s["name"] == current_session:
            s["active"] = True

    return sessions


def _session_hour_bounds(session_name: str) -> tuple[int, int]:
    if session_name == "ASIA":
        return 0, 8
    if session_name == "LONDON":
        return 8, 13
    if session_name == "NEW_YORK":
        return 13, 22
    return 22, 24


def build_session_context() -> dict:
    df = _load_recent_price_data()

    now_utc = datetime.now(timezone.utc)
    current_session = _get_current_session(now_utc)
    start_hour, end_hour = _session_hour_bounds(current_session)

    df["dt"] = pd.to_datetime(df["timestamp"], utc=True)
    today = now_utc.date()

    if current_session == "LATE_HOURS":
        session_df = df[
            (df["dt"].dt.date == today) &
            (df["dt"].dt.hour >= start_hour)
        ]
    else:
        session_df = df[
            (df["dt"].dt.date == today) &
            (df["dt"].dt.hour >= start_hour) &
            (df["dt"].dt.hour < end_hour)
        ]

    if session_df.empty:
        latest = df.iloc[-1]
        session_high = float(latest["high"])
        session_low = float(latest["low"])
    else:
        session_high = float(session_df["high"].max())
        session_low = float(session_df["low"].min())

    return {
        "session_context": {
            "current_session": current_session,
            "sessions": _session_schedule(current_session),
            "session_high": session_high,
            "session_low": session_low,
        }
    }