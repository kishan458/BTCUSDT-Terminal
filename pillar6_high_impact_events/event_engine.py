import sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from core.db import resolve_db_path
DB_PATH = "database/btc_terminal.db"

PRE_EVENT_WINDOW_HOURS = 24
POST_EVENT_WINDOW_HOURS = 4

UTC = ZoneInfo("UTC")

def update_event_states():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    now = datetime.now(tz=UTC)

    rows = cur.execute("SELECT event_uid, scheduled_time_utc FROM macro_events").fetchall()

    for event_uid, scheduled_time_utc in rows:
        event_time = datetime.strptime(scheduled_time_utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)

        if now < event_time - timedelta(hours=PRE_EVENT_WINDOW_HOURS):
            state = "IDLE"
        elif event_time - timedelta(hours=PRE_EVENT_WINDOW_HOURS) <= now < event_time:
            state = "PRE_EVENT"
        elif event_time <= now < event_time + timedelta(hours=POST_EVENT_WINDOW_HOURS):
            state = "LIVE"
        elif event_time + timedelta(hours=POST_EVENT_WINDOW_HOURS) <= now < event_time + timedelta(hours=24):
            state = "POST_EVENT"
        else:
            state = "COOL_OFF"

        cur.execute("UPDATE macro_events SET state=?, updated_at=CURRENT_TIMESTAMP WHERE event_uid=?",
                    (state, event_uid))

    conn.commit()
    conn.close()

def get_active_events():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT event_uid, event_name, scheduled_time_utc, state, importance
        FROM macro_events
        WHERE state IN ('PRE_EVENT','LIVE','POST_EVENT')
        ORDER BY scheduled_time_utc ASC
    """).fetchall()
    conn.close()
    return rows