import sqlite3
from typing import List, Dict
from macro.providers.base_provider import BaseEventProvider


DB_PATH = "database/btc_terminal.db"

UPSERT_SQL = """
INSERT INTO macro_events (
  event_uid, provider, provider_event_id, event_name, event_type, country,
  scheduled_time_utc, importance, actual, forecast, previous, raw_json, state, updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'IDLE', CURRENT_TIMESTAMP)
ON CONFLICT(event_uid) DO UPDATE SET
  provider=excluded.provider,
  provider_event_id=excluded.provider_event_id,
  event_name=excluded.event_name,
  event_type=excluded.event_type,
  country=excluded.country,
  scheduled_time_utc=excluded.scheduled_time_utc,
  importance=excluded.importance,
  actual=excluded.actual,
  forecast=excluded.forecast,
  previous=excluded.previous,
  raw_json=excluded.raw_json,
  updated_at=CURRENT_TIMESTAMP;
"""

def upsert_events(events: List[Dict]) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    n = 0

    for ev in events:
        cur.execute(UPSERT_SQL, (
            ev["event_uid"],
            ev["provider"],
            ev.get("provider_event_id"),
            ev["event_name"],
            ev.get("event_type"),
            ev.get("country"),
            ev["scheduled_time_utc"],
            ev.get("importance"),
            ev.get("actual"),
            ev.get("forecast"),
            ev.get("previous"),
            ev["raw_json"],
        ))
        n += 1

    conn.commit()
    conn.close()
    return n

def ingest(provider: BaseEventProvider) -> int:
    """
    Fetches events from provider (LEGIT source), normalizes inside provider,
    and upserts them into macro_events.
    """
    events = provider.fetch_events()
    return upsert_events(events)