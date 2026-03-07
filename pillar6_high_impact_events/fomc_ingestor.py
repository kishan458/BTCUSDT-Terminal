import sqlite3
import json
import hashlib
from pillar6_high_impact_events.fomc_event_builder import build_fomc_events

DB_PATH = "database/btc_terminal.db"


def _event_uid(event_name, country, scheduled_time_utc):
    raw = f"fed_fomc|{event_name}|{country}|{scheduled_time_utc}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def ingest_fomc_events():
    events = build_fomc_events()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    inserted = 0

    for e in events:
        event_uid = _event_uid(
            e["event_name"],
            e["country"],
            e["scheduled_time_utc"]
        )

        cur.execute("""
        INSERT OR IGNORE INTO macro_events
        (
            event_uid,
            provider,
            provider_event_id,
            event_name,
            event_type,
            country,
            scheduled_time_utc,
            importance,
            actual,
            forecast,
            previous,
            raw_json,
            state
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event_uid,
            "fed_fomc",
            None,
            e["event_name"],
            e["event_type"],
            e["country"],
            e["scheduled_time_utc"],
            e["importance"],
            None,
            None,
            None,
            json.dumps(e["raw_json"]),
            "IDLE"
        ))

        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    conn.close()

    return inserted