import os
import json
import sqlite3
from typing import Any

import requests

DB_PATH = "database/btc_terminal.db"
FMP_URL = "https://financialmodelingprep.com/stable/economic-calendar"


def _to_float_or_none(x: Any):
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"none", "null", "nan", "-"}:
        return None
    try:
        return float(s.replace("%", "").replace(",", ""))
    except Exception:
        return None


def _normalize_event_name(name: str) -> str:
    n = (name or "").strip().lower()

    if "employment situation" in n or "nonfarm" in n or "non-farm" in n or "nfp" in n:
        return "US Employment Situation (NFP)"

    if "consumer price index" in n or "cpi" in n:
        return "US CPI"

    return name.strip()


def fetch_calendar_rows() -> list[dict]:
    """
    Returns raw rows from FMP economics calendar.
    Safe behavior:
    - if FMP_API_KEY is missing, returns []
    - if request fails, returns []
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return []

    try:
        r = requests.get(
            FMP_URL,
            params={"apikey": api_key},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []


def update_event_outcomes() -> int:
    """
    Updates macro_events.actual / forecast / previous where event names match.
    Non-breaking:
    - if API unavailable or no key -> returns 0
    """
    rows = fetch_calendar_rows()
    if not rows:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    updates = 0

    for row in rows:
        raw_name = (
            row.get("event")
            or row.get("name")
            or row.get("title")
            or ""
        ).strip()

        if not raw_name:
            continue

        event_name = _normalize_event_name(raw_name)

        actual = _to_float_or_none(row.get("actual"))
        forecast = _to_float_or_none(row.get("forecast"))
        previous = _to_float_or_none(row.get("previous"))

        # keep raw provider payload for traceability
        raw_json = json.dumps(row)

        cur.execute(
            """
            UPDATE macro_events
            SET actual = COALESCE(?, actual),
                forecast = COALESCE(?, forecast),
                previous = COALESCE(?, previous),
                raw_json = CASE
                    WHEN raw_json IS NULL OR raw_json = '' THEN ?
                    ELSE raw_json
                END,
                updated_at = CURRENT_TIMESTAMP
            WHERE event_name = ?
            """,
            (actual, forecast, previous, raw_json, event_name),
        )

        updates += cur.rowcount

    conn.commit()
    conn.close()

    return updates