import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from macro.uncertainty_engine import compute_base_uncertainty
from macro.scenario_engine import build_scenarios

from core.db import resolve_db_path
DB_PATH = "database/btc_terminal.db"

UTC = ZoneInfo("UTC")


def _get_next_event() -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # pick next upcoming HIGH first, else next upcoming
    row = cur.execute(
        """
        SELECT event_uid, event_name, event_type, country, scheduled_time_utc, importance, state
        FROM macro_events
        WHERE scheduled_time_utc >= strftime('%Y-%m-%d %H:%M:%S','now')
        ORDER BY
          CASE importance WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END,
          scheduled_time_utc ASC
        LIMIT 1
        """
    ).fetchone()

    conn.close()

    if not row:
        return None

    return {
        "event_uid": row[0],
        "event_name": row[1],
        "event_type": row[2],
        "country": row[3],
        "scheduled_time_utc": row[4],
        "importance": row[5],
        "state": row[6],
    }


def _dominant_skew(scenarios: list[dict]) -> str:
    # only if probabilities exist
    probs = [(s.get("risk_bias"), s.get("probability")) for s in scenarios]
    if any(p is None for _, p in probs):
        return "UNKNOWN"

    # sum probabilities by bias
    bucket = {}
    for bias, p in probs:
        bucket[bias] = bucket.get(bias, 0.0) + float(p)

    # pick max
    best = max(bucket.items(), key=lambda x: x[1])[0]
    return best


def build_pillar6_output() -> dict:
    event = _get_next_event()
    if not event:
        return {
            "event": None,
            "state": "NO_EVENTS",
            "base_uncertainty": 0.0,
            "scenarios": [],
            "dominant_risk_skew": "UNKNOWN",
            "terminal_guidance": "No upcoming macro events found in the database.",
        }

    unc = compute_base_uncertainty(event["scheduled_time_utc"])
    scen = build_scenarios(event)

    base_unc = unc["base_uncertainty"]
    dom = _dominant_skew(scen["scenarios"])

    # guidance (data-backed via uncertainty + time-to-event)
    minutes_to_event = unc["components"]["minutes_to_event"]
    if minutes_to_event <= 0:
        guidance = "Event time has passed or is live. Expect volatility. Avoid impulse entries; wait for structure confirmation."
    elif base_unc >= 0.75:
        guidance = "High uncertainty. Prepare for expansion. Avoid positioning before release."
    elif base_unc >= 0.55:
        guidance = "Moderate uncertainty. Reduce leverage and wait for post-release confirmation."
    else:
        guidance = "Lower uncertainty (relative). Still respect event risk; keep sizing controlled."

    return {
        "event": event["event_name"],
        "state": event["state"],
        "base_uncertainty": base_unc,
        "scenarios": scen["scenarios"],
        "dominant_risk_skew": dom,
        "terminal_guidance": guidance,
        # optional debug you can keep or remove:
        "debug": {
            "scheduled_time_utc": event["scheduled_time_utc"],
            "importance": event["importance"],
            "uncertainty_components": unc["components"],
            "probability_method": scen["probability_method"],
            "historical_samples": scen["historical_samples"],
        },
    }