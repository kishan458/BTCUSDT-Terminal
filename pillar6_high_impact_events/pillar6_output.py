import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo

from pillar6_high_impact_events.ai_reasoning_engine import build_ai_reasoning
from pillar6_high_impact_events.trade_restriction_engine import build_trade_restrictions
from pillar6_high_impact_events.confidence_engine import build_confidence_score
from pillar6_high_impact_events.uncertainty_engine import compute_base_uncertainty
from pillar6_high_impact_events.scenario_engine import build_scenarios

from core.db import resolve_db_path

DB_PATH = str(resolve_db_path())
UTC = ZoneInfo("UTC")


def _get_next_event() -> dict | None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    now_utc = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")

    row = cur.execute(
        """
        SELECT event_uid, event_name, event_type, country, scheduled_time_utc, importance, state
        FROM macro_events
        WHERE scheduled_time_utc >= ?
        ORDER BY
          CASE importance WHEN 'HIGH' THEN 0 WHEN 'MEDIUM' THEN 1 ELSE 2 END,
          scheduled_time_utc ASC
        LIMIT 1
        """,
        (now_utc,),
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
    probs = [(s.get("risk_bias"), s.get("probability")) for s in scenarios]
    if any(p is None for _, p in probs):
        return "UNKNOWN"

    bucket = {}
    for bias, p in probs:
        bucket[bias] = bucket.get(bias, 0.0) + float(p)

    if not bucket:
        return "UNKNOWN"

    best = max(bucket.items(), key=lambda x: x[1])[0]
    return best


def build_pillar6_output() -> dict:
    event = _get_next_event()
    if not event:
        return {
            "event": None,
            "state": "NO_EVENTS",
            "base_uncertainty": 0.0,
            "confidence_score": 0.0,
            "trade_restrictions": {
                "allow_trade": False,
                "size_multiplier": 0.0,
                "leverage_cap": 0.0,
                "restriction_reason": "No upcoming macro events found in the database.",
            },
            "scenarios": [],
            "dominant_risk_skew": "UNKNOWN",
            "terminal_guidance": "No upcoming macro events found in the database.",
            "ai_reasoning": "No upcoming macro events found in the database.",
            "debug": {
                "db_path": DB_PATH,
                "scheduled_time_utc": None,
                "importance": None,
                "uncertainty_components": {},
                "probability_method": None,
                "historical_samples": 0,
                "confidence_components": {},
            },
        }

    unc = compute_base_uncertainty(event["scheduled_time_utc"])
    scen = build_scenarios(event)

    base_unc = unc["base_uncertainty"]
    dom = _dominant_skew(scen["scenarios"])

    conf = build_confidence_score(
        event=event,
        historical_samples=scen["historical_samples"],
        base_uncertainty=base_unc,
    )

    trade_restrictions = build_trade_restrictions(
        base_uncertainty=base_unc,
        confidence_score=conf["confidence_score"],
        event_state=event["state"],
    )

    ai_reasoning = build_ai_reasoning(
        {
            "event_name": event["event_name"],
            "state": event["state"],
            "base_uncertainty": base_unc,
            "confidence_score": conf["confidence_score"],
            "dominant_risk_skew": dom,
            "trade_restrictions": trade_restrictions,
            "scenarios": scen["scenarios"],
        }
    )

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
        "confidence_score": conf["confidence_score"],
        "trade_restrictions": trade_restrictions,
        "scenarios": scen["scenarios"],
        "dominant_risk_skew": dom,
        "terminal_guidance": guidance,
        "ai_reasoning": ai_reasoning,
        "debug": {
            "db_path": DB_PATH,
            "scheduled_time_utc": event["scheduled_time_utc"],
            "importance": event["importance"],
            "uncertainty_components": unc["components"],
            "probability_method": scen["probability_method"],
            "historical_samples": scen["historical_samples"],
            "confidence_components": conf["confidence_components"],
        },
    }