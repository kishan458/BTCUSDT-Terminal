import sqlite3
import pandas as pd
from pillar6_high_impact_events.cpi_probability_engine import build_cpi_probabilities

DB_PATH = "database/btc_terminal.db"


def _historical_surprise_probs(event_name: str) -> dict:
    """
    Returns historical probability buckets for supported events.
    """

    if event_name == "US CPI":
        cpi = build_cpi_probabilities("2024", "2026")
        if cpi["available"]:
            return {
                "available": True,
                "probs": cpi["probabilities"],
                "n": cpi["historical_samples"]
            }
        return {"available": False, "probs": None, "n": cpi["historical_samples"]}

    # fallback: old DB-based method for future expansion
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        """
        SELECT actual, forecast
        FROM macro_events
        WHERE event_name = ?
          AND actual IS NOT NULL
          AND forecast IS NOT NULL
        """,
        conn,
        params=(event_name,),
    )
    conn.close()

    n = len(df)
    if n < 30:
        return {"available": False, "probs": None, "n": n}

    df["diff"] = df["actual"] - df["forecast"]

    band = df["diff"].std()
    if band == 0 or pd.isna(band):
        return {"available": False, "probs": None, "n": n}

    up = (df["diff"] > band * 0.25).mean()
    down = (df["diff"] < -band * 0.25).mean()
    inline = 1.0 - up - down

    return {
        "available": True,
        "probs": {
            "UP": float(up),
            "INLINE": float(inline),
            "DOWN": float(down)
        },
        "n": n
    }


def build_scenarios(event: dict) -> dict:
    event_name = event["event_name"]
    event_type = event.get("event_type") or "UNKNOWN"

    hist = _historical_surprise_probs(event_name)

    probs = hist["probs"] if hist["available"] else {"UP": None, "INLINE": None, "DOWN": None}

    scenarios = [
        {
            "case": "Upside surprise / Hawkish tilt",
            "probability": probs["UP"],
            "macro_interpretation": "Stronger-than-expected inflation/labor or tighter stance",
            "risk_bias": "RISK_OFF",
            "expected_btc_reaction": {
                "initial_move": "DOWN",
                "volatility": "HIGH",
                "follow_through": "Higher if liquidity is thin",
            },
            "strategy_adjustment": {
                "pre_event": "Flat / reduce exposure",
                "post_release": "Avoid first impulse; consider breakdown continuation only after confirmation",
            },
        },
        {
            "case": "In-line outcome / Neutral tone",
            "probability": probs["INLINE"],
            "macro_interpretation": "Meets expectations; market looks for guidance elsewhere",
            "risk_bias": "NEUTRAL",
            "expected_btc_reaction": {
                "initial_move": "WHIPSAW",
                "volatility": "MEDIUM",
                "follow_through": "Often fades toward range",
            },
            "strategy_adjustment": {
                "pre_event": "Flat",
                "post_release": "Trade range edges; wait for structure confirmation",
            },
        },
        {
            "case": "Downside surprise / Dovish tilt",
            "probability": probs["DOWN"],
            "macro_interpretation": "Weaker inflation/labor or easier stance",
            "risk_bias": "RISK_ON",
            "expected_btc_reaction": {
                "initial_move": "UP",
                "volatility": "HIGH",
                "follow_through": "Higher if risk assets bid",
            },
            "strategy_adjustment": {
                "pre_event": "No trade / reduce leverage",
                "post_release": "Look for long pullback entry after first expansion settles",
            },
        },
    ]

    method = "historical_actual_change_distribution" if hist["available"] else "insufficient_actual_forecast_data"

    return {
        "scenarios": scenarios,
        "probability_method": method,
        "historical_samples": hist["n"],
        "event_type": event_type,
    }