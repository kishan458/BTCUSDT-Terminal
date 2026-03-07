import sqlite3
import pandas as pd
from pillar6_high_impact_events.cpi_probability_engine import build_cpi_probabilities
from pillar6_high_impact_events.nfp_probability_engine import build_nfp_probabilities

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

    if event_name == "US Employment Situation (NFP)":
        nfp = build_nfp_probabilities("2024", "2026")
        if nfp["available"]:
            return {
                "available": True,
                "probs": nfp["probabilities"],
                "n": nfp["historical_samples"]
            }
        return {"available": False, "probs": None, "n": nfp["historical_samples"]}

    # fallback: DB-based method for future expansion
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


def _build_cpi_scenarios(probs: dict) -> list:
    return [
        {
            "case": "Hot CPI / Hawkish repricing",
            "probability": probs["UP"],
            "macro_interpretation": "Inflation prints above expectations, raising the chance of tighter financial conditions and pressuring risk assets.",
            "risk_bias": "RISK_OFF",
            "expected_btc_reaction": {
                "initial_move": "DOWN",
                "volatility": "HIGH",
                "follow_through": "Selling can extend if yields and dollar strength accelerate after the release",
            },
            "strategy_adjustment": {
                "pre_event": "Reduce exposure and avoid aggressive pre-release positioning",
                "post_release": "Avoid chasing first impulse; consider continuation only after structure confirms weakness",
            },
        },
        {
            "case": "In-line CPI / Mixed macro read",
            "probability": probs["INLINE"],
            "macro_interpretation": "Inflation broadly matches expectations, so traders shift attention to positioning, liquidity, and broader risk sentiment.",
            "risk_bias": "NEUTRAL",
            "expected_btc_reaction": {
                "initial_move": "WHIPSAW",
                "volatility": "MEDIUM",
                "follow_through": "Initial move often fades unless broader market correlations strengthen",
            },
            "strategy_adjustment": {
                "pre_event": "Stay light and let the market reveal direction",
                "post_release": "Prefer range reactions and confirmation entries rather than breakout chasing",
            },
        },
        {
            "case": "Soft CPI / Dovish repricing",
            "probability": probs["DOWN"],
            "macro_interpretation": "Inflation comes in below expectations, easing policy pressure and supporting a more constructive tone for risk assets.",
            "risk_bias": "RISK_ON",
            "expected_btc_reaction": {
                "initial_move": "UP",
                "volatility": "HIGH",
                "follow_through": "Upside can persist if equities and broader risk assets also catch a bid",
            },
            "strategy_adjustment": {
                "pre_event": "Do not over-position before release",
                "post_release": "Watch for pullback long setups after the first expansion leg stabilizes",
            },
        },
    ]


def _build_nfp_scenarios(probs: dict) -> list:
    return [
        {
            "case": "Strong NFP / Hawkish labor signal",
            "probability": probs["UP"],
            "macro_interpretation": "Labor data beats expectations, reinforcing economic strength but also increasing the chance of tighter policy expectations.",
            "risk_bias": "RISK_OFF",
            "expected_btc_reaction": {
                "initial_move": "DOWN",
                "volatility": "HIGH",
                "follow_through": "BTC can remain heavy if rates markets reprice toward higher-for-longer conditions",
            },
            "strategy_adjustment": {
                "pre_event": "Keep positioning defensive into the release",
                "post_release": "Avoid reacting to the first spike; look for confirmation before following downside",
            },
        },
        {
            "case": "In-line NFP / Balanced labor read",
            "probability": probs["INLINE"],
            "macro_interpretation": "Employment data lands near expectations, limiting macro repricing and leaving BTC more sensitive to technical flows.",
            "risk_bias": "NEUTRAL",
            "expected_btc_reaction": {
                "initial_move": "WHIPSAW",
                "volatility": "MEDIUM",
                "follow_through": "Follow-through is often weak unless other risk markets break from range",
            },
            "strategy_adjustment": {
                "pre_event": "Stay patient and keep risk small",
                "post_release": "Let range structure develop before entering",
            },
        },
        {
            "case": "Weak NFP / Dovish labor signal",
            "probability": probs["DOWN"],
            "macro_interpretation": "Employment misses expectations, reducing growth momentum and increasing the odds of easier policy expectations.",
            "risk_bias": "RISK_ON",
            "expected_btc_reaction": {
                "initial_move": "UP",
                "volatility": "HIGH",
                "follow_through": "Upside can continue if macro traders interpret the release as supportive for liquidity conditions",
            },
            "strategy_adjustment": {
                "pre_event": "Avoid oversized exposure before the number",
                "post_release": "Look for long continuation only after volatility compresses and trend confirms",
            },
        },
    ]


def _build_fomc_scenarios() -> list:
    return [
        {
            "case": "Hawkish hold / Higher-for-longer signal",
            "probability": None,
            "macro_interpretation": "The Fed keeps policy tight or signals reluctance to ease, which can pressure liquidity-sensitive assets.",
            "risk_bias": "RISK_OFF",
            "expected_btc_reaction": {
                "initial_move": "DOWN",
                "volatility": "HIGH",
                "follow_through": "Downside can extend if yields rise and the dollar strengthens after the statement or press conference",
            },
            "strategy_adjustment": {
                "pre_event": "Stay underexposed into the decision and avoid aggressive leverage",
                "post_release": "Wait for the first reaction and only engage if macro pressure confirms through price structure",
            },
        },
        {
            "case": "Neutral hold / Limited policy surprise",
            "probability": None,
            "macro_interpretation": "The Fed broadly matches market expectations, leaving BTC to react more to positioning, tone, and Q&A nuance.",
            "risk_bias": "NEUTRAL",
            "expected_btc_reaction": {
                "initial_move": "WHIPSAW",
                "volatility": "HIGH",
                "follow_through": "Initial volatility may fade unless Powell guidance materially shifts rate expectations",
            },
            "strategy_adjustment": {
                "pre_event": "Stay light and avoid predicting the first move",
                "post_release": "Trade only after the statement and press conference direction become clearer",
            },
        },
        {
            "case": "Dovish shift / Easing signal",
            "probability": None,
            "macro_interpretation": "The Fed signals softer policy expectations or a more supportive liquidity backdrop, which can help risk assets reprice higher.",
            "risk_bias": "RISK_ON",
            "expected_btc_reaction": {
                "initial_move": "UP",
                "volatility": "HIGH",
                "follow_through": "Upside can persist if markets price in easier conditions and correlated risk assets break higher",
            },
            "strategy_adjustment": {
                "pre_event": "Avoid overcommitting before the announcement",
                "post_release": "Look for pullback continuation entries only after the initial volatility settles",
            },
        },
    ]


def _build_generic_scenarios(probs: dict) -> list:
    return [
        {
            "case": "Upside surprise / Hawkish tilt",
            "probability": probs["UP"],
            "macro_interpretation": "Stronger-than-expected macro outcome or tighter stance",
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
            "macro_interpretation": "Outcome broadly meets expectations",
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
            "macro_interpretation": "Weaker-than-expected macro outcome or easier stance",
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


def build_scenarios(event: dict) -> dict:
    event_name = event["event_name"]
    event_type = event.get("event_type") or "UNKNOWN"

    hist = _historical_surprise_probs(event_name)
    probs = hist["probs"] if hist["available"] else {"UP": None, "INLINE": None, "DOWN": None}

    if event_name == "US CPI":
        scenarios = _build_cpi_scenarios(probs)
    elif event_name == "US Employment Situation (NFP)":
        scenarios = _build_nfp_scenarios(probs)
    elif event_name == "FOMC Rate Decision":
        scenarios = _build_fomc_scenarios()
    else:
        scenarios = _build_generic_scenarios(probs)

    method = "historical_actual_change_distribution" if hist["available"] else "insufficient_actual_forecast_data"

    if event_name == "FOMC Rate Decision":
        method = "qualitative_fomc_scenario_framework"

    return {
        "scenarios": scenarios,
        "probability_method": method,
        "historical_samples": hist["n"],
        "event_type": event_type,
    }