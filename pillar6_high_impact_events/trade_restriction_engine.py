def build_trade_restrictions(base_uncertainty: float, confidence_score: float, event_state: str) -> dict:
    """
    Returns trade restriction guidance for macro events.
    """

    if event_state == "LIVE_EVENT":
        return {
            "allow_trade": False,
            "size_multiplier": 0.0,
            "leverage_cap": 1.0,
            "restriction_reason": "Live macro release window. Avoid execution during peak event volatility."
        }

    if base_uncertainty >= 0.80:
        return {
            "allow_trade": False,
            "size_multiplier": 0.0,
            "leverage_cap": 1.0,
            "restriction_reason": "Extreme uncertainty regime. Stand down until post-event structure appears."
        }

    if base_uncertainty >= 0.65:
        return {
            "allow_trade": True,
            "size_multiplier": 0.30,
            "leverage_cap": 2.0,
            "restriction_reason": "High uncertainty regime. Reduce size and keep leverage tightly capped."
        }

    if confidence_score < 0.55:
        return {
            "allow_trade": True,
            "size_multiplier": 0.25,
            "leverage_cap": 1.5,
            "restriction_reason": "Low model confidence. Trade smaller until signal quality improves."
        }

    if base_uncertainty >= 0.50:
        return {
            "allow_trade": True,
            "size_multiplier": 0.50,
            "leverage_cap": 3.0,
            "restriction_reason": "Moderate uncertainty regime. Trade lighter than normal."
        }

    return {
        "allow_trade": True,
        "size_multiplier": 1.0,
        "leverage_cap": 5.0,
        "restriction_reason": "Conditions acceptable relative to current macro risk model."
    }