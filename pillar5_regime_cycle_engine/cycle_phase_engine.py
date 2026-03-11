def classify_cycle_phase(
    metrics: dict,
    directional_regime: str,
    volatility_regime: str,
    market_state: str,
) -> dict:
    ma = metrics["market_metrics"]["moving_average_structure"]
    momentum = metrics["market_metrics"]["momentum"]
    swing = metrics["market_metrics"]["swing_structure"]
    dist = metrics["market_metrics"]["distance_from_key_mas"]
    returns = metrics["market_metrics"]["returns"]

    ma_order = ma["ma_order"]
    structure_state = swing["structure_state"]
    momentum_score = momentum["momentum_score"]

    d20 = dist["distance_to_ema20_pct"]
    d50 = dist["distance_to_ema50_pct"]
    d200 = dist["distance_to_ema200_pct"]

    ret_24 = returns["return_24bar"]
    ret_7d = returns["return_7d"]

    if (
        directional_regime == "STRONG_UPTREND"
        and market_state == "TRENDING"
        and ma_order == "BULLISH_STACKED"
        and structure_state == "HIGHER_HIGH_HIGHER_LOW"
        and momentum_score is not None
        and momentum_score >= 0.35
        and d20 is not None and d20 > 0
        and d50 is not None and d50 > 0
    ):
        if volatility_regime == "DISLOCATED" and ret_7d is not None and ret_7d > 0.08:
            phase = "EXHAUSTION"
        else:
            phase = "EXPANSION"

    elif (
        directional_regime == "WEAK_UPTREND"
        and market_state in ["TRENDING", "BREAKOUT_TRANSITION"]
        and d200 is not None
        and d200 < 0
    ):
        phase = "RECOVERY"

    elif (
        directional_regime == "RANGE"
        and market_state in ["RANGING", "MEAN_REVERTING"]
        and ma_order == "MIXED"
        and structure_state in ["MIXED_STRUCTURE", "LOWER_HIGH_HIGHER_LOW"]
    ):
        phase = "ACCUMULATION"

    elif (
        directional_regime == "RANGE"
        and volatility_regime in ["EXPANDING", "DISLOCATED"]
        and structure_state in ["HIGHER_HIGH_LOWER_LOW", "MIXED_STRUCTURE"]
    ):
        phase = "DISTRIBUTION"

    elif (
        directional_regime in ["WEAK_DOWNTREND", "STRONG_DOWNTREND"]
        and market_state in ["TRENDING", "BREAKDOWN_TRANSITION"]
    ):
        phase = "MARKDOWN"

    else:
        if ret_24 is not None and ret_24 > 0:
            phase = "RECOVERY"
        elif ret_24 is not None and ret_24 < 0:
            phase = "MARKDOWN"
        else:
            phase = "ACCUMULATION"

    return {
        "cycle_phase": phase,
        "cycle_phase_debug": {
            "ma_order": ma_order,
            "structure_state": structure_state,
            "momentum_score": momentum_score,
            "distance_to_ema20_pct": d20,
            "distance_to_ema50_pct": d50,
            "distance_to_ema200_pct": d200,
            "return_24bar": ret_24,
            "return_7d": ret_7d,
        },
    }