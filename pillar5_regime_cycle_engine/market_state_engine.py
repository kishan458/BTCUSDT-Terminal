def classify_market_state(metrics: dict, directional_regime: str, volatility_regime: str) -> dict:
    momentum = metrics["market_metrics"]["momentum"]
    comp = metrics["market_metrics"]["compression_expansion"]
    swing = metrics["market_metrics"]["swing_structure"]
    ma = metrics["market_metrics"]["moving_average_structure"]

    momentum_score = momentum["momentum_score"]
    breakout_pressure = comp["breakout_pressure_score"]
    compression_score = comp["range_compression_score"]
    expansion_score = comp["expansion_score"]
    structure_state = swing["structure_state"]
    ma_order = ma["ma_order"]

    if directional_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
        state = "TRENDING"

    elif (
        compression_score is not None
        and compression_score >= 0.45
        and breakout_pressure is not None
        and breakout_pressure >= 0.45
    ):
        if directional_regime in ["WEAK_UPTREND", "STRONG_UPTREND"]:
            state = "BREAKOUT_TRANSITION"
        elif directional_regime in ["WEAK_DOWNTREND", "STRONG_DOWNTREND"]:
            state = "BREAKDOWN_TRANSITION"
        else:
            state = "CHOPPY"

    elif (
        directional_regime == "RANGE"
        and structure_state in ["MIXED_STRUCTURE", "LOWER_HIGH_HIGHER_LOW", "HIGHER_HIGH_LOWER_LOW"]
    ):
        state = "RANGING"

    elif (
        volatility_regime in ["NORMAL", "COMPRESSED"]
        and directional_regime == "RANGE"
        and momentum_score is not None
        and abs(momentum_score) < 0.15
    ):
        state = "MEAN_REVERTING"

    else:
        state = "CHOPPY"

    return {
        "market_state": state,
        "market_state_debug": {
            "momentum_score": momentum_score,
            "breakout_pressure_score": breakout_pressure,
            "range_compression_score": compression_score,
            "expansion_score": expansion_score,
            "structure_state": structure_state,
            "ma_order": ma_order,
        },
    }