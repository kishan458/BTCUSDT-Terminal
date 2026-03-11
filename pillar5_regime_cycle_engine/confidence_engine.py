def _clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def build_regime_confidence(
    metrics: dict,
    directional_regime: str,
    volatility_regime: str,
    market_state: str,
    cycle_phase: str,
) -> dict:
    ma = metrics["market_metrics"]["moving_average_structure"]
    momentum = metrics["market_metrics"]["momentum"]
    swing = metrics["market_metrics"]["swing_structure"]
    vol = metrics["market_metrics"]["volatility"]

    ma_order = ma["ma_order"]
    structure_state = swing["structure_state"]
    momentum_score = momentum["momentum_score"]
    vol_pctile = vol["volatility_percentile"]

    # 1) trend clarity
    trend_score = 0.5
    if directional_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
        trend_score = 0.9
    elif directional_regime in ["WEAK_UPTREND", "WEAK_DOWNTREND"]:
        trend_score = 0.72
    elif directional_regime == "RANGE":
        trend_score = 0.6

    # 2) structure clarity
    structure_score = 0.5
    if structure_state in ["HIGHER_HIGH_HIGHER_LOW", "LOWER_HIGH_LOWER_LOW"]:
        structure_score = 0.9
    elif structure_state in ["HIGHER_HIGH_LOWER_LOW", "LOWER_HIGH_HIGHER_LOW"]:
        structure_score = 0.6
    elif structure_state == "MIXED_STRUCTURE":
        structure_score = 0.45

    # 3) MA clarity
    ma_score = 0.5
    if ma_order in ["BULLISH_STACKED", "BEARISH_STACKED"]:
        ma_score = 0.9
    elif ma_order == "MIXED":
        ma_score = 0.5

    # 4) volatility clarity
    volatility_score = 0.65
    if volatility_regime in ["NORMAL", "EXPANDING"]:
        volatility_score = 0.8
    elif volatility_regime == "COMPRESSED":
        volatility_score = 0.7
    elif volatility_regime == "DISLOCATED":
        volatility_score = 0.58

    # 5) state confidence boost
    state_score = 0.6
    if market_state in ["TRENDING", "RANGING"]:
        state_score = 0.82
    elif market_state in ["BREAKOUT_TRANSITION", "BREAKDOWN_TRANSITION"]:
        state_score = 0.7
    elif market_state in ["MEAN_REVERTING", "CHOPPY"]:
        state_score = 0.55

    # 6) cycle clarity
    cycle_score = 0.65
    if cycle_phase in ["EXPANSION", "MARKDOWN"]:
        cycle_score = 0.82
    elif cycle_phase in ["RECOVERY", "DISTRIBUTION", "ACCUMULATION"]:
        cycle_score = 0.68
    elif cycle_phase == "EXHAUSTION":
        cycle_score = 0.6

    # 7) momentum boost
    momentum_boost = 0.0
    if momentum_score is not None:
        momentum_boost = min(abs(momentum_score) * 0.08, 0.08)

    confidence_score = (
        trend_score * 0.22
        + structure_score * 0.18
        + ma_score * 0.18
        + volatility_score * 0.14
        + state_score * 0.14
        + cycle_score * 0.14
        + momentum_boost
    )

    confidence_score = _clamp(confidence_score)

    return {
        "confidence_score": float(confidence_score),
        "confidence_debug": {
            "trend_score": float(trend_score),
            "structure_score": float(structure_score),
            "ma_score": float(ma_score),
            "volatility_score": float(volatility_score),
            "state_score": float(state_score),
            "cycle_score": float(cycle_score),
            "momentum_score": momentum_score,
            "momentum_boost": float(momentum_boost),
            "volatility_percentile": vol_pctile,
        },
    }