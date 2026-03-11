def build_strategy_compatibility(
    directional_regime: str,
    volatility_regime: str,
    market_state: str,
    cycle_phase: str,
    confidence_score: float,
) -> dict:
    trend_following = "NEUTRAL"
    breakout_trading = "NEUTRAL"
    mean_reversion = "NEUTRAL"
    stand_down = False

    # Trend following
    if directional_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"] and market_state == "TRENDING":
        trend_following = "FAVORED"
    elif directional_regime in ["WEAK_UPTREND", "WEAK_DOWNTREND"]:
        trend_following = "MODERATELY_FAVORED"
    elif directional_regime == "RANGE":
        trend_following = "NOT_FAVORED"

    # Breakout trading
    if market_state in ["BREAKOUT_TRANSITION", "BREAKDOWN_TRANSITION"]:
        breakout_trading = "FAVORED"
    elif market_state == "TRENDING" and volatility_regime in ["NORMAL", "EXPANDING"]:
        breakout_trading = "MODERATELY_FAVORED"
    elif volatility_regime == "DISLOCATED":
        breakout_trading = "NOT_FAVORED"

    # Mean reversion
    if market_state in ["RANGING", "MEAN_REVERTING"] and directional_regime == "RANGE":
        mean_reversion = "FAVORED"
    elif market_state == "CHOPPY":
        mean_reversion = "MODERATELY_FAVORED"
    elif directional_regime in ["STRONG_UPTREND", "STRONG_DOWNTREND"]:
        mean_reversion = "NOT_FAVORED"

    # Stand down conditions
    if confidence_score < 0.55:
        stand_down = True

    if volatility_regime == "DISLOCATED" and market_state == "CHOPPY":
        stand_down = True

    if cycle_phase == "EXHAUSTION" and volatility_regime == "DISLOCATED":
        stand_down = True

    return {
        "strategy_compatibility": {
            "trend_following": trend_following,
            "breakout_trading": breakout_trading,
            "mean_reversion": mean_reversion,
            "stand_down": stand_down,
        }
    }