def build_risk_flags(
    metrics: dict,
    directional_regime: str,
    volatility_regime: str,
    market_state: str,
    cycle_phase: str,
) -> dict:

    flags = []

    momentum = metrics["market_metrics"]["momentum"]
    vol = metrics["market_metrics"]["volatility"]
    dist = metrics["market_metrics"]["distance_from_key_mas"]

    momentum_score = momentum["momentum_score"]
    vol_pct = vol["volatility_percentile"]

    d20 = dist["distance_to_ema20_pct"]
    d50 = dist["distance_to_ema50_pct"]

    # Late trend acceleration
    if cycle_phase == "EXPANSION" and volatility_regime == "DISLOCATED":
        flags.append("Late-trend acceleration risk")

    # Volatility shock
    if vol_pct is not None and vol_pct > 0.95:
        flags.append("Volatility shock risk")

    # Breakout failure
    if market_state == "TRENDING" and volatility_regime == "DISLOCATED":
        flags.append("Breakout failure risk")

    # Trend exhaustion
    if momentum_score is not None and abs(momentum_score) > 0.7:
        flags.append("Trend exhaustion risk")

    # Liquidity vacuum
    if d20 is not None and abs(d20) > 0.04:
        flags.append("Liquidity vacuum risk")

    return {
        "risk_flags": flags
    }