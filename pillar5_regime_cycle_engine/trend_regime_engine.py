def classify_directional_regime(metrics: dict) -> dict:
    ma = metrics["market_metrics"]["moving_average_structure"]
    momentum = metrics["market_metrics"]["momentum"]
    swing = metrics["market_metrics"]["swing_structure"]
    dist = metrics["market_metrics"]["distance_from_key_mas"]

    ma_order = ma["ma_order"]
    price_vs_ema20 = ma["price_vs_ema20"]
    price_vs_ema50 = ma["price_vs_ema50"]
    price_vs_ema200 = ma["price_vs_ema200"]

    ema20_slope = momentum["ema20_slope"]
    ema50_slope = momentum["ema50_slope"]
    momentum_score = momentum["momentum_score"]

    structure_state = swing["structure_state"]

    d20 = dist["distance_to_ema20_pct"]
    d50 = dist["distance_to_ema50_pct"]
    d200 = dist["distance_to_ema200_pct"]

    bullish_alignment = (
        ma_order == "BULLISH_STACKED"
        and price_vs_ema20 == "ABOVE"
        and price_vs_ema50 == "ABOVE"
        and price_vs_ema200 == "ABOVE"
        and structure_state == "HIGHER_HIGH_HIGHER_LOW"
        and ema20_slope is not None and ema20_slope > 0
        and ema50_slope is not None and ema50_slope > 0
    )

    bearish_alignment = (
        ma_order == "BEARISH_STACKED"
        and price_vs_ema20 == "BELOW"
        and price_vs_ema50 == "BELOW"
        and price_vs_ema200 == "BELOW"
        and structure_state == "LOWER_HIGH_LOWER_LOW"
        and ema20_slope is not None and ema20_slope < 0
        and ema50_slope is not None and ema50_slope < 0
    )

    strong_uptrend = (
        bullish_alignment
        and momentum_score is not None and momentum_score >= 0.45
        and d20 is not None and d20 > 0
        and d50 is not None and d50 > 0
    )

    weak_uptrend = (
        not strong_uptrend
        and bullish_alignment
    )

    strong_downtrend = (
        bearish_alignment
        and momentum_score is not None and momentum_score <= -0.45
        and d20 is not None and d20 < 0
        and d50 is not None and d50 < 0
    )

    weak_downtrend = (
        not strong_downtrend
        and bearish_alignment
    )

    if strong_uptrend:
        regime = "STRONG_UPTREND"
    elif weak_uptrend:
        regime = "WEAK_UPTREND"
    elif strong_downtrend:
        regime = "STRONG_DOWNTREND"
    elif weak_downtrend:
        regime = "WEAK_DOWNTREND"
    else:
        regime = "RANGE"

    return {
        "directional_regime": regime,
        "trend_debug": {
            "ma_order": ma_order,
            "structure_state": structure_state,
            "ema20_slope": ema20_slope,
            "ema50_slope": ema50_slope,
            "momentum_score": momentum_score,
            "distance_to_ema20_pct": d20,
            "distance_to_ema50_pct": d50,
            "distance_to_ema200_pct": d200,
        },
    }