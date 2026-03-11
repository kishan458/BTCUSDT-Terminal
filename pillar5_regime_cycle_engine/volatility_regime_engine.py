def classify_volatility_regime(metrics: dict) -> dict:
    vol = metrics["market_metrics"]["volatility"]

    atr_pct = vol["atr_pct"]
    realized_vol = vol["realized_vol"]
    vol_pctile = vol["volatility_percentile"]

    if atr_pct is None or vol_pctile is None:
        regime = "NORMAL"
    elif vol_pctile >= 0.95 or (atr_pct is not None and atr_pct >= 0.02):
        regime = "DISLOCATED"
    elif vol_pctile >= 0.70 or (atr_pct is not None and atr_pct >= 0.012):
        regime = "EXPANDING"
    elif vol_pctile <= 0.25 or (atr_pct is not None and atr_pct <= 0.006):
        regime = "COMPRESSED"
    else:
        regime = "NORMAL"

    return {
        "volatility_regime": regime,
        "volatility_debug": {
            "atr_pct": atr_pct,
            "realized_vol": realized_vol,
            "volatility_percentile": vol_pctile,
        },
    }