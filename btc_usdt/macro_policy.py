def btc_policy_from_macro(macro):
    regime = macro["regime"]

    if regime == "RISK_ON":
        return {
            "allow_long": True,
            "allow_short": False,
            "position_size": 1.0,
            "note": "Macro tailwind"
        }

    if regime == "NEUTRAL":
        return {
            "allow_long": True,
            "allow_short": True,
            "position_size": 0.4,
            "note": "Low conviction, range trading"
        }

    if regime == "RISK_OFF":
        return {
            "allow_long": False,
            "allow_short": True,
            "position_size": 0.6,
            "note": "Macro headwind"
        }
