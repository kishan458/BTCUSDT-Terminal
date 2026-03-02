# pillar6_high_impact_events/processors/phase_classifier.py

def classify_phase(events, macro=None):
    """
    Classify the current market phase based on macro regime
    and proximity to high-impact events.

    Returns:
        dict with phase, confidence, and explanation
    """

    # Default assumptions
    phase = "RANGE"
    confidence = 0.4
    reasons = []

    # --- Event-based logic ---
    high_impact_events = [
        e for e in events
        if e.get("impact") == "HIGH"
    ]

    if high_impact_events:
        reasons.append("High-impact macro events present")

    # --- Macro regime logic (if provided) ---
    if macro:
        regime = macro.get("regime", "NEUTRAL")
        score = macro.get("score", 0)

        if regime == "RISK_ON":
            phase = "TREND"
            confidence = 0.7
            reasons.append("Risk-on macro regime")

        elif regime == "RISK_OFF":
            phase = "DEFENSIVE"
            confidence = 0.7
            reasons.append("Risk-off macro regime")

        else:  # NEUTRAL
            phase = "RANGE"
            confidence = 0.4
            reasons.append("Neutral macro regime")

        # Adjust confidence slightly using score
        confidence += min(abs(score) * 0.05, 0.2)

    return {
        "phase": phase,
        "confidence": round(min(confidence, 1.0), 2),
        "reasons": reasons
    }
