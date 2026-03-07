def _clamp(x: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, x))


def build_confidence_score(event: dict, historical_samples: int, base_uncertainty: float) -> dict:
    """
    Confidence score for the event model.

    Higher confidence when:
    - we have more historical samples
    - uncertainty is not extreme
    - event type/modeling quality is stronger
    """

    event_name = event.get("event_name", "")
    event_type = event.get("event_type", "UNKNOWN")

    # 1) sample-size score
    if event_name == "FOMC Rate Decision":
        sample_score = 0.55
    else:
        sample_score = _clamp(historical_samples / 36.0)

    # 2) uncertainty score
    uncertainty_score = 1.0 - (base_uncertainty * 0.35)
    uncertainty_score = _clamp(uncertainty_score)

    # 3) event-model score
    if event_name == "US CPI":
        event_score = 0.90
        model_quality = "STRONG"
    elif event_name == "US Employment Situation (NFP)":
        event_score = 0.85
        model_quality = "STRONG"
    elif event_name == "FOMC Rate Decision":
        event_score = 0.72
        model_quality = "MODERATE"
    else:
        event_score = 0.60
        model_quality = "BASIC"

    # weighted blend
    confidence_score = (
        sample_score * 0.45
        + uncertainty_score * 0.30
        + event_score * 0.25
    )
    confidence_score = _clamp(confidence_score)

    return {
        "confidence_score": float(confidence_score),
        "confidence_components": {
            "sample_score": float(sample_score),
            "uncertainty_score": float(uncertainty_score),
            "event_score": float(event_score),
            "historical_samples": int(historical_samples),
            "event_type": event_type,
            "model_quality": model_quality,
            "event_name": event_name,
        },
    }