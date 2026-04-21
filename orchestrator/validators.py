from __future__ import annotations

from typing import Any

from orchestrator.pillar_registry import get_enabled_pillar_keys
from orchestrator.schema import (
    BiasEnum,
    FreshnessEnum,
    StatusEnum,
)


REQUIRED_PILLAR_WRAPPER_KEYS = {
    "pillar",
    "timestamp_utc",
    "status",
    "confidence",
    "summary",
    "signal",
    "data_quality",
    "payload",
}

REQUIRED_SIGNAL_KEYS = {
    "bias",
    "strength",
    "state",
}

REQUIRED_DATA_QUALITY_KEYS = {
    "is_fresh",
    "is_complete",
    "freshness_status",
    "warnings",
}


def validate_required_keys(obj: dict[str, Any], required_keys: set[str]) -> list[str]:
    """Return list of missing keys."""
    return sorted(required_keys - set(obj.keys()))


def validate_confidence(confidence: Any) -> list[str]:
    """Validate confidence is numeric and between 0 and 1."""
    errors: list[str] = []

    if not isinstance(confidence, (int, float)):
        return ["confidence must be numeric"]

    if confidence < 0.0 or confidence > 1.0:
        errors.append("confidence must be between 0.0 and 1.0")

    return errors


def validate_signal(signal: Any) -> list[str]:
    """Validate signal object shape and enums."""
    errors: list[str] = []

    if not isinstance(signal, dict):
        return ["signal must be a dictionary"]

    missing = validate_required_keys(signal, REQUIRED_SIGNAL_KEYS)
    if missing:
        errors.append(f"signal missing keys: {missing}")
        return errors

    allowed_biases = {item.value for item in BiasEnum}
    if signal["bias"] not in allowed_biases:
        errors.append(f"signal.bias must be one of {sorted(allowed_biases)}")

    strength = signal["strength"]
    if not isinstance(strength, (int, float)):
        errors.append("signal.strength must be numeric")
    elif strength < 0.0 or strength > 1.0:
        errors.append("signal.strength must be between 0.0 and 1.0")

    if not isinstance(signal["state"], str):
        errors.append("signal.state must be a string")

    return errors


def validate_data_quality(data_quality: Any) -> list[str]:
    """Validate data_quality object shape and enums."""
    errors: list[str] = []

    if not isinstance(data_quality, dict):
        return ["data_quality must be a dictionary"]

    missing = validate_required_keys(data_quality, REQUIRED_DATA_QUALITY_KEYS)
    if missing:
        errors.append(f"data_quality missing keys: {missing}")
        return errors

    if not isinstance(data_quality["is_fresh"], bool):
        errors.append("data_quality.is_fresh must be a boolean")

    if not isinstance(data_quality["is_complete"], bool):
        errors.append("data_quality.is_complete must be a boolean")

    allowed_freshness = {item.value for item in FreshnessEnum}
    if data_quality["freshness_status"] not in allowed_freshness:
        errors.append(
            f"data_quality.freshness_status must be one of {sorted(allowed_freshness)}"
        )

    if not isinstance(data_quality["warnings"], list):
        errors.append("data_quality.warnings must be a list")

    return errors


def validate_status(status: Any) -> list[str]:
    """Validate status enum."""
    allowed_statuses = {item.value for item in StatusEnum}
    if status not in allowed_statuses:
        return [f"status must be one of {sorted(allowed_statuses)}"]
    return []


def validate_pillar_key(pillar_key: Any) -> list[str]:
    """Validate pillar key exists in registry."""
    if not isinstance(pillar_key, str):
        return ["pillar must be a string"]

    allowed_keys = set(get_enabled_pillar_keys())
    if pillar_key not in allowed_keys:
        return [f"pillar must be one of {sorted(allowed_keys)}"]

    return []


def validate_pillar_wrapper(wrapper: Any) -> list[str]:
    """Validate top-level pillar wrapper structure."""
    errors: list[str] = []

    if not isinstance(wrapper, dict):
        return ["pillar wrapper must be a dictionary"]

    missing = validate_required_keys(wrapper, REQUIRED_PILLAR_WRAPPER_KEYS)
    if missing:
        errors.append(f"pillar wrapper missing keys: {missing}")
        return errors

    errors.extend(validate_pillar_key(wrapper["pillar"]))
    errors.extend(validate_status(wrapper["status"]))
    errors.extend(validate_confidence(wrapper["confidence"]))

    if not isinstance(wrapper["summary"], str):
        errors.append("summary must be a string")

    if not isinstance(wrapper["timestamp_utc"], str):
        errors.append("timestamp_utc must be a string")

    if not isinstance(wrapper["payload"], dict):
        errors.append("payload must be a dictionary")

    errors.extend(validate_signal(wrapper["signal"]))
    errors.extend(validate_data_quality(wrapper["data_quality"]))

    return errors


def is_valid_pillar_wrapper(wrapper: Any) -> bool:
    """Return True if wrapper passes validation."""
    return len(validate_pillar_wrapper(wrapper)) == 0