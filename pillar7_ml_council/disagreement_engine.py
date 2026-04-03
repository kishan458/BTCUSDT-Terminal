from __future__ import annotations

from typing import Any, Dict, Optional


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_agent_direction(agent_label: Optional[str]) -> str:
    if agent_label is None:
        return "UNKNOWN"

    label = str(agent_label).upper()

    if label in {"LONG", "CHASE_LONG"}:
        return "LONG"

    if label in {"SHORT", "CHASE_SHORT"}:
        return "SHORT"

    if label in {"NO_TRADE", "NO_ACTION", "WATCH", "HOLD"}:
        return "INACTIVE"

    return "UNKNOWN"


def _extract_agent_probability(agent_result: Dict[str, Any]) -> float:
    return _safe_float(
        agent_result.get("calibrated_probability"),
        default=_safe_float(agent_result.get("raw_probability"), default=0.0),
    )


def compute_disagreement(
    professor_result: Dict[str, Any],
    retail_result: Dict[str, Any],
) -> Dict[str, Any]:
    professor_label = professor_result.get("predicted_label")
    retail_label = retail_result.get("predicted_label")

    professor_dir = _normalize_agent_direction(professor_label)
    retail_dir = _normalize_agent_direction(retail_label)

    professor_prob = _extract_agent_probability(professor_result)
    retail_prob = _extract_agent_probability(retail_result)

    confidence_dispersion = abs(professor_prob - retail_prob)

    if professor_dir == "UNKNOWN" or retail_dir == "UNKNOWN":
        alignment_class = "UNKNOWN_ALIGNMENT"
        agreement_score = 0.0
        conflict_score = 1.0
    elif professor_dir == retail_dir:
        if professor_dir == "LONG":
            alignment_class = "BULLISH_ALIGNMENT"
            agreement_score = 1.0
            conflict_score = 0.0
        elif professor_dir == "SHORT":
            alignment_class = "BEARISH_ALIGNMENT"
            agreement_score = 1.0
            conflict_score = 0.0
        else:
            alignment_class = "INACTIVE_ALIGNMENT"
            agreement_score = 0.75
            conflict_score = 0.25
    else:
        alignment_class = "DIRECTIONAL_CONFLICT"
        agreement_score = 0.0
        conflict_score = 1.0

    if professor_prob >= retail_prob:
        dominant_agent = "PROFESSOR_AGENT"
        dominant_probability = professor_prob
        minority_agent = "RETAIL_AGENT"
        minority_probability = retail_prob
    else:
        dominant_agent = "RETAIL_AGENT"
        dominant_probability = retail_prob
        minority_agent = "PROFESSOR_AGENT"
        minority_probability = professor_prob

    dominance_gap = dominant_probability - minority_probability

    return {
        "alignment_class": alignment_class,
        "agreement_score": agreement_score,
        "conflict_score": conflict_score,
        "confidence_dispersion": confidence_dispersion,
        "dominant_agent": dominant_agent,
        "minority_agent": minority_agent,
        "dominance_gap": dominance_gap,
        "professor_direction": professor_dir,
        "retail_direction": retail_dir,
        "professor_probability": professor_prob,
        "retail_probability": retail_prob,
    }


def summarize_disagreement(disagreement_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "alignment_class": disagreement_result.get("alignment_class"),
        "agreement_score": disagreement_result.get("agreement_score"),
        "conflict_score": disagreement_result.get("conflict_score"),
        "confidence_dispersion": disagreement_result.get("confidence_dispersion"),
        "dominant_agent": disagreement_result.get("dominant_agent"),
        "minority_agent": disagreement_result.get("minority_agent"),
        "dominance_gap": disagreement_result.get("dominance_gap"),
    }