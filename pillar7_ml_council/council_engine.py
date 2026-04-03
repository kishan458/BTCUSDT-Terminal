from __future__ import annotations

from typing import Any, Dict, Optional

from pillar7_ml_council.agents.professor_agent.infer import (
    infer_professor_from_shared_state,
    summarize_professor_inference,
)
from pillar7_ml_council.agents.retail_agent.infer import (
    infer_retail_from_shared_state,
    summarize_retail_inference,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_trade_context(shared_state: Dict[str, Any]) -> Dict[str, Any]:
    trade_restrictions = shared_state.get("trade_restrictions", {}) or {}
    risk_flags = shared_state.get("risk_flags", []) or []
    meta = shared_state.get("meta", {}) or {}

    allow_trade = trade_restrictions.get("allow_trade")
    event_uncertainty = meta.get("event_base_uncertainty")

    return {
        "allow_trade": allow_trade,
        "event_base_uncertainty": _safe_float(event_uncertainty, default=0.0),
        "risk_flags": risk_flags,
        "risk_flag_count": len(risk_flags),
        "trade_restrictions": trade_restrictions,
    }


def _compute_agreement(professor_result: Dict[str, Any], retail_result: Dict[str, Any]) -> Dict[str, Any]:
    professor_label = professor_result.get("predicted_label")
    retail_label = retail_result.get("predicted_label")

    professor_long = professor_label == "LONG"
    retail_long = retail_label == "CHASE_LONG"

    if professor_long and retail_long:
        return {
            "agreement_score": 1.0,
            "conflict_score": 0.0,
            "agreement_state": "BULLISH_ALIGNMENT",
        }

    if (not professor_long) and (not retail_long):
        return {
            "agreement_score": 0.75,
            "conflict_score": 0.25,
            "agreement_state": "INACTIVE_ALIGNMENT",
        }

    return {
        "agreement_score": 0.0,
        "conflict_score": 1.0,
        "agreement_state": "DISAGREEMENT",
    }


def _compute_council_bias(
    professor_result: Dict[str, Any],
    retail_result: Dict[str, Any],
    trade_context: Dict[str, Any],
) -> Dict[str, Any]:
    professor_prob = _safe_float(professor_result.get("calibrated_probability"))
    retail_prob = _safe_float(retail_result.get("calibrated_probability"))

    allow_trade = trade_context.get("allow_trade")
    event_uncertainty = _safe_float(trade_context.get("event_base_uncertainty"))
    risk_flag_count = int(trade_context.get("risk_flag_count", 0))

    if allow_trade is False:
        return {
            "council_bias": "NO_TRADE",
            "tradeability_score": 0.0,
            "dominant_agent": "RISK_VETO",
            "reason": "trade_restricted",
        }

    professor_weight = 0.70
    retail_weight = 0.30

    raw_score = (professor_prob * professor_weight) + (retail_prob * retail_weight)

    penalty = 0.0
    if event_uncertainty >= 0.70:
        penalty += 0.35
    elif event_uncertainty >= 0.50:
        penalty += 0.15

    penalty += min(risk_flag_count * 0.05, 0.20)

    tradeability_score = max(0.0, min(1.0, raw_score - penalty))

    if professor_prob >= retail_prob:
        dominant_agent = "PROFESSOR_AGENT"
    else:
        dominant_agent = "RETAIL_AGENT"

    if tradeability_score >= 0.60:
        council_bias = "LONG"
    elif tradeability_score >= 0.40:
        council_bias = "WATCH"
    else:
        council_bias = "NO_TRADE"

    return {
        "council_bias": council_bias,
        "tradeability_score": tradeability_score,
        "dominant_agent": dominant_agent,
        "reason": "weighted_agent_blend",
    }


def _build_reason_stack(
    professor_result: Dict[str, Any],
    retail_result: Dict[str, Any],
    agreement_block: Dict[str, Any],
    council_block: Dict[str, Any],
    trade_context: Dict[str, Any],
) -> list[str]:
    reasons = []

    reasons.append(f"professor={professor_result.get('predicted_label')}")
    reasons.append(f"retail={retail_result.get('predicted_label')}")
    reasons.append(f"agreement_state={agreement_block.get('agreement_state')}")
    reasons.append(f"council_bias={council_block.get('council_bias')}")

    if trade_context.get("allow_trade") is False:
        reasons.append("trade_restricted")

    if _safe_float(trade_context.get("event_base_uncertainty")) >= 0.70:
        reasons.append("event_uncertainty_high")

    if int(trade_context.get("risk_flag_count", 0)) > 0:
        reasons.append(f"risk_flags={int(trade_context.get('risk_flag_count', 0))}")

    return reasons


def run_council_engine(
    *,
    shared_state: Dict[str, Any],
    professor_artifact_path: str,
    retail_artifact_path: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    professor_result = infer_professor_from_shared_state(
        shared_state=shared_state,
        artifact_path=professor_artifact_path,
        threshold=threshold,
    )

    retail_result = infer_retail_from_shared_state(
        shared_state=shared_state,
        artifact_path=retail_artifact_path,
        threshold=threshold,
    )

    trade_context = _extract_trade_context(shared_state)
    agreement_block = _compute_agreement(professor_result, retail_result)
    council_block = _compute_council_bias(professor_result, retail_result, trade_context)

    reason_stack = _build_reason_stack(
        professor_result=professor_result,
        retail_result=retail_result,
        agreement_block=agreement_block,
        council_block=council_block,
        trade_context=trade_context,
    )

    return {
        "asset": shared_state.get("asset", "BTCUSDT"),
        "timestamp_utc": shared_state.get("timestamp_utc"),
        "shared_state_summary": {
            "regime_state": shared_state.get("regime_cycle", {}).get("state"),
            "event_state": shared_state.get("high_impact_event", {}).get("state"),
            "structure_state": shared_state.get("structure_liquidity", {}).get("state"),
            "sentiment_state": shared_state.get("sentiment", {}).get("state"),
            "memory_state": shared_state.get("market_memory", {}).get("state"),
            "candle_state": shared_state.get("candle", {}).get("state"),
        },
        "professor_agent": summarize_professor_inference(professor_result),
        "retail_agent": summarize_retail_inference(retail_result),
        "agreement": agreement_block,
        "council": council_block,
        "trade_context": trade_context,
        "reason_stack": reason_stack,
    }


def summarize_council_output(council_output: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "asset": council_output.get("asset"),
        "timestamp_utc": council_output.get("timestamp_utc"),
        "council_bias": council_output.get("council", {}).get("council_bias"),
        "tradeability_score": council_output.get("council", {}).get("tradeability_score"),
        "dominant_agent": council_output.get("council", {}).get("dominant_agent"),
        "agreement_state": council_output.get("agreement", {}).get("agreement_state"),
        "agreement_score": council_output.get("agreement", {}).get("agreement_score"),
        "conflict_score": council_output.get("agreement", {}).get("conflict_score"),
        "reason_stack": council_output.get("reason_stack"),
    }