from __future__ import annotations

from typing import Any, Dict, List


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_alignment_text(agreement_state: str, agreement_score: float, conflict_score: float) -> str:
    if agreement_state == "BULLISH_ALIGNMENT":
        return (
            f"Professor and retail agents are aligned on the bullish side "
            f"(agreement={agreement_score:.2f}, conflict={conflict_score:.2f})."
        )

    if agreement_state == "INACTIVE_ALIGNMENT":
        return (
            f"Both agents are leaning inactive / non-committal "
            f"(agreement={agreement_score:.2f}, conflict={conflict_score:.2f})."
        )

    return (
        f"The agents are in disagreement "
        f"(agreement={agreement_score:.2f}, conflict={conflict_score:.2f})."
    )


def _build_tradeability_text(council_bias: str, tradeability_score: float) -> str:
    if council_bias == "LONG":
        return f"Council bias is LONG with a tradeability score of {tradeability_score:.2f}."
    if council_bias == "WATCH":
        return f"Council bias is WATCH with a tradeability score of {tradeability_score:.2f}."
    return f"Council bias is NO_TRADE with a tradeability score of {tradeability_score:.2f}."


def _build_risk_text(trade_context: Dict[str, Any]) -> str:
    allow_trade = trade_context.get("allow_trade")
    event_uncertainty = _safe_float(trade_context.get("event_base_uncertainty"))
    risk_flag_count = int(trade_context.get("risk_flag_count", 0))

    parts: List[str] = []

    if allow_trade is False:
        parts.append("Trading is currently restricted by the risk layer.")

    if event_uncertainty >= 0.70:
        parts.append(f"Event uncertainty is high at {event_uncertainty:.2f}.")
    elif event_uncertainty >= 0.50:
        parts.append(f"Event uncertainty is moderate at {event_uncertainty:.2f}.")
    else:
        parts.append(f"Event uncertainty is contained at {event_uncertainty:.2f}.")

    if risk_flag_count > 0:
        parts.append(f"There are {risk_flag_count} active risk flags.")
    else:
        parts.append("No active risk flags are present.")

    return " ".join(parts)


def _build_agent_text(council_output: Dict[str, Any]) -> str:
    professor = council_output.get("professor_agent", {})
    retail = council_output.get("retail_agent", {})
    dominant_agent = council_output.get("council", {}).get("dominant_agent", "UNKNOWN")

    professor_label = professor.get("predicted_label", "UNKNOWN")
    professor_prob = _safe_float(professor.get("calibrated_probability"))

    retail_label = retail.get("predicted_label", "UNKNOWN")
    retail_prob = _safe_float(retail.get("calibrated_probability"))

    return (
        f"Professor agent suggests {professor_label} "
        f"(calibrated_prob={professor_prob:.2f}). "
        f"Retail agent suggests {retail_label} "
        f"(calibrated_prob={retail_prob:.2f}). "
        f"Dominant agent in the blend is {dominant_agent}."
    )


def build_council_explanation(council_output: Dict[str, Any]) -> Dict[str, Any]:
    agreement = council_output.get("agreement", {})
    council = council_output.get("council", {})
    trade_context = council_output.get("trade_context", {})
    reason_stack = council_output.get("reason_stack", [])

    agreement_state = agreement.get("agreement_state", "UNKNOWN")
    agreement_score = _safe_float(agreement.get("agreement_score"))
    conflict_score = _safe_float(agreement.get("conflict_score"))

    council_bias = council.get("council_bias", "UNKNOWN")
    tradeability_score = _safe_float(council.get("tradeability_score"))

    alignment_text = _build_alignment_text(
        agreement_state=agreement_state,
        agreement_score=agreement_score,
        conflict_score=conflict_score,
    )

    tradeability_text = _build_tradeability_text(
        council_bias=council_bias,
        tradeability_score=tradeability_score,
    )

    risk_text = _build_risk_text(trade_context)
    agent_text = _build_agent_text(council_output)

    narrative = " ".join([
        alignment_text,
        tradeability_text,
        agent_text,
        risk_text,
    ])

    return {
        "headline": f"{council_bias} | {agreement_state}",
        "narrative": narrative,
        "alignment_text": alignment_text,
        "tradeability_text": tradeability_text,
        "agent_text": agent_text,
        "risk_text": risk_text,
        "reason_stack": reason_stack,
    }


def summarize_council_explanation(explanation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "headline": explanation.get("headline"),
        "narrative": explanation.get("narrative"),
        "reason_stack": explanation.get("reason_stack", []),
    }