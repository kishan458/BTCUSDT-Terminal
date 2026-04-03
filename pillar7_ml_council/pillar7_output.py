from __future__ import annotations

from typing import Any, Dict

from pillar7_ml_council.council_output import build_council_output, summarize_council_payload
from pillar7_ml_council.ai_overview_engine import build_ai_overview, summarize_ai_overview


def build_pillar7_output(
    *,
    shared_state: Dict[str, Any],
    professor_artifact_path: str,
    retail_artifact_path: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    council_payload = build_council_output(
        shared_state=shared_state,
        professor_artifact_path=professor_artifact_path,
        retail_artifact_path=retail_artifact_path,
        threshold=threshold,
    )

    ai_overview = build_ai_overview(council_payload)

    return {
        "asset": council_payload.get("asset", "BTCUSDT"),
        "timestamp_utc": council_payload.get("timestamp_utc"),
        "shared_state_summary": council_payload.get("shared_state_summary", {}),
        "agent_outputs": council_payload.get("agent_outputs", {}),
        "disagreement": council_payload.get("disagreement", {}),
        "council": council_payload.get("council", {}),
        "trade_context": council_payload.get("trade_context", {}),
        "explanation": council_payload.get("explanation", {}),
        "ai_overview": ai_overview,
        "reason_stack": council_payload.get("reason_stack", []),
    }


def summarize_pillar7_output(payload: Dict[str, Any]) -> Dict[str, Any]:
    council_summary = summarize_council_payload(payload)
    ai_summary = summarize_ai_overview(payload.get("ai_overview", {}))

    return {
        "asset": payload.get("asset"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "council_bias": council_summary.get("council_bias"),
        "tradeability_score": council_summary.get("tradeability_score"),
        "dominant_agent": council_summary.get("dominant_agent"),
        "alignment_class": council_summary.get("alignment_class"),
        "agreement_score": council_summary.get("agreement_score"),
        "conflict_score": council_summary.get("conflict_score"),
        "headline": ai_summary.get("headline"),
        "ai_source": ai_summary.get("source"),
        "reason_stack": payload.get("reason_stack", []),
    }