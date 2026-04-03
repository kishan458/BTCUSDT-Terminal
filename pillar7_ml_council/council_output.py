from __future__ import annotations

from typing import Any, Dict

from pillar7_ml_council.council_engine import run_council_engine
from pillar7_ml_council.council_explainer import build_council_explanation
from pillar7_ml_council.disagreement_engine import compute_disagreement, summarize_disagreement


def build_council_output(
    *,
    shared_state: Dict[str, Any],
    professor_artifact_path: str,
    retail_artifact_path: str,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    council_core = run_council_engine(
        shared_state=shared_state,
        professor_artifact_path=professor_artifact_path,
        retail_artifact_path=retail_artifact_path,
        threshold=threshold,
    )

    professor_result = council_core.get("professor_agent", {})
    retail_result = council_core.get("retail_agent", {})

    disagreement = compute_disagreement(
        professor_result=professor_result,
        retail_result=retail_result,
    )

    explanation = build_council_explanation(council_core)

    return {
        "asset": council_core.get("asset", "BTCUSDT"),
        "timestamp_utc": council_core.get("timestamp_utc"),
        "shared_state_summary": council_core.get("shared_state_summary", {}),
        "agent_outputs": {
            "professor_agent": professor_result,
            "retail_agent": retail_result,
        },
        "disagreement": disagreement,
        "council": council_core.get("council", {}),
        "trade_context": council_core.get("trade_context", {}),
        "explanation": explanation,
        "reason_stack": council_core.get("reason_stack", []),
    }


def summarize_council_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    disagreement_summary = summarize_disagreement(payload.get("disagreement", {}))
    explanation = payload.get("explanation", {})

    return {
        "asset": payload.get("asset"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "council_bias": payload.get("council", {}).get("council_bias"),
        "tradeability_score": payload.get("council", {}).get("tradeability_score"),
        "dominant_agent": payload.get("council", {}).get("dominant_agent"),
        "alignment_class": disagreement_summary.get("alignment_class"),
        "agreement_score": disagreement_summary.get("agreement_score"),
        "conflict_score": disagreement_summary.get("conflict_score"),
        "confidence_dispersion": disagreement_summary.get("confidence_dispersion"),
        "headline": explanation.get("headline"),
        "reason_stack": payload.get("reason_stack", []),
    }