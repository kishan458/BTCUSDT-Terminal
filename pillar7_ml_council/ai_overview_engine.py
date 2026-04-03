from __future__ import annotations

import os
from typing import Any


def _fallback_overview(payload: dict[str, Any]) -> dict[str, Any]:
    shared = payload.get("shared_state_summary", {}) or {}
    council = payload.get("council", {}) or {}
    disagreement = payload.get("disagreement", {}) or {}
    trade_context = payload.get("trade_context", {}) or {}
    agents = payload.get("agent_outputs", {}) or {}

    professor = agents.get("professor_agent", {}) or {}
    retail = agents.get("retail_agent", {}) or {}

    regime_state = shared.get("regime_state", "UNKNOWN")
    event_state = shared.get("event_state", "UNKNOWN")
    structure_state = shared.get("structure_state", "UNKNOWN")
    sentiment_state = shared.get("sentiment_state", "UNKNOWN")
    memory_state = shared.get("memory_state", "UNKNOWN")
    candle_state = shared.get("candle_state", "UNKNOWN")

    council_bias = council.get("council_bias", "UNKNOWN")
    tradeability_score = council.get("tradeability_score", "UNKNOWN")
    dominant_agent = council.get("dominant_agent", "UNKNOWN")

    alignment_class = disagreement.get("alignment_class", "UNKNOWN_ALIGNMENT")
    agreement_score = disagreement.get("agreement_score", "UNKNOWN")
    conflict_score = disagreement.get("conflict_score", "UNKNOWN")
    dominance_gap = disagreement.get("dominance_gap", "UNKNOWN")

    professor_label = professor.get("predicted_label", "UNKNOWN")
    professor_prob = professor.get("calibrated_probability", "UNKNOWN")
    retail_label = retail.get("predicted_label", "UNKNOWN")
    retail_prob = retail.get("calibrated_probability", "UNKNOWN")

    allow_trade = trade_context.get("allow_trade", True)
    event_uncertainty = trade_context.get("event_base_uncertainty", "UNKNOWN")
    risk_flag_count = trade_context.get("risk_flag_count", 0)
    risk_flags = trade_context.get("risk_flags", []) or []

    risk_text = ", ".join(risk_flags) if risk_flags else "No active risk flags"

    overview = (
        f"BTCUSDT is currently sitting in a {regime_state} regime with {structure_state} structure, "
        f"{sentiment_state} sentiment, {memory_state} historical bias, and {candle_state} candle context. "
        f"Event state is {event_state}, with event uncertainty at {event_uncertainty}. "
        f"The professor agent is leaning {professor_label} ({professor_prob}), while the retail agent is leaning "
        f"{retail_label} ({retail_prob}). Alignment is {alignment_class} with agreement score {agreement_score}, "
        f"conflict score {conflict_score}, and dominance gap {dominance_gap}. "
        f"Council bias is {council_bias} with tradeability score {tradeability_score}, and the dominant agent is "
        f"{dominant_agent}. Trading allowed = {allow_trade}. Risk context: {risk_text}. "
        f"Overall read: respect the council bias, but do not ignore uncertainty, crowd behavior, or future event shifts."
    )

    return {
        "headline": f"{council_bias} | {alignment_class}",
        "overview": overview,
        "source": "fallback",
    }


def build_ai_overview(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Gemini-powered Pillar 7 council commentary.

    Returns fallback overview if:
    - GEMINI_API_KEY is missing
    - google-genai is not installed
    - Gemini request fails
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return _fallback_overview(payload)

    try:
        from google import genai
    except Exception:
        return _fallback_overview(payload)

    try:
        client = genai.Client(api_key=api_key)

        shared = payload.get("shared_state_summary", {}) or {}
        council = payload.get("council", {}) or {}
        disagreement = payload.get("disagreement", {}) or {}
        trade_context = payload.get("trade_context", {}) or {}
        agents = payload.get("agent_outputs", {}) or {}
        explanation = payload.get("explanation", {}) or {}
        reason_stack = payload.get("reason_stack", []) or []

        prompt = f"""
You are a professional BTC quantitative strategist writing a council note for an advanced BTCUSDT intelligence terminal.

Write exactly 3 short paragraphs:
1) State diagnosis
2) Agent disagreement / alignment interpretation
3) Tactical implication and risk

Rules:
- Sound like a serious institutional market strategist.
- No bullet points.
- No hype.
- No generic chatbot tone.
- Do not just restate fields mechanically.
- Interpret the interaction between professor agent and retail agent.
- Mention whether the move looks structurally healthy, fragile, crowded, or risk-suppressed.
- If retail dominates, say what that implies.
- If agreement is high, explain whether that is constructive or potentially late-stage.
- Aim for roughly 170 to 240 words total.

Structured input:
asset: {payload.get("asset")}
timestamp_utc: {payload.get("timestamp_utc")}

shared_state_summary: {shared}

professor_agent: {agents.get("professor_agent", {})}
retail_agent: {agents.get("retail_agent", {})}

disagreement: {disagreement}
council: {council}
trade_context: {trade_context}

existing_headline: {explanation.get("headline")}
reason_stack: {reason_stack}
""".strip()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text and text.strip():
            return {
                "headline": explanation.get("headline", "COUNCIL OVERVIEW"),
                "overview": text.strip(),
                "source": "gemini",
            }

        return _fallback_overview(payload)

    except Exception:
        return _fallback_overview(payload)


def summarize_ai_overview(overview_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline": overview_payload.get("headline"),
        "overview": overview_payload.get("overview"),
        "source": overview_payload.get("source"),
    }