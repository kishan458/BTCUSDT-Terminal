from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _fallback_overview(payload: dict[str, Any]) -> dict[str, Any]:
    shared      = payload.get("shared_state_summary", {}) or {}
    council     = payload.get("council", {}) or {}
    disagreement = payload.get("disagreement", {}) or {}
    trade_context = payload.get("trade_context", {}) or {}
    agents      = payload.get("agent_outputs", {}) or {}
    explanation = payload.get("explanation", {}) or {}

    professor = agents.get("professor_agent", {}) or {}
    retail    = agents.get("retail_agent", {}) or {}

    regime_state    = shared.get("regime_state", "UNKNOWN")
    event_state     = shared.get("event_state", "UNKNOWN")
    structure_state = shared.get("structure_state", "UNKNOWN")
    sentiment_state = shared.get("sentiment_state", "UNKNOWN")
    memory_state    = shared.get("memory_state", "UNKNOWN")
    candle_state    = shared.get("candle_state", "UNKNOWN")

    council_bias       = council.get("council_bias", "UNKNOWN")
    tradeability_score = council.get("tradeability_score", "UNKNOWN")
    dominant_agent     = council.get("dominant_agent", "UNKNOWN")

    alignment_class    = disagreement.get("alignment_class", "UNKNOWN_ALIGNMENT")
    agreement_score    = disagreement.get("agreement_score", "UNKNOWN")
    conflict_score     = disagreement.get("conflict_score", "UNKNOWN")
    dominance_gap      = disagreement.get("dominance_gap", "UNKNOWN")

    professor_label = professor.get("predicted_label", "UNKNOWN")
    professor_prob  = professor.get("calibrated_probability", "UNKNOWN")
    retail_label    = retail.get("predicted_label", "UNKNOWN")
    retail_prob     = retail.get("calibrated_probability", "UNKNOWN")

    allow_trade       = trade_context.get("allow_trade", True)
    event_uncertainty = trade_context.get("event_base_uncertainty", "UNKNOWN")
    risk_flag_count   = trade_context.get("risk_flag_count", 0)
    risk_flags        = trade_context.get("risk_flags", []) or []
    risk_text         = ", ".join(risk_flags) if risk_flags else "No active risk flags"

    overview = (
        f"BTCUSDT is in a {regime_state} regime with {structure_state} structure, "
        f"{sentiment_state} sentiment, {memory_state} memory bias, and {candle_state} candle context. "
        f"Event state is {event_state}, uncertainty at {event_uncertainty}. "
        f"Professor: {professor_label} ({professor_prob}). Retail: {retail_label} ({retail_prob}). "
        f"Alignment: {alignment_class} (agreement={agreement_score}, conflict={conflict_score}, gap={dominance_gap}). "
        f"Council bias: {council_bias}, tradeability: {tradeability_score}, dominant: {dominant_agent}. "
        f"Trade allowed: {allow_trade}. Risk flags: {risk_text}."
    )

    return {
        "headline": explanation.get("headline", f"{council_bias} | {alignment_class}"),
        "overview": overview,
        "source": "fallback",
    }


def build_ai_overview(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Groq-powered Pillar 7 council commentary.
    Falls back to rule-based text if Groq is unavailable.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_overview(payload)

    try:
        from groq import Groq
    except ImportError:
        return _fallback_overview(payload)

    try:
        client = Groq(api_key=api_key)

        shared        = payload.get("shared_state_summary", {}) or {}
        council       = payload.get("council", {}) or {}
        disagreement  = payload.get("disagreement", {}) or {}
        trade_context = payload.get("trade_context", {}) or {}
        agents        = payload.get("agent_outputs", {}) or {}
        explanation   = payload.get("explanation", {}) or {}
        reason_stack  = payload.get("reason_stack", []) or []

        prompt = f"""You are a professional BTC quantitative strategist writing a council note for an institutional BTCUSDT terminal.

Write exactly 3 short paragraphs:
1) State diagnosis — what the current regime, structure, sentiment, and memory are saying together
2) Agent alignment / disagreement interpretation — what the professor vs retail divergence or alignment means
3) Tactical implication and risk — what a serious trader should do with this council output

Rules:
- Sound like a serious institutional market strategist.
- No bullet points. No hype. No generic chatbot tone.
- Interpret the interaction between professor agent and retail agent.
- Mention whether the move looks structurally healthy, fragile, crowded, or risk-suppressed.
- If agreement is high, explain whether that is constructive or potentially late-stage.
- 170 to 240 words total.

Structured input:
asset: {payload.get("asset")}
timestamp_utc: {payload.get("timestamp_utc")}
shared_state_summary: {shared}
professor_agent: {agents.get("professor_agent", {})}
retail_agent: {agents.get("retail_agent", {})}
disagreement: {disagreement}
council: {council}
trade_context: {trade_context}
headline: {explanation.get("headline")}
reason_stack: {reason_stack}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior BTC quantitative strategist writing institutional-grade "
                        "council notes for a Bloomberg-style trading terminal. "
                        "Be direct, interpretive, and avoid restating fields mechanically."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.3,
            max_tokens=400,
        )

        text = response.choices[0].message.content
        if text and text.strip():
            return {
                "headline": explanation.get("headline", "COUNCIL OVERVIEW"),
                "overview": text.strip(),
                "source": "groq",
            }

        return _fallback_overview(payload)

    except Exception as e:
        print(f"[AI Overview P7 ERROR] {e}")
        return _fallback_overview(payload)


def summarize_ai_overview(overview_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "headline": overview_payload.get("headline"),
        "overview": overview_payload.get("overview"),
        "source": overview_payload.get("source"),
    }