from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _fmt(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.{decimals}f}"
    return f"{number:.{decimals}f}"


def _fallback_overview(payload: dict[str, Any]) -> str:
    summary      = payload.get("candle_summary", {})
    features     = payload.get("latest_candle_features", {})
    volatility   = payload.get("volatility_context", {})
    context      = payload.get("multi_candle_context", {})
    intent_scores = payload.get("intent_scores", {})
    absorption   = payload.get("absorption", {})
    breakout     = payload.get("breakout_analysis", {})
    pressure     = payload.get("pressure", {})
    alignment    = payload.get("context_alignment", {})
    risk_flags   = payload.get("risk_flags", [])

    dominant_intent       = summary.get("dominant_intent", "UNKNOWN")
    momentum_state        = summary.get("momentum_state", "UNKNOWN")
    control_state         = summary.get("control_state", "UNKNOWN")
    expansion_state       = summary.get("expansion_state", "UNKNOWN")
    follow_through_quality = summary.get("follow_through_quality", "UNKNOWN")
    exhaustion_state      = summary.get("exhaustion_state", "UNKNOWN")
    intent_confidence     = _fmt(summary.get("intent_confidence"))

    body_to_range    = _fmt(features.get("body_to_range_ratio"))
    close_location   = _fmt(features.get("close_location_value"))
    atr_scaled_range = _fmt(features.get("atr_scaled_range"))
    range_expansion  = _fmt(features.get("range_expansion_score"))

    pressure_bias     = pressure.get("pressure_bias", "UNKNOWN")
    pressure_strength = pressure.get("pressure_strength", "UNKNOWN")
    buying_pressure   = _fmt(pressure.get("buying_pressure_score"))
    selling_pressure  = _fmt(pressure.get("selling_pressure_score"))

    breakout_state    = breakout.get("breakout_state", "UNKNOWN")
    breakout_validity = breakout.get("breakout_validity", "UNKNOWN")
    fake_breakout_risk = _fmt(breakout.get("fake_breakout_risk"))

    dominant_absorption = absorption.get("dominant_absorption", "UNKNOWN")
    absorption_confidence = _fmt(absorption.get("absorption_confidence"))

    risk_text = "; ".join(risk_flags) if risk_flags else "No material candle-structure warnings."

    return (
        f"Latest BTC candle reads as {dominant_intent} (confidence {intent_confidence}). "
        f"Momentum is {momentum_state}, control is {control_state}, expansion is {expansion_state}, "
        f"follow-through is {follow_through_quality}. "
        f"Body-to-range {body_to_range}, close-location {close_location}, "
        f"ATR-scaled range {atr_scaled_range}, range expansion {range_expansion}.\n\n"
        f"Pressure bias is {pressure_bias} ({pressure_strength}) — "
        f"buying {buying_pressure} vs selling {selling_pressure}. "
        f"Dominant absorption: {dominant_absorption} (confidence {absorption_confidence}). "
        f"Breakout state: {breakout_state}, validity: {breakout_validity}, "
        f"fake breakout risk: {fake_breakout_risk}.\n\n"
        f"Exhaustion: {exhaustion_state}. Flags: {risk_text}"
    )


def build_ai_overview(payload: dict[str, Any]) -> str:
    """
    Groq-powered Pillar 4 candle intelligence commentary.
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

        # Send a focused subset of the payload — not the entire thing
        # to stay within token limits while keeping signal high
        focused_payload = {
            "asset":           payload.get("asset"),
            "timestamp_utc":   payload.get("timestamp_utc"),
            "timeframe":       payload.get("timeframe"),
            "candle_summary":  payload.get("candle_summary"),
            "latest_candle_features": {
                k: payload.get("latest_candle_features", {}).get(k)
                for k in [
                    "direction", "body_to_range_ratio", "close_location_value",
                    "atr_scaled_range", "range_expansion_score", "body_expansion_score",
                    "upper_wick_to_range_ratio", "lower_wick_to_range_ratio",
                    "bar_efficiency", "overlap_ratio_vs_prev_bar",
                ]
            },
            "volatility_context": {
                k: payload.get("volatility_context", {}).get(k)
                for k in ["atr", "realized_volatility", "realized_volatility_percentile"]
            },
            "intent_scores":   payload.get("intent_scores"),
            "absorption":      payload.get("absorption"),
            "breakout_analysis": {
                k: payload.get("breakout_analysis", {}).get(k)
                for k in [
                    "breakout_direction", "breakout_validity", "breakout_state",
                    "acceptance_score", "failure_score", "fake_breakout_risk",
                    "breakout_quality_score",
                ]
            },
            "pressure":        payload.get("pressure"),
            "risk_flags":      payload.get("risk_flags"),
        }

        prompt = f"""You are a professional BTC quantitative strategist writing a candle-intelligence note for an institutional trading terminal.

Write exactly 3 paragraphs:
1) Candle mechanics and sequence diagnosis — what this candle and recent sequence are actually saying
2) Trading implication — what a serious trader should think about right now
3) Risk / failure path — what invalidates this read

Rules:
- Ground every statement strictly in the structured payload.
- Do NOT invent levels, events, support/resistance, or context not in the data.
- Do NOT summarize fields line by line. Synthesize.
- Use rounded readable numbers when mentioning metrics.
- No bullet points. No hype. No emojis. No chatbot tone.
- 170 to 230 words total.

Structured payload:
{json.dumps(focused_payload, indent=2, default=str)}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior BTC quantitative strategist writing professional "
                        "candle-intelligence notes for a Bloomberg-style trading terminal. "
                        "Be direct, specific, and synthesize rather than summarize."
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
            return text.strip()

        return _fallback_overview(payload)

    except Exception as e:
        print(f"[AI Overview P4 ERROR] {e}")
        return _fallback_overview(payload)