from __future__ import annotations

import json
import os
from typing import Any


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
    summary = payload.get("candle_summary", {})
    features = payload.get("latest_candle_features", {})
    volatility = payload.get("volatility_context", {})
    context = payload.get("multi_candle_context", {})
    intent_scores = payload.get("intent_scores", {})
    absorption = payload.get("absorption", {})
    breakout = payload.get("breakout_analysis", {})
    pressure = payload.get("pressure", {})
    alignment = payload.get("context_alignment", {})
    risk_flags = payload.get("risk_flags", [])

    dominant_intent = summary.get("dominant_intent", "UNKNOWN")
    momentum_state = summary.get("momentum_state", "UNKNOWN")
    control_state = summary.get("control_state", "UNKNOWN")
    expansion_state = summary.get("expansion_state", "UNKNOWN")
    overlap_state = summary.get("overlap_state", "UNKNOWN")
    follow_through_quality = summary.get("follow_through_quality", "UNKNOWN")
    exhaustion_state = summary.get("exhaustion_state", "UNKNOWN")
    intent_confidence = _fmt(summary.get("intent_confidence"))

    body_to_range = _fmt(features.get("body_to_range_ratio"))
    close_location = _fmt(features.get("close_location_value"))
    atr_scaled_range = _fmt(features.get("atr_scaled_range"))
    overlap_ratio = _fmt(features.get("overlap_ratio_vs_prev_bar"))
    range_expansion = _fmt(features.get("range_expansion_score"))
    body_expansion = _fmt(features.get("body_expansion_score"))

    atr_value = _fmt(volatility.get("atr"))
    realized_vol = _fmt(volatility.get("realized_volatility"), 4)
    realized_vol_pct = _fmt(volatility.get("realized_volatility_percentile"))

    progress_short = _fmt(context.get("progress_efficiency_short"))
    progress_medium = _fmt(context.get("progress_efficiency_medium"))
    avg_overlap_short = _fmt(context.get("avg_overlap_ratio_short"))
    post_expansion_fade = _fmt(context.get("post_expansion_fade_score"))

    bullish_cont = _fmt(intent_scores.get("bullish_continuation_score"))
    bearish_cont = _fmt(intent_scores.get("bearish_continuation_score"))
    indecision_score = _fmt(intent_scores.get("indecision_score"))
    buy_absorption_candidate = _fmt(intent_scores.get("buy_absorption_candidate_score"))
    sell_absorption_candidate = _fmt(intent_scores.get("sell_absorption_candidate_score"))

    dominant_absorption = absorption.get("dominant_absorption", "UNKNOWN")
    dominant_rejection = absorption.get("dominant_rejection", "UNKNOWN")
    buy_absorption_score = _fmt(absorption.get("buy_absorption_score"))
    sell_absorption_score = _fmt(absorption.get("sell_absorption_score"))
    absorption_confidence = _fmt(absorption.get("absorption_confidence"))
    failed_upside_extension_count = _fmt(absorption.get("failed_upside_extension_count"), 0)
    failed_downside_extension_count = _fmt(absorption.get("failed_downside_extension_count"), 0)

    breakout_direction = breakout.get("breakout_direction", "UNKNOWN")
    breakout_validity = breakout.get("breakout_validity", "UNKNOWN")
    breakout_state = breakout.get("breakout_state", "UNKNOWN")
    acceptance_score = _fmt(breakout.get("acceptance_score"))
    failure_score = _fmt(breakout.get("failure_score"))
    fake_breakout_risk = _fmt(breakout.get("fake_breakout_risk"))
    breakout_quality_score = _fmt(breakout.get("breakout_quality_score"))

    buying_pressure = _fmt(pressure.get("buying_pressure_score"))
    selling_pressure = _fmt(pressure.get("selling_pressure_score"))
    net_pressure = _fmt(pressure.get("net_pressure_score"))
    pressure_bias = pressure.get("pressure_bias", "UNKNOWN")
    pressure_strength = pressure.get("pressure_strength", "UNKNOWN")

    liquidity_alignment = alignment.get("pillar3_liquidity_alignment", "NOT_AVAILABLE")
    liquidity_story = alignment.get("candle_vs_liquidity_story", "NOT_AVAILABLE")

    risk_text = "; ".join(risk_flags) if risk_flags else "No material candle-structure warnings detected."

    return (
        f"Latest BTC candle behavior is best read as {dominant_intent} with confidence {intent_confidence}, "
        f"but the broader tape remains contested rather than cleanly directional. Momentum is {momentum_state}, "
        f"control is {control_state}, expansion sits in {expansion_state}, and overlap is {overlap_state}. "
        f"The latest bar printed body-to-range {body_to_range}, close-location {close_location}, ATR-scaled range {atr_scaled_range}, "
        f"range expansion {range_expansion}, body expansion {body_expansion}, and overlap versus the prior bar at {overlap_ratio}. "
        f"That combination describes a recovery candle with poor auction separation rather than decisive trend continuation.\n\n"
        f"Sequence quality remains soft. Short-horizon progress efficiency is {progress_short}, medium-horizon progress efficiency is {progress_medium}, "
        f"average short overlap is {avg_overlap_short}, and post-expansion fade is {post_expansion_fade}, which is consistent with {follow_through_quality.lower()} follow-through. "
        f"Intent scores still lean more toward bullish continuation ({bullish_cont}) than bearish continuation ({bearish_cont}), but absorption-style behavior remains relevant "
        f"with buy-absorption candidate {buy_absorption_candidate}, sell-absorption candidate {sell_absorption_candidate}, and indecision at {indecision_score}. "
        f"Absorption diagnostics currently favor {dominant_absorption} over {dominant_rejection}, with buy absorption {buy_absorption_score}, sell absorption {sell_absorption_score}, "
        f"confidence {absorption_confidence}, failed upside extensions {failed_upside_extension_count}, and failed downside extensions {failed_downside_extension_count}.\n\n"
        f"Breakout context is {breakout_state} with direction {breakout_direction} and validity {breakout_validity}. Acceptance is {acceptance_score}, failure is {failure_score}, "
        f"fake-breakout risk is {fake_breakout_risk}, and breakout quality is {breakout_quality_score}. Pressure is classified as {pressure_bias} with {pressure_strength.lower()} strength "
        f"(buying pressure {buying_pressure}, selling pressure {selling_pressure}, net pressure {net_pressure}). ATR is {atr_value} and realized volatility is {realized_vol} "
        f"with percentile {realized_vol_pct}. Pillar 3 alignment is {liquidity_alignment}. Liquidity read-through: {liquidity_story}. "
        f"Exhaustion state is {exhaustion_state}. Main flags: {risk_text}"
    )


def build_ai_overview(payload: dict[str, Any]) -> str:
    """
    Gemini-powered Pillar 4 candle intelligence commentary.

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

        prompt = f"""
You are a professional BTC quantitative strategist writing a candle-intelligence and market-microstructure note for an institutional BTC/USDT terminal.

Write exactly 3 paragraphs:
1) Candle mechanics and sequence diagnosis
2) Trading implication
3) Risk / failure path

Rules:
- Sound like a serious BTC market professional with strong quantitative intuition.
- Be sharp, high-signal, and specific.
- Ground every statement strictly in the structured payload.
- Do NOT invent any levels, events, support/resistance, catalysts, or market context not explicitly provided.
- Do NOT claim certainty.
- Do NOT use generic TA fluff.
- Do NOT summarize fields mechanically line by line.
- Use rounded readable numbers when mentioning metrics.
- No bullet points.
- No hype, no emojis, no chatbot tone.
- Write like a top-tier quant desk note.
- Aim for roughly 170 to 230 words total.

Structured payload:
{json.dumps(payload, indent=2, default=str)}
""".strip()

        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
        )

        text = getattr(response, "text", None)
        if text and text.strip():
            return text.strip()

        return _fallback_overview(payload)

    except Exception:
        return _fallback_overview(payload)