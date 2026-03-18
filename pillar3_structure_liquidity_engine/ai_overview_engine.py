import os
from typing import Any


def _fallback_overview(payload: dict[str, Any]) -> str:
    summary = payload.get("structure_liquidity_summary", {})
    liquidity = payload.get("liquidity_levels", {})
    structure = payload.get("structure_state", {})
    trap = payload.get("trap_detection", {})
    liquidation = payload.get("liquidation_risk", {})
    targets = payload.get("liquidity_targets", [])
    risk_flags = payload.get("risk_flags", [])

    dominant_side = summary.get("dominant_liquidity_side", "UNKNOWN")
    liquidity_env = summary.get("liquidity_environment", "UNKNOWN")
    trap_risk = summary.get("trap_risk", "UNKNOWN")

    market_structure = structure.get("market_structure", "UNKNOWN")
    range_state = structure.get("range_state", "UNKNOWN")
    compression_state = structure.get("compression_state", "UNKNOWN")

    breakout_trap = trap.get("breakout_trap_probability")
    breakdown_trap = trap.get("breakdown_trap_probability")
    likely_trap_side = trap.get("likely_trap_side", "UNKNOWN")

    long_liq = liquidation.get("long_liquidation_risk", "UNKNOWN")
    short_liq = liquidation.get("short_liquidation_risk", "UNKNOWN")
    cascade = liquidation.get("cascade_probability")

    buy_side = liquidity.get("buy_side_liquidity")
    sell_side = liquidity.get("sell_side_liquidity")
    nearest_magnet = liquidity.get("nearest_liquidity_magnet")

    risk_text = ", ".join(risk_flags) if risk_flags else "No major liquidity warnings detected"

    return (
        f"BTC is trading in a {range_state} tape with {market_structure} structure and a {compression_state} volatility backdrop. "
        f"Liquidity is currently skewed toward {dominant_side}, with the environment classified as {liquidity_env}. "
        f"The nearest actionable magnet sits at {nearest_magnet}, with buy-side liquidity at {buy_side} and sell-side liquidity at {sell_side}. "
        f"That configuration suggests the market is still orienting around stop location rather than clean directional discovery.\n\n"
        f"Trap conditions remain controlled rather than extreme. Breakout trap probability is {breakout_trap}, breakdown trap probability is {breakdown_trap}, "
        f"and the current trap profile is {likely_trap_side}. That argues against blindly chasing expansion unless price first engages the nearest liquidity pool "
        f"and proves it can hold above or below the sweep area with follow-through.\n\n"
        f"Liquidation pressure is asymmetrical but not disorderly. Long liquidation risk is {long_liq}, short liquidation risk is {short_liq}, "
        f"and cascade probability is {cascade}. The practical implication is that the first move into liquidity is likely tradable, but the higher-value signal will come "
        f"from the reaction after the sweep rather than from the sweep itself. Primary liquidity targets: {targets}. Main flags: {risk_text}."
    )


def build_ai_overview(payload: dict[str, Any]) -> str:
    """
    Gemini-powered Pillar 3 structure/liquidity commentary.

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

        summary = payload.get("structure_liquidity_summary", {})
        liquidity = payload.get("liquidity_levels", {})
        structure = payload.get("structure_state", {})
        trap = payload.get("trap_detection", {})
        liquidation = payload.get("liquidation_risk", {})
        targets = payload.get("liquidity_targets", [])
        risk_flags = payload.get("risk_flags", [])

        prompt = f"""
You are a professional BTC quantitative strategist writing a structure and liquidity note for an advanced institutional trading terminal.

Write exactly 3 paragraphs:
1) Structure and liquidity diagnosis
2) Trading implication
3) Failure mode / risk path

Rules:
- Sound like a serious BTC market professional with strong quantitative intuition.
- Be sharp, high-signal, and specific.
- Focus on liquidity attraction, stop location, trap risk, and forced-move potential.
- Explain what side of the market is vulnerable and why.
- Mention structure quality, liquidity asymmetry, trap conditions, and liquidation risk.
- Do not summarize fields mechanically.
- No bullet points.
- No hype, no emojis, no generic chatbot wording.
- Write like a top-tier quant desk note.
- Aim for roughly 170 to 230 words total.

Structured input:
dominant_liquidity_side: {summary.get("dominant_liquidity_side")}
liquidity_environment: {summary.get("liquidity_environment")}
trap_risk: {summary.get("trap_risk")}

buy_side_liquidity: {liquidity.get("buy_side_liquidity")}
sell_side_liquidity: {liquidity.get("sell_side_liquidity")}
nearest_liquidity_magnet: {liquidity.get("nearest_liquidity_magnet")}

market_structure: {structure.get("market_structure")}
range_state: {structure.get("range_state")}
compression_state: {structure.get("compression_state")}

breakout_trap_probability: {trap.get("breakout_trap_probability")}
breakdown_trap_probability: {trap.get("breakdown_trap_probability")}
likely_trap_side: {trap.get("likely_trap_side")}

long_liquidation_risk: {liquidation.get("long_liquidation_risk")}
short_liquidation_risk: {liquidation.get("short_liquidation_risk")}
cascade_probability: {liquidation.get("cascade_probability")}

liquidity_targets: {targets}
risk_flags: {risk_flags}
timestamp_utc: {payload.get("timestamp_utc")}
asset: {payload.get("asset")}
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