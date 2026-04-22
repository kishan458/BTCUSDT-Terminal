import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()


def _fallback_overview(payload: dict[str, Any]) -> str:
    summary     = payload.get("structure_liquidity_summary", {})
    liquidity   = payload.get("liquidity_levels", {})
    structure   = payload.get("structure_state", {})
    trap        = payload.get("trap_detection", {})
    liquidation = payload.get("liquidation_risk", {})
    targets     = payload.get("liquidity_targets", [])
    risk_flags  = payload.get("risk_flags", [])

    dominant_side  = summary.get("dominant_liquidity_side", "UNKNOWN")
    liquidity_env  = summary.get("liquidity_environment", "UNKNOWN")
    trap_risk      = summary.get("trap_risk", "UNKNOWN")

    market_structure  = structure.get("market_structure", "UNKNOWN")
    range_state       = structure.get("range_state", "UNKNOWN")
    compression_state = structure.get("compression_state", "UNKNOWN")

    breakout_trap    = trap.get("breakout_trap_probability")
    breakdown_trap   = trap.get("breakdown_trap_probability")
    likely_trap_side = trap.get("likely_trap_side", "UNKNOWN")

    long_liq  = liquidation.get("long_liquidation_risk", "UNKNOWN")
    short_liq = liquidation.get("short_liquidation_risk", "UNKNOWN")
    cascade   = liquidation.get("cascade_probability")

    buy_side       = liquidity.get("buy_side_liquidity")
    sell_side      = liquidity.get("sell_side_liquidity")
    nearest_magnet = liquidity.get("nearest_liquidity_magnet")

    risk_text = ", ".join(risk_flags) if risk_flags else "No major liquidity warnings detected"

    return (
        f"BTC is trading in a {range_state} tape with {market_structure} structure and a {compression_state} volatility backdrop. "
        f"Liquidity is skewed toward {dominant_side}, environment classified as {liquidity_env}. "
        f"Nearest magnet: {nearest_magnet}, buy-side liquidity: {buy_side}, sell-side liquidity: {sell_side}.\n\n"
        f"Trap conditions: breakout trap probability={breakout_trap}, breakdown trap probability={breakdown_trap}, "
        f"trap profile={likely_trap_side}. Avoid chasing expansion unless price engages the nearest liquidity pool "
        f"and proves follow-through.\n\n"
        f"Liquidation: long risk={long_liq}, short risk={short_liq}, cascade probability={cascade}. "
        f"Targets: {targets}. Flags: {risk_text}."
    )


def build_ai_overview(payload: dict[str, Any]) -> str:
    """
    Groq-powered Pillar 3 structure/liquidity commentary.
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

        summary     = payload.get("structure_liquidity_summary", {})
        liquidity   = payload.get("liquidity_levels", {})
        structure   = payload.get("structure_state", {})
        trap        = payload.get("trap_detection", {})
        liquidation = payload.get("liquidation_risk", {})
        targets     = payload.get("liquidity_targets", [])
        risk_flags  = payload.get("risk_flags", [])

        prompt = f"""You are a professional BTC quantitative strategist writing a structure and liquidity note for an institutional trading terminal.

Write exactly 3 paragraphs:
1) Structure and liquidity diagnosis — what the market structure looks like right now
2) Trading implication — what a serious trader should be thinking
3) Failure mode / risk path — what invalidates the current setup

Rules:
- Sound like a serious BTC market professional with strong quantitative intuition.
- Focus on liquidity attraction, stop location, trap risk, and forced-move potential.
- Explain what side of the market is vulnerable and why.
- Do not summarize fields mechanically. Infer what matters.
- No bullet points. No hype. No emojis.
- 170 to 230 words total.

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
risk_flags: {risk_flags}"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior BTC quantitative strategist writing professional "
                        "structure and liquidity notes for a Bloomberg-style trading terminal. "
                        "Be direct, specific, and high-signal."
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
        print(f"[AI Overview P3 ERROR] {e}")
        return _fallback_overview(payload)