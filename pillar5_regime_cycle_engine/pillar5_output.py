from pillar5_regime_cycle_engine.market_metrics_engine import build_market_metrics
from pillar5_regime_cycle_engine.trend_regime_engine import classify_directional_regime
from pillar5_regime_cycle_engine.volatility_regime_engine import classify_volatility_regime
from pillar5_regime_cycle_engine.market_state_engine import classify_market_state
from pillar5_regime_cycle_engine.cycle_phase_engine import classify_cycle_phase
from pillar5_regime_cycle_engine.session_engine import build_session_context
from pillar5_regime_cycle_engine.confidence_engine import build_regime_confidence
from pillar5_regime_cycle_engine.strategy_compatibility_engine import build_strategy_compatibility
from pillar5_regime_cycle_engine.risk_flag_engine import build_risk_flags
from pillar5_regime_cycle_engine.ai_overview_engine import build_ai_overview


def _build_regime_explanation(
    directional_regime: str,
    volatility_regime: str,
    market_state: str,
    cycle_phase: str,
) -> dict:
    trend_context = {
        "STRONG_UPTREND": "Price is in a strong bullish trend with supportive structure and moving average alignment.",
        "WEAK_UPTREND": "Price is trending upward but with less conviction and weaker structural support.",
        "RANGE": "Price is not in a clean directional trend and is behaving more like a range-bound market.",
        "WEAK_DOWNTREND": "Price is leaning bearish, though trend quality is not fully established.",
        "STRONG_DOWNTREND": "Price is in a strong bearish trend with persistent downside structure.",
    }.get(directional_regime, "Trend context unclear.")

    volatility_context = {
        "COMPRESSED": "Volatility is compressed, which can precede expansion but currently limits movement.",
        "NORMAL": "Volatility is in a stable regime without major disorder.",
        "EXPANDING": "Volatility is expanding, increasing the chance of larger directional movement.",
        "DISLOCATED": "Volatility is elevated and unstable, increasing the risk of sharp moves and failed continuation.",
    }.get(volatility_regime, "Volatility context unclear.")

    cycle_context = {
        "ACCUMULATION": "Market behavior resembles accumulation, where price is building a base rather than expanding strongly.",
        "EXPANSION": "Market behavior resembles expansion, with constructive trend continuation and directional follow-through.",
        "DISTRIBUTION": "Market behavior resembles distribution, where trend strength is fading and upside may be sold into.",
        "MARKDOWN": "Market behavior resembles markdown, with price under persistent downside pressure.",
        "RECOVERY": "Market behavior resembles recovery, where price is improving but trend maturity is still developing.",
        "EXHAUSTION": "Market behavior suggests exhaustion, where strong trend conditions may be approaching instability or reversal risk.",
    }.get(cycle_phase, "Cycle context unclear.")

    return {
        "regime_explanation": {
            "trend_context": trend_context,
            "volatility_context": volatility_context,
            "cycle_context": cycle_context,
        }
    }


def build_pillar5_output() -> dict:
    metrics = build_market_metrics()

    trend = classify_directional_regime(metrics)
    vol = classify_volatility_regime(metrics)

    state = classify_market_state(
        metrics=metrics,
        directional_regime=trend["directional_regime"],
        volatility_regime=vol["volatility_regime"],
    )

    cycle = classify_cycle_phase(
        metrics=metrics,
        directional_regime=trend["directional_regime"],
        volatility_regime=vol["volatility_regime"],
        market_state=state["market_state"],
    )

    session = build_session_context()

    conf = build_regime_confidence(
        metrics=metrics,
        directional_regime=trend["directional_regime"],
        volatility_regime=vol["volatility_regime"],
        market_state=state["market_state"],
        cycle_phase=cycle["cycle_phase"],
    )

    strategy = build_strategy_compatibility(
        directional_regime=trend["directional_regime"],
        volatility_regime=vol["volatility_regime"],
        market_state=state["market_state"],
        cycle_phase=cycle["cycle_phase"],
        confidence_score=conf["confidence_score"],
    )

    risk = build_risk_flags(
        metrics=metrics,
        directional_regime=trend["directional_regime"],
        volatility_regime=vol["volatility_regime"],
        market_state=state["market_state"],
        cycle_phase=cycle["cycle_phase"],
    )

    explanation = _build_regime_explanation(
        directional_regime=trend["directional_regime"],
        volatility_regime=vol["volatility_regime"],
        market_state=state["market_state"],
        cycle_phase=cycle["cycle_phase"],
    )

    ai_overview = build_ai_overview({
        "asset": "BTCUSDT",
        "timestamp_utc": metrics["timestamp_utc"],
        "regime_summary": {
            "directional_regime": trend["directional_regime"],
            "volatility_regime": vol["volatility_regime"],
            "market_state": state["market_state"],
            "cycle_phase": cycle["cycle_phase"],
        },
        "confidence_score": conf["confidence_score"],
        "strategy_compatibility": strategy["strategy_compatibility"],
        "session_context": session["session_context"],
        "market_metrics": metrics["market_metrics"],
        "risk_flags": risk["risk_flags"],
    })

    return {
        "asset": "BTCUSDT",
        "timestamp_utc": metrics["timestamp_utc"],
        "regime_summary": {
            "directional_regime": trend["directional_regime"],
            "volatility_regime": vol["volatility_regime"],
            "market_state": state["market_state"],
            "cycle_phase": cycle["cycle_phase"],
        },
        "confidence_score": conf["confidence_score"],
        "strategy_compatibility": strategy["strategy_compatibility"],
        "regime_explanation": explanation["regime_explanation"],
        "ai_overview": ai_overview,
        "session_context": session["session_context"],
        "market_metrics": metrics["market_metrics"],
        "risk_flags": risk["risk_flags"],
    }