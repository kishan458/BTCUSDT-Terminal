from __future__ import annotations

from typing import Any

from orchestrator.schema import (
    ActionEnum,
    BiasEnum,
    RiskEnum,
    TradabilityEnum,
)


BULLISH_PILLARS = {
    "pillar_1_sentiment",
    "pillar_2_memory",
    "pillar_3_structure",
    "pillar_4_candles",
    "pillar_5_regime",
    "pillar_7_ml_council",
    "pillar_8_decision",
}

BEARISH_CONTEXT_PILLARS = {
    "pillar_6_events",
}


def _safe_signal(wrapper: dict[str, Any]) -> dict[str, Any]:
    signal = wrapper.get("signal", {})
    return signal if isinstance(signal, dict) else {}


def _safe_payload(wrapper: dict[str, Any]) -> dict[str, Any]:
    payload = wrapper.get("payload", {})
    return payload if isinstance(payload, dict) else {}


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _collect_bias_strengths(
    pillar_outputs: dict[str, dict[str, Any]],
) -> dict[str, list[float]]:
    bullish_strengths: list[float] = []
    bearish_strengths: list[float] = []
    neutral_strengths: list[float] = []

    for wrapper in pillar_outputs.values():
        if not isinstance(wrapper, dict):
            continue

        signal = _safe_signal(wrapper)
        bias = signal.get("bias")
        strength = signal.get("strength", 0.0)

        if not isinstance(strength, (int, float)):
            continue

        if bias == BiasEnum.BULLISH.value:
            bullish_strengths.append(float(strength))
        elif bias == BiasEnum.BEARISH.value:
            bearish_strengths.append(float(strength))
        else:
            neutral_strengths.append(float(strength))

    return {
        "bullish": bullish_strengths,
        "bearish": bearish_strengths,
        "neutral": neutral_strengths,
    }


def derive_directional_bias(
    pillar_outputs: dict[str, dict[str, Any]],
) -> tuple[str, float]:
    """
    Derive a simple top-level directional bias from available pillar signals.
    Returns: (bias, confidence)
    """
    strengths = _collect_bias_strengths(pillar_outputs)
    bullish_score = sum(strengths["bullish"])
    bearish_score = sum(strengths["bearish"])

    if bullish_score > bearish_score:
        return BiasEnum.BULLISH.value, _average(strengths["bullish"])
    if bearish_score > bullish_score:
        return BiasEnum.BEARISH.value, _average(strengths["bearish"])
    return BiasEnum.NEUTRAL.value, _average(strengths["neutral"])


def build_market_snapshot(
    pillar_outputs: dict[str, dict[str, Any]],
    *,
    symbol: str = "BTCUSDT",
) -> dict[str, Any]:
    """
    Build a high-level market snapshot using only real available pillar data.
    """
    directional_bias, confidence = derive_directional_bias(pillar_outputs)

    price = None
    change_24h_pct = None
    volume_24h = None
    volatility_state = "UNKNOWN"
    market_regime = "UNKNOWN"
    cycle_state = "UNKNOWN"
    tradability = TradabilityEnum.UNFAVORABLE.value
    risk_level = RiskEnum.MEDIUM.value
    timestamp_utc = None

    # Pillar 5 can contribute market regime / cycle / volatility
    regime_wrapper = pillar_outputs.get("pillar_5_regime")
    if isinstance(regime_wrapper, dict):
        regime_payload = _safe_payload(regime_wrapper)
        regime_metrics = regime_payload.get("metrics", {})
        if isinstance(regime_metrics, dict):
            volatility_state = regime_metrics.get("volatility_regime", volatility_state)
            market_regime = regime_metrics.get("regime_state", market_regime)
            cycle_state = regime_metrics.get("cycle_phase", cycle_state)
        timestamp_utc = regime_wrapper.get("timestamp_utc", timestamp_utc)

    # Pillar 8 can contribute tradability / risk
    decision_wrapper = pillar_outputs.get("pillar_8_decision")
    if isinstance(decision_wrapper, dict):
        decision_payload = _safe_payload(decision_wrapper)
        decision_metrics = decision_payload.get("metrics", {})
        if isinstance(decision_metrics, dict):
            tradability = decision_metrics.get("tradability", tradability)
            risk_level = decision_metrics.get("risk_level", risk_level)
        timestamp_utc = decision_wrapper.get("timestamp_utc", timestamp_utc) or timestamp_utc

    # If a pillar provides market data, use it.
    for wrapper in pillar_outputs.values():
        if not isinstance(wrapper, dict):
            continue
        payload = _safe_payload(wrapper)
        market_data = payload.get("market_data", {})
        if isinstance(market_data, dict):
            price = market_data.get("price", price)
            change_24h_pct = market_data.get("change_24h_pct", change_24h_pct)
            volume_24h = market_data.get("volume_24h", volume_24h)
            timestamp_utc = wrapper.get("timestamp_utc", timestamp_utc)

    return {
        "symbol": symbol,
        "timestamp_utc": timestamp_utc,
        "price": price,
        "change_24h_pct": change_24h_pct,
        "volume_24h": volume_24h,
        "volatility_state": volatility_state,
        "market_regime": market_regime,
        "cycle_state": cycle_state,
        "directional_bias": directional_bias,
        "confidence": confidence,
        "tradability": tradability,
        "risk_level": risk_level,
    }


def _extract_signal_stack(
    pillar_outputs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    mapping = {
        "pillar_1_sentiment": "sentiment",
        "pillar_6_events": "events",
        "pillar_5_regime": "regime",
        "pillar_3_structure": "structure",
        "pillar_4_candles": "candles",
        "pillar_2_memory": "memory",
        "pillar_7_ml_council": "ml_council",
        "pillar_8_decision": "decision_engine",
    }

    signal_stack: dict[str, dict[str, Any]] = {}

    for pillar_key, label in mapping.items():
        wrapper = pillar_outputs.get(pillar_key)
        if not isinstance(wrapper, dict):
            continue

        signal = _safe_signal(wrapper)
        signal_stack[label] = {
            "state": signal.get("state", "UNKNOWN"),
            "bias": signal.get("bias", BiasEnum.NEUTRAL.value),
            "strength": signal.get("strength", 0.0),
        }

    return signal_stack


def _collect_text_list(
    pillar_outputs: dict[str, dict[str, Any]],
    field_name: str,
) -> list[str]:
    items: list[str] = []

    for wrapper in pillar_outputs.values():
        if not isinstance(wrapper, dict):
            continue
        payload = _safe_payload(wrapper)
        value = payload.get(field_name, [])
        if isinstance(value, list):
            items.extend([str(item) for item in value])

    # Preserve order while deduping
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def _extract_decision_block(
    pillar_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    wrapper = pillar_outputs.get("pillar_8_decision")
    if not isinstance(wrapper, dict):
        return {
            "action": ActionEnum.WAIT.value,
            "action_strength": 0.0,
            "position_size_fraction": 0.0,
            "max_leverage": 0.0,
            "execution_style": "UNKNOWN",
            "thesis": "",
            "time_horizon": "UNKNOWN",
        }

    signal = _safe_signal(wrapper)
    payload = _safe_payload(wrapper)
    decision = payload.get("decision", {})
    metrics = payload.get("metrics", {})

    if not isinstance(decision, dict):
        decision = {}
    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "action": decision.get("action", ActionEnum.WAIT.value),
        "action_strength": signal.get("strength", wrapper.get("confidence", 0.0)),
        "position_size_fraction": metrics.get("size_fraction", 0.0),
        "max_leverage": metrics.get("max_leverage", 0.0),
        "execution_style": decision.get("execution_style", "UNKNOWN"),
        "thesis": decision.get("thesis", wrapper.get("summary", "")),
        "time_horizon": decision.get("time_horizon", "UNKNOWN"),
    }


def _extract_ai_summary_block(
    pillar_outputs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    decision_wrapper = pillar_outputs.get("pillar_8_decision")
    if isinstance(decision_wrapper, dict):
        payload = _safe_payload(decision_wrapper)
        ai_summary = payload.get("ai_summary", {})
        if isinstance(ai_summary, dict):
            return {
                "short_summary": ai_summary.get("short_summary", decision_wrapper.get("summary", "")),
                "market_story": ai_summary.get("market_story", ""),
                "trader_takeaway": ai_summary.get("trader_takeaway", ""),
                "risk_warning": ai_summary.get("risk_warning", ""),
            }

    return {
        "short_summary": "",
        "market_story": "",
        "trader_takeaway": "",
        "risk_warning": "",
    }


def build_control_panel(
    pillar_outputs: dict[str, dict[str, Any]],
    market_snapshot: dict[str, Any],
) -> dict[str, Any]:
    """
    Build the main control panel block using only available real pillar outputs.
    """
    signal_stack = _extract_signal_stack(pillar_outputs)
    decision_block = _extract_decision_block(pillar_outputs)
    ai_summary = _extract_ai_summary_block(pillar_outputs)

    key_drivers = _collect_text_list(pillar_outputs, "key_drivers")
    conflicts = _collect_text_list(pillar_outputs, "conflicts")
    restrictions = _collect_text_list(pillar_outputs, "restrictions")

    strengths = _collect_bias_strengths(pillar_outputs)
    total_signals = (
        len(strengths["bullish"]) +
        len(strengths["bearish"]) +
        len(strengths["neutral"])
    )

    agreement_score = 0.0
    if total_signals > 0:
        agreement_score = max(
            len(strengths["bullish"]),
            len(strengths["bearish"]),
            len(strengths["neutral"]),
        ) / total_signals

    contradiction_penalty = 0.0 if len(conflicts) == 0 else min(0.25, len(conflicts) * 0.05)
    stale_data_penalty = 0.0
    missing_data_penalty = 0.0
    macro_uncertainty_penalty = 0.0

    events_wrapper = pillar_outputs.get("pillar_6_events")
    if isinstance(events_wrapper, dict):
        signal = _safe_signal(events_wrapper)
        if signal.get("state") == "EVENT_RISK_ELEVATED":
            macro_uncertainty_penalty = 0.07

    headline_state = {
        "market_regime": market_snapshot.get("market_regime", "UNKNOWN"),
        "cycle_state": market_snapshot.get("cycle_state", "UNKNOWN"),
        "directional_bias": market_snapshot.get("directional_bias", BiasEnum.NEUTRAL.value),
        "confidence": market_snapshot.get("confidence", 0.0),
        "tradability": market_snapshot.get("tradability", TradabilityEnum.UNFAVORABLE.value),
        "risk_level": market_snapshot.get("risk_level", RiskEnum.MEDIUM.value),
    }

    return {
        "headline_state": headline_state,
        "decision": decision_block,
        "ai_summary": ai_summary,
        "signal_stack": signal_stack,
        "key_drivers": key_drivers,
        "conflicts": conflicts,
        "restrictions": restrictions,
        "confidence_model": {
            "global_confidence": market_snapshot.get("confidence", 0.0),
            "agreement_score": agreement_score,
            "contradiction_penalty": contradiction_penalty,
            "stale_data_penalty": stale_data_penalty,
            "missing_data_penalty": missing_data_penalty,
            "macro_uncertainty_penalty": macro_uncertainty_penalty,
        },
    }