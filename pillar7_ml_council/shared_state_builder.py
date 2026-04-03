from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ============================================================
# SHARED STATE SCHEMA
# ============================================================

@dataclass
class SharedStateSection:
    state: str = "UNKNOWN"
    confidence: Optional[float] = None
    summary: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class SharedState:
    asset: str
    timestamp_utc: str
    sentiment: SharedStateSection
    market_memory: SharedStateSection
    structure_liquidity: SharedStateSection
    candle: SharedStateSection
    regime_cycle: SharedStateSection
    high_impact_event: SharedStateSection
    risk_flags: List[str]
    trade_restrictions: Dict[str, Any]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "asset": self.asset,
            "timestamp_utc": self.timestamp_utc,
            "sentiment": asdict(self.sentiment),
            "market_memory": asdict(self.market_memory),
            "structure_liquidity": asdict(self.structure_liquidity),
            "candle": asdict(self.candle),
            "regime_cycle": asdict(self.regime_cycle),
            "high_impact_event": asdict(self.high_impact_event),
            "risk_flags": self.risk_flags,
            "trade_restrictions": self.trade_restrictions,
            "meta": self.meta,
        }


# ============================================================
# HELPERS
# ============================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_non_null(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _coerce_dict(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {}


def _append_if_present(items: List[str], value: Optional[str]) -> None:
    if value and value not in items:
        items.append(value)


def _dig(payload: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return current if current is not None else default


# ============================================================
# NORMALIZERS FOR EACH PILLAR
# ============================================================

def _normalize_pillar1_sentiment(payload: Optional[Dict[str, Any]]) -> SharedStateSection:
    data = _coerce_dict(payload)

    agg = data.get("aggregate_sentiment", {})
    sentiment_state = _first_non_null(
        _dig(data, "aggregate_sentiment", "label"),
        data.get("sentiment_state"),
        data.get("final_sentiment"),
        data.get("sentiment"),
        data.get("institutional_sentiment"),
        "UNKNOWN",
    )

    confidence = _safe_float(
        _first_non_null(
            _dig(data, "aggregate_sentiment", "confidence"),
            data.get("confidence"),
            data.get("sentiment_confidence"),
            data.get("final_confidence"),
        )
    )

    narrative = _first_non_null(
        data.get("narrative_state"),
        data.get("narrative_maturity"),
        data.get("summary"),
    )

    summary = f"sentiment={sentiment_state}"
    if narrative:
        summary += f" | narrative={narrative}"

    return SharedStateSection(
        state=str(sentiment_state),
        confidence=confidence,
        summary=summary,
        raw=data,
    )


def _normalize_pillar2_memory(payload: Optional[Dict[str, Any]]) -> SharedStateSection:
    data = _coerce_dict(payload)

    memory_summary = data.get("memory_summary", {})
    forward_outcomes = data.get("forward_outcomes", {})
    stability_diagnostics = data.get("stability_diagnostics", {})
    context_memory = data.get("context_memory", {})

    memory_state = _first_non_null(
        _dig(data, "memory_summary", "memory_bias"),
        _dig(data, "memory_summary", "bias"),
        _dig(data, "forward_outcomes", "forward_bias"),
        _dig(data, "forward_outcomes", "bias"),
        data.get("memory_state"),
        data.get("memory_bias"),
        data.get("state"),
        "UNKNOWN",
    )

    confidence = _safe_float(
        _first_non_null(
            _dig(data, "stability_diagnostics", "stability_confidence"),
            _dig(data, "stability_diagnostics", "confidence"),
            data.get("confidence"),
            data.get("memory_confidence"),
            data.get("stability_confidence"),
        )
    )

    analog_quality = _first_non_null(
        _dig(data, "historical_analogs", "analog_quality"),
        _dig(data, "memory_summary", "analog_quality"),
        data.get("analog_quality"),
        data.get("historical_match_quality"),
        data.get("match_quality"),
    )

    session_tendency = _first_non_null(
        _dig(context_memory, "session_tendency"),
        _dig(context_memory, "calendar_tendency"),
    )

    summary = f"memory={memory_state}"
    if analog_quality is not None:
        summary += f" | analog_quality={analog_quality}"
    if session_tendency:
        summary += f" | context={session_tendency}"

    return SharedStateSection(
        state=str(memory_state),
        confidence=confidence,
        summary=summary,
        raw=data,
    )


def _normalize_pillar3_structure(payload: Optional[Dict[str, Any]]) -> SharedStateSection:
    data = _coerce_dict(payload)

    struct_block = data.get("structure_state", {})
    if isinstance(struct_block, dict):
        structure_state = _first_non_null(
            struct_block.get("market_structure"),
            struct_block.get("range_state"),
            "UNKNOWN",
        )
    else:
        structure_state = _first_non_null(
            data.get("structure_state"),
            data.get("market_structure"),
            data.get("state"),
            "UNKNOWN",
        )

    liquidity_state = _first_non_null(
        _dig(data, "liquidity_levels", "nearest_liquidity_magnet"),
        _dig(data, "structure_liquidity_summary", "liquidity_environment"),
        data.get("nearest_liquidity_magnet"),
        data.get("liquidity_state"),
    )

    summary = f"struct={structure_state}"
    if liquidity_state:
        summary += f" | liquidity={liquidity_state}"

    return SharedStateSection(
        state=str(structure_state),
        confidence=None,
        summary=summary,
        raw=data,
    )


def _normalize_pillar4_candle(payload: Optional[Dict[str, Any]]) -> SharedStateSection:
    data = _coerce_dict(payload)

    candle_summary = data.get("candle_summary", {})
    breakout_analysis = data.get("breakout_analysis", {})
    pressure = data.get("pressure", {})

    if isinstance(candle_summary, dict):
        candle_state = _first_non_null(
            candle_summary.get("dominant_intent"),
            candle_summary.get("momentum_state"),
            "UNKNOWN",
        )
        confidence = _safe_float(candle_summary.get("intent_confidence"))
        follow_through_quality = candle_summary.get("follow_through_quality")
    else:
        candle_state = _first_non_null(
            data.get("candle_state"),
            data.get("candle_intent"),
            "UNKNOWN",
        )
        confidence = _safe_float(
            _first_non_null(
                data.get("confidence"),
                data.get("candle_confidence"),
                data.get("intent_confidence"),
            )
        )
        follow_through_quality = data.get("follow_through_quality")

    breakout_quality = _first_non_null(
        _dig(data, "breakout_analysis", "breakout_validity"),
        _dig(data, "breakout_analysis", "breakout_state"),
        follow_through_quality,
        _dig(pressure, "pressure_bias"),
    )

    summary = f"candle={candle_state}"
    if breakout_quality:
        summary += f" | breakout={breakout_quality}"

    return SharedStateSection(
        state=str(candle_state),
        confidence=confidence,
        summary=summary,
        raw=data,
    )


def _normalize_pillar5_regime(payload: Optional[Dict[str, Any]]) -> SharedStateSection:
    data = _coerce_dict(payload)

    regime_summary = data.get("regime_summary", {})
    directional_regime = _first_non_null(
        _dig(regime_summary, "directional_regime"),
        data.get("market_regime"),
        data.get("regime_state"),
        data.get("trend_regime"),
        "UNKNOWN",
    )

    cycle_phase = _first_non_null(
        _dig(regime_summary, "cycle_phase"),
        data.get("cycle_phase"),
        _dig(regime_summary, "market_state"),
        data.get("market_state"),
    )

    confidence = _safe_float(
        _first_non_null(
            data.get("confidence_score"),
            data.get("confidence"),
            data.get("regime_confidence"),
        )
    )

    summary = f"regime={directional_regime}"
    if cycle_phase:
        summary += f" | cycle={cycle_phase}"

    return SharedStateSection(
        state=str(directional_regime),
        confidence=confidence,
        summary=summary,
        raw=data,
    )


def _normalize_pillar6_event(payload: Optional[Dict[str, Any]]) -> SharedStateSection:
    data = _coerce_dict(payload)

    event_state = _first_non_null(
        data.get("state"),
        data.get("event_state"),
        data.get("high_impact_state"),
        "UNKNOWN",
    )

    confidence = _safe_float(
        _first_non_null(
            data.get("confidence_score"),
            data.get("confidence"),
            data.get("event_confidence"),
        )
    )

    event_name = _first_non_null(
        data.get("event"),
        data.get("event_name"),
    )

    summary = f"event_state={event_state}"
    if event_name:
        summary += f" | event={event_name}"

    return SharedStateSection(
        state=str(event_state),
        confidence=confidence,
        summary=summary,
        raw=data,
    )


# ============================================================
# EXTRACTION LOGIC
# ============================================================

def _collect_risk_flags(
    pillar3: Dict[str, Any],
    pillar4: Dict[str, Any],
    pillar5: Dict[str, Any],
    pillar6: Dict[str, Any],
) -> List[str]:
    flags: List[str] = []

    _append_if_present(flags, pillar3.get("risk_flag"))
    _append_if_present(flags, pillar5.get("risk_flag"))

    p3_trap_risk = pillar3.get("trap_risk")
    if p3_trap_risk is None and isinstance(pillar3.get("structure_liquidity_summary"), dict):
        p3_trap_risk = pillar3["structure_liquidity_summary"].get("trap_risk")

    if isinstance(p3_trap_risk, str) and p3_trap_risk in {"HIGH", "ELEVATED", "MODERATE"}:
        _append_if_present(flags, "TRAP_RISK_HIGH")

    p3_liq = pillar3.get("liquidation_risk")
    if isinstance(p3_liq, str):
        if p3_liq in {"HIGH", "ELEVATED"}:
            _append_if_present(flags, "LIQUIDATION_RISK_HIGH")
    elif isinstance(p3_liq, dict):
        long_liq = p3_liq.get("long_liquidation_risk")
        short_liq = p3_liq.get("short_liquidation_risk")
        cascade_prob = _safe_float(p3_liq.get("cascade_probability"))

        if long_liq in {"HIGH", "ELEVATED"} or short_liq in {"HIGH", "ELEVATED"}:
            _append_if_present(flags, "LIQUIDATION_RISK_HIGH")

        if cascade_prob is not None and cascade_prob >= 0.50:
            _append_if_present(flags, "LIQUIDATION_CASCADE_RISK")

    breakout_quality = pillar4.get("breakout_quality")
    if breakout_quality is None and isinstance(pillar4.get("breakout_analysis"), dict):
        breakout_quality = pillar4["breakout_analysis"].get("breakout_validity")

    follow_through_quality = pillar4.get("follow_through_quality")
    if follow_through_quality is None and isinstance(pillar4.get("candle_summary"), dict):
        follow_through_quality = pillar4["candle_summary"].get("follow_through_quality")

    if breakout_quality in {"WEAK", "UNCONFIRMED", "FAILING"} or follow_through_quality in {"WEAK", "FAILED"}:
        _append_if_present(flags, "BREAKOUT_QUALITY_WEAK")

    if isinstance(pillar4.get("risk_flags"), list):
        for f in pillar4["risk_flags"]:
            _append_if_present(flags, str(f))

    strategy_compatibility = pillar5.get("strategy_compatibility")
    if isinstance(strategy_compatibility, dict):
        stand_down = strategy_compatibility.get("stand_down")
        if stand_down in {"HIGH", "VERY_HIGH", "TRUE", True}:
            _append_if_present(flags, "STRATEGY_MISMATCH_RISK")
    elif strategy_compatibility == "LOW":
        _append_if_present(flags, "STRATEGY_MISMATCH_RISK")

    if isinstance(pillar5.get("risk_flags"), list):
        for f in pillar5["risk_flags"]:
            _append_if_present(flags, str(f))

    uncertainty = _safe_float(pillar6.get("base_uncertainty"))
    if uncertainty is not None and uncertainty >= 0.70:
        _append_if_present(flags, "EVENT_UNCERTAINTY_HIGH")

    trade_restrictions = pillar6.get("trade_restrictions", {})
    if isinstance(trade_restrictions, dict):
        if trade_restrictions.get("allow_trade") is False:
            _append_if_present(flags, "EVENT_TRADE_RESTRICTED")

    return flags


def _extract_trade_restrictions(pillar6: Dict[str, Any]) -> Dict[str, Any]:
    trade_restrictions = pillar6.get("trade_restrictions", {})
    if isinstance(trade_restrictions, dict):
        return trade_restrictions

    return {
        "allow_trade": True,
        "size_multiplier": 1.0,
        "leverage_cap": None,
        "restriction_reason": None,
    }


def _build_meta(
    pillar1: Dict[str, Any],
    pillar2: Dict[str, Any],
    pillar3: Dict[str, Any],
    pillar4: Dict[str, Any],
    pillar5: Dict[str, Any],
    pillar6: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "institutional_vs_hype_spread": _first_non_null(
            pillar1.get("institutional_vs_hype_spread"),
            pillar1.get("sentiment_divergence"),
            _dig(pillar1, "aggregate_sentiment", "sentiment_divergence"),
        ),
        "analog_quality": _first_non_null(
            pillar2.get("analog_quality"),
            pillar2.get("historical_match_quality"),
            pillar2.get("match_quality"),
            _dig(pillar2, "historical_analogs", "analog_quality"),
            _dig(pillar2, "memory_summary", "analog_quality"),
        ),
        "trap_risk": _first_non_null(
            pillar3.get("trap_risk"),
            _dig(pillar3, "structure_liquidity_summary", "trap_risk"),
        ),
        "liquidation_risk": pillar3.get("liquidation_risk"),
        "candle_intent": _first_non_null(
            pillar4.get("candle_intent"),
            pillar4.get("candle_state"),
            _dig(pillar4, "candle_summary", "dominant_intent"),
        ),
        "cycle_phase": _first_non_null(
            pillar5.get("cycle_phase"),
            _dig(pillar5, "regime_summary", "cycle_phase"),
        ),
        "event_name": _first_non_null(
            pillar6.get("event"),
            pillar6.get("event_name"),
        ),
        "event_base_uncertainty": pillar6.get("base_uncertainty"),
    }


# ============================================================
# MAIN BUILDER
# ============================================================

def build_shared_state(
    *,
    asset: str = "BTCUSDT",
    timestamp_utc: Optional[str] = None,
    pillar1_output: Optional[Dict[str, Any]] = None,
    pillar2_output: Optional[Dict[str, Any]] = None,
    pillar3_output: Optional[Dict[str, Any]] = None,
    pillar4_output: Optional[Dict[str, Any]] = None,
    pillar5_output: Optional[Dict[str, Any]] = None,
    pillar6_output: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    p1 = _coerce_dict(pillar1_output)
    p2 = _coerce_dict(pillar2_output)
    p3 = _coerce_dict(pillar3_output)
    p4 = _coerce_dict(pillar4_output)
    p5 = _coerce_dict(pillar5_output)
    p6 = _coerce_dict(pillar6_output)

    resolved_timestamp = (
        timestamp_utc
        or p6.get("timestamp_utc")
        or p5.get("timestamp_utc")
        or p4.get("timestamp_utc")
        or p3.get("timestamp_utc")
        or p2.get("timestamp_utc")
        or p1.get("timestamp")
        or p1.get("timestamp_utc")
        or _utc_now_iso()
    )

    sentiment = _normalize_pillar1_sentiment(p1)
    market_memory = _normalize_pillar2_memory(p2)
    structure_liquidity = _normalize_pillar3_structure(p3)
    candle = _normalize_pillar4_candle(p4)
    regime_cycle = _normalize_pillar5_regime(p5)
    high_impact_event = _normalize_pillar6_event(p6)

    risk_flags = _collect_risk_flags(p3, p4, p5, p6)
    trade_restrictions = _extract_trade_restrictions(p6)
    meta = _build_meta(p1, p2, p3, p4, p5, p6)

    state = SharedState(
        asset=asset,
        timestamp_utc=resolved_timestamp,
        sentiment=sentiment,
        market_memory=market_memory,
        structure_liquidity=structure_liquidity,
        candle=candle,
        regime_cycle=regime_cycle,
        high_impact_event=high_impact_event,
        risk_flags=risk_flags,
        trade_restrictions=trade_restrictions,
        meta=meta,
    )

    return state.to_dict()


# ============================================================
# OPTIONAL VALIDATOR
# ============================================================

def validate_shared_state(shared_state: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(shared_state, dict):
        return {
            "is_valid": False,
            "errors": ["shared_state must be a dict"],
            "warnings": [],
        }

    required_top_keys = [
        "asset",
        "timestamp_utc",
        "sentiment",
        "market_memory",
        "structure_liquidity",
        "candle",
        "regime_cycle",
        "high_impact_event",
        "risk_flags",
        "trade_restrictions",
        "meta",
    ]

    for key in required_top_keys:
        if key not in shared_state:
            errors.append(f"missing top-level key: {key}")

    for section_name in [
        "sentiment",
        "market_memory",
        "structure_liquidity",
        "candle",
        "regime_cycle",
        "high_impact_event",
    ]:
        section = shared_state.get(section_name)
        if not isinstance(section, dict):
            errors.append(f"{section_name} must be a dict")
            continue

        if "state" not in section:
            errors.append(f"{section_name}.state missing")

        if section.get("state") == "UNKNOWN":
            warnings.append(f"{section_name}.state is UNKNOWN")

    trade_restrictions = shared_state.get("trade_restrictions", {})
    if isinstance(trade_restrictions, dict):
        if trade_restrictions.get("allow_trade") is False:
            warnings.append("trade is currently restricted")

    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }