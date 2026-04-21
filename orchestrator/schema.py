from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 format with Z suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class StatusEnum(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


class BiasEnum(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


class RiskEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class TradabilityEnum(str, Enum):
    FAVORABLE = "FAVORABLE"
    MODERATE = "MODERATE"
    UNFAVORABLE = "UNFAVORABLE"
    BLOCKED = "BLOCKED"


class FreshnessEnum(str, Enum):
    FRESH = "FRESH"
    PARTIAL = "PARTIAL"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class ActionEnum(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NO_TRADE = "NO_TRADE"
    REDUCE_RISK = "REDUCE_RISK"
    WAIT = "WAIT"


class SeverityEnum(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class PillarSignal:
    bias: str = BiasEnum.NEUTRAL.value
    strength: float = 0.0
    state: str = "UNKNOWN"


@dataclass
class DataQuality:
    is_fresh: bool = False
    is_complete: bool = False
    freshness_status: str = FreshnessEnum.UNKNOWN.value
    warnings: list[str] = field(default_factory=list)


@dataclass
class PillarWrapper:
    pillar: str
    timestamp_utc: str = field(default_factory=utc_now_iso)
    status: str = StatusEnum.DEGRADED.value
    confidence: float = 0.0
    summary: str = ""
    signal: PillarSignal = field(default_factory=PillarSignal)
    data_quality: DataQuality = field(default_factory=DataQuality)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PillarTabHeader:
    title: str
    pillar_key: str
    status: str = StatusEnum.DEGRADED.value
    confidence: float = 0.0
    last_updated_utc: str = field(default_factory=utc_now_iso)
    summary: str = ""


@dataclass
class PillarCoreState:
    primary_state: str = "UNKNOWN"
    bias: str = BiasEnum.NEUTRAL.value
    strength: float = 0.0
    market_implication: str = ""


@dataclass
class AISummary:
    headline: str = ""
    summary: str = ""
    interpretation: str = ""
    trade_relevance: str = ""


@dataclass
class PillarTab:
    header: PillarTabHeader
    core_state: PillarCoreState = field(default_factory=PillarCoreState)
    metrics: dict[str, Any] = field(default_factory=dict)
    display_blocks: list[dict[str, Any]] = field(default_factory=list)
    charts: list[dict[str, Any]] = field(default_factory=list)
    ai_summary: AISummary = field(default_factory=AISummary)
    warnings: list[str] = field(default_factory=list)
    restrictions: list[str] = field(default_factory=list)
    raw_payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pillar_wrapper(
    pillar: str,
    *,
    status: str = StatusEnum.DEGRADED.value,
    confidence: float = 0.0,
    summary: str = "",
    bias: str = BiasEnum.NEUTRAL.value,
    strength: float = 0.0,
    state: str = "UNKNOWN",
    is_fresh: bool = False,
    is_complete: bool = False,
    freshness_status: str = FreshnessEnum.UNKNOWN.value,
    warnings: list[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized pillar wrapper dictionary."""
    wrapper = PillarWrapper(
        pillar=pillar,
        status=status,
        confidence=confidence,
        summary=summary,
        signal=PillarSignal(
            bias=bias,
            strength=strength,
            state=state,
        ),
        data_quality=DataQuality(
            is_fresh=is_fresh,
            is_complete=is_complete,
            freshness_status=freshness_status,
            warnings=warnings or [],
        ),
        payload=payload or {},
    )
    return wrapper.to_dict()


def build_pillar_tab(
    pillar_key: str,
    title: str,
    *,
    status: str = StatusEnum.DEGRADED.value,
    confidence: float = 0.0,
    summary: str = "",
    primary_state: str = "UNKNOWN",
    bias: str = BiasEnum.NEUTRAL.value,
    strength: float = 0.0,
    market_implication: str = "",
    metrics: dict[str, Any] | None = None,
    display_blocks: list[dict[str, Any]] | None = None,
    charts: list[dict[str, Any]] | None = None,
    ai_summary: dict[str, str] | None = None,
    warnings: list[str] | None = None,
    restrictions: list[str] | None = None,
    raw_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a standardized pillar tab dictionary for frontend/backend use."""
    tab = PillarTab(
        header=PillarTabHeader(
            title=title,
            pillar_key=pillar_key,
            status=status,
            confidence=confidence,
            summary=summary,
        ),
        core_state=PillarCoreState(
            primary_state=primary_state,
            bias=bias,
            strength=strength,
            market_implication=market_implication,
        ),
        metrics=metrics or {},
        display_blocks=display_blocks or [],
        charts=charts or [],
        ai_summary=AISummary(**(ai_summary or {})),
        warnings=warnings or [],
        restrictions=restrictions or [],
        raw_payload=raw_payload or {},
    )
    return tab.to_dict()


def build_terminal_payload_skeleton() -> dict[str, Any]:
    """Build an empty top-level orchestrator payload skeleton."""
    return {
        "metadata": {
            "product": "BTCUSDT Terminal",
            "version": "1.0.0",
            "run_id": "",
            "generated_at_utc": utc_now_iso(),
            "symbol": "BTCUSDT",
            "timeframe": "",
            "environment": "",
            "orchestrator_status": StatusEnum.DEGRADED.value,
            "total_latency_ms": None,
            "active_pillars": 0,
            "notes": [],
        },
        "system_health": {
            "status": StatusEnum.DEGRADED.value,
            "severity": SeverityEnum.WARNING.value,
            "summary": "",
            "active_pillars": 0,
            "degraded_pillars": 0,
            "failed_pillars": 0,
            "critical_failures": [],
            "warnings": [],
            "data_freshness": {},
            "pillar_statuses": {},
        },
        "market_snapshot": {
            "symbol": "BTCUSDT",
            "timestamp_utc": utc_now_iso(),
            "price": None,
            "change_24h_pct": None,
            "volume_24h": None,
            "volatility_state": "UNKNOWN",
            "market_regime": "UNKNOWN",
            "cycle_state": "UNKNOWN",
            "directional_bias": BiasEnum.NEUTRAL.value,
            "confidence": 0.0,
            "tradability": TradabilityEnum.UNFAVORABLE.value,
            "risk_level": RiskEnum.MEDIUM.value,
        },
        "control_panel": {
            "headline_state": {
                "market_regime": "UNKNOWN",
                "cycle_state": "UNKNOWN",
                "directional_bias": BiasEnum.NEUTRAL.value,
                "confidence": 0.0,
                "tradability": TradabilityEnum.UNFAVORABLE.value,
                "risk_level": RiskEnum.MEDIUM.value,
            },
            "decision": {
                "action": ActionEnum.WAIT.value,
                "action_strength": 0.0,
                "position_size_fraction": 0.0,
                "max_leverage": 0.0,
                "execution_style": "UNKNOWN",
                "thesis": "",
                "time_horizon": "UNKNOWN",
            },
            "ai_summary": {
                "short_summary": "",
                "market_story": "",
                "trader_takeaway": "",
                "risk_warning": "",
            },
            "signal_stack": {},
            "key_drivers": [],
            "conflicts": [],
            "restrictions": [],
            "confidence_model": {
                "global_confidence": 0.0,
                "agreement_score": 0.0,
                "contradiction_penalty": 0.0,
                "stale_data_penalty": 0.0,
                "missing_data_penalty": 0.0,
                "macro_uncertainty_penalty": 0.0,
            },
        },
        "pillar_tabs": {},
        "ui_blocks": {},
        "raw_pillar_outputs": {},
    }


def deep_copy_terminal_payload_skeleton() -> dict[str, Any]:
    """Return a safe deep copy of the top-level payload skeleton."""
    return deepcopy(build_terminal_payload_skeleton())