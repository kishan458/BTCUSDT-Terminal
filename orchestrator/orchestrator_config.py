from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from orchestrator.pillar_registry import get_critical_pillar_keys, get_enabled_pillar_keys
from orchestrator.schema import RiskEnum, SeverityEnum, StatusEnum, TradabilityEnum


@dataclass(frozen=True)
class ConfidenceConfig:
    min_confidence: float = 0.0
    max_confidence: float = 1.0
    contradiction_penalty_default: float = 0.08
    stale_data_penalty_default: float = 0.05
    missing_data_penalty_default: float = 0.10
    macro_uncertainty_penalty_default: float = 0.07


@dataclass(frozen=True)
class HealthConfig:
    ok_status: str = StatusEnum.OK.value
    degraded_status: str = StatusEnum.DEGRADED.value
    failed_status: str = StatusEnum.FAILED.value
    default_severity: str = SeverityEnum.WARNING.value
    unsafe_if_critical_pillars_failed_count_gte: int = 2


@dataclass(frozen=True)
class MarketDefaultsConfig:
    default_risk_level: str = RiskEnum.MEDIUM.value
    default_tradability: str = TradabilityEnum.UNFAVORABLE.value


@dataclass(frozen=True)
class OrchestratorConfig:
    product_name: str = "BTCUSDT Terminal"
    version: str = "1.0.0"
    symbol: str = "BTCUSDT"
    default_timeframe: str = "1h"
    environment: str = "development"

    enabled_pillar_keys: tuple[str, ...] = field(default_factory=lambda: tuple(get_enabled_pillar_keys()))
    critical_pillar_keys: tuple[str, ...] = field(default_factory=lambda: tuple(get_critical_pillar_keys()))

    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    market_defaults: MarketDefaultsConfig = field(default_factory=MarketDefaultsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_orchestrator_config() -> OrchestratorConfig:
    """Return the default orchestrator configuration."""
    return OrchestratorConfig()


def build_orchestrator_config_dict() -> dict[str, Any]:
    """Return the default orchestrator configuration as a dictionary."""
    return build_orchestrator_config().to_dict()