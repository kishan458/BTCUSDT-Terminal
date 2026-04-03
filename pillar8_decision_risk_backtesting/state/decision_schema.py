from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FinalAction(str, Enum):
    NO_TRADE = "NO_TRADE"
    WATCHLIST = "WATCHLIST"
    PROBE_LONG = "PROBE_LONG"
    PROBE_SHORT = "PROBE_SHORT"
    LONG = "LONG"
    SHORT = "SHORT"
    REDUCE = "REDUCE"
    EXIT = "EXIT"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class RiskState(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class SentimentState:
    sentiment_state: str = "NEUTRAL"
    confidence: float = 0.0
    drivers: List[str] = field(default_factory=list)
    institutional_summary: str = ""


@dataclass
class MarketMemoryState:
    memory_state: str = "NEUTRAL"
    analog_quality: float = 0.0
    forward_bias: str = "NEUTRAL"
    stability_score: float = 0.0
    context_notes: List[str] = field(default_factory=list)


@dataclass
class StructureLiquidityState:
    structure_state: str = "NEUTRAL"
    liquidity_levels: List[Dict[str, Any]] = field(default_factory=list)
    trap_risk: float = 0.0
    liquidation_risk: float = 0.0
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class CandleIntelligenceState:
    dominant_intent: str = "NEUTRAL"
    momentum_state: str = "NEUTRAL"
    breakout_quality: float = 0.0
    pressure_bias: str = "NEUTRAL"
    absorption_signals: List[str] = field(default_factory=list)
    failure_risk: float = 0.0


@dataclass
class RegimeCycleState:
    regime_state: str = "UNKNOWN"
    cycle_phase: str = "UNKNOWN"
    strategy_compatibility: float = 0.0


@dataclass
class EventState:
    event_state: str = "IDLE"
    base_uncertainty: float = 0.0
    trade_restrictions: Dict[str, Any] = field(default_factory=dict)
    scenarios: List[Dict[str, Any]] = field(default_factory=list)
    ai_reasoning: str = ""


@dataclass
class CouncilState:
    final_bias: str = "NEUTRAL"
    final_decision: str = "NO_TRADE"
    confidence: float = 0.0
    agreement_score: float = 0.0
    conflict_score: float = 0.0
    dominant_agent: str = "NONE"
    reasoning: str = ""


@dataclass
class DecisionState:
    timestamp_utc: str = ""

    sentiment: SentimentState = field(default_factory=SentimentState)
    memory: MarketMemoryState = field(default_factory=MarketMemoryState)
    structure: StructureLiquidityState = field(default_factory=StructureLiquidityState)
    candle: CandleIntelligenceState = field(default_factory=CandleIntelligenceState)
    regime: RegimeCycleState = field(default_factory=RegimeCycleState)
    events: EventState = field(default_factory=EventState)
    council: CouncilState = field(default_factory=CouncilState)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    entry_style: str = ""
    stop_framework: str = ""
    target_framework: str = ""
    invalidators: List[str] = field(default_factory=list)
    time_stop_bars: Optional[int] = None


@dataclass
class BacktestContext:
    matched_template: str = ""
    sample_size: int = 0
    expectancy: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    oos_quality: str = "UNKNOWN"


@dataclass
class StressContext:
    ruin_probability: float = 0.0
    mc_p05_equity: float = 0.0
    cost_fragility: str = "UNKNOWN"
    regime_fragility: str = "UNKNOWN"


@dataclass
class AuditTrace:
    dominant_positive_drivers: List[str] = field(default_factory=list)
    dominant_negative_drivers: List[str] = field(default_factory=list)
    policy_version: str = "v1"
    backtest_version: str = "v1"


@dataclass
class Pillar8DecisionOutput:
    timestamp_utc: str = ""
    final_action: FinalAction = FinalAction.NO_TRADE
    direction: Direction = Direction.NONE

    decision_confidence: float = 0.0
    tradability_score: float = 0.0
    risk_score: float = 0.0
    risk_state: RiskState = RiskState.HIGH

    size_fraction: float = 0.0
    max_leverage_allowed: float = 1.0
    decision_archetype: str = "NONE"
    thesis_summary: str = ""

    alignment: Dict[str, Any] = field(default_factory=dict)
    vetoes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)
    backtest_context: BacktestContext = field(default_factory=BacktestContext)
    stress_context: StressContext = field(default_factory=StressContext)
    audit_trace: AuditTrace = field(default_factory=AuditTrace)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "final_action": self.final_action.value,
            "direction": self.direction.value,
            "decision_confidence": self.decision_confidence,
            "tradability_score": self.tradability_score,
            "risk_score": self.risk_score,
            "risk_state": self.risk_state.value,
            "size_fraction": self.size_fraction,
            "max_leverage_allowed": self.max_leverage_allowed,
            "decision_archetype": self.decision_archetype,
            "thesis_summary": self.thesis_summary,
            "alignment": self.alignment,
            "vetoes": self.vetoes,
            "warnings": self.warnings,
            "execution_plan": {
                "entry_style": self.execution_plan.entry_style,
                "stop_framework": self.execution_plan.stop_framework,
                "target_framework": self.execution_plan.target_framework,
                "invalidators": self.execution_plan.invalidators,
                "time_stop_bars": self.execution_plan.time_stop_bars,
            },
            "backtest_context": {
                "matched_template": self.backtest_context.matched_template,
                "sample_size": self.backtest_context.sample_size,
                "expectancy": self.backtest_context.expectancy,
                "max_drawdown": self.backtest_context.max_drawdown,
                "profit_factor": self.backtest_context.profit_factor,
                "oos_quality": self.backtest_context.oos_quality,
            },
            "stress_context": {
                "ruin_probability": self.stress_context.ruin_probability,
                "mc_p05_equity": self.stress_context.mc_p05_equity,
                "cost_fragility": self.stress_context.cost_fragility,
                "regime_fragility": self.stress_context.regime_fragility,
            },
            "audit_trace": {
                "dominant_positive_drivers": self.audit_trace.dominant_positive_drivers,
                "dominant_negative_drivers": self.audit_trace.dominant_negative_drivers,
                "policy_version": self.audit_trace.policy_version,
                "backtest_version": self.audit_trace.backtest_version,
            },
        }