from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from pillar8_decision_risk_backtesting.decision.decision_gate_engine import DecisionGateResult
from pillar8_decision_risk_backtesting.risk.risk_score_engine import RiskScoreResult
from pillar8_decision_risk_backtesting.decision.alignment_engine import AlignmentResult
from pillar8_decision_risk_backtesting.execution.trade_constructor import TradeConstructionResult
from pillar8_decision_risk_backtesting.sizing.size_allocator import SizeAllocationResult
from pillar8_decision_risk_backtesting.sizing.leverage_cap_engine import LeverageCapResult

from pillar8_decision_risk_backtesting.backtesting.validation_report import ValidationReportResult
from pillar8_decision_risk_backtesting.stress.drawdown_stress_engine import DrawdownStressResult
from pillar8_decision_risk_backtesting.stress.cost_shock_engine import CostShockResult
from pillar8_decision_risk_backtesting.stress.regime_stress_engine import RegimeStressResult
from pillar8_decision_risk_backtesting.stress.ruin_probability_engine import RuinProbabilityResult


@dataclass
class Pillar8Output:
    timestamp_utc: str

    final_action: str
    direction: str
    decision_confidence: float
    tradability_score: float
    decision_archetype: str

    risk_score: float
    risk_state: str

    size_fraction: float
    max_leverage_allowed: float

    thesis_summary: str

    alignment: Dict[str, object]
    vetoes: List[str]
    warnings: List[str]

    execution_plan: Dict[str, object]

    validation: Dict[str, object]
    stress: Dict[str, object]

    audit_trace: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "final_action": self.final_action,
            "direction": self.direction,
            "decision_confidence": self.decision_confidence,
            "tradability_score": self.tradability_score,
            "decision_archetype": self.decision_archetype,
            "risk_score": self.risk_score,
            "risk_state": self.risk_state,
            "size_fraction": self.size_fraction,
            "max_leverage_allowed": self.max_leverage_allowed,
            "thesis_summary": self.thesis_summary,
            "alignment": self.alignment,
            "vetoes": self.vetoes,
            "warnings": self.warnings,
            "execution_plan": self.execution_plan,
            "validation": self.validation,
            "stress": self.stress,
            "audit_trace": self.audit_trace,
        }


def build_pillar8_output(
    *,
    timestamp_utc: str,
    gate: DecisionGateResult,
    risk: RiskScoreResult,
    alignment: AlignmentResult,
    trade_plan: TradeConstructionResult,
    size: SizeAllocationResult,
    leverage: LeverageCapResult,
    vetoes: List[str],
    warnings: List[str],
    thesis_summary: str,
    validation_report: ValidationReportResult,
    drawdown_stress: DrawdownStressResult,
    cost_shock: CostShockResult,
    regime_stress: RegimeStressResult,
    ruin: RuinProbabilityResult,
) -> Pillar8Output:
    return Pillar8Output(
        timestamp_utc=timestamp_utc,
        final_action=gate.final_action.value,
        direction=gate.direction.value,
        decision_confidence=round(gate.decision_confidence, 6),
        tradability_score=round(gate.tradability_score, 6),
        decision_archetype=gate.decision_archetype,
        risk_score=round(risk.risk_score, 6),
        risk_state=risk.risk_state.value,
        size_fraction=round(size.final_size_fraction, 6),
        max_leverage_allowed=round(leverage.final_leverage_cap, 6),
        thesis_summary=thesis_summary,
        alignment=alignment.to_dict(),
        vetoes=vetoes,
        warnings=warnings,
        execution_plan=trade_plan.to_dict(),
        validation=validation_report.to_dict(),
        stress={
            "drawdown": drawdown_stress.to_dict(),
            "cost_shock": cost_shock.to_dict(),
            "regime": regime_stress.to_dict(),
            "ruin": ruin.to_dict(),
        },
        audit_trace={
            "policy_version": "v2_production",
            "backtest_version": "v2_production",
        },
    )