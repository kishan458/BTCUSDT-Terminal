from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pillar8_decision_risk_backtesting.decision.alignment_engine import AlignmentResult
from pillar8_decision_risk_backtesting.state.decision_schema import DecisionState


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class ModelRiskResult:
    council_risk: float
    memory_risk: float
    alignment_risk: float
    aggregate_model_risk: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "council_risk": self.council_risk,
            "memory_risk": self.memory_risk,
            "alignment_risk": self.alignment_risk,
            "aggregate_model_risk": self.aggregate_model_risk,
        }


def _compute_council_risk(state: DecisionState) -> float:
    conflict = _clip(state.council.conflict_score)
    confidence = _clip(state.council.confidence)
    agreement = _clip(state.council.agreement_score)

    return _clip(
        (0.50 * conflict)
        + (0.30 * (1.0 - confidence))
        + (0.20 * (1.0 - agreement))
    )


def _compute_memory_risk(state: DecisionState) -> float:
    analog_quality = _clip(state.memory.analog_quality)
    stability_score = _clip(state.memory.stability_score)

    return _clip(
        (0.55 * (1.0 - analog_quality))
        + (0.45 * (1.0 - stability_score))
    )


def _compute_alignment_risk(alignment: AlignmentResult) -> float:
    directional_conflict = _clip(alignment.directional_conflict)
    edge_strength = _clip(abs(alignment.net_directional_edge))

    return _clip(
        (0.70 * directional_conflict)
        + (0.30 * (1.0 - edge_strength))
    )


def compute_model_risk(
    *,
    state: DecisionState,
    alignment: AlignmentResult,
) -> ModelRiskResult:
    council_risk = _compute_council_risk(state)
    memory_risk = _compute_memory_risk(state)
    alignment_risk = _compute_alignment_risk(alignment)

    aggregate_model_risk = _clip(
        (0.38 * council_risk)
        + (0.27 * memory_risk)
        + (0.35 * alignment_risk)
    )

    return ModelRiskResult(
        council_risk=round(council_risk, 6),
        memory_risk=round(memory_risk, 6),
        alignment_risk=round(alignment_risk, 6),
        aggregate_model_risk=round(aggregate_model_risk, 6),
    )