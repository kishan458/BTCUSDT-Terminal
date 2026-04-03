from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class CostShockResult:
    original_total_pnl: float
    shocked_total_pnl: float
    original_average_pnl: float
    shocked_average_pnl: float
    pnl_degradation: float
    degradation_ratio: float
    edge_survives: bool

    def to_dict(self) -> Dict[str, float | bool]:
        return {
            "original_total_pnl": self.original_total_pnl,
            "shocked_total_pnl": self.shocked_total_pnl,
            "original_average_pnl": self.original_average_pnl,
            "shocked_average_pnl": self.shocked_average_pnl,
            "pnl_degradation": self.pnl_degradation,
            "degradation_ratio": self.degradation_ratio,
            "edge_survives": self.edge_survives,
        }


def _validate_numeric_list(values: List[float], name: str) -> List[float]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")

    clean: List[float] = []
    for value in values:
        try:
            clean.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"All values in {name} must be numeric") from exc
    return clean


def run_cost_shock_stress(
    *,
    gross_pnls: List[float],
    original_total_costs: List[float],
    cost_multiplier: float,
) -> CostShockResult:
    gross = _validate_numeric_list(gross_pnls, "gross_pnls")
    costs = _validate_numeric_list(original_total_costs, "original_total_costs")

    if len(gross) != len(costs):
        raise ValueError("gross_pnls and original_total_costs must have the same length")

    if cost_multiplier < 0:
        raise ValueError("cost_multiplier must be >= 0")

    if len(gross) == 0:
        return CostShockResult(
            original_total_pnl=0.0,
            shocked_total_pnl=0.0,
            original_average_pnl=0.0,
            shocked_average_pnl=0.0,
            pnl_degradation=0.0,
            degradation_ratio=0.0,
            edge_survives=False,
        )

    original_net_pnls = [g - c for g, c in zip(gross, costs)]
    shocked_net_pnls = [g - (c * cost_multiplier) for g, c in zip(gross, costs)]

    original_total_pnl = sum(original_net_pnls)
    shocked_total_pnl = sum(shocked_net_pnls)

    original_average_pnl = original_total_pnl / len(original_net_pnls)
    shocked_average_pnl = shocked_total_pnl / len(shocked_net_pnls)

    pnl_degradation = original_total_pnl - shocked_total_pnl

    degradation_ratio = 0.0
    if original_total_pnl != 0:
        degradation_ratio = pnl_degradation / abs(original_total_pnl)

    edge_survives = shocked_total_pnl > 0

    return CostShockResult(
        original_total_pnl=round(original_total_pnl, 6),
        shocked_total_pnl=round(shocked_total_pnl, 6),
        original_average_pnl=round(original_average_pnl, 6),
        shocked_average_pnl=round(shocked_average_pnl, 6),
        pnl_degradation=round(pnl_degradation, 6),
        degradation_ratio=round(degradation_ratio, 6),
        edge_survives=edge_survives,
    )