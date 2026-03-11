from market_metrics_engine import build_market_metrics
from trend_regime_engine import classify_directional_regime
from volatility_regime_engine import classify_volatility_regime
from market_state_engine import classify_market_state
from cycle_phase_engine import classify_cycle_phase
from session_engine import build_session_context
from confidence_engine import build_regime_confidence
from strategy_compatibility_engine import build_strategy_compatibility
from risk_flag_engine import build_risk_flags
import json

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

out = {
    "timestamp_utc": metrics["timestamp_utc"],
    "directional_regime": trend["directional_regime"],
    "volatility_regime": vol["volatility_regime"],
    "market_state": state["market_state"],
    "cycle_phase": cycle["cycle_phase"],
    "confidence_score": conf["confidence_score"],
    "strategy_compatibility": strategy["strategy_compatibility"],
    "session_context": session["session_context"],
    "risk_flags": risk["risk_flags"],
}

print(json.dumps(out, indent=2))