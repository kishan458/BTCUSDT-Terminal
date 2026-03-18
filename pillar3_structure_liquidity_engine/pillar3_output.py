import pandas as pd

from pillar3_structure_liquidity_engine.structure_engine import run_structure_engine
from pillar3_structure_liquidity_engine.liquidity_pool_engine import run_liquidity_pool_engine
from pillar3_structure_liquidity_engine.trap_detection_engine import run_trap_detection_engine
from pillar3_structure_liquidity_engine.liquidation_risk_engine import run_liquidation_risk_engine
from pillar3_structure_liquidity_engine.liquidity_target_engine import run_liquidity_target_engine
from pillar3_structure_liquidity_engine.risk_flag_engine import run_risk_flag_engine
from pillar3_structure_liquidity_engine.ai_overview_engine import build_ai_overview


def _trap_risk_label(trap_detection: dict) -> str:
    breakout = float(trap_detection.get("breakout_trap_probability") or 0.0)
    breakdown = float(trap_detection.get("breakdown_trap_probability") or 0.0)
    max_trap = max(breakout, breakdown)

    if max_trap >= 0.70:
        return "HIGH"
    if max_trap >= 0.40:
        return "MODERATE"
    return "LOW"


def _liquidity_environment(liquidity: dict) -> str:
    side = liquidity.get("dominant_liquidity_side")

    if side == "BUY_SIDE":
        return "STOP_CLUSTER_ABOVE"
    if side == "SELL_SIDE":
        return "STOP_CLUSTER_BELOW"
    return "NEUTRAL"


def run_pillar3_output(df: pd.DataFrame, asset: str = "BTCUSDT") -> dict:
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")

    df = df.reset_index(drop=True).copy()

    structure = run_structure_engine(df)
    liquidity = run_liquidity_pool_engine(df)
    trap = run_trap_detection_engine(df)
    liquidation = run_liquidation_risk_engine(df)
    targets = run_liquidity_target_engine(df)
    flags = run_risk_flag_engine(df)

    timestamp_value = None
    if "timestamp" in df.columns:
        timestamp_value = str(df["timestamp"].iloc[-1])

    payload = {
        "asset": asset,
        "timestamp_utc": timestamp_value,
        "structure_liquidity_summary": {
            "dominant_liquidity_side": liquidity["dominant_liquidity_side"],
            "liquidity_environment": _liquidity_environment(liquidity),
            "trap_risk": _trap_risk_label(trap)
        },
        "liquidity_levels": {
            "buy_side_liquidity": liquidity["buy_side_liquidity"],
            "sell_side_liquidity": liquidity["sell_side_liquidity"],
            "nearest_liquidity_magnet": liquidity["nearest_liquidity_magnet"]
        },
        "structure_state": {
            "market_structure": structure["market_structure"],
            "range_state": structure["range_state"],
            "compression_state": structure["compression_state"]
        },
        "trap_detection": {
            "breakout_trap_probability": trap["breakout_trap_probability"],
            "breakdown_trap_probability": trap["breakdown_trap_probability"],
            "likely_trap_side": trap["likely_trap_side"]
        },
        "liquidation_risk": {
            "long_liquidation_risk": liquidation["long_liquidation_risk"],
            "short_liquidation_risk": liquidation["short_liquidation_risk"],
            "cascade_probability": liquidation["cascade_probability"]
        },
        "liquidity_targets": targets["liquidity_targets"],
        "risk_flags": flags["risk_flags"]
    }

    payload["ai_overview"] = build_ai_overview(payload)
    return payload