import pandas as pd

from pillar3_structure_liquidity_engine.liquidity_pool_engine import run_liquidity_pool_engine
from pillar3_structure_liquidity_engine.stop_hunt_engine import run_stop_hunt_engine
from pillar3_structure_liquidity_engine.trap_detection_engine import run_trap_detection_engine
from pillar3_structure_liquidity_engine.liquidation_risk_engine import run_liquidation_risk_engine


def run_risk_flag_engine(df: pd.DataFrame):
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")

    df = df.reset_index(drop=True).copy()

    liquidity = run_liquidity_pool_engine(df)
    stop_hunt = run_stop_hunt_engine(df)
    trap = run_trap_detection_engine(df)
    liquidation = run_liquidation_risk_engine(df)

    flags = []

    if stop_hunt["stop_hunt_probability"] is not None and stop_hunt["stop_hunt_probability"] >= 0.60:
        if stop_hunt["likely_sweep_direction"] == "BUY_SIDE_SWEEP":
            flags.append("Stop hunt above equal highs")
        elif stop_hunt["likely_sweep_direction"] == "SELL_SIDE_SWEEP":
            flags.append("Stop hunt below equal lows")
        else:
            flags.append("Elevated stop hunt risk")

    if trap["breakout_trap_probability"] is not None and trap["breakout_trap_probability"] >= 0.55:
        flags.append("Breakout failure risk")

    if trap["breakdown_trap_probability"] is not None and trap["breakdown_trap_probability"] >= 0.55:
        flags.append("Breakdown failure risk")

    if trap["likely_trap_side"] == "LONG_TRAP":
        flags.append("Breakout longs vulnerable")
    elif trap["likely_trap_side"] == "SHORT_TRAP":
        flags.append("Breakdown shorts vulnerable")

    if liquidation["short_liquidation_risk"] == "HIGH":
        flags.append("Short squeeze risk")

    if liquidation["long_liquidation_risk"] == "HIGH":
        flags.append("Long liquidation cascade risk")

    if liquidation["cascade_probability"] is not None and liquidation["cascade_probability"] >= 0.70:
        flags.append("Cascade risk elevated")

    if liquidity["dominant_liquidity_side"] == "BUY_SIDE":
        flags.append("Liquidity magnet above")
    elif liquidity["dominant_liquidity_side"] == "SELL_SIDE":
        flags.append("Liquidity magnet below")

    # remove duplicates while preserving order
    deduped_flags = list(dict.fromkeys(flags))

    return {
        "risk_flags": deduped_flags
    }