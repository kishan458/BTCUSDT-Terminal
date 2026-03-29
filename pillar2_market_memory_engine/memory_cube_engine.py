from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict
import sys

import numpy as np
import pandas as pd


# ============================================================
# IMPORT FIX FOR DIRECT SCRIPT RUN
# ============================================================

CURRENT_FILE = Path(__file__).resolve()
REPO_ROOT = CURRENT_FILE.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pillar2_market_memory_engine.memory_feature_engine import (  # noqa: E402
    MemoryFeatureConfig,
)
from pillar2_market_memory_engine.memory_outcome_engine import (  # noqa: E402
    MemoryOutcomeConfig,
    run_memory_outcome_engine,
)
from pillar2_market_memory_engine.state_signature_engine import (  # noqa: E402
    build_state_signature_from_row,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class MemoryCubeConfig:
    min_samples_per_group: int = 25
    output_dir: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def _safe_median(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.median())


def _safe_std(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=1))


def _safe_quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.quantile(q))


def _safe_prob(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def _sample_reliability_label(n: int) -> str:
    if n >= 200:
        return "HIGH"
    if n >= 75:
        return "MODERATE"
    if n >= 25:
        return "LOW"
    return "INSUFFICIENT"


def _prepare_cube_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Derived categorical fields for cube building
    out["path_efficiency_state"] = out["directional_efficiency_6"].apply(
        lambda x: (
            "VERY_HIGH_EFFICIENCY" if pd.notna(x) and x >= 0.70 else
            "HIGH_EFFICIENCY" if pd.notna(x) and x >= 0.55 else
            "MODERATE_EFFICIENCY" if pd.notna(x) and x >= 0.35 else
            "LOW_EFFICIENCY" if pd.notna(x) else None
        )
    )

    out["overlap_state"] = out["overlap_pct_prev_bar"].apply(
        lambda x: (
            "HIGH_OVERLAP" if pd.notna(x) and x >= 0.70 else
            "MODERATE_OVERLAP" if pd.notna(x) and x >= 0.35 else
            "LOW_OVERLAP" if pd.notna(x) else None
        )
    )

    def _follow_quality(row: pd.Series) -> Optional[str]:
        pvr = row["progress_vs_range_6"]
        eff = row["directional_efficiency_6"]
        if pd.isna(pvr) or pd.isna(eff):
            return None
        composite = 0.5 * float(pvr) + 0.5 * float(eff)
        if composite >= 0.75:
            return "STRONG"
        if composite >= 0.50:
            return "MODERATE"
        if composite >= 0.30:
            return "WEAK"
        return "FAILING"

    def _pressure_bias(row: pd.Series) -> Optional[str]:
        body = row["body"]
        clv = row["close_location_value"]
        if pd.isna(body) or pd.isna(clv):
            return None
        if float(body) > 0 and float(clv) >= 0.6:
            return "BUY_PRESSURE"
        if float(body) < 0 and float(clv) <= 0.4:
            return "SELL_PRESSURE"
        return "BALANCED"

    def _breakout_state(row: pd.Series) -> Optional[str]:
        rer = row["range_expansion_ratio"]
        clv = row["close_location_value"]
        bpr = row["body_pct_of_range"]
        if pd.isna(rer) or pd.isna(clv) or pd.isna(bpr):
            return None
        rer = float(rer)
        clv = float(clv)
        bpr = float(bpr)
        if rer >= 1.5 and bpr >= 0.6:
            if clv >= 0.8 or clv <= 0.2:
                return "CONFIRMED"
            return "ATTEMPT"
        if rer >= 1.2 and bpr < 0.35:
            return "FAILED"
        return "NONE"

    out["follow_through_quality"] = out.apply(_follow_quality, axis=1)
    out["pressure_bias"] = out.apply(_pressure_bias, axis=1)
    out["breakout_state_runtime"] = out.apply(_breakout_state, axis=1)

    return out


def _aggregate_cube(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    usable = df.copy()

    required_outcome_cols = [
        "fwd_return_3",
        "fwd_return_6",
        "fwd_mfe_6",
        "fwd_mae_6",
        "next_3_bar_up",
        "next_6_bar_up",
        "continuation_label_3",
        "reversal_label_3",
        "mean_reversion_label_3",
        "volatility_expansion_label_6",
    ]

    usable = usable.dropna(subset=group_cols, how="any")

    grouped_rows = []

    for keys, group in usable.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n = len(group)
        row: Dict[str, object] = dict(zip(group_cols, keys))
        row["sample_size"] = int(n)
        row["sample_reliability"] = _sample_reliability_label(n)

        row["next_3_bar_up_probability"] = _safe_prob(group["next_3_bar_up"])
        row["next_6_bar_up_probability"] = _safe_prob(group["next_6_bar_up"])

        row["continuation_probability"] = _safe_prob(group["continuation_label_3"])
        row["reversal_probability"] = _safe_prob(group["reversal_label_3"])
        row["mean_reversion_probability"] = _safe_prob(group["mean_reversion_label_3"])
        row["volatility_expansion_probability"] = _safe_prob(group["volatility_expansion_label_6"])

        row["mean_forward_return_3"] = _safe_mean(group["fwd_return_3"])
        row["median_forward_return_3"] = _safe_median(group["fwd_return_3"])
        row["std_forward_return_3"] = _safe_std(group["fwd_return_3"])

        row["mean_forward_return_6"] = _safe_mean(group["fwd_return_6"])
        row["median_forward_return_6"] = _safe_median(group["fwd_return_6"])
        row["std_forward_return_6"] = _safe_std(group["fwd_return_6"])

        row["left_tail_10pct_6"] = _safe_quantile(group["fwd_return_6"], 0.10)
        row["right_tail_90pct_6"] = _safe_quantile(group["fwd_return_6"], 0.90)

        row["mean_mfe_6"] = _safe_mean(group["fwd_mfe_6"])
        row["median_mfe_6"] = _safe_median(group["fwd_mfe_6"])
        row["mean_mae_6"] = _safe_mean(group["fwd_mae_6"])
        row["median_mae_6"] = _safe_median(group["fwd_mae_6"])

        row["usable_fwd_return_3_count"] = int(group["fwd_return_3"].notna().sum())
        row["usable_fwd_return_6_count"] = int(group["fwd_return_6"].notna().sum())
        row["usable_mfe_mae_6_count"] = int((group["fwd_mfe_6"].notna() & group["fwd_mae_6"].notna()).sum())

        grouped_rows.append(row)

    result = pd.DataFrame(grouped_rows)

    return result.sort_values(by="sample_size", ascending=False).reset_index(drop=True)


# ============================================================
# CUBE BUILDERS
# ============================================================

def build_session_vol_momentum_cube(df: pd.DataFrame, min_samples_per_group: int) -> pd.DataFrame:
    cube = _aggregate_cube(
        df,
        group_cols=["session_label", "volatility_bucket", "momentum_state"],
    )
    return cube[cube["sample_size"] >= min_samples_per_group].reset_index(drop=True)


def build_weekday_hour_vol_cube(df: pd.DataFrame, min_samples_per_group: int) -> pd.DataFrame:
    cube = _aggregate_cube(
        df,
        group_cols=["weekday_name", "hour_utc", "volatility_bucket"],
    )
    return cube[cube["sample_size"] >= min_samples_per_group].reset_index(drop=True)


def build_overlap_follow_pressure_cube(df: pd.DataFrame, min_samples_per_group: int) -> pd.DataFrame:
    cube = _aggregate_cube(
        df,
        group_cols=["overlap_state", "follow_through_quality", "pressure_bias"],
    )
    return cube[cube["sample_size"] >= min_samples_per_group].reset_index(drop=True)


def build_breakout_expansion_momentum_cube(df: pd.DataFrame, min_samples_per_group: int) -> pd.DataFrame:
    cube = _aggregate_cube(
        df,
        group_cols=["breakout_state_runtime", "expansion_state", "momentum_state"],
    )
    return cube[cube["sample_size"] >= min_samples_per_group].reset_index(drop=True)


# ============================================================
# FINAL BUILD
# ============================================================

def build_memory_cubes(df_outcomes: pd.DataFrame, config: MemoryCubeConfig) -> Dict[str, pd.DataFrame]:
    base = _prepare_cube_base_dataframe(df_outcomes)

    cubes = {
        "session_vol_momentum_cube": build_session_vol_momentum_cube(base, config.min_samples_per_group),
        "weekday_hour_vol_cube": build_weekday_hour_vol_cube(base, config.min_samples_per_group),
        "overlap_follow_pressure_cube": build_overlap_follow_pressure_cube(base, config.min_samples_per_group),
        "breakout_expansion_momentum_cube": build_breakout_expansion_momentum_cube(base, config.min_samples_per_group),
    }

    return cubes


def run_memory_cube_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    cube_config: Optional[MemoryCubeConfig] = None,
) -> Dict[str, pd.DataFrame]:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    if cube_config is None:
        cube_config = MemoryCubeConfig()

    df_outcomes = run_memory_outcome_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
    )

    cubes = build_memory_cubes(df_outcomes, cube_config)

    if cube_config.output_dir:
        out_dir = Path(cube_config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for cube_name, cube_df in cubes.items():
            cube_df.to_parquet(out_dir / f"{cube_name}.parquet", index=False)

    return cubes


if __name__ == "__main__":
    feature_cfg = MemoryFeatureConfig(
        db_path="database/btc_terminal.db",
        table_name="btc_price_1h",
        timestamp_col="timestamp",
        output_parquet_path=None,
    )

    outcome_cfg = MemoryOutcomeConfig(
        horizons=(1, 3, 6, 12),
        output_parquet_path=None,
    )

    cube_cfg = MemoryCubeConfig(
        min_samples_per_group=25,
        output_dir=None,
    )

    cubes = run_memory_cube_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        cube_config=cube_cfg,
    )

    print("\n=== MEMORY CUBE ENGINE SUCCESS ===")
    for name, cube in cubes.items():
        print(f"\n{name}:")
        print(f"rows={len(cube)} cols={len(cube.columns)}")
        if not cube.empty:
            print(cube.head(5).to_string(index=False))