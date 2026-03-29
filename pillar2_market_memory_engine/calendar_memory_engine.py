from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
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
    StateSignatureConfig,
    run_state_signature_engine,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class CalendarMemoryConfig:
    min_samples_per_group: int = 25


# ============================================================
# HELPERS
# ============================================================

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    s = _safe_numeric(series).dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def _safe_median(series: pd.Series) -> float:
    s = _safe_numeric(series).dropna()
    if s.empty:
        return np.nan
    return float(s.median())


def _safe_std(series: pd.Series) -> float:
    s = _safe_numeric(series).dropna()
    if len(s) < 2:
        return np.nan
    return float(s.std(ddof=1))


def _safe_prob(series: pd.Series) -> float:
    s = _safe_numeric(series).dropna()
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


def _calendar_tendency_text(
    continuation_probability: float,
    reversal_probability: float,
    mean_reversion_probability: float,
    next_6_up_probability: float,
) -> str:
    vals = {
        "continuation": continuation_probability,
        "reversal": reversal_probability,
        "mean_reversion": mean_reversion_probability,
        "upside": next_6_up_probability,
        "downside": (1.0 - next_6_up_probability) if pd.notna(next_6_up_probability) else np.nan,
    }
    clean = {k: v for k, v in vals.items() if pd.notna(v)}
    if not clean:
        return "NO_CLEAR_CALENDAR_MEMORY"

    best = max(clean.items(), key=lambda kv: kv[1])
    if best[1] < 0.52:
        return "NO_CLEAR_CALENDAR_MEMORY"

    mapping = {
        "continuation": "CONTINUATION_TENDENCY",
        "reversal": "REVERSAL_TENDENCY",
        "mean_reversion": "MEAN_REVERSION_TENDENCY",
        "upside": "UPSIDE_DRIFT_TENDENCY",
        "downside": "DOWNSIDE_DRIFT_TENDENCY",
    }
    return mapping[best[0]]


# ============================================================
# PREP
# ============================================================

def _prepare_calendar_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "weekday_name" not in out.columns and "weekday" in out.columns:
        weekday_map = {
            0: "MONDAY",
            1: "TUESDAY",
            2: "WEDNESDAY",
            3: "THURSDAY",
            4: "FRIDAY",
            5: "SATURDAY",
            6: "SUNDAY",
        }
        out["weekday_name"] = out["weekday"].map(weekday_map)

    out["week_part"] = np.where(out["is_weekend"], "WEEKEND", "WEEKDAY")

    out["open_window_bucket"] = np.select(
        [
            out["is_sunday_open_window"] == True,
            out["is_monday_open_window"] == True,
        ],
        [
            "SUNDAY_OPEN_WINDOW",
            "MONDAY_OPEN_WINDOW",
        ],
        default="NORMAL_WINDOW",
    )

    out["weekday_hour_bucket"] = (
        out["weekday_name"].astype(str) + "__" + out["hour_utc"].astype(str)
    )

    return out


# ============================================================
# AGGREGATION
# ============================================================

def _aggregate_calendar_memory(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    usable = df.dropna(subset=group_cols, how="any").copy()

    rows = []

    for keys, group in usable.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        n = len(group)

        row: Dict[str, Any] = dict(zip(group_cols, keys))
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

        row["mean_mfe_6"] = _safe_mean(group["fwd_mfe_6"])
        row["mean_mae_6"] = _safe_mean(group["fwd_mae_6"])

        row["calendar_tendency"] = _calendar_tendency_text(
            continuation_probability=row["continuation_probability"],
            reversal_probability=row["reversal_probability"],
            mean_reversion_probability=row["mean_reversion_probability"],
            next_6_up_probability=row["next_6_bar_up_probability"],
        )

        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values("sample_size", ascending=False).reset_index(drop=True)


# ============================================================
# CALENDAR MEMORY TABLES
# ============================================================

def build_weekday_memory(df: pd.DataFrame, cfg: CalendarMemoryConfig) -> pd.DataFrame:
    out = _aggregate_calendar_memory(df, ["weekday_name"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


def build_hour_memory(df: pd.DataFrame, cfg: CalendarMemoryConfig) -> pd.DataFrame:
    out = _aggregate_calendar_memory(df, ["hour_utc"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


def build_weekpart_memory(df: pd.DataFrame, cfg: CalendarMemoryConfig) -> pd.DataFrame:
    out = _aggregate_calendar_memory(df, ["week_part"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


def build_weekday_hour_memory(df: pd.DataFrame, cfg: CalendarMemoryConfig) -> pd.DataFrame:
    out = _aggregate_calendar_memory(df, ["weekday_name", "hour_utc"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


def build_open_window_memory(df: pd.DataFrame, cfg: CalendarMemoryConfig) -> pd.DataFrame:
    out = _aggregate_calendar_memory(df, ["open_window_bucket"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


# ============================================================
# CURRENT CONTEXT LOOKUP
# ============================================================

def _pick_matching_row(df: pd.DataFrame, filters: Dict[str, Any]) -> Optional[pd.Series]:
    if df.empty:
        return None

    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        if col not in df.columns:
            return None
        mask &= df[col] == val

    matched = df[mask]
    if matched.empty:
        return None

    return matched.sort_values("sample_size", ascending=False).iloc[0]


def build_calendar_context_memory(
    current_signature: Dict[str, Any],
    df_outcomes: pd.DataFrame,
    weekday_memory: pd.DataFrame,
    hour_memory: pd.DataFrame,
    weekpart_memory: pd.DataFrame,
    weekday_hour_memory: pd.DataFrame,
    open_window_memory: pd.DataFrame,
) -> Dict[str, Any]:
    latest_row = df_outcomes.iloc[-1]

    weekday = current_signature.get("weekday")
    hour_utc = int(latest_row["hour_utc"]) if "hour_utc" in latest_row.index and pd.notna(latest_row["hour_utc"]) else None
    weekend_flag = current_signature.get("weekend_flag")

    week_part = "WEEKEND" if weekend_flag else "WEEKDAY"

    if "is_sunday_open_window" in latest_row.index and latest_row["is_sunday_open_window"] == True:
        open_window_bucket = "SUNDAY_OPEN_WINDOW"
    elif "is_monday_open_window" in latest_row.index and latest_row["is_monday_open_window"] == True:
        open_window_bucket = "MONDAY_OPEN_WINDOW"
    else:
        open_window_bucket = "NORMAL_WINDOW"

    weekday_row = _pick_matching_row(
        weekday_memory,
        {"weekday_name": weekday},
    )
    hour_row = _pick_matching_row(
        hour_memory,
        {"hour_utc": hour_utc},
    )
    weekpart_row = _pick_matching_row(
        weekpart_memory,
        {"week_part": week_part},
    )
    weekday_hour_row = _pick_matching_row(
        weekday_hour_memory,
        {"weekday_name": weekday, "hour_utc": hour_utc},
    )
    open_window_row = _pick_matching_row(
        open_window_memory,
        {"open_window_bucket": open_window_bucket},
    )

    return {
        "weekday_tendency": weekday_row["calendar_tendency"] if weekday_row is not None else "NO_MATCH",
        "hour_tendency": hour_row["calendar_tendency"] if hour_row is not None else "NO_MATCH",
        "weekpart_tendency": weekpart_row["calendar_tendency"] if weekpart_row is not None else "NO_MATCH",
        "weekday_hour_tendency": weekday_hour_row["calendar_tendency"] if weekday_hour_row is not None else "NO_MATCH",
        "open_window_tendency": open_window_row["calendar_tendency"] if open_window_row is not None else "NO_MATCH",
        "weekday_next_6_up_probability": float(weekday_row["next_6_bar_up_probability"]) if weekday_row is not None and pd.notna(weekday_row["next_6_bar_up_probability"]) else np.nan,
        "hour_next_6_up_probability": float(hour_row["next_6_bar_up_probability"]) if hour_row is not None and pd.notna(hour_row["next_6_bar_up_probability"]) else np.nan,
    }


# ============================================================
# FINAL BUILD
# ============================================================

def build_calendar_memory_payload(
    df_outcomes: pd.DataFrame,
    current_signature: Dict[str, Any],
    cfg: Optional[CalendarMemoryConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = CalendarMemoryConfig()

    base = _prepare_calendar_base(df_outcomes)

    weekday_memory = build_weekday_memory(base, cfg)
    hour_memory = build_hour_memory(base, cfg)
    weekpart_memory = build_weekpart_memory(base, cfg)
    weekday_hour_memory = build_weekday_hour_memory(base, cfg)
    open_window_memory = build_open_window_memory(base, cfg)

    context_memory = build_calendar_context_memory(
        current_signature=current_signature,
        df_outcomes=base,
        weekday_memory=weekday_memory,
        hour_memory=hour_memory,
        weekpart_memory=weekpart_memory,
        weekday_hour_memory=weekday_hour_memory,
        open_window_memory=open_window_memory,
    )

    return {
        "weekday_memory": weekday_memory,
        "hour_memory": hour_memory,
        "weekpart_memory": weekpart_memory,
        "weekday_hour_memory": weekday_hour_memory,
        "open_window_memory": open_window_memory,
        "calendar_context_memory": context_memory,
    }


# ============================================================
# RUNTIME
# ============================================================

def run_calendar_memory_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
    calendar_config: Optional[CalendarMemoryConfig] = None,
) -> Dict[str, Any]:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    if signature_config is None:
        signature_config = StateSignatureConfig()

    if calendar_config is None:
        calendar_config = CalendarMemoryConfig()

    df_outcomes = run_memory_outcome_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
    )

    current_signature = run_state_signature_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
    )

    return build_calendar_memory_payload(
        df_outcomes=df_outcomes,
        current_signature=current_signature,
        cfg=calendar_config,
    )


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

    signature_cfg = StateSignatureConfig(
        require_complete_core_state=True,
    )

    calendar_cfg = CalendarMemoryConfig(
        min_samples_per_group=25,
    )

    result = run_calendar_memory_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
        calendar_config=calendar_cfg,
    )

    print("\n=== CALENDAR MEMORY ENGINE SUCCESS ===")

    print("\nCalendar context memory:\n")
    for k, v in result["calendar_context_memory"].items():
        print(f"{k}: {v}")

    print("\nWeekday memory preview:\n")
    print(result["weekday_memory"].head(10).to_string(index=False))