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
class SessionMemoryConfig:
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


def _session_tendency_text(
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
        return "NO_CLEAR_SESSION_MEMORY"

    best = max(clean.items(), key=lambda kv: kv[1])
    if best[1] < 0.52:
        return "NO_CLEAR_SESSION_MEMORY"

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

def _prepare_session_base(df: pd.DataFrame) -> pd.DataFrame:
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

    out["session_day_bucket"] = (
        out["session_label"].astype(str) + "__" + out["weekday_name"].astype(str)
    )

    out["session_weekpart_bucket"] = (
        out["session_label"].astype(str) + "__" + out["week_part"].astype(str)
    )

    return out


# ============================================================
# AGGREGATION
# ============================================================

def _aggregate_session_memory(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
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

        row["session_tendency"] = _session_tendency_text(
            continuation_probability=row["continuation_probability"],
            reversal_probability=row["reversal_probability"],
            mean_reversion_probability=row["mean_reversion_probability"],
            next_6_up_probability=row["next_6_bar_up_probability"],
        )

        rows.append(row)

    result = pd.DataFrame(rows)
    return result.sort_values("sample_size", ascending=False).reset_index(drop=True)


# ============================================================
# SESSION MEMORY TABLES
# ============================================================

def build_session_core_memory(df: pd.DataFrame, cfg: SessionMemoryConfig) -> pd.DataFrame:
    out = _aggregate_session_memory(df, ["session_label"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


def build_session_weekpart_memory(df: pd.DataFrame, cfg: SessionMemoryConfig) -> pd.DataFrame:
    out = _aggregate_session_memory(df, ["session_label", "week_part"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


def build_session_weekday_memory(df: pd.DataFrame, cfg: SessionMemoryConfig) -> pd.DataFrame:
    out = _aggregate_session_memory(df, ["session_label", "weekday_name"])
    return out[out["sample_size"] >= cfg.min_samples_per_group].reset_index(drop=True)


def build_transition_memory(df: pd.DataFrame, cfg: SessionMemoryConfig) -> pd.DataFrame:
    out = _aggregate_session_memory(df, ["session_transition_label"])
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


def build_session_context_memory(
    current_signature: Dict[str, Any],
    session_core_memory: pd.DataFrame,
    session_weekpart_memory: pd.DataFrame,
    session_weekday_memory: pd.DataFrame,
    transition_memory: pd.DataFrame,
) -> Dict[str, Any]:
    session = current_signature.get("session")
    weekday = current_signature.get("weekday")
    weekend_flag = current_signature.get("weekend_flag")
    transition = current_signature.get("session_transition")

    week_part = "WEEKEND" if weekend_flag else "WEEKDAY"

    core_row = _pick_matching_row(session_core_memory, {"session_label": session})
    weekpart_row = _pick_matching_row(
        session_weekpart_memory,
        {"session_label": session, "week_part": week_part},
    )
    weekday_row = _pick_matching_row(
        session_weekday_memory,
        {"session_label": session, "weekday_name": weekday},
    )
    transition_row = _pick_matching_row(
        transition_memory,
        {"session_transition_label": transition},
    )

    return {
        "session_tendency": core_row["session_tendency"] if core_row is not None else "NO_MATCH",
        "session_weekpart_tendency": weekpart_row["session_tendency"] if weekpart_row is not None else "NO_MATCH",
        "session_weekday_tendency": weekday_row["session_tendency"] if weekday_row is not None else "NO_MATCH",
        "session_transition_tendency": transition_row["session_tendency"] if transition_row is not None else "NO_MATCH",
        "session_forward_return_6_mean": float(core_row["mean_forward_return_6"]) if core_row is not None and pd.notna(core_row["mean_forward_return_6"]) else np.nan,
        "session_next_6_up_probability": float(core_row["next_6_bar_up_probability"]) if core_row is not None and pd.notna(core_row["next_6_bar_up_probability"]) else np.nan,
    }


# ============================================================
# FINAL BUILD
# ============================================================

def build_session_memory_payload(
    df_outcomes: pd.DataFrame,
    current_signature: Dict[str, Any],
    cfg: Optional[SessionMemoryConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = SessionMemoryConfig()

    base = _prepare_session_base(df_outcomes)

    session_core_memory = build_session_core_memory(base, cfg)
    session_weekpart_memory = build_session_weekpart_memory(base, cfg)
    session_weekday_memory = build_session_weekday_memory(base, cfg)
    transition_memory = build_transition_memory(base, cfg)

    context_memory = build_session_context_memory(
        current_signature=current_signature,
        session_core_memory=session_core_memory,
        session_weekpart_memory=session_weekpart_memory,
        session_weekday_memory=session_weekday_memory,
        transition_memory=transition_memory,
    )

    return {
        "session_core_memory": session_core_memory,
        "session_weekpart_memory": session_weekpart_memory,
        "session_weekday_memory": session_weekday_memory,
        "transition_memory": transition_memory,
        "session_context_memory": context_memory,
    }


# ============================================================
# RUNTIME
# ============================================================

def run_session_memory_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
    session_config: Optional[SessionMemoryConfig] = None,
) -> Dict[str, Any]:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    if signature_config is None:
        signature_config = StateSignatureConfig()

    if session_config is None:
        session_config = SessionMemoryConfig()

    df_outcomes = run_memory_outcome_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
    )

    current_signature = run_state_signature_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
    )

    return build_session_memory_payload(
        df_outcomes=df_outcomes,
        current_signature=current_signature,
        cfg=session_config,
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

    session_cfg = SessionMemoryConfig(
        min_samples_per_group=25,
    )

    result = run_session_memory_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
        session_config=session_cfg,
    )

    print("\n=== SESSION MEMORY ENGINE SUCCESS ===")

    print("\nSession context memory:\n")
    for k, v in result["session_context_memory"].items():
        print(f"{k}: {v}")

    print("\nSession core memory preview:\n")
    print(result["session_core_memory"].head(10).to_string(index=False))