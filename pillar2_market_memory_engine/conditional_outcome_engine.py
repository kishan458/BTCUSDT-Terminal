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
)
from pillar2_market_memory_engine.state_signature_engine import (  # noqa: E402
    StateSignatureConfig,
)
from pillar2_market_memory_engine.analog_retrieval_engine import (  # noqa: E402
    AnalogRetrievalConfig,
    run_analog_retrieval_engine,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class ConditionalOutcomeConfig:
    use_weighted_matches: bool = True
    weighted_score_column: str = "analog_similarity_score"

    small_sample_threshold: int = 25
    moderate_sample_threshold: int = 75
    high_sample_threshold: int = 200

    high_dispersion_std_threshold_6: float = 0.03
    very_high_dispersion_std_threshold_6: float = 0.05

    no_clear_edge_band: float = 0.03


# ============================================================
# HELPERS
# ============================================================

def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


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


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")

    mask = (~v.isna()) & (~w.isna()) & (w > 0)
    if not mask.any():
        return np.nan

    v = v[mask].to_numpy(dtype=np.float64)
    w = w[mask].to_numpy(dtype=np.float64)

    return float(np.average(v, weights=w))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")

    mask = (~v.isna()) & (~w.isna()) & (w > 0)
    if mask.sum() < 2:
        return np.nan

    v = v[mask].to_numpy(dtype=np.float64)
    w = w[mask].to_numpy(dtype=np.float64)

    avg = np.average(v, weights=w)
    var = np.average((v - avg) ** 2, weights=w)

    return float(np.sqrt(var))


def _weighted_prob(values: pd.Series, weights: pd.Series) -> float:
    return _weighted_mean(values, weights)


def _sample_reliability_label(
    n: int,
    cfg: ConditionalOutcomeConfig,
) -> str:
    if n >= cfg.high_sample_threshold:
        return "HIGH"
    if n >= cfg.moderate_sample_threshold:
        return "MODERATE"
    if n >= cfg.small_sample_threshold:
        return "LOW"
    return "INSUFFICIENT"


def _historical_match_quality(avg_score: float) -> str:
    if pd.isna(avg_score):
        return "LOW"
    if avg_score >= 0.80:
        return "HIGH"
    if avg_score >= 0.60:
        return "MODERATE"
    return "LOW"


def _memory_bias(
    next_6_up_probability: float,
    continuation_probability: float,
    reversal_probability: float,
    mean_reversion_probability: float,
    no_clear_edge_band: float,
) -> str:
    candidates = {
        "CONTINUATION_BIAS": continuation_probability,
        "MEAN_REVERSION_BIAS": mean_reversion_probability,
        "REVERSAL_BIAS": reversal_probability,
        "UP_BIAS": next_6_up_probability,
        "DOWN_BIAS": (1.0 - next_6_up_probability) if pd.notna(next_6_up_probability) else np.nan,
    }

    clean = {k: v for k, v in candidates.items() if pd.notna(v)}
    if not clean:
        return "NO_CLEAR_EDGE"

    ordered = sorted(clean.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_value = ordered[0]
    second_value = ordered[1][1] if len(ordered) > 1 else np.nan

    if pd.notna(second_value) and abs(top_value - second_value) < no_clear_edge_band:
        return "NO_CLEAR_EDGE"

    if top_value < 0.52:
        return "NO_CLEAR_EDGE"

    return top_label


def _dispersion_flag(std_6: float, cfg: ConditionalOutcomeConfig) -> Optional[str]:
    if pd.isna(std_6):
        return None
    if std_6 >= cfg.very_high_dispersion_std_threshold_6:
        return "VERY_HIGH"
    if std_6 >= cfg.high_dispersion_std_threshold_6:
        return "HIGH"
    return "NORMAL"


# ============================================================
# CORE OUTCOME COMPUTATION
# ============================================================

def _select_base_analogs(
    analog_result: Dict[str, Any],
    cfg: ConditionalOutcomeConfig,
) -> pd.DataFrame:
    if cfg.use_weighted_matches:
        return analog_result["analog_sets"]["weighted_matches"].copy()
    return analog_result["analog_sets"]["partial_matches"].copy()


def compute_forward_outcomes(
    analog_df: pd.DataFrame,
    cfg: ConditionalOutcomeConfig,
) -> Dict[str, Any]:
    if analog_df.empty:
        return {
            "next_bar_up_probability": np.nan,
            "next_3_bar_up_probability": np.nan,
            "next_6_bar_up_probability": np.nan,
            "next_12_bar_up_probability": np.nan,
            "mean_forward_return_3": np.nan,
            "median_forward_return_3": np.nan,
            "mean_forward_return_6": np.nan,
            "median_forward_return_6": np.nan,
            "mean_forward_return_12": np.nan,
            "median_forward_return_12": np.nan,
            "mean_mfe_6": np.nan,
            "mean_mae_6": np.nan,
            "mfe_mae_ratio_6": np.nan,
            "continuation_probability": np.nan,
            "reversal_probability": np.nan,
            "mean_reversion_probability": np.nan,
            "volatility_expansion_probability": np.nan,
            "failure_probability": np.nan,
        }

    weights = _safe_series(analog_df, cfg.weighted_score_column)

    next_1 = _safe_series(analog_df, "next_1_bar_up")
    next_3 = _safe_series(analog_df, "next_3_bar_up")
    next_6 = _safe_series(analog_df, "next_6_bar_up")
    next_12 = _safe_series(analog_df, "next_12_bar_up")

    fwd_3 = _safe_series(analog_df, "fwd_return_3")
    fwd_6 = _safe_series(analog_df, "fwd_return_6")
    fwd_12 = _safe_series(analog_df, "fwd_return_12")

    mfe_6 = _safe_series(analog_df, "fwd_mfe_6")
    mae_6 = _safe_series(analog_df, "fwd_mae_6")

    continuation = _safe_series(analog_df, "continuation_label_3")
    reversal = _safe_series(analog_df, "reversal_label_3")
    mean_reversion = _safe_series(analog_df, "mean_reversion_label_3")
    vol_expansion = _safe_series(analog_df, "volatility_expansion_label_6")

    breakout_state = analog_df["breakout_state_runtime"] if "breakout_state_runtime" in analog_df.columns else pd.Series(dtype="object")
    failure_prob = _safe_prob((breakout_state == "FAILED").astype("float64")) if not breakout_state.empty else np.nan

    return {
        "next_bar_up_probability": _weighted_prob(next_1, weights),
        "next_3_bar_up_probability": _weighted_prob(next_3, weights),
        "next_6_bar_up_probability": _weighted_prob(next_6, weights),
        "next_12_bar_up_probability": _weighted_prob(next_12, weights),

        "mean_forward_return_3": _weighted_mean(fwd_3, weights),
        "median_forward_return_3": _safe_median(fwd_3),

        "mean_forward_return_6": _weighted_mean(fwd_6, weights),
        "median_forward_return_6": _safe_median(fwd_6),

        "mean_forward_return_12": _weighted_mean(fwd_12, weights),
        "median_forward_return_12": _safe_median(fwd_12),

        "mean_mfe_6": _weighted_mean(mfe_6, weights),
        "mean_mae_6": _weighted_mean(mae_6, weights),
        "mfe_mae_ratio_6": _weighted_mean(mfe_6, weights) / abs(_weighted_mean(mae_6, weights)) if pd.notna(_weighted_mean(mae_6, weights)) and _weighted_mean(mae_6, weights) != 0 else np.nan,

        "continuation_probability": _weighted_prob(continuation, weights),
        "reversal_probability": _weighted_prob(reversal, weights),
        "mean_reversion_probability": _weighted_prob(mean_reversion, weights),
        "volatility_expansion_probability": _weighted_prob(vol_expansion, weights),
        "failure_probability": failure_prob,
    }


def compute_distribution_diagnostics(
    analog_df: pd.DataFrame,
    cfg: ConditionalOutcomeConfig,
) -> Dict[str, Any]:
    if analog_df.empty:
        return {
            "return_std_3": np.nan,
            "return_std_6": np.nan,
            "return_iqr_3": np.nan,
            "left_tail_10pct_6": np.nan,
            "right_tail_90pct_6": np.nan,
            "skew_proxy_6": np.nan,
            "path_dispersion_score": np.nan,
        }

    fwd_3 = _safe_series(analog_df, "fwd_return_3")
    fwd_6 = _safe_series(analog_df, "fwd_return_6")
    weights = _safe_series(analog_df, cfg.weighted_score_column)

    std_3 = _weighted_std(fwd_3, weights)
    std_6 = _weighted_std(fwd_6, weights)

    q25_3 = _safe_quantile(fwd_3, 0.25)
    q75_3 = _safe_quantile(fwd_3, 0.75)
    q10_6 = _safe_quantile(fwd_6, 0.10)
    q50_6 = _safe_quantile(fwd_6, 0.50)
    q90_6 = _safe_quantile(fwd_6, 0.90)

    skew_proxy = np.nan
    if pd.notna(q10_6) and pd.notna(q50_6) and pd.notna(q90_6):
        skew_proxy = float((q90_6 - q50_6) - (q50_6 - q10_6))

    return {
        "return_std_3": std_3,
        "return_std_6": std_6,
        "return_iqr_3": (q75_3 - q25_3) if pd.notna(q75_3) and pd.notna(q25_3) else np.nan,
        "left_tail_10pct_6": q10_6,
        "right_tail_90pct_6": q90_6,
        "skew_proxy_6": skew_proxy,
        "path_dispersion_score": std_6,
    }


def compute_memory_summary(
    analog_df: pd.DataFrame,
    analog_meta: Dict[str, Any],
    forward_outcomes: Dict[str, Any],
    cfg: ConditionalOutcomeConfig,
) -> Dict[str, Any]:
    n = int(len(analog_df))
    avg_score = analog_meta.get("analog_quality_score", np.nan)

    return {
        "current_memory_state": "STATE_MEMORY_ACTIVE",
        "historical_match_quality": _historical_match_quality(avg_score),
        "sample_size": n,
        "effective_sample_size": float(analog_df[cfg.weighted_score_column].sum()) if (not analog_df.empty and cfg.weighted_score_column in analog_df.columns) else np.nan,
        "memory_bias": _memory_bias(
            next_6_up_probability=forward_outcomes["next_6_bar_up_probability"],
            continuation_probability=forward_outcomes["continuation_probability"],
            reversal_probability=forward_outcomes["reversal_probability"],
            mean_reversion_probability=forward_outcomes["mean_reversion_probability"],
            no_clear_edge_band=cfg.no_clear_edge_band,
        ),
        "headline_confidence": avg_score,
    }


def compute_reliability_flags(
    analog_df: pd.DataFrame,
    analog_meta: Dict[str, Any],
    distribution_diag: Dict[str, Any],
    cfg: ConditionalOutcomeConfig,
) -> Dict[str, Any]:
    n = int(len(analog_df))
    sample_reliability = _sample_reliability_label(n, cfg)
    dispersion_flag = _dispersion_flag(distribution_diag["return_std_6"], cfg)

    risk_flags: list[str] = []

    if sample_reliability in {"LOW", "INSUFFICIENT"}:
        risk_flags.append("Historical sample is limited")

    if dispersion_flag == "HIGH":
        risk_flags.append("Forward path dispersion is high")
    elif dispersion_flag == "VERY_HIGH":
        risk_flags.append("Forward path dispersion is extremely high")

    avg_score = analog_meta.get("analog_quality_score", np.nan)
    if pd.notna(avg_score) and avg_score < 0.60:
        risk_flags.append("Analog match quality is only moderate to weak")

    if not risk_flags:
        risk_flags.append("No major reliability warning from analog sample")

    return {
        "sample_reliability": sample_reliability,
        "dispersion_flag": dispersion_flag,
        "risk_flags": risk_flags,
    }


def build_conditional_outcome_payload(
    analog_result: Dict[str, Any],
    cfg: Optional[ConditionalOutcomeConfig] = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = ConditionalOutcomeConfig()

    analog_df = _select_base_analogs(analog_result, cfg)
    analog_meta = analog_result["historical_analogs"]

    forward_outcomes = compute_forward_outcomes(analog_df, cfg)
    distribution_diag = compute_distribution_diagnostics(analog_df, cfg)
    memory_summary = compute_memory_summary(analog_df, analog_meta, forward_outcomes, cfg)
    reliability = compute_reliability_flags(analog_df, analog_meta, distribution_diag, cfg)

    return {
        "memory_summary": memory_summary,
        "historical_analogs": analog_meta,
        "forward_outcomes": forward_outcomes,
        "distribution_diagnostics": distribution_diag,
        "risk_flags": reliability["risk_flags"],
        "sample_reliability": reliability["sample_reliability"],
        "dispersion_flag": reliability["dispersion_flag"],
        "base_analogs_used": analog_df,
    }


# ============================================================
# RUNTIME
# ============================================================

def run_conditional_outcome_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
    retrieval_config: Optional[AnalogRetrievalConfig] = None,
    conditional_config: Optional[ConditionalOutcomeConfig] = None,
) -> Dict[str, Any]:
    if feature_config is None:
        feature_config = MemoryFeatureConfig()

    if outcome_config is None:
        outcome_config = MemoryOutcomeConfig()

    if signature_config is None:
        signature_config = StateSignatureConfig()

    if retrieval_config is None:
        retrieval_config = AnalogRetrievalConfig()

    if conditional_config is None:
        conditional_config = ConditionalOutcomeConfig()

    analog_result = run_analog_retrieval_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
        retrieval_config=retrieval_config,
    )

    payload = build_conditional_outcome_payload(analog_result, conditional_config)
    payload["current_state_signature"] = analog_result["current_state_signature"]

    return payload


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

    retrieval_cfg = AnalogRetrievalConfig(
        min_similarity_score=0.45,
        max_weighted_matches=300,
        exclude_current_bar=True,
    )

    conditional_cfg = ConditionalOutcomeConfig(
        use_weighted_matches=True,
    )

    result = run_conditional_outcome_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
        retrieval_config=retrieval_cfg,
        conditional_config=conditional_cfg,
    )

    print("\n=== CONDITIONAL OUTCOME ENGINE SUCCESS ===")

    print("\nMemory summary:\n")
    for k, v in result["memory_summary"].items():
        print(f"{k}: {v}")

    print("\nForward outcomes:\n")
    for k, v in result["forward_outcomes"].items():
        print(f"{k}: {v}")

    print("\nDistribution diagnostics:\n")
    for k, v in result["distribution_diagnostics"].items():
        print(f"{k}: {v}")

    print("\nRisk flags:\n")
    for item in result["risk_flags"]:
        print(f"- {item}")