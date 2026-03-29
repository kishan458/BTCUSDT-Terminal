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
)
from pillar2_market_memory_engine.conditional_outcome_engine import (  # noqa: E402
    ConditionalOutcomeConfig,
    run_conditional_outcome_engine,
)


# ============================================================
# CONFIG
# ============================================================

@dataclass
class StabilityConfig:
    minimum_window_samples: int = 10
    no_clear_edge_band: float = 0.03
    strong_bias_threshold: float = 0.55
    weak_bias_threshold: float = 0.52


# ============================================================
# HELPERS
# ============================================================

def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = _safe_numeric(values)
    w = _safe_numeric(weights)

    mask = (~v.isna()) & (~w.isna()) & (w > 0)
    if not mask.any():
        return np.nan

    v = v[mask].to_numpy(dtype=np.float64)
    w = w[mask].to_numpy(dtype=np.float64)

    return float(np.average(v, weights=w))


def _safe_mean(series: pd.Series) -> float:
    s = _safe_numeric(series).dropna()
    if s.empty:
        return np.nan
    return float(s.mean())


def _bias_label_from_probability(
    up_probability: float,
    no_clear_edge_band: float,
    strong_bias_threshold: float,
    weak_bias_threshold: float,
) -> str:
    if pd.isna(up_probability):
        return "INSUFFICIENT"

    if abs(up_probability - 0.5) < no_clear_edge_band:
        return "MIXED"

    if up_probability >= strong_bias_threshold:
        return "UP_BIAS"
    if up_probability <= (1.0 - strong_bias_threshold):
        return "DOWN_BIAS"

    if up_probability >= weak_bias_threshold:
        return "SLIGHT_UP_BIAS"
    if up_probability <= (1.0 - weak_bias_threshold):
        return "SLIGHT_DOWN_BIAS"

    return "MIXED"


def _agreement_score(labels: list[str]) -> float:
    clean = [x for x in labels if x != "INSUFFICIENT"]
    if len(clean) <= 1:
        return np.nan

    unique = len(set(clean))
    if unique == 1:
        return 1.0
    if unique == 2:
        return 0.5
    return 0.0


def _sample_reliability_label(counts: list[int], minimum_window_samples: int) -> str:
    valid_windows = sum(int(c >= minimum_window_samples) for c in counts)

    if valid_windows == 3:
        return "HIGH"
    if valid_windows == 2:
        return "MODERATE"
    if valid_windows == 1:
        return "LOW"
    return "INSUFFICIENT"


def _temporal_stability_score(
    probs: list[float],
    labels: list[str],
    minimum_window_samples_flags: list[bool],
) -> float:
    valid_probs = [p for p, ok in zip(probs, minimum_window_samples_flags) if ok and pd.notna(p)]
    valid_labels = [l for l, ok in zip(labels, minimum_window_samples_flags) if ok and l != "INSUFFICIENT"]

    if len(valid_probs) <= 1:
        return np.nan

    agreement = _agreement_score(valid_labels)
    prob_std = float(np.std(valid_probs, ddof=0)) if len(valid_probs) > 1 else 0.0

    if pd.isna(agreement):
        return np.nan

    stability = agreement * (1.0 / (1.0 + 5.0 * prob_std))
    return float(max(0.0, min(1.0, stability)))


def _regime_dependency_score(probs: list[float], minimum_window_samples_flags: list[bool]) -> float:
    valid_probs = [p for p, ok in zip(probs, minimum_window_samples_flags) if ok and pd.notna(p)]
    if len(valid_probs) <= 1:
        return np.nan

    spread = max(valid_probs) - min(valid_probs)
    # bigger spread => more regime/time dependency
    score = min(1.0, max(0.0, spread / 0.25))
    return float(score)


# ============================================================
# WINDOW SPLIT
# ============================================================

def split_analogs_into_temporal_windows(analog_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    if analog_df.empty:
        return {
            "older_window": analog_df.copy(),
            "middle_window": analog_df.copy(),
            "recent_window": analog_df.copy(),
        }

    out = analog_df.copy()

    if "timestamp_utc" not in out.columns:
        raise ValueError("timestamp_utc column missing from analog DataFrame.")

    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce")
    out = out.sort_values("timestamp_utc").reset_index(drop=True)

    n = len(out)
    one_third = n // 3
    two_third = 2 * n // 3

    older = out.iloc[:one_third].copy()
    middle = out.iloc[one_third:two_third].copy()
    recent = out.iloc[two_third:].copy()

    return {
        "older_window": older,
        "middle_window": middle,
        "recent_window": recent,
    }


# ============================================================
# WINDOW METRICS
# ============================================================

def compute_window_directional_bias(
    df_window: pd.DataFrame,
    cfg: StabilityConfig,
    score_column: str = "analog_similarity_score",
) -> Dict[str, Any]:
    if df_window.empty:
        return {
            "sample_size": 0,
            "next_6_bar_up_probability": np.nan,
            "mean_forward_return_6": np.nan,
            "bias_label": "INSUFFICIENT",
        }

    weights = _safe_numeric(df_window[score_column]) if score_column in df_window.columns else pd.Series(np.ones(len(df_window)))
    next_6 = _safe_numeric(df_window["next_6_bar_up"]) if "next_6_bar_up" in df_window.columns else pd.Series(dtype="float64")
    fwd_6 = _safe_numeric(df_window["fwd_return_6"]) if "fwd_return_6" in df_window.columns else pd.Series(dtype="float64")

    up_prob = _weighted_mean(next_6, weights)
    mean_fwd_6 = _weighted_mean(fwd_6, weights)

    if len(df_window) < cfg.minimum_window_samples:
        bias_label = "INSUFFICIENT"
    else:
        bias_label = _bias_label_from_probability(
            up_probability=up_prob,
            no_clear_edge_band=cfg.no_clear_edge_band,
            strong_bias_threshold=cfg.strong_bias_threshold,
            weak_bias_threshold=cfg.weak_bias_threshold,
        )

    return {
        "sample_size": int(len(df_window)),
        "next_6_bar_up_probability": up_prob,
        "mean_forward_return_6": mean_fwd_6,
        "bias_label": bias_label,
    }


# ============================================================
# FINAL STABILITY BUILD
# ============================================================

def build_stability_diagnostics(
    analog_df: pd.DataFrame,
    cfg: Optional[StabilityConfig] = None,
    score_column: str = "analog_similarity_score",
) -> Dict[str, Any]:
    if cfg is None:
        cfg = StabilityConfig()

    windows = split_analogs_into_temporal_windows(analog_df)

    older = compute_window_directional_bias(windows["older_window"], cfg, score_column=score_column)
    middle = compute_window_directional_bias(windows["middle_window"], cfg, score_column=score_column)
    recent = compute_window_directional_bias(windows["recent_window"], cfg, score_column=score_column)

    probs = [
        older["next_6_bar_up_probability"],
        middle["next_6_bar_up_probability"],
        recent["next_6_bar_up_probability"],
    ]
    labels = [
        older["bias_label"],
        middle["bias_label"],
        recent["bias_label"],
    ]
    counts = [
        older["sample_size"],
        middle["sample_size"],
        recent["sample_size"],
    ]
    min_flags = [c >= cfg.minimum_window_samples for c in counts]

    temporal_stability = _temporal_stability_score(probs, labels, min_flags)
    regime_dependency = _regime_dependency_score(probs, min_flags)
    sample_reliability = _sample_reliability_label(counts, cfg.minimum_window_samples)

    return {
        "older_window_bias": older["bias_label"],
        "middle_window_bias": middle["bias_label"],
        "recent_window_bias": recent["bias_label"],
        "older_window_up_probability": older["next_6_bar_up_probability"],
        "middle_window_up_probability": middle["next_6_bar_up_probability"],
        "recent_window_up_probability": recent["next_6_bar_up_probability"],
        "older_window_sample_size": older["sample_size"],
        "middle_window_sample_size": middle["sample_size"],
        "recent_window_sample_size": recent["sample_size"],
        "temporal_stability_score": temporal_stability,
        "regime_dependency_score": regime_dependency,
        "sample_reliability": sample_reliability,
    }


# ============================================================
# RUNTIME
# ============================================================

def run_stability_engine(
    feature_config: Optional[MemoryFeatureConfig] = None,
    outcome_config: Optional[MemoryOutcomeConfig] = None,
    signature_config: Optional[StateSignatureConfig] = None,
    retrieval_config: Optional[AnalogRetrievalConfig] = None,
    conditional_config: Optional[ConditionalOutcomeConfig] = None,
    stability_config: Optional[StabilityConfig] = None,
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

    if stability_config is None:
        stability_config = StabilityConfig()

    conditional_payload = run_conditional_outcome_engine(
        feature_config=feature_config,
        outcome_config=outcome_config,
        signature_config=signature_config,
        retrieval_config=retrieval_config,
        conditional_config=conditional_config,
    )

    base_analogs = conditional_payload["base_analogs_used"].copy()

    stability_diag = build_stability_diagnostics(
        base_analogs,
        cfg=stability_config,
        score_column=conditional_config.weighted_score_column,
    )

    conditional_payload["stability_diagnostics"] = stability_diag
    return conditional_payload


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

    stability_cfg = StabilityConfig(
        minimum_window_samples=10,
    )

    result = run_stability_engine(
        feature_config=feature_cfg,
        outcome_config=outcome_cfg,
        signature_config=signature_cfg,
        retrieval_config=retrieval_cfg,
        conditional_config=conditional_cfg,
        stability_config=stability_cfg,
    )

    print("\n=== STABILITY ENGINE SUCCESS ===")
    print("\nStability diagnostics:\n")
    for k, v in result["stability_diagnostics"].items():
        print(f"{k}: {v}")