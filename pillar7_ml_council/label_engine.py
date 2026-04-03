from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_price_df(price_df: pd.DataFrame) -> None:
    required_cols = ["close"]
    missing = [col for col in required_cols if col not in price_df.columns]
    if missing:
        raise ValueError(f"price_df missing required columns: {missing}")

    if len(price_df) == 0:
        raise ValueError("price_df is empty")


def build_forward_return_labels(
    price_df: pd.DataFrame,
    horizons: List[int] | None = None,
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Adds forward return labels:
    - fwd_ret_{h}
    - fwd_dir_{h}

    fwd_ret_h = (close[t+h] / close[t]) - 1
    fwd_dir_h = 1 if fwd_ret_h > 0 else 0, NaN where unavailable
    """
    if horizons is None:
        horizons = [1, 3, 6, 12]

    _validate_price_df(price_df)

    df = price_df.copy()

    for h in horizons:
        future_close = df[close_col].shift(-h)
        fwd_ret = (future_close / df[close_col]) - 1.0

        df[f"fwd_ret_{h}"] = fwd_ret
        df[f"fwd_dir_{h}"] = fwd_ret.apply(
            lambda x: 1 if pd.notna(x) and x > 0 else (0 if pd.notna(x) else pd.NA)
        )

    return df


def build_continuation_reversal_labels(
    price_df: pd.DataFrame,
    lookback: int = 3,
    horizon: int = 3,
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Builds:
    - past_ret_{lookback}
    - fwd_ret_{horizon}
    - continuation_label_{lookback}_{horizon}
    - reversal_label_{lookback}_{horizon}

    continuation = past and future return have same sign
    reversal = past and future return have opposite sign
    """
    _validate_price_df(price_df)

    df = price_df.copy()

    past_ret = (df[close_col] / df[close_col].shift(lookback)) - 1.0
    future_close = df[close_col].shift(-horizon)
    fwd_ret = (future_close / df[close_col]) - 1.0

    df[f"past_ret_{lookback}"] = past_ret
    df[f"fwd_ret_{horizon}"] = fwd_ret

    cont_col = f"continuation_label_{lookback}_{horizon}"
    rev_col = f"reversal_label_{lookback}_{horizon}"

    continuation_values: List[Any] = []
    reversal_values: List[Any] = []

    for past_val, fwd_val in zip(past_ret.tolist(), fwd_ret.tolist()):
        if pd.isna(past_val) or pd.isna(fwd_val):
            continuation_values.append(pd.NA)
            reversal_values.append(pd.NA)
            continue

        if past_val == 0 or fwd_val == 0:
            continuation_values.append(0)
            reversal_values.append(0)
            continue

        same_sign = (past_val > 0 and fwd_val > 0) or (past_val < 0 and fwd_val < 0)
        opposite_sign = (past_val > 0 and fwd_val < 0) or (past_val < 0 and fwd_val > 0)

        continuation_values.append(1 if same_sign else 0)
        reversal_values.append(1 if opposite_sign else 0)

    df[cont_col] = continuation_values
    df[rev_col] = reversal_values

    return df


def build_trade_quality_labels(
    price_df: pd.DataFrame,
    horizon: int = 6,
    long_threshold: float = 0.01,
    short_threshold: float = -0.01,
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Builds simple V1 trade quality labels from forward returns.

    Labels:
    - fwd_ret_{horizon}
    - long_quality_{horizon}
    - short_quality_{horizon}
    - no_trade_quality_{horizon}

    Rules:
    long_quality = 1 if forward return >= long_threshold
    short_quality = 1 if forward return <= short_threshold
    no_trade_quality = 1 if short_threshold < forward return < long_threshold
    """
    _validate_price_df(price_df)

    if short_threshold >= long_threshold:
        raise ValueError("short_threshold must be less than long_threshold")

    df = price_df.copy()

    future_close = df[close_col].shift(-horizon)
    fwd_ret = (future_close / df[close_col]) - 1.0
    df[f"fwd_ret_{horizon}"] = fwd_ret

    long_col = f"long_quality_{horizon}"
    short_col = f"short_quality_{horizon}"
    no_trade_col = f"no_trade_quality_{horizon}"

    long_vals: List[Any] = []
    short_vals: List[Any] = []
    no_trade_vals: List[Any] = []

    for val in fwd_ret.tolist():
        if pd.isna(val):
            long_vals.append(pd.NA)
            short_vals.append(pd.NA)
            no_trade_vals.append(pd.NA)
            continue

        long_vals.append(1 if val >= long_threshold else 0)
        short_vals.append(1 if val <= short_threshold else 0)
        no_trade_vals.append(1 if short_threshold < val < long_threshold else 0)

    df[long_col] = long_vals
    df[short_col] = short_vals
    df[no_trade_col] = no_trade_vals

    return df


def build_label_set(
    price_df: pd.DataFrame,
    close_col: str = "close",
) -> pd.DataFrame:
    """
    Full V1 label builder.
    Keeps things simple and deterministic.
    """
    _validate_price_df(price_df)

    df = price_df.copy()

    df = build_forward_return_labels(df, horizons=[1, 3, 6, 12], close_col=close_col)
    df = build_continuation_reversal_labels(df, lookback=3, horizon=3, close_col=close_col)
    df = build_trade_quality_labels(
        df,
        horizon=6,
        long_threshold=0.01,
        short_threshold=-0.01,
        close_col=close_col,
    )

    return df


def summarize_labels(label_df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": len(label_df),
        "columns": list(label_df.columns),
    }

    interesting_cols = [
        "fwd_ret_1",
        "fwd_ret_3",
        "fwd_ret_6",
        "fwd_ret_12",
        "continuation_label_3_3",
        "reversal_label_3_3",
        "long_quality_6",
        "short_quality_6",
        "no_trade_quality_6",
    ]

    for col in interesting_cols:
        if col in label_df.columns:
            non_null = label_df[col].dropna()
            summary[col] = {
                "non_null_count": int(non_null.shape[0]),
            }
            if pd.api.types.is_numeric_dtype(non_null):
                summary[col]["mean"] = _safe_float(non_null.mean())

    return summary