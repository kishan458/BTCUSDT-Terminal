import pandas as pd

from pillar4_candle_engine.candle_features_engine import (
    CandleFeatureConfig,
    OhlcColumns,
    add_candle_features,
)


def main() -> None:
    # Replace this with your real BTC dataframe loading later.
    # For now this is only a test dataset to verify the engine works correctly.
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=12, freq="15min"),
            "open":   [100, 102, 101, 104, 106, 105, 107, 110, 109, 112, 111, 115],
            "high":   [103, 103, 105, 107, 108, 108, 111, 112, 113, 114, 116, 118],
            "low":    [ 99, 100, 100, 103, 104, 103, 106, 108, 107, 110, 109, 113],
            "close":  [102, 101, 104, 106, 105, 107, 110, 109, 112, 111, 115, 117],
            "volume": [10, 12, 11, 13, 12, 14, 16, 15, 18, 17, 20, 22],
        }
    )

    config = CandleFeatureConfig(
        atr_window=14,
        range_mean_window=10,
        body_mean_window=10,
        zscore_window=10,
        overlap_window_short=3,
        overlap_window_long=5,
        progress_window_short=3,
        progress_window_medium=5,
        progress_window_long=8,
        persistence_window_short=3,
        persistence_window_medium=5,
        persistence_window_long=8,
        realized_vol_window=10,
        volatility_percentile_window=10,
        range_percentile_window=10,
        body_percentile_window=10,
        contraction_window=5,
        inside_outside_window=5,
        entropy_window=5,
        rolling_wick_window=5,
    )

    feature_df = add_candle_features(
        df=df,
        config=config,
        columns=OhlcColumns(
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            timestamp="timestamp",
        ),
        atr_method="wilder",
        copy=True,
    )

    print("\nENGINE RAN SUCCESSFULLY\n")
    print("Input shape :", df.shape)
    print("Output shape:", feature_df.shape)

    important_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "body_size",
        "full_range",
        "upper_wick",
        "lower_wick",
        "body_to_range_ratio",
        "close_location_value",
        "true_range",
        "atr",
        "atr_scaled_range",
        "range_expansion_score",
        "overlap_ratio_vs_prev_bar",
        "progress_efficiency_short",
        "same_direction_body_count",
        "inside_bar_flag",
        "outside_bar_flag",
        "realized_volatility",
        "parkinson_vol",
        "garman_klass_vol",
        "rogers_satchell_vol",
        "yang_zhang_vol",
        "ohlc_constraints_ok",
        "data_quality_ok",
    ]

    print("\nLAST 5 ROWS OF IMPORTANT COLUMNS:\n")
    print(feature_df[important_cols].tail(5).to_string(index=False))

    print("\nCHECKING FOR REQUIRED COLUMNS:\n")
    required_check_cols = [
        "body_size",
        "full_range",
        "upper_wick",
        "lower_wick",
        "body_to_range_ratio",
        "upper_wick_to_range_ratio",
        "lower_wick_to_range_ratio",
        "wick_imbalance",
        "close_location_value",
        "midpoint_displacement",
        "true_range",
        "atr",
        "atr_scaled_range",
        "atr_scaled_body",
        "range_expansion_score",
        "body_expansion_score",
        "overlap_ratio_vs_prev_bar",
        "progress_efficiency_short",
        "progress_efficiency_medium",
        "progress_efficiency_long",
        "realized_volatility",
        "parkinson_vol",
        "garman_klass_vol",
        "rogers_satchell_vol",
        "yang_zhang_vol",
        "direction_entropy",
        "close_location_entropy",
        "ohlc_constraints_ok",
        "data_quality_ok",
    ]

    missing = [col for col in required_check_cols if col not in feature_df.columns]
    if missing:
        print("Missing columns:", missing)
    else:
        print("All required columns are present.")

    print("\nNAN COUNTS FOR SOME KEY FEATURES:\n")
    nan_cols = [
        "body_size",
        "full_range",
        "body_to_range_ratio",
        "close_location_value",
        "true_range",
        "atr",
        "overlap_ratio_vs_prev_bar",
        "realized_volatility",
        "yang_zhang_vol",
    ]
    print(feature_df[nan_cols].isna().sum())

    print("\nLATEST ROW FULL SNAPSHOT:\n")
    print(feature_df.iloc[-1].to_string())


if __name__ == "__main__":
    main()