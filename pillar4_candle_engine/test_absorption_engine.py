import pandas as pd

from pillar4_candle_engine.candle_features_engine import (
    CandleFeatureConfig,
    OhlcColumns,
    add_candle_features,
)
from pillar4_candle_engine.candle_intent_engine import (
    CandleIntentConfig,
    classify_candle_intents,
)
from pillar4_candle_engine.multi_candle_context_engine import (
    MultiCandleContextConfig,
    build_multi_candle_context,
)
from pillar4_candle_engine.absorption_engine import (
    AbsorptionConfig,
    build_absorption_context,
)


def main() -> None:
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

    feature_config = CandleFeatureConfig(
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

    intent_config = CandleIntentConfig()
    context_config = MultiCandleContextConfig()
    absorption_config = AbsorptionConfig()

    feature_df = add_candle_features(
        df=df,
        config=feature_config,
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

    intent_df = classify_candle_intents(
        feature_df=feature_df,
        config=intent_config,
    )

    context_df = build_multi_candle_context(
        feature_df=intent_df,
        config=context_config,
    )

    absorption_df = build_absorption_context(
        feature_df=context_df,
        config=absorption_config,
    )

    print("\nABSORPTION ENGINE RAN SUCCESSFULLY\n")
    print("Input shape      :", df.shape)
    print("Feature shape    :", feature_df.shape)
    print("Intent shape     :", intent_df.shape)
    print("Context shape    :", context_df.shape)
    print("Absorption shape :", absorption_df.shape)

    important_cols = [
        "timestamp",
        "close",
        "dominant_intent",
        "momentum_state",
        "control_state",
        "upper_wick_to_range_ratio",
        "lower_wick_to_range_ratio",
        "close_location_value",
        "overlap_ratio_vs_prev_bar",
        "progress_efficiency_short",
        "failed_upside_extension_flag",
        "failed_downside_extension_flag",
        "failed_upside_extension_count",
        "failed_downside_extension_count",
        "buy_rejection_score",
        "sell_rejection_score",
        "buy_absorption_score",
        "sell_absorption_score",
        "dominant_rejection",
        "dominant_absorption",
        "absorption_confidence",
    ]

    print("\nLAST 5 ROWS:\n")
    print(absorption_df[important_cols].tail(5).to_string(index=False))

    latest = absorption_df.iloc[-1]

    print("\nLATEST ROW SUMMARY:\n")
    print("dominant_intent              :", latest["dominant_intent"])
    print("momentum_state               :", latest["momentum_state"])
    print("control_state                :", latest["control_state"])
    print("buy_rejection_score          :", latest["buy_rejection_score"])
    print("sell_rejection_score         :", latest["sell_rejection_score"])
    print("buy_absorption_score         :", latest["buy_absorption_score"])
    print("sell_absorption_score        :", latest["sell_absorption_score"])
    print("dominant_rejection           :", latest["dominant_rejection"])
    print("dominant_absorption          :", latest["dominant_absorption"])
    print("absorption_confidence        :", latest["absorption_confidence"])
    print("failed_upside_extension_count:", latest["failed_upside_extension_count"])
    print("failed_downside_extension_count:", latest["failed_downside_extension_count"])

    print("\nCHECKING REQUIRED OUTPUT COLUMNS:\n")
    required_cols = [
        "failed_upside_extension_flag",
        "failed_downside_extension_flag",
        "failed_upside_extension_count",
        "failed_downside_extension_count",
        "buy_rejection_score",
        "sell_rejection_score",
        "buy_absorption_score",
        "sell_absorption_score",
        "dominant_rejection",
        "dominant_absorption",
        "absorption_confidence",
    ]

    missing = [col for col in required_cols if col not in absorption_df.columns]
    if missing:
        print("Missing columns:", missing)
    else:
        print("All required absorption columns are present.")


if __name__ == "__main__":
    main()