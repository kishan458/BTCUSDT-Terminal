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

    print("\nINTENT ENGINE RAN SUCCESSFULLY\n")
    print("Input shape :", df.shape)
    print("Feature shape:", feature_df.shape)
    print("Intent shape :", intent_df.shape)

    important_cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "body_to_range_ratio",
        "upper_wick_to_range_ratio",
        "lower_wick_to_range_ratio",
        "close_location_value",
        "range_expansion_score",
        "overlap_ratio_vs_prev_bar",
        "progress_efficiency_short",
        "dominant_intent",
        "intent_confidence",
        "intent_score_top",
        "intent_score_second",
        "intent_score_strong_bullish_continuation",
        "intent_score_weak_bullish_continuation",
        "intent_score_strong_bearish_continuation",
        "intent_score_weak_bearish_continuation",
        "intent_score_buy_rejection",
        "intent_score_sell_rejection",
        "intent_score_buy_absorption_candidate",
        "intent_score_sell_absorption_candidate",
        "intent_score_indecision",
        "intent_score_inside_compression",
        "intent_score_outside_expansion",
        "intent_score_exhaustion_up_candidate",
        "intent_score_exhaustion_down_candidate",
    ]

    print("\nLAST 5 ROWS:\n")
    print(intent_df[important_cols].tail(5).to_string(index=False))

    print("\nLATEST ROW SUMMARY:\n")
    latest = intent_df.iloc[-1]
    print("dominant_intent   :", latest["dominant_intent"])
    print("intent_confidence :", latest["intent_confidence"])

    score_cols = [col for col in intent_df.columns if col.startswith("intent_score_")]
    print("\nLATEST ROW INTENT SCORES:\n")
    print(intent_df[score_cols].tail(1).to_string(index=False))

    print("\nCHECKING REQUIRED OUTPUT COLUMNS:\n")
    required_cols = [
        "dominant_intent",
        "intent_confidence",
        "intent_score_top",
        "intent_score_second",
        "intent_score_strong_bullish_continuation",
        "intent_score_weak_bullish_continuation",
        "intent_score_strong_bearish_continuation",
        "intent_score_weak_bearish_continuation",
        "intent_score_buy_rejection",
        "intent_score_sell_rejection",
        "intent_score_buy_absorption_candidate",
        "intent_score_sell_absorption_candidate",
        "intent_score_indecision",
        "intent_score_inside_compression",
        "intent_score_outside_expansion",
        "intent_score_exhaustion_up_candidate",
        "intent_score_exhaustion_down_candidate",
    ]
    missing = [col for col in required_cols if col not in intent_df.columns]
    if missing:
        print("Missing columns:", missing)
    else:
        print("All required intent columns are present.")


if __name__ == "__main__":
    main()