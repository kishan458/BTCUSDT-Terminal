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

    print("\nMULTI-CANDLE CONTEXT ENGINE RAN SUCCESSFULLY\n")
    print("Input shape   :", df.shape)
    print("Feature shape :", feature_df.shape)
    print("Intent shape  :", intent_df.shape)
    print("Context shape :", context_df.shape)

    important_cols = [
        "timestamp",
        "close",
        "dominant_intent",
        "intent_confidence",
        "momentum_state",
        "control_state",
        "expansion_state",
        "overlap_state",
        "follow_through_quality",
        "exhaustion_state",
        "directional_bias_score",
        "momentum_continuation_score",
        "momentum_acceleration_score",
        "stalling_score",
        "reversal_risk_score",
        "buyer_control_score",
        "seller_control_score",
        "follow_through_score",
        "upside_exhaustion_score",
        "downside_exhaustion_score",
    ]

    print("\nLAST 5 ROWS:\n")
    print(context_df[important_cols].tail(5).to_string(index=False))

    print("\nLATEST ROW SUMMARY:\n")
    latest = context_df.iloc[-1]
    print("dominant_intent            :", latest["dominant_intent"])
    print("momentum_state             :", latest["momentum_state"])
    print("control_state              :", latest["control_state"])
    print("expansion_state            :", latest["expansion_state"])
    print("overlap_state              :", latest["overlap_state"])
    print("follow_through_quality     :", latest["follow_through_quality"])
    print("exhaustion_state           :", latest["exhaustion_state"])
    print("momentum_continuation_score:", latest["momentum_continuation_score"])
    print("buyer_control_score        :", latest["buyer_control_score"])
    print("seller_control_score       :", latest["seller_control_score"])
    print("follow_through_score       :", latest["follow_through_score"])

    print("\nCHECKING REQUIRED OUTPUT COLUMNS:\n")
    required_cols = [
        "momentum_state",
        "control_state",
        "expansion_state",
        "overlap_state",
        "follow_through_quality",
        "exhaustion_state",
        "directional_bias_score",
        "momentum_continuation_score",
        "momentum_acceleration_score",
        "stalling_score",
        "reversal_risk_score",
        "buyer_control_score",
        "seller_control_score",
        "follow_through_score",
        "upside_exhaustion_score",
        "downside_exhaustion_score",
    ]

    missing = [col for col in required_cols if col not in context_df.columns]
    if missing:
        print("Missing columns:", missing)
    else:
        print("All required multi-candle context columns are present.")


if __name__ == "__main__":
    main()