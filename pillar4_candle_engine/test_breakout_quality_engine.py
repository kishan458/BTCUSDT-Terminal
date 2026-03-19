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
from pillar4_candle_engine.breakout_quality_engine import (
    BreakoutQualityConfig,
    build_breakout_quality_context,
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
    breakout_config = BreakoutQualityConfig()

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

    breakout_df = build_breakout_quality_context(
        feature_df=absorption_df,
        config=breakout_config,
    )

    print("\nBREAKOUT QUALITY ENGINE RAN SUCCESSFULLY\n")
    print("Input shape      :", df.shape)
    print("Feature shape    :", feature_df.shape)
    print("Intent shape     :", intent_df.shape)
    print("Context shape    :", context_df.shape)
    print("Absorption shape :", absorption_df.shape)
    print("Breakout shape   :", breakout_df.shape)

    important_cols = [
        "timestamp",
        "close",
        "dominant_intent",
        "momentum_state",
        "dominant_absorption",
        "reference_range_high",
        "reference_range_low",
        "breakout_direction",
        "upside_breach_magnitude",
        "downside_breach_magnitude",
        "breach_magnitude",
        "close_outside_amount",
        "close_outside_range_ratio",
        "retrace_ratio",
        "acceptance_score",
        "failure_score",
        "wick_penalty",
        "fake_breakout_risk",
        "breakout_quality_score",
        "breakout_validity",
        "breakout_state",
    ]

    print("\nLAST 5 ROWS:\n")
    print(breakout_df[important_cols].tail(5).to_string(index=False))

    latest = breakout_df.iloc[-1]

    print("\nLATEST ROW SUMMARY:\n")
    print("dominant_intent        :", latest["dominant_intent"])
    print("momentum_state         :", latest["momentum_state"])
    print("dominant_absorption    :", latest["dominant_absorption"])
    print("reference_range_high   :", latest["reference_range_high"])
    print("reference_range_low    :", latest["reference_range_low"])
    print("breakout_direction     :", latest["breakout_direction"])
    print("breach_magnitude       :", latest["breach_magnitude"])
    print("close_outside_amount   :", latest["close_outside_amount"])
    print("close_outside_ratio    :", latest["close_outside_range_ratio"])
    print("acceptance_score       :", latest["acceptance_score"])
    print("failure_score          :", latest["failure_score"])
    print("fake_breakout_risk     :", latest["fake_breakout_risk"])
    print("breakout_quality_score :", latest["breakout_quality_score"])
    print("breakout_validity      :", latest["breakout_validity"])
    print("breakout_state         :", latest["breakout_state"])

    print("\nCHECKING REQUIRED OUTPUT COLUMNS:\n")
    required_cols = [
        "reference_range_high",
        "reference_range_low",
        "breakout_direction",
        "upside_breach_magnitude",
        "downside_breach_magnitude",
        "breach_magnitude",
        "close_outside_amount",
        "close_outside_range_ratio",
        "retrace_ratio",
        "acceptance_score",
        "failure_score",
        "wick_penalty",
        "fake_breakout_risk",
        "breakout_quality_score",
        "breakout_validity",
        "breakout_state",
    ]

    missing = [col for col in required_cols if col not in breakout_df.columns]
    if missing:
        print("Missing columns:", missing)
    else:
        print("All required breakout columns are present.")


if __name__ == "__main__":
    main()