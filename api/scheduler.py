"""
Background scheduler — runs each pillar independently on its own interval.

Refresh schedule:
  Fast  (60s)  : P3 Structure, P4 Candle, P5 Regime, P6 Events
  Medium (300s) : P1 Sentiment, P7 Council, P8 Decision
  Slow  (900s)  : P2 Memory (heavy analog retrieval)
"""

from __future__ import annotations

import asyncio
import sqlite3
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from api.cache import cache

# ── Repo root & DB ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = str(REPO_ROOT / "database" / "btc_terminal.db")

PROFESSOR_ARTIFACT = str(
    REPO_ROOT / "pillar7_ml_council" / "artifacts" /
    "professor_policy_rf_long_vs_no_trade_v1_test.pkl"
)
RETAIL_ARTIFACT = str(
    REPO_ROOT / "pillar7_ml_council" / "artifacts" /
    "retail_policy_rf_long_vs_no_action_v1_test.pkl"
)

INTERVALS = {
    "pillar1": 300,   # 5 min — FinBERT is slow
    "pillar2": 900,   # 15 min — analog retrieval is heavy
    "pillar3": 60,    # 1 min
    "pillar4": 60,    # 1 min
    "pillar5": 60,    # 1 min
    "pillar6": 60,    # 1 min
    "pillar7": 300,   # 5 min — depends on P1–P6
    "pillar8": 300,   # 5 min — depends on P7
}


# ── OHLCV loader ──────────────────────────────────────────────────────────────
def _load_ohlcv(limit: int = 300) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            f"SELECT timestamp, open, high, low, close, volume FROM btc_price_1h ORDER BY timestamp DESC LIMIT {limit}",
            conn,
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Pillar runners ────────────────────────────────────────────────────────────
def _run_pillar1() -> Dict[str, Any]:
    from pillar1_sentiment.run_pillar1 import run_pillar1
    return run_pillar1()


def _run_pillar2() -> Dict[str, Any]:
    from pillar2_market_memory_engine.memory_feature_engine import MemoryFeatureConfig
    from pillar2_market_memory_engine.memory_outcome_engine import MemoryOutcomeConfig
    from pillar2_market_memory_engine.state_signature_engine import StateSignatureConfig
    from pillar2_market_memory_engine.analog_retrieval_engine import AnalogRetrievalConfig
    from pillar2_market_memory_engine.conditional_outcome_engine import ConditionalOutcomeConfig
    from pillar2_market_memory_engine.stability_engine import StabilityConfig
    from pillar2_market_memory_engine.session_memory_engine import SessionMemoryConfig
    from pillar2_market_memory_engine.calendar_memory_engine import CalendarMemoryConfig
    from pillar2_market_memory_engine.pillar2_output import Pillar2OutputConfig, run_pillar2_output
    return run_pillar2_output(
        feature_config=MemoryFeatureConfig(db_path=DB_PATH, table_name="btc_price_1h", timestamp_col="timestamp"),
        outcome_config=MemoryOutcomeConfig(horizons=(1, 3, 6, 12)),
        signature_config=StateSignatureConfig(require_complete_core_state=True),
        retrieval_config=AnalogRetrievalConfig(min_similarity_score=0.45, max_weighted_matches=300),
        conditional_config=ConditionalOutcomeConfig(use_weighted_matches=True),
        stability_config=StabilityConfig(minimum_window_samples=10),
        session_config=SessionMemoryConfig(min_samples_per_group=25),
        calendar_config=CalendarMemoryConfig(min_samples_per_group=25),
        output_config=Pillar2OutputConfig(asset="BTCUSDT", round_decimals=4),
    )


def _run_pillar3(df: pd.DataFrame) -> Dict[str, Any]:
    from pillar3_structure_liquidity_engine.pillar3_output import run_pillar3_output
    return run_pillar3_output(df)


def _run_pillar4(df: pd.DataFrame) -> Dict[str, Any]:
    from pillar4_candle_engine.absorption_engine import AbsorptionConfig
    from pillar4_candle_engine.breakout_quality_engine import BreakoutQualityConfig
    from pillar4_candle_engine.candle_features_engine import CandleFeatureConfig, OhlcColumns
    from pillar4_candle_engine.candle_intent_engine import CandleIntentConfig
    from pillar4_candle_engine.multi_candle_context_engine import MultiCandleContextConfig
    from pillar4_candle_engine.pillar4_output import Pillar4Config, run_pillar4_candle_intelligence
    from pillar4_candle_engine.pressure_engine import PressureConfig
    config = Pillar4Config(
        candle_features=CandleFeatureConfig(
            atr_window=14, range_mean_window=20, body_mean_window=20, zscore_window=20,
            overlap_window_short=3, overlap_window_long=5, progress_window_short=3,
            progress_window_medium=5, progress_window_long=8, persistence_window_short=3,
            persistence_window_medium=5, persistence_window_long=8, realized_vol_window=20,
            volatility_percentile_window=50, range_percentile_window=50, body_percentile_window=50,
            contraction_window=5, inside_outside_window=5, entropy_window=5, rolling_wick_window=5,
        ),
        candle_intent=CandleIntentConfig(),
        multi_candle_context=MultiCandleContextConfig(),
        absorption=AbsorptionConfig(),
        breakout_quality=BreakoutQualityConfig(
            range_window=5, acceptance_close_threshold=0.55, strong_breakout_threshold=0.65,
            weak_breakout_threshold=0.40, fake_breakout_overlap_threshold=0.65, minimum_breach_threshold=0.0,
        ),
        pressure=PressureConfig(),
    )
    return run_pillar4_candle_intelligence(
        df=df, pillar4_config=config,
        columns=OhlcColumns(open="open", high="high", low="low", close="close", volume="volume", timestamp="timestamp"),
        asset="BTCUSDT", timeframe="1h", lookback_bars_used=min(30, len(df)), atr_method="wilder",
    )


def _run_pillar5() -> Dict[str, Any]:
    from pillar5_regime_cycle_engine.pillar5_output import build_pillar5_output
    return build_pillar5_output()


def _run_pillar6() -> Dict[str, Any]:
    from pillar6_high_impact_events.pillar6_output import build_pillar6_output
    return build_pillar6_output()


def _run_pillar7(p1, p2, p3, p4, p5, p6) -> Dict[str, Any]:
    from pillar7_ml_council.shared_state_builder import build_shared_state
    from pillar7_ml_council.pillar7_output import build_pillar7_output
    shared_state = build_shared_state(
        asset="BTCUSDT",
        pillar1_output=p1, pillar2_output=p2, pillar3_output=p3,
        pillar4_output=p4, pillar5_output=p5, pillar6_output=p6,
    )
    return build_pillar7_output(
        shared_state=shared_state,
        professor_artifact_path=PROFESSOR_ARTIFACT,
        retail_artifact_path=RETAIL_ARTIFACT,
        threshold=0.5,
    )


def _run_pillar8(p1, p2, p3, p4, p5, p6, p7) -> Dict[str, Any]:
    from datetime import datetime, timezone
    from pillar8_decision_risk_backtesting.pillar8_engine import run_pillar8_engine
    from pillar8_decision_risk_backtesting.run_pillar8 import (
        _adapt_p1, _adapt_p2, _adapt_p3, _adapt_p4, _adapt_p5, _adapt_p6, _adapt_p7
    )
    result = run_pillar8_engine(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        sentiment_payload=_adapt_p1(p1 or {}),
        memory_payload=_adapt_p2(p2 or {}),
        structure_payload=_adapt_p3(p3 or {}),
        candle_payload=_adapt_p4(p4 or {}),
        regime_payload=_adapt_p5(p5 or {}),
        events_payload=_adapt_p6(p6 or {}),
        council_payload=_adapt_p7(p7 or {}),
        realized_volatility=None,
        target_volatility=0.20,
        win_probability=0.52,
        payoff_ratio=1.5,
        kelly_fraction=0.25,
        max_size_fraction=1.0,
        max_leverage=3.0,
        min_leverage=1.0,
        thesis_summary="Multi-pillar BTC/USDT terminal decision output.",
    )
    return result.to_dict()


# ── Sentiment history storage ─────────────────────────────────────────────────
def _store_sentiment_snapshot(p1: Dict[str, Any]) -> None:
    """Store sentiment score in SQLite for 7-day history chart."""
    try:
        agg = p1.get("aggregate_sentiment", {}) if isinstance(p1, dict) else {}
        label = agg.get("label", "NEUTRAL")
        confidence = float(agg.get("confidence", 0.0))
        score_map = {"POSITIVE": confidence, "NEGATIVE": -confidence, "NEUTRAL": 0.0}
        score = score_map.get(label.upper(), 0.0)

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    label TEXT,
                    score REAL,
                    confidence REAL,
                    article_count INTEGER
                )
            """)
            conn.execute(
                "INSERT INTO sentiment_history (timestamp, label, score, confidence, article_count) VALUES (?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    label,
                    score,
                    confidence,
                    p1.get("article_count", 0) if isinstance(p1, dict) else 0,
                )
            )
            conn.commit()
    except Exception as e:
        print(f"[Scheduler] Failed to store sentiment snapshot: {e}")


# ── Individual pillar async tasks ─────────────────────────────────────────────
async def _task_pillar1():
    """Run Pillar 1 every 5 minutes."""
    # Import broadcast lazily to avoid circular imports
    from api.main import broadcast
    while True:
        try:
            print(f"[Scheduler] Running Pillar 1 — Sentiment...")
            result = await asyncio.get_event_loop().run_in_executor(None, _run_pillar1)
            cache.set("pillar1", result)
            _store_sentiment_snapshot(result)
            await broadcast("pillar1", result)
            print(f"[Scheduler] ✓ Pillar 1 complete")
        except Exception:
            print(f"[Scheduler] ✗ Pillar 1 failed:\n{traceback.format_exc()}")
        await asyncio.sleep(INTERVALS["pillar1"])


async def _task_pillar2():
    """Run Pillar 2 every 15 minutes."""
    from api.main import broadcast
    while True:
        try:
            print(f"[Scheduler] Running Pillar 2 — Memory...")
            result = await asyncio.get_event_loop().run_in_executor(None, _run_pillar2)
            cache.set("pillar2", result)
            await broadcast("pillar2", result)
            print(f"[Scheduler] ✓ Pillar 2 complete")
        except Exception:
            print(f"[Scheduler] ✗ Pillar 2 failed:\n{traceback.format_exc()}")
        await asyncio.sleep(INTERVALS["pillar2"])


async def _task_fast_pillars():
    """Run P3, P4, P5, P6 every 60 seconds."""
    from api.main import broadcast
    while True:
        try:
            df = await asyncio.get_event_loop().run_in_executor(None, _load_ohlcv)

            print(f"[Scheduler] Running fast pillars (P3, P4, P5, P6)...")

            p3 = await asyncio.get_event_loop().run_in_executor(None, _run_pillar3, df)
            cache.set("pillar3", p3)
            await broadcast("pillar3", p3)

            p4 = await asyncio.get_event_loop().run_in_executor(None, _run_pillar4, df)
            cache.set("pillar4", p4)
            await broadcast("pillar4", p4)

            p5 = await asyncio.get_event_loop().run_in_executor(None, _run_pillar5)
            cache.set("pillar5", p5)
            await broadcast("pillar5", p5)

            p6 = await asyncio.get_event_loop().run_in_executor(None, _run_pillar6)
            cache.set("pillar6", p6)
            await broadcast("pillar6", p6)

            print(f"[Scheduler] ✓ Fast pillars complete")

        except Exception:
            print(f"[Scheduler] ✗ Fast pillars failed:\n{traceback.format_exc()}")

        await asyncio.sleep(INTERVALS["pillar3"])


async def _task_council_and_decision():
    """Run P7 + P8 every 5 minutes, using cached upstream data."""
    from api.main import broadcast
    # Wait 60s on startup for fast pillars to populate
    await asyncio.sleep(60)
    while True:
        try:
            print(f"[Scheduler] Running Pillar 7 — Council...")
            p1 = cache.get("pillar1") or {}
            p2 = cache.get("pillar2") or {}
            p3 = cache.get("pillar3") or {}
            p4 = cache.get("pillar4") or {}
            p5 = cache.get("pillar5") or {}
            p6 = cache.get("pillar6") or {}

            p7 = await asyncio.get_event_loop().run_in_executor(
                None, _run_pillar7, p1, p2, p3, p4, p5, p6
            )
            cache.set("pillar7", p7)
            await broadcast("pillar7", p7)
            print(f"[Scheduler] ✓ Pillar 7 complete")

            print(f"[Scheduler] Running Pillar 8 — Decision...")
            p8 = await asyncio.get_event_loop().run_in_executor(
                None, _run_pillar8, p1, p2, p3, p4, p5, p6, p7
            )
            cache.set("pillar8", p8)
            await broadcast("pillar8", p8)
            print(f"[Scheduler] ✓ Pillar 8 complete")

        except Exception:
            print(f"[Scheduler] ✗ Council/Decision failed:\n{traceback.format_exc()}")

        await asyncio.sleep(INTERVALS["pillar7"])


# ── Main scheduler entry point ────────────────────────────────────────────────
async def start_scheduler():
    """
    Launch all pillar tasks concurrently.
    Called once on FastAPI startup.
    """
    print("[Scheduler] Starting all pillar tasks...")
    await asyncio.gather(
        _task_pillar1(),
        _task_pillar2(),
        _task_fast_pillars(),
        _task_council_and_decision(),
    )