#  Updated System Overview (LOCKED)

Your **BTC/USDT terminal** now consists of **8 CORE ENGINES** (each independent, but connected):

### BTC/USDT TERMINAL
* **Pillar 1:** Global Sentiment & Narrative Engine
* **Pillar 2:** BTC Market Memory Engine
* **Pillar 3:** Structure & Liquidity Engine
* **Pillar 4:** Candle Intelligence Engine
* **Pillar 5:** Regime & Cycle Engine
* **Pillar 6:** High-Impact Event Intelligence
* **Pillar 7:** Multi-Agent ML Council
* **Pillar 8:** Decision, Risk & Backtesting Engine

---

##  PILLAR 1 — GLOBAL SENTIMENT & NARRATIVE ENGINE (BTC-ONLY)
This is where we start. Always.

**Goal** Understand:
* What the world is saying about BTC
* Who is saying it
* Whether it’s real or hype
* Whether the narrative is early, mature, or exhausted

**Covers:**
* Institutional sentiment
* Retail / hype sentiment
* Narrative tracking
* Sentiment divergence
* Event-driven sentiment shocks

**This pillar alone should already help you:**
* avoid tops
* avoid fake bullish news
* respect uncertainty

*We will go extreme depth here.*

---

##  PILLAR 2 — BTC MARKET MEMORY ENGINE
**Goal:** “Has BTC seen this before? And what usually happened next?”

**This pillar builds BTC-specific intuition:**
* How BTC reacts to funding extremes
* How it behaves on weekends
* How it reacts post-event vs pre-event
* Session behavior (Asia / London / NY)

*This turns raw data into context.*

---

##  PILLAR 3 — STRUCTURE & LIQUIDITY ENGINE
**Goal:** “Where is pain? Who is trapped? Where is price attracted?”

**Covers:**
* Liquidity pools
* Stop hunts
* Fake breakouts
* Liquidation cascades
* Funding exploitation

*This is where retail loses and smart money wins.*

---

##  PILLAR 4 — CANDLE INTELLIGENCE ENGINE
**Goal:** “What does THIS candle mean in THIS context?”

**Covers:**
* Candle anatomy
* Absorption vs aggression
* Context-aware interpretation
* Candle intent (continuation vs manipulation)

*This pillar makes charts talk.*

---

##  PILLAR 5 — REGIME & CYCLE ENGINE
**Goal:** “What kind of market is BTC in right now?”

**Covers:**
* Macro regime
* BTC cycle phase
* Range vs trend
* Distribution vs accumulation

*This prevents strategy mismatch.*

---

##  PILLAR 6 — HIGH-IMPACT EVENT ENGINE
**Goal:** “Is price reacting to information or uncertainty?”

**Covers:**
* Fed decisions
* Powell tone analysis
* Political shocks
* ETF events
* Risk-off vs risk-on transitions

*This pillar tells you when NOT to trade.*

---

##  PILLAR 7 — MULTI-AGENT ML COUNCIL
**Goal:** “What would different market participants do right now?”

**Agents:**
*  Professor trader
*  Retail trader
*  Institutional actor

*This pillar is comparison, not prediction.*

---

##  PILLAR 8 — DECISION, RISK & BACKTESTING ENGINE
**Goal:** “Should I trade? How much? Or stand down?”

**Covers:**
* Trade / no-trade
* Conviction sizing
* Risk warnings
* Monte Carlo stress tests
* AI chart backtesting

*This is where money is protected.*

---

##  IMPORTANT: Build Order (LOCK THIS)
We will build in this order — no exceptions:

1.  **Pillar 1** – Sentiment & Narrative
2.  **Pillar 6** – High-Impact Events
3.  **Pillar 5** – Regime & Cycle
4.  **Pillar 3** – Structure & Liquidity
5.  **Pillar 4** – Candle Intelligence
6.  **Pillar 2** – Market Memory
7.  **Pillar 7** – ML Council
8.  **Pillar 8** – Decision & Backtesting


Curently working Pillar-4 Ideal Output:
{
  "asset": "BTCUSDT",
  "timestamp_utc": "2026-03-19T05:45:00Z",
  "timeframe": "15m",
  "lookback_bars_used": 30,

  "candle_summary": {
    "dominant_intent": "STRONG_BULLISH_CONTINUATION",
    "intent_confidence": 0.78,
    "momentum_state": "ACCELERATING",
    "control_state": "BUYERS_IN_CONTROL",
    "expansion_state": "EXPANDING",
    "overlap_state": "LOW_OVERLAP",
    "follow_through_quality": "MODERATE",
    "exhaustion_state": "NONE"
  },

  "latest_candle_features": {
    "direction": 1,
    "body_size": 182.4,
    "full_range": 256.8,
    "upper_wick": 21.3,
    "lower_wick": 53.1,

    "body_to_range_ratio": 0.7103,
    "upper_wick_to_range_ratio": 0.0830,
    "lower_wick_to_range_ratio": 0.2068,
    "total_wick_to_range_ratio": 0.2898,
    "wick_imbalance": 0.1238,
    "bar_efficiency": 0.7103,

    "close_location_value": 0.9172,
    "open_location_value": 0.2068,
    "midpoint_displacement": 0.4172,

    "gap_from_prev_close": 12.6,
    "close_to_close_return": 165.8,
    "high_extension_vs_prev_high": 84.0,
    "low_extension_vs_prev_low": -22.0,
    "inside_bar_flag": 0,
    "outside_bar_flag": 0,

    "true_range": 271.4,
    "atr_scaled_range": 1.34,
    "atr_scaled_body": 0.95,
    "atr_scaled_gap": 0.07,

    "range_expansion_score": 1.41,
    "body_expansion_score": 1.58,
    "range_zscore_20": 1.92,
    "body_zscore_20": 2.11,

    "overlap_ratio_vs_prev_bar": 0.18
  },

  "volatility_context": {
    "atr_14": 191.8,
    "realized_volatility_20": 0.023,
    "realized_volatility_percentile_90d": 0.71,
    "parkinson_vol": 0.019,
    "garman_klass_vol": 0.021,
    "rogers_satchell_vol": 0.022,
    "yang_zhang_vol": 0.024,
    "range_shock_percentile_90d": 0.83,
    "body_shock_percentile_90d": 0.88
  },

  "multi_candle_context": {
    "same_direction_body_count": 4,
    "same_direction_close_count": 4,
    "close_near_high_count_3": 2,
    "close_near_high_count_5": 3,
    "close_near_low_count_5": 0,

    "rolling_body_dominance_3": 0.61,
    "rolling_body_dominance_5": 0.57,
    "rolling_wick_dominance_5": -0.09,
    "rolling_sign_consistency_5": 0.80,

    "avg_overlap_ratio_3": 0.22,
    "avg_overlap_ratio_5": 0.31,

    "net_progress_3": 286.0,
    "net_progress_5": 354.0,
    "progress_efficiency_3": 0.63,
    "progress_efficiency_5": 0.49,

    "inside_bar_frequency_10": 0.10,
    "outside_bar_frequency_10": 0.20,
    "range_contraction_streak": 0,
    "expansion_after_compression_score": 0.72,
    "post_expansion_fade_score": 0.18
  },

  "intent_scores": {
    "bullish_continuation_score": 0.78,
    "bearish_continuation_score": 0.04,
    "indecision_score": 0.11,
    "buy_rejection_score": 0.09,
    "sell_rejection_score": 0.19,
    "buy_absorption_candidate_score": 0.08,
    "sell_absorption_candidate_score": 0.22,
    "inside_compression_score": 0.03,
    "outside_expansion_score": 0.34,
    "exhaustion_up_candidate_score": 0.16,
    "exhaustion_down_candidate_score": 0.02
  },

  "absorption": {
    "buy_absorption_score": 0.18,
    "sell_absorption_score": 0.61,
    "buy_rejection_score": 0.24,
    "sell_rejection_score": 0.57,
    "dominant_absorption": "SELL_ABSORPTION",
    "dominant_rejection": "SELL_REJECTION",
    "absorption_confidence": 0.63,
    "failed_upside_extension_count_5": 2,
    "failed_downside_extension_count_5": 0
  },

  "breakout_analysis": {
    "reference_range_high": 84210.5,
    "reference_range_low": 83660.2,
    "breakout_direction": "UPSIDE",
    "breach_magnitude": 74.5,
    "close_outside_range_ratio": 0.62,
    "acceptance_score": 0.54,
    "failure_score": 0.29,
    "retrace_ratio": 0.21,
    "wick_penalty": 0.11,
    "breakout_quality_score": 0.59,
    "fake_breakout_risk": 0.38,
    "breakout_validity": "UNCONFIRMED",
    "breakout_state": "ACCEPTED_BUT_EARLY"
  },

  "pressure": {
    "buying_pressure_score": 0.68,
    "selling_pressure_score": 0.29,
    "net_pressure_score": 0.39,
    "pressure_bias": "BUY_PRESSURE",
    "pressure_strength": "MODERATE"
  },

  "sequence_similarity": {
    "enabled": false,
    "matched_pattern_family": null,
    "similarity_score": null,
    "historical_outcome_bias": null
  },

  "context_alignment": {
    "pillar3_liquidity_alignment": "ALIGNED_BULLISH",
    "nearest_liquidity_magnet": "BUY_SIDE_LIQUIDITY_ABOVE",
    "distance_to_nearest_liquidity_magnet_atr": 0.84,
    "candle_vs_liquidity_story": "Current candle behavior supports upside liquidity draw, but breakout acceptance remains incomplete.",
    "pillar5_regime_alignment": "ALIGNED_WITH_EXPANSION_REGIME",
    "pillar6_event_context": "NOT_AVAILABLE",
    "pillar2_memory_alignment": "NOT_AVAILABLE"
  },

  "risk_flags": [
    "Breakout acceptance not fully confirmed",
    "Sell absorption proxy elevated above local highs",
    "Recent expansion is strong, but follow-through is still only moderate"
  ],

  "diagnostics": {
    "data_quality_ok": true,
    "zero_range_bars_in_window": 0,
    "nan_count": 0,
    "ohlc_constraints_ok": true
  },

  "ai_overview": "Latest candle behavior is directionally bullish and mechanically strong, with high close quality, above-average body expansion, and low overlap versus the prior bar. Recent sequencing supports buyer control and continuation, but acceptance above the local range is not fully proven yet. Upside progress remains efficient, though the engine still detects some sell-absorption risk near the highs, which keeps breakout validity in the unconfirmed bucket rather than clean acceptance."
}
