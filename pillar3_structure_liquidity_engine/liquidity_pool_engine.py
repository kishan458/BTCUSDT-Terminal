import pandas as pd


def validate_ohlcv(df: pd.DataFrame):
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError("Missing required OHLCV columns")
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")


def _to_native(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def find_swing_highs(highs, left=3, right=3):
    idx = []
    for i in range(left, len(highs) - right):
        if highs[i] > max(highs[i - left:i]) and highs[i] >= max(highs[i + 1:i + right + 1]):
            idx.append(i)
    return idx


def find_swing_lows(lows, left=3, right=3):
    idx = []
    for i in range(left, len(lows) - right):
        if lows[i] < min(lows[i - left:i]) and lows[i] <= min(lows[i + 1:i + right + 1]):
            idx.append(i)
    return idx


def cluster_levels(levels, indices, current_price, total_bars, threshold_pct=0.0015, is_major=False):
    if not levels:
        return []

    combined = sorted(zip(levels, indices), key=lambda x: x[0])
    clusters = [[combined[0]]]

    for level, idx in combined[1:]:
        cluster_levels_only = [x[0] for x in clusters[-1]]
        cluster_mean = sum(cluster_levels_only) / len(cluster_levels_only)

        if abs(level - cluster_mean) / max(cluster_mean, 1e-9) <= threshold_pct:
            clusters[-1].append((level, idx))
        else:
            clusters.append([(level, idx)])

    results = []

    for cluster in clusters:
        levels_only = [x[0] for x in cluster]
        indices_only = [x[1] for x in cluster]

        level = sum(levels_only) / len(levels_only)
        touches = len(cluster)
        last_touch_index = max(indices_only)
        first_touch_index = min(indices_only)

        recency_score = last_touch_index / max(total_bars - 1, 1)
        spacing_score = min(1.0, (last_touch_index - first_touch_index) / max(total_bars * 0.25, 1))
        distance_pct = abs(level - current_price) / max(current_price, 1e-9)
        proximity_score = max(0.0, 1.0 - (distance_pct / 0.03))
        touch_score = min(1.0, touches / 3.0)
        major_bonus = 1.0 if is_major else 0.0

        score = (
            0.30 * touch_score +
            0.20 * recency_score +
            0.15 * spacing_score +
            0.15 * proximity_score +
            0.20 * major_bonus
        )

        results.append({
            "level": float(level),
            "touches": int(touches),
            "last_touch_index": int(last_touch_index),
            "score": float(score),
            "distance_pct": float(distance_pct * 100.0),
            "is_major": bool(is_major),
        })

    return results


def _merge_cluster_lists(clusters):
    merged = {}
    for c in clusters:
        key = round(c["level"], 2)
        if key not in merged or c["score"] > merged[key]["score"]:
            merged[key] = c
    return list(merged.values())


def select_side_candidate(clusters, current_price, side):
    if side == "BUY":
        candidates = [c for c in clusters if c["level"] > current_price]
    else:
        candidates = [c for c in clusters if c["level"] < current_price]

    if not candidates:
        return None, []

    strong_candidates = [
        c for c in candidates
        if (
            c["touches"] >= 2
            or c["is_major"]
            or (c["score"] >= 0.52 and c["distance_pct"] <= 1.5)
        )
    ]

    if not strong_candidates:
        strong_candidates = candidates

    selected = max(
        strong_candidates,
        key=lambda x: (
            x["score"],
            -x["distance_pct"],
            x["touches"],
            1 if x["is_major"] else 0,
        ),
    )
    return selected, candidates


def _effective_magnet_strength(cluster):
    if not cluster:
        return None

    raw_score = float(cluster["score"])
    distance_pct = float(cluster["distance_pct"])

    if distance_pct <= 1.0:
        distance_penalty = 0.0
    else:
        distance_penalty = min(0.50, (distance_pct - 1.0) * 0.07)

    major_bonus = 0.03 if cluster["is_major"] else 0.0
    touch_bonus = min(0.10, max(cluster["touches"] - 1, 0) * 0.05)

    effective_score = raw_score - distance_penalty + major_bonus + touch_bonus
    return effective_score


def _is_actionable(cluster):
    if not cluster:
        return False
    return float(cluster["distance_pct"]) <= 2.0


def run_liquidity_pool_engine(df: pd.DataFrame):
    validate_ohlcv(df)
    df = df.reset_index(drop=True).copy()

    current_price = float(df["close"].iloc[-1])
    total_bars = len(df)

    internal_high_idx = find_swing_highs(df["high"].values, left=3, right=3)
    internal_low_idx = find_swing_lows(df["low"].values, left=3, right=3)

    major_high_idx = find_swing_highs(df["high"].values, left=5, right=5)
    major_low_idx = find_swing_lows(df["low"].values, left=5, right=5)

    internal_highs = [float(df.loc[i, "high"]) for i in internal_high_idx]
    internal_lows = [float(df.loc[i, "low"]) for i in internal_low_idx]

    major_highs = [float(df.loc[i, "high"]) for i in major_high_idx]
    major_lows = [float(df.loc[i, "low"]) for i in major_low_idx]

    internal_high_clusters = cluster_levels(
        internal_highs, internal_high_idx, current_price, total_bars, is_major=False
    )
    internal_low_clusters = cluster_levels(
        internal_lows, internal_low_idx, current_price, total_bars, is_major=False
    )
    major_high_clusters = cluster_levels(
        major_highs, major_high_idx, current_price, total_bars, is_major=True
    )
    major_low_clusters = cluster_levels(
        major_lows, major_low_idx, current_price, total_bars, is_major=True
    )

    high_clusters = _merge_cluster_lists(internal_high_clusters + major_high_clusters)
    low_clusters = _merge_cluster_lists(internal_low_clusters + major_low_clusters)

    nearest_buy_side, buy_candidates = select_side_candidate(high_clusters, current_price, "BUY")
    nearest_sell_side, sell_candidates = select_side_candidate(low_clusters, current_price, "SELL")

    nearest_liquidity_magnet = None
    dominant_side = "NONE"

    buy_effective = _effective_magnet_strength(nearest_buy_side)
    sell_effective = _effective_magnet_strength(nearest_sell_side)

    buy_actionable = _is_actionable(nearest_buy_side)
    sell_actionable = _is_actionable(nearest_sell_side)

    if nearest_buy_side and nearest_sell_side:
        if buy_actionable and not sell_actionable:
            nearest_liquidity_magnet = nearest_buy_side["level"]
            dominant_side = "BUY_SIDE"
        elif sell_actionable and not buy_actionable:
            nearest_liquidity_magnet = nearest_sell_side["level"]
            dominant_side = "SELL_SIDE"
        else:
            if buy_effective > sell_effective:
                nearest_liquidity_magnet = nearest_buy_side["level"]
                dominant_side = "BUY_SIDE"
            elif sell_effective > buy_effective:
                nearest_liquidity_magnet = nearest_sell_side["level"]
                dominant_side = "SELL_SIDE"
            else:
                if nearest_buy_side["distance_pct"] <= nearest_sell_side["distance_pct"]:
                    nearest_liquidity_magnet = nearest_buy_side["level"]
                    dominant_side = "BUY_SIDE"
                else:
                    nearest_liquidity_magnet = nearest_sell_side["level"]
                    dominant_side = "SELL_SIDE"
    elif nearest_buy_side:
        nearest_liquidity_magnet = nearest_buy_side["level"]
        dominant_side = "BUY_SIDE"
    elif nearest_sell_side:
        nearest_liquidity_magnet = nearest_sell_side["level"]
        dominant_side = "SELL_SIDE"

    return {
        "current_price": _to_native(current_price),
        "buy_side_liquidity": _to_native(nearest_buy_side["level"]) if nearest_buy_side else None,
        "sell_side_liquidity": _to_native(nearest_sell_side["level"]) if nearest_sell_side else None,
        "nearest_liquidity_magnet": _to_native(nearest_liquidity_magnet),
        "dominant_liquidity_side": dominant_side,
        "buy_side_clusters": buy_candidates,
        "sell_side_clusters": sell_candidates,
    }