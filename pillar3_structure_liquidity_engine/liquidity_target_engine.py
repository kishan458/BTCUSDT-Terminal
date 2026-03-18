import pandas as pd

from pillar3_structure_liquidity_engine.liquidity_pool_engine import run_liquidity_pool_engine


def _to_native(value):
    if value is None:
        return None
    if pd.isna(value):
        return None
    return float(value)


def _target_rank_score(cluster):
    score = float(cluster["score"])
    distance_pct = float(cluster["distance_pct"])
    touches = int(cluster["touches"])
    is_major = bool(cluster["is_major"])

    proximity_bonus = max(0.0, 1.2 - min(distance_pct, 6.0) / 6.0)
    touch_bonus = min(0.15, max(touches - 1, 0) * 0.075)
    major_bonus = 0.08 if is_major else 0.0

    return score + proximity_bonus + touch_bonus + major_bonus


def run_liquidity_target_engine(df: pd.DataFrame):
    if len(df) < 30:
        raise ValueError("Need at least 30 rows of data")

    df = df.reset_index(drop=True).copy()

    liquidity = run_liquidity_pool_engine(df)
    current_price = float(df["close"].iloc[-1])

    buy_clusters = liquidity["buy_side_clusters"]
    sell_clusters = liquidity["sell_side_clusters"]

    upside_candidates = [c for c in buy_clusters if float(c["level"]) > current_price]
    downside_candidates = [c for c in sell_clusters if float(c["level"]) < current_price]

    upside_ranked = sorted(
        upside_candidates,
        key=lambda x: (-_target_rank_score(x), x["distance_pct"])
    )
    downside_ranked = sorted(
        downside_candidates,
        key=lambda x: (-_target_rank_score(x), x["distance_pct"])
    )

    # Keep a manageable ladder: top 3 each side
    top_upside = upside_ranked[:3]
    top_downside = downside_ranked[:3]

    next_upside_target = top_upside[0]["level"] if top_upside else None
    next_downside_target = top_downside[0]["level"] if top_downside else None

    nearest_target = None
    directional_candidates = []

    if next_upside_target is not None:
        directional_candidates.append(next_upside_target)
    if next_downside_target is not None:
        directional_candidates.append(next_downside_target)

    if directional_candidates:
        nearest_target = min(directional_candidates, key=lambda x: abs(x - current_price))

    # Final ordered list = nearest directional first, then remaining ranked ladder
    all_targets = []
    seen = set()

    for cluster in top_upside + top_downside:
        level = float(cluster["level"])
        key = round(level, 2)
        if key not in seen:
            seen.add(key)
            all_targets.append(level)

    ordered_targets = sorted(all_targets, key=lambda x: abs(x - current_price))

    return {
        "nearest_liquidity_target": _to_native(nearest_target),
        "next_upside_target": _to_native(next_upside_target),
        "next_downside_target": _to_native(next_downside_target),
        "liquidity_targets": [_to_native(x) for x in ordered_targets]
    }