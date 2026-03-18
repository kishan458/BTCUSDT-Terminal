import sqlite3
import pandas as pd

from pillar3_structure_liquidity_engine.pillar3_output import run_pillar3_output
from pillar3_structure_liquidity_engine.liquidity_pool_engine import run_liquidity_pool_engine


def load_btc_1h(limit=300):
    conn = sqlite3.connect("database/btc_terminal.db")

    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM btc_price_1h
    ORDER BY timestamp DESC
    LIMIT {limit}
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        raise ValueError("No BTC 1h data found in btc_price_1h table")

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def pct_distance(current_price, level):
    if level is None:
        return None
    return ((level - current_price) / current_price) * 100.0


def format_level_with_pct(label, current_price, level):
    if level is None:
        return f"{label} = None"

    pct = pct_distance(current_price, level)
    sign = "+" if pct >= 0 else ""
    return f"{label} = {level:.2f} (~ {sign}{pct:.2f}%)"


def get_liquidity_comments(current_price, buy_side, sell_side):
    comments = []

    buy_pct = pct_distance(current_price, buy_side) if buy_side is not None else None
    sell_pct = pct_distance(current_price, sell_side) if sell_side is not None else None
    sell_abs = abs(sell_pct) if sell_pct is not None else None

    if buy_pct is not None and sell_abs is not None:
        if buy_pct < sell_abs * 0.35:
            comments.append("Upside liquidity is materially closer than downside liquidity; an upward sweep is the easier path right now.")
        elif buy_pct < sell_abs * 0.60:
            comments.append("Buy-side liquidity is meaningfully closer than sell-side liquidity; near-term price attraction favors the upside.")
        elif buy_pct < sell_abs * 0.85:
            comments.append("Upside liquidity still has the proximity advantage, but the imbalance is moderate rather than extreme.")
        elif abs(buy_pct - sell_abs) <= 0.50:
            comments.append("Liquidity is relatively balanced on both sides; directional resolution likely needs a stronger trigger.")
        elif sell_abs < buy_pct * 0.60:
            comments.append("Downside liquidity is materially closer than upside liquidity; the easier path is lower.")
        else:
            comments.append("Liquidity is not one-sided enough to justify a strong directional claim from distance alone.")

        spread = abs(sell_abs - buy_pct)
        if spread >= 4.0:
            comments.append("The liquidity distance gap is wide, which usually makes the nearer side the dominant short-horizon magnet.")
        elif spread >= 2.0:
            comments.append("There is a clear but not extreme distance advantage toward one side of the book.")
        else:
            comments.append("The liquidity distance spread is not large; follow-through quality matters more here than raw location.")

    elif buy_pct is not None:
        comments.append("Only actionable buy-side liquidity is being detected; the upside is the main visible magnet.")
    elif sell_abs is not None:
        comments.append("Only actionable sell-side liquidity is being detected; the downside is the main visible magnet.")
    else:
        comments.append("No actionable liquidity pools are being detected under current rules; structure may be too noisy or too diffuse.")

    return comments


def print_terminology_index():
    print("\nTERMINOLOGY INDEX")
    print("- BUY SIDE LIQUIDITY: stop-rich liquidity resting above current price, usually around swing highs / short stops.")
    print("- SELL SIDE LIQUIDITY: stop-rich liquidity resting below current price, usually around swing lows / long stops.")
    print("- TOUCHES: number of structural interactions contributing to a liquidity cluster.")
    print("- LAST TOUCH INDEX: how recent the last structural interaction was inside the analysis window.")
    print("- SCORE: internal strength score of the liquidity pool based on structure, proximity, recency, and significance.")
    print("- DISTANCE PCT: percentage distance between current price and that liquidity level.")
    print("- IS MAJOR: whether the level comes from a major swing rather than a smaller internal swing.")


def test_pillar3_real_data():
    df = load_btc_1h(limit=300)

    liq = run_liquidity_pool_engine(df)
    result = run_pillar3_output(df, asset="BTCUSDT")

    current_price = float(liq["current_price"])
    buy_side = liq["buy_side_liquidity"]
    sell_side = liq["sell_side_liquidity"]

    print("ROWS LOADED:", len(df))
    print("FIRST TIMESTAMP:", df["timestamp"].iloc[0])
    print("LAST TIMESTAMP:", df["timestamp"].iloc[-1])

    print("\nLIQUIDITY SNAPSHOT")
    print(f"CURRENT PRICE = {current_price:.2f}")
    print(format_level_with_pct("BUY SIDE LIQUIDITY", current_price, buy_side))
    print(format_level_with_pct("SELL SIDE LIQUIDITY", current_price, sell_side))

    print("\nLIQUIDITY COMMENTS")
    for comment in get_liquidity_comments(current_price, buy_side, sell_side):
        print(f"- {comment}")

    print_terminology_index()

    print("\nBUY CLUSTERS (top 5):")
    for row in liq["buy_side_clusters"][:5]:
        print(row)

    print("\nSELL CLUSTERS (top 5):")
    for row in liq["sell_side_clusters"][:5]:
        print(row)

    print("\nFINAL PILLAR 3 OUTPUT:")
    print(result)


if __name__ == "__main__":
    test_pillar3_real_data()