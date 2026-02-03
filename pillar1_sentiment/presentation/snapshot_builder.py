def build_institutional_snapshot(articles: list) -> dict:
    """
    articles: output AFTER sentiment + tone processing
    """

    return {
        "asset": "BTC",
        "articles": articles
    }
