from collections import Counter
from statistics import mean

def aggregate_institutional_sentiment(articles):
    """
    articles: list of dicts with keys:
      - sentiment: 'positive' | 'negative' | 'neutral'
      - confidence: float (0–1)
      - source: str
      - headline: str
    """

    if not articles:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "drivers": []
        }

    # weighted vote
    weighted_scores = []
    drivers = []

    for a in articles:
        score = a["confidence"]
        if a["sentiment"] == "negative":
            score *= -1
        elif a["sentiment"] == "neutral":
            score *= 0

        weighted_scores.append(score)
        drivers.append(a["headline"])

    avg_score = mean(weighted_scores)

    if avg_score > 0.15:
        final_sentiment = "positive"
    elif avg_score < -0.15:
        final_sentiment = "negative"
    else:
        final_sentiment = "neutral"

    return {
        "sentiment": final_sentiment,
        "confidence": round(abs(avg_score), 3),
        "drivers": drivers[:5]
    }
