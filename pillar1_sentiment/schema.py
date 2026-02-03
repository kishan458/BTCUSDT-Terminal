# pillar1_sentiment/schema.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class InstitutionalArticle:
    source: str                 # e.g. "Federal Reserve", "Reuters"
    category: str               # macro | regulation | corporate | legal
    title: str
    content: str
    published_at: datetime
    url: Optional[str] = None

@dataclass
class SentimentResult:
    article: InstitutionalArticle
    sentiment: str              # positive | neutral | negative
    confidence: float           # 0.0 – 1.0

@dataclass
class Pillar1SummaryInput:
    overall_sentiment: str      # bullish | neutral | bearish
    confidence: float
    drivers: list[str]          # short reasons
    articles: list[SentimentResult]
