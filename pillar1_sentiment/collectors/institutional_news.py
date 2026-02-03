from pillar1_sentiment.schema import InstitutionalArticle
from datetime import datetime

def collect_institutional_news() -> list[InstitutionalArticle]:
    articles = []

    articles.append(
        InstitutionalArticle(
            source="Federal Reserve",
            category="macro",
            title="Fed maintains restrictive policy stance",
            content="Full cleaned article text here...",
            published_at=datetime.utcnow(),
            url="https://..."
        )
    )

    return articles
