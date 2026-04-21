import feedparser
import requests
from datetime import datetime, timezone
from pillar1_sentiment.schema import InstitutionalArticle
from pillar1_sentiment.config import (
    NEWSAPI_KEY,
    NEWSAPI_ENDPOINT,
    NEWSAPI_QUERY,
    NEWSAPI_LANGUAGE,
    NEWSAPI_PAGE_SIZE,
    NEWSAPI_SORT_BY,
    RSS_FEEDS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_date(date_str: str | None) -> datetime:
    """Best-effort UTC datetime from an ISO or RSS date string."""
    if not date_str:
        return datetime.now(timezone.utc)
    try:
        # NewsAPI format: "2024-01-15T10:30:00Z"
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except Exception:
        return datetime.now(timezone.utc)


def _clean(text: str | None) -> str:
    return (text or "").strip()


# ── Source 1: NewsAPI ─────────────────────────────────────────────────────────

def _fetch_from_newsapi() -> list[InstitutionalArticle]:
    """Pull BTC/macro institutional news from NewsAPI."""
    if not NEWSAPI_KEY:
        print("[WARNING] NEWSAPI_KEY not set — skipping NewsAPI fetch.")
        return []

    try:
        response = requests.get(
            NEWSAPI_ENDPOINT,
            params={
                "q":        NEWSAPI_QUERY,
                "language": NEWSAPI_LANGUAGE,
                "pageSize": NEWSAPI_PAGE_SIZE,
                "sortBy":   NEWSAPI_SORT_BY,
                "apiKey":   NEWSAPI_KEY,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[ERROR] NewsAPI fetch failed: {e}")
        return []

    articles = []
    for item in data.get("articles", []):
        title   = _clean(item.get("title"))
        content = _clean(item.get("description") or item.get("content"))
        source  = _clean(item.get("source", {}).get("name")) or "NewsAPI"
        url     = _clean(item.get("url"))
        pub_at  = _parse_date(item.get("publishedAt"))

        if not title:
            continue

        articles.append(
            InstitutionalArticle(
                source=source,
                category="macro",
                title=title,
                content=content,
                published_at=pub_at,
                url=url,
            )
        )

    print(f"[NewsAPI] Fetched {len(articles)} articles.")
    return articles


# ── Source 2: RSS Feeds ───────────────────────────────────────────────────────

def _fetch_from_rss() -> list[InstitutionalArticle]:
    """Pull articles from CoinDesk, Cointelegraph, Decrypt, Federal Reserve RSS."""
    articles = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            count = 0

            for entry in feed.entries[:10]:          # max 10 per feed
                title   = _clean(entry.get("title"))
                content = _clean(
                    entry.get("summary")
                    or entry.get("description")
                    or entry.get("content", [{}])[0].get("value", "")
                )
                url     = _clean(entry.get("link"))

                # Parse published date
                published_parsed = entry.get("published_parsed")
                if published_parsed:
                    pub_at = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                else:
                    pub_at = datetime.now(timezone.utc)

                if not title:
                    continue

                # Tag Fed as policy, others as crypto
                category = "policy" if source_name == "Federal Reserve" else "crypto"

                articles.append(
                    InstitutionalArticle(
                        source=source_name,
                        category=category,
                        title=title,
                        content=content,
                        published_at=pub_at,
                        url=url,
                    )
                )
                count += 1

            print(f"[RSS:{source_name}] Fetched {count} articles.")

        except Exception as e:
            print(f"[ERROR] RSS fetch failed for {source_name}: {e}")

    return articles


# ── Main Collector ────────────────────────────────────────────────────────────

def collect_institutional_news() -> list[InstitutionalArticle]:
    """
    Collects live institutional news from:
      - NewsAPI  (macro + institutional, BTC-filtered)
      - RSS      (CoinDesk, Cointelegraph, Decrypt, Federal Reserve)

    Returns a deduplicated list of InstitutionalArticle objects.
    """
    newsapi_articles = _fetch_from_newsapi()
    rss_articles     = _fetch_from_rss()

    all_articles = newsapi_articles + rss_articles

    # Deduplicate by title (case-insensitive)
    seen_titles: set[str] = set()
    unique_articles: list[InstitutionalArticle] = []

    for article in all_articles:
        key = article.title.lower().strip()
        if key not in seen_titles:
            seen_titles.add(key)
            unique_articles.append(article)

    print(f"\n[Collector] Total unique articles: {len(unique_articles)}")
    return unique_articles