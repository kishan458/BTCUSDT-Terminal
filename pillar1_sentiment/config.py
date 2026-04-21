import os
from dotenv import load_dotenv

load_dotenv()  # loads your .env file automatically

# ── API Keys ──────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY")

# ── Model Settings ────────────────────────────────────────
FINBERT_MODEL      = "ProsusAI/finbert"
GEMINI_MODEL       = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.3
FINBERT_MAX_TOKENS = 512

# ── NewsAPI Settings ──────────────────────────────────────
NEWSAPI_ENDPOINT    = "https://newsapi.org/v2/everything"
NEWSAPI_QUERY       = "bitcoin BTC cryptocurrency federal reserve inflation"
NEWSAPI_LANGUAGE    = "en"
NEWSAPI_PAGE_SIZE   = 20                        # max per call on free tier
NEWSAPI_SORT_BY     = "publishedAt"

# ── RSS Feed Sources (no key needed) ─────────────────────
RSS_FEEDS = {
    "CoinDesk":        "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph":   "https://cointelegraph.com/rss",
    "Decrypt":         "https://decrypt.co/feed",
    "Federal Reserve": "https://www.federalreserve.gov/feeds/press_all.xml",
}

# ── Aggregator Thresholds ─────────────────────────────────
SENTIMENT_POSITIVE_THRESHOLD = 0.15   # avg score above this → positive
SENTIMENT_NEGATIVE_THRESHOLD = -0.15  # avg score below this → negative

# ── How many articles to surface as drivers ───────────────
MAX_DRIVERS = 5