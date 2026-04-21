import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pillar1_sentiment.config import FINBERT_MODEL, FINBERT_MAX_TOKENS

# ── Load once at module level ─────────────────────────────────────────────────
# (no duplicate pipeline — single clean load)
_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
_model     = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
_model.eval()

LABELS = ["negative", "neutral", "positive"]


# ── Core FinBERT Inference ────────────────────────────────────────────────────

def _analyze_text(text: str) -> tuple[str, float]:
    """
    Run FinBERT on a single text string.
    Returns (sentiment_label, confidence_score).
    """
    if not text.strip():
        return "neutral", 0.0

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=FINBERT_MAX_TOKENS,
    )

    with torch.no_grad():
        outputs = _model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)[0]

    confidence, idx = torch.max(probs, dim=0)
    sentiment       = LABELS[idx.item()]

    return sentiment, round(confidence.item(), 4)


def _chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    """
    Split long content into overlapping chunks so FinBERT
    sees more of the article than just the first 512 tokens.
    Simple sentence-boundary split.
    """
    sentences = text.replace("\n", " ").split(". ")
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) < max_chars:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "

    if current:
        chunks.append(current.strip())

    return chunks or [text[:max_chars]]


def _analyze_long_text(text: str) -> tuple[str, float]:
    """
    For longer content: chunk → analyze each chunk → average scores.
    Returns weighted (sentiment, confidence).
    """
    chunks = _chunk_text(text)

    if len(chunks) == 1:
        return _analyze_text(chunks[0])

    # Score each chunk
    scores = []
    for chunk in chunks:
        sentiment, confidence = _analyze_text(chunk)
        if sentiment == "negative":
            scores.append(-confidence)
        elif sentiment == "positive":
            scores.append(confidence)
        else:
            scores.append(0.0)

    avg_score = sum(scores) / len(scores)

    if avg_score > 0.05:
        final_label = "positive"
    elif avg_score < -0.05:
        final_label = "negative"
    else:
        final_label = "neutral"

    return final_label, round(abs(avg_score), 4)


# ── Main Public Function ──────────────────────────────────────────────────────

def analyze_articles(articles) -> list[dict]:
    """
    articles: list of InstitutionalArticle dataclass objects

    Strategy:
      - Prefer full content (article body) for analysis
      - Fall back to title only if content is missing/short
      - For long content, use chunked analysis

    Returns list of dicts with sentiment metadata.
    """
    results = []

    for article in articles:
        title   = (article.title   or "").strip()
        content = (article.content or "").strip()

        # Pick best text to analyze
        if len(content) > 80:
            # Full content available — use chunked analysis
            sentiment, confidence = _analyze_long_text(content)
            analyzed_field = "content"
        elif title:
            # Only title available — analyze title
            sentiment, confidence = _analyze_text(title)
            analyzed_field = "title"
        else:
            # Nothing to analyze
            continue

        results.append({
            "sentiment":      sentiment,
            "confidence":     confidence,
            "source":         article.source,
            "category":       article.category,
            "headline":       title,
            "analyzed_field": analyzed_field,   # transparency: what did we actually analyze?
            "url":            article.url or "",
        })

    print(f"[FinBERT] Analyzed {len(results)} articles.")
    return results