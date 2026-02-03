from transformers import pipeline

# Load once (important for performance)
_finbert = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load once (module-level, fast)
MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

LABELS = ["negative", "neutral", "positive"]

def analyze_text_with_finbert(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    confidence, idx = torch.max(probs, dim=0)
    sentiment = LABELS[idx.item()]

    return sentiment, round(confidence.item(), 3)





def analyze_articles(articles):
    results = []

    for article in articles:
        text = article.title or article.content or ""

        if not text.strip():
            continue

        sentiment, confidence = analyze_text_with_finbert(text)

        results.append({
            "sentiment": sentiment,
            "confidence": confidence,
            "source": article.source,
            "headline": article.title
        })

    return results


    return results
