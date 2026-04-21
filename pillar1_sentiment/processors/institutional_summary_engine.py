import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

def generate_institutional_summary(articles, aggregate_result):
    """
    articles: list of dicts
    aggregate_result: output from aggregator
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in environment")

    client = Groq(api_key=api_key)

    prompt = f"""You are a senior institutional financial strategist.
Below is verified institutional market information with quantified sentiment.

RAW ARTICLES:
{json.dumps(articles, indent=2)}

AGGREGATED SENTIMENT:
{json.dumps(aggregate_result, indent=2)}

TASK:
Summarize the institutional sentiment in 4-5 professional bullet points.

RULES:
- No hype
- No price targets
- No trading advice
- Focus on macro tone, policy risk, and institutional behavior
- Write like a Bloomberg macro note
- Start each bullet point with a dash (-)"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior institutional financial strategist writing Bloomberg-style macro notes. Be concise, professional, and data-driven."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"API Error: {str(e)}"


# Quick test block
if __name__ == "__main__":
    test_articles = [{"headline": "Fed holds rates steady amid inflation concerns", "sentiment": "negative", "confidence": 0.82, "source": "Federal Reserve"}]
    test_aggregate = {"sentiment": "neutral", "confidence": 0.45, "drivers": ["Fed policy uncertainty", "ETF inflows"]}
    print(generate_institutional_summary(test_articles, test_aggregate))