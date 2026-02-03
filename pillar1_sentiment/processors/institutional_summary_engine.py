import json
import os
from google import genai
from google.genai import types

def generate_institutional_summary(articles, aggregate_result):
    """
    articles: list of dicts
    aggregate_result: output from aggregator
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")

    # Initialize Gemini client (NEW SDK)
    client = genai.Client(api_key=api_key)

    # UPDATED: Using 'gemini-2.5-flash' for 2026 performance and reliability.
    # If you need high-level reasoning, use 'gemini-3-pro-preview'.
    MODEL_ID = "gemini-2.5-flash"

    prompt = f"""
    You are a senior institutional financial strategist.
    Below is verified institutional market information with quantified sentiment.

    RAW ARTICLES:
    {json.dumps(articles, indent=2)}

    AGGREGATED SENTIMENT:
    {json.dumps(aggregate_result, indent=2)}

    TASK:
    Summarize the institutional sentiment in 4–5 professional bullet points.

    RULES:
    - No hype
    - No price targets
    - No trading advice
    - Focus on macro tone, policy risk, and institutional behavior
    - Write like a Bloomberg macro note
    """

    try:
        # The generate_content method in the new SDK
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3, # Low temperature for professional, factual tone
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"API Error: {str(e)}"

# Quick test block
if __name__ == "__main__":
    test_articles = [{"title": "Fed Interest Rate Decision", "content": "The Fed held rates steady..."}]
    test_aggregate = {"sentiment": "Neutral/Hawkish", "score": 0.65}
    print(generate_institutional_summary(test_articles, test_aggregate))