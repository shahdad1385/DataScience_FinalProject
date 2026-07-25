import os
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')

analyzer = SentimentIntensityAnalyzer()

TICKER_KEYWORDS = {
    "NVDA": ["nvidia", "nvda"],
    "META": ["meta platforms", "facebook", "meta"],
    "GOOGL": ["google", "alphabet", "googl"],
    "TSM": ["tsmc", "taiwan semiconductor"],
    "VRT": ["vertiv"],
    "MOD": ["modine"],
    "SMCI": ["super micro", "supermicro", "smci"],
    "AMD": ["amd"],
    "INTC": ["intel", "intc"],
    "AVGO": ["broadcom", "avgo"],
    "QCOM": ["qualcomm", "qcom"],
    "MRVL": ["marvell"],
    "AMAT": ["applied materials", "amat"],
    "MU": ["micron", "mu stock"],
    "LRCX": ["lam research"],
    "KLAC": ["kla"],
    "ASML": ["asml"],
    "ARM": ["arm holdings", "arm ltd"],
    "NXPI": ["nxp"],
    "ON": ["onsemi", "on semiconductor"],
}


def extract_tickers(headline):
    """Extract matching tickers from a headline."""
    text = str(headline).lower()
    found = []
    for ticker, keywords in TICKER_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            found.append(ticker)
    return found


def score_sentiment(text):
    """Return VADER compound score."""
    if not text or not isinstance(text, str):
        return 0.0
    return analyzer.polarity_scores(text)["compound"]


def generate_social_sentiment():
    print("Generating social media sentiment scores...\n")

    social_path = os.path.join(OUTPUT_DIR, "social_media_posts.csv")
    if not os.path.exists(social_path):
        print(f"No social media posts found at {social_path}")
        return

    df = pd.read_csv(social_path)
    print(f"Loaded {len(df)} social media posts")

    rows = []
    for _, post in df.iterrows():
        headline = str(post.get("headline", ""))
        tickers = extract_tickers(headline)
        sentiment = score_sentiment(headline)
        date = post.get("date", "")
        source = post.get("source", "")

        # Create one row per matching ticker
        if tickers:
            for ticker in tickers:
                rows.append({
                    "date": date,
                    "ticker": ticker,
                    "headline": headline,
                    "source": source,
                    "url": post.get("url", ""),
                    "sentiment_score": sentiment,
                    "post_length": len(headline),
                })
        else:
            # No ticker match — still record with industry-level sentiment
            rows.append({
                "date": date,
                "ticker": None,
                "headline": headline,
                "source": source,
                "url": post.get("url", ""),
                "sentiment_score": sentiment,
                "post_length": len(headline),
            })

    df_out = pd.DataFrame(rows)
    df_out.sort_values("date", ascending=False, inplace=True)
    df_out.drop_duplicates(subset=["date", "headline", "ticker"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "social_sentiment.csv")
    df_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\n{'=' * 50}")
    print(f"Social Sentiment Saved!")
    print(f"Total rows: {len(df_out)} ({df_out['ticker'].notna().sum()} with ticker, {df_out['ticker'].isna().sum()} industry-level)")
    print(f"File: {os.path.abspath(output_path)}")
    print(f"{'=' * 50}")

    print(f"\nTicker distribution:")
    ticker_counts = df_out["ticker"].value_counts().head(10)
    for t, cnt in ticker_counts.items():
        print(f"  {t}: {cnt}")

    print(f"\nSentiment stats:")
    print(f"  Mean: {df_out['sentiment_score'].mean():.3f}")
    print(f"  Median: {df_out['sentiment_score'].median():.3f}")
    print(f"  Positive (>0.05): {(df_out['sentiment_score'] > 0.05).sum()}")
    print(f"  Negative (<-0.05): {(df_out['sentiment_score'] < -0.05).sum()}")
    print(f"  Neutral: {((df_out['sentiment_score'] >= -0.05) & (df_out['sentiment_score'] <= 0.05)).sum()}")


if __name__ == "__main__":
    generate_social_sentiment()
