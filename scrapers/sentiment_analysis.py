import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')

analyzer = SentimentIntensityAnalyzer()


def score_text(text):
    """Return VADER compound, pos, neg, neu scores for a text string."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return 0.0, 0.0, 0.0, 0.0
    scores = analyzer.polarity_scores(text)
    return scores['compound'], scores['pos'], scores['neg'], scores['neu']


def analyze_news_sentiment():
    print("Analyzing news sentiment with VADER...\n")
    rows = []

    # Process main news articles
    news_path = os.path.join(OUTPUT_DIR, "ai_infrastructure_news.csv")
    if os.path.exists(news_path):
        df_news = pd.read_csv(news_path)
        print(f"Processing {len(df_news)} infrastructure news articles...")
        for _, row in df_news.iterrows():
            headline = str(row.get("headline", ""))
            summary = str(row.get("summary", ""))

            h_compound, h_pos, h_neg, h_neu = score_text(headline)
            s_compound, s_pos, s_neg, s_neu = score_text(summary)

            # Weighted overall: headline counts more (0.6) than summary (0.4)
            if summary and summary != "nan":
                overall = 0.6 * h_compound + 0.4 * s_compound
            else:
                overall = h_compound

            rows.append({
                "date": row.get("date", ""),
                "headline": headline,
                "source": row.get("source", ""),
                "ticker": None,  # Industry-level news, no specific ticker
                "url": row.get("url", ""),
                "headline_compound": h_compound,
                "headline_pos": h_pos,
                "headline_neg": h_neg,
                "headline_neu": h_neu,
                "summary_compound": s_compound,
                "summary_pos": s_pos,
                "summary_neg": s_neg,
                "summary_neu": s_neu,
                "overall_compound": overall,
            })

    # Process HF / ticker-specific news
    hf_path = os.path.join(OUTPUT_DIR, "hf_financial_news.csv")
    if os.path.exists(hf_path):
        df_hf = pd.read_csv(hf_path)
        print(f"Processing {len(df_hf)} ticker-specific news articles...")
        for _, row in df_hf.iterrows():
            headline = str(row.get("headline", ""))

            h_compound, h_pos, h_neg, h_neu = score_text(headline)

            rows.append({
                "date": row.get("date", ""),
                "headline": headline,
                "source": row.get("source", ""),
                "ticker": row.get("ticker", None),
                "url": row.get("url", ""),
                "headline_compound": h_compound,
                "headline_pos": h_pos,
                "headline_neg": h_neg,
                "headline_neu": h_neu,
                "summary_compound": 0.0,
                "summary_pos": 0.0,
                "summary_neg": 0.0,
                "summary_neu": 0.0,
                "overall_compound": h_compound,
            })

    if not rows:
        print("No news data found to analyze.")
        return

    df_sentiment = pd.DataFrame(rows)
    df_sentiment.sort_values("date", ascending=False, inplace=True)
    df_sentiment.reset_index(drop=True, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "news_sentiment.csv")
    df_sentiment.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {output_path} ({len(df_sentiment)} rows)")
    print(f"\nSentiment stats:")
    print(f"  Mean compound: {df_sentiment['overall_compound'].mean():.3f}")
    print(f"  Median compound: {df_sentiment['overall_compound'].median():.3f}")
    print(f"  Std compound: {df_sentiment['overall_compound'].std():.3f}")

    return df_sentiment


if __name__ == "__main__":
    analyze_news_sentiment()
