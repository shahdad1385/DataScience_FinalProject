import os
import re
import hashlib
import time
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote, quote_plus

import feedparser
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')

TARGET_TICKERS = {
    "NVDA": ["nvidia", "nvda"],
    "META": ["meta", "meta platforms", "facebook"],
    "GOOGL": ["google", "alphabet", "googl"],
    "TSM": ["tsmc", "taiwan semiconductor"],
    "VRT": ["vertiv"],
    "MOD": ["modine"],
    "SMCI": ["super micro", "supermicro", "smci"],
    "AMD": ["amd", "advanced micro devices"],
    "INTC": ["intel"],
    "AVGO": ["broadcom", "avgo"],
    "QCOM": ["qualcomm"],
    "MRVL": ["marvell"],
    "AMAT": ["applied materials", "amat"],
    "MU": ["micron", "micron technology"],
    "LRCX": ["lam research"],
    "KLAC": ["kla corporation", "kla"],
    "ASML": ["asml"],
    "ARM": ["arm holdings", "arm ltd"],
    "NXPI": ["nxp semiconductors", "nxp"],
    "ON": ["on semiconductor", "onsemi"],
}

START_DATE = "2018-01-01"


def _clean(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_date(entry):
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6]).strftime("%Y-%m-%d")
        except Exception:
            pass
    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        try:
            return datetime(*entry.updated_parsed[:6]).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None


def _resolve_google_news_url(google_url):
    parsed = urlparse(google_url)
    if "news.google.com" not in parsed.netloc:
        return google_url
    qs = parse_qs(parsed.query)
    if "url" in qs:
        return unquote(qs["url"][0])
    return google_url


def _headline_id(headline, url):
    key = f"{headline.lower().strip()}|{url}"
    return hashlib.md5(key.encode()).hexdigest()


def fetch_ticker_news_google_rss(ticker, keywords, max_queries=3):
    """Fetch recent news for a specific ticker via Google News RSS."""
    articles = []
    seen_urls = set()

    queries = [
        f"{keywords[0]} stock",
        f"{keywords[0]} earnings",
        f"{' '.join(keywords[:2])} news",
    ][:max_queries]

    for query in queries:
        rss_url = (
            f"https://news.google.com/rss/search?q={quote_plus(query)}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )
        try:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries:
                title = _clean(getattr(entry, "title", ""))
                raw_link = getattr(entry, "link", "")
                link = _resolve_google_news_url(raw_link)
                date = _parse_date(entry)

                if not title or len(title) < 10 or not link:
                    continue
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                # Check if any keyword matches
                title_lower = title.lower()
                if not any(kw in title_lower for kw in keywords):
                    continue

                publisher = getattr(entry, "source", {})
                publisher_name = publisher.get("title", "") if isinstance(publisher, dict) else "Google News"

                articles.append({
                    "date": date,
                    "headline": title,
                    "source": publisher_name,
                    "ticker": ticker,
                    "url": link,
                })
        except Exception as e:
            print(f"  Error fetching {query}: {e}")

    return articles


def download_and_filter():
    print("Downloading financial news for all tickers via Google News RSS...\n")
    all_articles = []

    for ticker, keywords in TARGET_TICKERS.items():
        print(f"Fetching news for {ticker} ({', '.join(keywords[:2])})...")
        articles = fetch_ticker_news_google_rss(ticker, keywords)
        print(f"  -> {len(articles)} articles")
        all_articles.extend(articles)
        time.sleep(2)  # Rate limiting

    print(f"\nTotal raw articles: {len(all_articles)}")

    df = pd.DataFrame(all_articles)

    # Deduplicate
    df["hid"] = df.apply(lambda r: _headline_id(r["headline"], r["url"]), axis=1)
    df.drop_duplicates(subset=["hid"], inplace=True)
    df.drop(columns=["hid"], inplace=True)
    print(f"After deduplication: {len(df)}")

    # Filter by date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df = df[df["date"] >= START_DATE]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    print(f"After date filter (>= {START_DATE}): {len(df)}")

    # Sort
    df.sort_values("date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Also load existing HF data if it exists
    existing_path = os.path.join(OUTPUT_DIR, "hf_financial_news.csv")
    if os.path.exists(existing_path):
        df_existing = pd.read_csv(existing_path)
        df_existing["date"] = pd.to_datetime(df_existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = pd.concat([df_existing, df], ignore_index=True)
        df.drop_duplicates(subset=["headline", "ticker"], inplace=True)
        df.sort_values("date", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        print(f"Merged with existing data: {len(df)} total")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "hf_financial_news.csv")
    cols = ["date", "headline", "source", "ticker", "url"]
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {output_path} ({len(df)} rows)")
    print(f"\nTicker distribution:")
    print(df["ticker"].value_counts().to_string())
    print(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique sources: {df['source'].nunique()}")

    return df


if __name__ == "__main__":
    download_and_filter()
