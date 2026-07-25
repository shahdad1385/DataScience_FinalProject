import os
import re
import time
import hashlib
from datetime import datetime
from urllib.parse import urlparse, parse_qs, unquote, quote_plus

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    from scrapling.fetchers import Fetcher as ScraplingFetcher
    HAS_SCRAPLING = True
except ImportError:
    HAS_SCRAPLING = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data_samples')

# ---------------------------------------------------------------------------
# Keyword matrix — articles must match at least one to be kept
# ---------------------------------------------------------------------------
TARGET_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "nvidia", "amd", "intel", "tsmc", "semiconductor", "chip", "gpu", "tpu",
    "data center", "datacenter", "cloud", "hyperscaler",
    "cooling", "liquid cooling", "immersion",
    "power", "grid", "electricity", "energy", "utility", "utilities",
    "infrastructure", "server", "rack",
    "supply chain", "foundry", "fab",
]

# ---------------------------------------------------------------------------
# Source definitions
# ---------------------------------------------------------------------------
SOURCES = [
    {
        "name": "TechCrunch AI",
        "type": "rss",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "categories": ["artificial-intelligence", "tech"],
    },
    {
        "name": "Semiconductor Engineering",
        "type": "rss",
        "url": "https://semiengineering.com/feed/",
        "categories": ["semiconductor", "chips"],
    },
    {
        "name": "The Register",
        "type": "rss",
        "url": "https://www.theregister.com/headlines.atom",
        "categories": ["datacenter", "infrastructure"],
        "filter_tags": ["data centre", "data center", "ai", "cloud", "semiconductor"],
    },
    {
        "name": "Tom's Hardware",
        "type": "rss",
        "url": "https://www.tomshardware.com/feeds/all",
        "categories": ["hardware", "semiconductor"],
    },
    {
        "name": "SemiAnalysis",
        "type": "rss",
        "url": "https://semianalysis.com/feed/",
        "categories": ["semiconductor", "analysis"],
    },
    {
        "name": "Ars Technica",
        "type": "rss",
        "url": "https://feeds.arstechnica.com/arstechnica/technology-lab",
        "categories": ["technology", "infrastructure"],
    },
    {
        "name": "The Verge",
        "type": "rss",
        "url": "https://www.theverge.com/rss/index.xml",
        "categories": ["technology", "ai"],
    },
    {
        "name": "PCWorld",
        "type": "rss",
        "url": "https://www.pcworld.com/feed",
        "categories": ["hardware", "semiconductor"],
    },
    {
        "name": "DataCenterDynamics",
        "type": "html_scrapling",
        "url": "https://www.datacenterdynamics.com/en/news/",
        "categories": ["datacenter", "infrastructure"],
    },
    {
        "name": "Reuters",
        "type": "google_news_rss",
        "queries": [
            "reuters artificial intelligence data center",
            "reuters nvidia tsmc semiconductor chip",
            "reuters ai infrastructure power grid energy",
        ],
        "categories": ["reuters", "finance", "technology"],
    },
    {
        "name": "CNBC Tech",
        "type": "google_news_rss",
        "queries": [
            "cnbc artificial intelligence semiconductor chip",
            "cnbc nvidia tsmc data center",
        ],
        "categories": ["cnbc", "finance", "technology"],
    },
    {
        "name": "Yahoo Finance",
        "type": "google_news_rss",
        "queries": [
            "yahoo finance ai semiconductor chip nvidia",
            "yahoo finance data center infrastructure energy",
        ],
        "categories": ["yahoo", "finance"],
    },
    {
        "name": "Reddit r/wallstreetbets",
        "type": "reddit_rss",
        "url": "https://www.reddit.com/r/wallstreetbets/.rss",
        "categories": ["reddit", "market-sentiment"],
    },
    {
        "name": "Reddit r/nvidia",
        "type": "reddit_rss",
        "url": "https://www.reddit.com/r/nvidia/.rss",
        "categories": ["reddit", "nvidia", "gpu"],
    },
    {
        "name": "Reddit r/semiconductors",
        "type": "reddit_rss",
        "url": "https://www.reddit.com/r/semiconductors/.rss",
        "categories": ["reddit", "semiconductor"],
    },
    {
        "name": "Reddit r/hardware",
        "type": "reddit_rss",
        "url": "https://www.reddit.com/r/hardware/.rss",
        "categories": ["reddit", "hardware"],
    },
    {
        "name": "Reddit r/technology",
        "type": "reddit_rss",
        "url": "https://www.reddit.com/r/technology/.rss",
        "categories": ["reddit", "technology"],
    },
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _parse_date(raw):
    if hasattr(raw, "published_parsed") and raw.published_parsed:
        try:
            return datetime(*raw.published_parsed[:6]).strftime("%Y-%m-%d")
        except Exception:
            pass
    if hasattr(raw, "updated_parsed") and raw.updated_parsed:
        try:
            return datetime(*raw.updated_parsed[:6]).strftime("%Y-%m-%d")
        except Exception:
            pass
    return datetime.today().strftime("%Y-%m-%d")


def _clean(text):
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _headline_id(headline, url):
    key = f"{headline.lower().strip()}|{url}"
    return hashlib.md5(key.encode()).hexdigest()


def _extract_date_from_url(url):
    """Try to extract date from URL patterns like /2024/01/15/..."""
    match = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


# ---------------------------------------------------------------------------
# Source-specific parsers
# ---------------------------------------------------------------------------
def fetch_rss(source):
    feed = feedparser.parse(source["url"])
    articles = []
    for entry in feed.entries:
        title = _clean(getattr(entry, "title", ""))
        link = getattr(entry, "link", "")
        summary = _clean(getattr(entry, "summary", getattr(entry, "description", "")))
        date = _parse_date(entry)

        if not title or len(title) < 10 or not link:
            continue

        articles.append({
            "date": date,
            "headline": title,
            "summary": summary[:2000],
            "url": link,
            "source": source["name"],
            "source_categories": ",".join(source["categories"]),
            "tags": ",".join([t.get("term", "") for t in getattr(entry, "tags", [])]),
        })
    return articles


def fetch_html_scrapling(source):
    """Scrape using Scrapling for Cloudflare-protected sites."""
    articles = []
    try:
        if HAS_SCRAPLING:
            page = ScraplingFetcher.get(source["url"], stealthy_headers=True)
            html_content = page.html_content if hasattr(page, 'html_content') else str(page.body)
        else:
            resp = requests.get(source["url"], headers=HEADERS, timeout=15)
            resp.raise_for_status()
            html_content = resp.text

        soup = BeautifulSoup(html_content, "html.parser")
        seen = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            text = _clean(a_tag.get_text())

            if not text or len(text) < 20:
                continue
            if not href.startswith("http"):
                href = source["url"].rstrip("/") + "/" + href.lstrip("/")

            parsed = urlparse(href)
            if "datacenterdynamics.com" not in parsed.netloc:
                continue

            path = parsed.path
            if not any(path.startswith(f"/en/{s}/") for s in ("news", "features", "analysis")):
                continue

            slug = path.rstrip("/").split("/")[-1]
            if len(slug) < 10 or slug.startswith("?"):
                continue

            skip_words = ("channel", "channels", "category", "tag", "author", "page")
            if any(w in slug.lower() for w in skip_words):
                continue

            if href in seen:
                continue
            seen.add(href)

            articles.append({
                "date": datetime.today().strftime("%Y-%m-%d"),
                "headline": text,
                "summary": "",
                "url": href,
                "source": source["name"],
                "source_categories": ",".join(source["categories"]),
                "tags": "",
            })
    except Exception as e:
        print(f"  Failed to scrape {source['name']}: {e}")
    return articles


def _resolve_google_news_url(google_url):
    parsed = urlparse(google_url)
    if "news.google.com" not in parsed.netloc:
        return google_url
    qs = parse_qs(parsed.query)
    if "url" in qs:
        return unquote(qs["url"][0])
    return google_url


def fetch_google_news_rss(source):
    articles = []
    seen_urls = set()

    for query in source["queries"]:
        rss_url = (
            f"https://news.google.com/rss/search?q={quote_plus(query)}"
            f"&hl=en-US&gl=US&ceid=US:en"
        )
        feed = feedparser.parse(rss_url)

        for entry in feed.entries:
            publisher = getattr(entry, "source", {})
            publisher_name = publisher.get("title", "") if isinstance(publisher, dict) else ""

            title = _clean(getattr(entry, "title", ""))

            raw_link = getattr(entry, "link", "")
            link = _resolve_google_news_url(raw_link)

            if not title or len(title) < 10 or not link:
                continue
            if link in seen_urls:
                continue
            seen_urls.add(link)

            summary = _clean(getattr(entry, "summary", getattr(entry, "description", "")))
            date = _parse_date(entry)

            articles.append({
                "date": date,
                "headline": title,
                "summary": summary[:2000],
                "url": link,
                "source": publisher_name if publisher_name else source["name"],
                "source_categories": ",".join(source["categories"]),
                "tags": "",
            })

    return articles


def fetch_reddit_rss(source):
    articles = []
    try:
        feed = feedparser.parse(source["url"])
        for entry in feed.entries:
            title = _clean(getattr(entry, "title", ""))
            link = getattr(entry, "link", "")
            summary = _clean(getattr(entry, "summary", getattr(entry, "description", "")))
            date = _parse_date(entry)

            if not title or len(title) < 10 or not link:
                continue

            articles.append({
                "date": date,
                "headline": title,
                "summary": summary[:2000],
                "url": link,
                "source": source["name"],
                "source_categories": ",".join(source["categories"]),
                "tags": "",
            })
    except Exception as e:
        print(f"  Failed to fetch {source['name']}: {e}")
    return articles


FETCHERS = {
    "rss": fetch_rss,
    "html_scrapling": fetch_html_scrapling,
    "google_news_rss": fetch_google_news_rss,
    "reddit_rss": fetch_reddit_rss,
}


# ---------------------------------------------------------------------------
# Filtering and deduplication
# ---------------------------------------------------------------------------
def matches_keywords(article, keywords):
    text = f"{article['headline']} {article['summary']} {article['tags']}".lower()
    return any(kw in text for kw in keywords)


def deduplicate(articles):
    seen = set()
    unique = []
    for a in articles:
        hid = _headline_id(a["headline"], a["url"])
        if hid not in seen:
            seen.add(hid)
            unique.append(a)
    return unique


# ---------------------------------------------------------------------------
# Main pipeline — produces TWO outputs:
#   1. ai_infrastructure_news.csv  (official news only)
#   2. social_media_posts.csv      (Reddit posts only, short titles)
# ---------------------------------------------------------------------------
REDDIT_MAX_TITLE_LEN = 150  # skip long Reddit rants


def collect_all():
    print("Multi-Source AI Infrastructure News Collector\n")
    print(f"Scrapling available: {HAS_SCRAPLING}\n")

    official_articles = []
    social_articles = []

    for source in SOURCES:
        print(f"Fetching from {source['name']} ({source['type']})...")
        fetcher = FETCHERS[source["type"]]
        articles = fetcher(source)
        print(f"  -> {len(articles)} raw articles")

        if source["type"] == "reddit_rss":
            social_articles.extend(articles)
            time.sleep(4)
        else:
            official_articles.extend(articles)

    print(f"\n--- Official news ---")
    print(f"Total raw: {len(official_articles)}")
    official_articles = deduplicate(official_articles)
    print(f"After dedup: {len(official_articles)}")

    df_official = pd.DataFrame(official_articles)
    if not df_official.empty:
        mask = df_official.apply(lambda row: matches_keywords(row, TARGET_KEYWORDS), axis=1)
        df_official = df_official[mask].copy()
    print(f"After keyword filter: {len(df_official)}")

    print(f"\n--- Social media (Reddit) ---")
    print(f"Total raw: {len(social_articles)}")
    social_articles = deduplicate(social_articles)
    print(f"After dedup: {len(social_articles)}")

    df_social = pd.DataFrame(social_articles)
    if not df_social.empty:
        # Keyword filter
        mask = df_social.apply(lambda row: matches_keywords(row, TARGET_KEYWORDS), axis=1)
        df_social = df_social[mask].copy()
        # Keep only short posts (title-based, no long rants)
        df_social = df_social[df_social["headline"].str.len() <= REDDIT_MAX_TITLE_LEN].copy()
        # Clear summaries — Reddit posts don't have article summaries
        df_social["summary"] = None
    print(f"After filter (keywords + short title <= {REDDIT_MAX_TITLE_LEN} chars): {len(df_social)}")

    # Sort both
    df_official.sort_values("date", ascending=False, inplace=True)
    df_official.reset_index(drop=True, inplace=True)
    df_social.sort_values("date", ascending=False, inplace=True)
    df_social.reset_index(drop=True, inplace=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save official news
    official_path = os.path.join(OUTPUT_DIR, "ai_infrastructure_news.csv")
    df_official.to_csv(official_path, index=False, encoding="utf-8-sig")
    print(f"\nOfficial news saved: {official_path} ({len(df_official)} rows)")

    # Save social media posts
    social_path = os.path.join(OUTPUT_DIR, "social_media_posts.csv")
    df_social.to_csv(social_path, index=False, encoding="utf-8-sig")
    print(f"Social media saved: {social_path} ({len(df_social)} rows)")

    return df_official, df_social


if __name__ == "__main__":
    collect_all()
