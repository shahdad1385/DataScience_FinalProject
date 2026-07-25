"""
Generate LLM summaries for news articles using NVIDIA NIM API (nemotron-super).
Parallel processing with ThreadPoolExecutor for speed.

Usage:
    1. Get an NVIDIA API key from https://build.nvidia.com
    2. Put it in .env: NVIDIA_API_KEY=nvapi-...
    3. Run: python3 scripts/generate_summaries.py
"""

import os
import sys
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database_connection import get_engine
from sqlalchemy import text

try:
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_root, ".env"))
except ImportError:
    pass

API_KEY = os.environ.get("NVIDIA_API_KEY", "")
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
NUM_WORKERS = 5          # parallel API calls
SAVE_INTERVAL = 100      # save to DB every N summaries
FETCH_TIMEOUT = 10
MAX_RETRIES = 3

PROMPT_TEMPLATE = """Write a 2-4 sentence summary of this financial news for stock market analysis. Mention which company/ticker is affected, what happened, and potential market impact. Do NOT include any reasoning, analysis, or step-by-step thinking. Output ONLY the summary — nothing else.

Headline: {content}

Summary:"""


def get_nvidia_client():
    if not API_KEY:
        print("ERROR: NVIDIA_API_KEY not set in .env file.")
        sys.exit(1)
    from openai import OpenAI
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=API_KEY, timeout=60)


def fetch_article_content(url):
    if not url or url == "None":
        return None
    try:
        import requests
        from bs4 import BeautifulSoup
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=FETCH_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        article = soup.find("article") or soup.find("main")
        text_content = article.get_text(separator=" ", strip=True) if article else soup.get_text(separator=" ", strip=True)
        text_content = re.sub(r"\s+", " ", text_content).strip()
        return text_content[:3000] if len(text_content) > 50 else None
    except Exception:
        return None


def generate_summary(client, headline, article_content):
    content = headline
    if article_content:
        content = f"{headline}\n\n{article_content[:1500]}"
    prompt = PROMPT_TEMPLATE.format(content=content)

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, top_p=0.9, max_tokens=200, stream=False,
            )
            result = completion.choices[0].message.content
            if not result:
                return None
            result = result.strip()
            # Strip prompt echo / reasoning
            for pat in [r"Must mention.*?impact\.\s*", r"No preamble.*?sentence\.\s*",
                        r"Headline:\s*.*?\n\s*Summary:\s*",
                        r"Okay,?\s*the user.*?(?:headline|summary)[^:]*:\s*[\"']?\s*",
                        r"Okay,?\s*the user.*?[\"']\s*\n?\s*"]:
                m = re.search(pat, result, re.IGNORECASE)
                if m:
                    result = result[m.end():]
                    break
            for prefix in ["Okay", "Let me", "The user", "I need to", "I should",
                           "First,", "Thinking:", "Analysis:", "Here is", "Here's",
                           "The summary", "Based on", "To summarize", "In summary",
                           "We need", "The article", "This article"]:
                if result.startswith(prefix):
                    for sep in [". ", "! ", "? ", "\n"]:
                        idx = result.find(sep, len(prefix))
                        if 0 < idx < 200:
                            result = result[idx + len(sep):]
                            break
            result = result.strip('"').strip("'").strip()
            return result if len(result) > 15 else None
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower():
                time.sleep(5 * (attempt + 1))
            else:
                return None
    return None


def process_one(client, article):
    article_id, headline, url, _ = article
    article_content = fetch_article_content(url)
    summary = generate_summary(client, headline, article_content)
    return (article_id, summary)


def save_summaries(engine, results):
    with engine.connect() as conn:
        for article_id, summary_text in results:
            if summary_text:
                conn.execute(
                    text("UPDATE news SET llm_summary = :summary WHERE id = :id"),
                    {"summary": summary_text, "id": article_id},
                )
        conn.commit()


def main():
    engine = get_engine()
    client = get_nvidia_client()

    articles = fetch_articles_without_summary(engine)
    print(f"Found {len(articles)} articles without summaries.")
    print(f"Running with {NUM_WORKERS} parallel workers\n")

    if not articles:
        print("All articles already have summaries. Nothing to do.")
        return

    total = len(articles)
    processed = []
    errors = 0
    done_count = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_one, client, a): a for a in articles}

        for future in as_completed(futures):
            article_id, summary = future.result()
            with lock:
                done_count += 1
                if summary:
                    processed.append((article_id, summary))
                    if done_count % 20 == 0:
                        print(f"  [{done_count}/{total}] {len(processed)} summaries saved, {errors} errors")
                else:
                    errors += 1

                if len(processed) >= SAVE_INTERVAL:
                    print(f"\n  Saving batch ({len(processed)} summaries)...")
                    save_summaries(engine, processed)
                    processed = []
                    print(f"  Progress: {done_count}/{total} ({done_count/total*100:.1f}%)")

    # Save remaining
    if processed:
        print(f"\nSaving final batch ({len(processed)} summaries)...")
        save_summaries(engine, processed)

    # Report
    with engine.connect() as conn:
        total_with = conn.execute(text("SELECT COUNT(*) FROM news WHERE llm_summary IS NOT NULL")).scalar()
        total_rows = conn.execute(text("SELECT COUNT(*) FROM news")).scalar()
    print(f"\nDone! {total_with}/{total_rows} articles have LLM summaries. ({errors} errors)")


def fetch_articles_without_summary(engine):
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT id, headline, url, summary FROM news WHERE llm_summary IS NULL ORDER BY id")
        ).fetchall()
    return [(r[0], r[1], r[2], r[3]) for r in rows]


if __name__ == "__main__":
    main()
