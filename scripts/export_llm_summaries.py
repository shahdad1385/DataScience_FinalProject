"""
Export news.llm_summary to data_samples/_cache/llm_summaries.csv so the
LLM-generated summaries survive any DB rebuild. Restored by load_data.py on
the next news load for any row whose llm_summary is NULL.

Output columns: id, headline, url, llm_summary
Only rows with non-null llm_summary are exported.

Usage: python3 scripts/export_llm_summaries.py
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database_connection import get_engine, get_db_path

CACHE_DIR = os.path.join(os.path.dirname(get_db_path()), "_cache")
CACHE_PATH = os.path.join(CACHE_DIR, "llm_summaries.csv")


def export_llm_summaries():
    engine = get_engine()
    if not os.path.exists(get_db_path()):
        print(f"Database not found: {get_db_path()}")
        sys.exit(1)

    try:
        df = pd.read_sql(
            "SELECT id, headline, url, llm_summary FROM news WHERE llm_summary IS NOT NULL",
            engine,
        )
    except Exception as e:
        print(f"Failed to read news table: {e}")
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Atomic write: temp file then rename, so an interrupted export never
    # leaves a truncated snapshot that load_data.py could restore from.
    tmp_path = CACHE_PATH + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, CACHE_PATH)

    print(f"Exported {len(df):,} llm_summaries -> {CACHE_PATH}")
    return len(df)


if __name__ == "__main__":
    export_llm_summaries()
