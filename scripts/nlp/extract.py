"""
NLP Feature Extraction Orchestrator.

Calls all extraction functions from features.py on each train/val/test split.
Trains models on train only, reuses on val/test.
Saves feature tables to DB: one table per feature type per split.
"""

import os
import sys
import pickle
import warnings

import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db import get_engine

from . import word2vec as w2v_module
from . import bert as bert_module
from .features import (
    extract_sentiment, extract_text_stats,
    extract_keywords_w2v, extract_keywords_bert,
    extract_topics, extract_entities,
    extract_word2vec, extract_tfidf_clusters, train_tfidf_kmeans,
    extract_bert, extract_volume, extract_word_vector_similarity,
    _get_texts,
)

warnings.filterwarnings("ignore")

engine = get_engine()
SPLITS = ["train", "val", "test"]
TICKERS = ["NVDA", "GOOGL", "AVGO", "AMD", "TSM"]
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def save_nlp_models(models):
    """Save all NLP models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if "lda" in models:
        with open(os.path.join(MODELS_DIR, "lda.pkl"), "wb") as f:
            pickle.dump(models["lda"], f)
    if "lda_vec" in models:
        with open(os.path.join(MODELS_DIR, "lda_vec.pkl"), "wb") as f:
            pickle.dump(models["lda_vec"], f)
    if "tfidf" in models:
        with open(os.path.join(MODELS_DIR, "tfidf.pkl"), "wb") as f:
            pickle.dump(models["tfidf"], f)
    if "kmeans" in models:
        with open(os.path.join(MODELS_DIR, "tfidf_kmeans.pkl"), "wb") as f:
            pickle.dump(models["kmeans"], f)
    if "bert" in models and "bert_pca" in models:
        bert_module.save_artifacts(models["bert"], models["bert_pca"])
    print(f"  NLP models saved to {MODELS_DIR}")


def load_nlp_models():
    """Load all NLP models from disk."""
    models = {}
    w2v_path = os.path.join(MODELS_DIR, "word2vec.model")
    if os.path.exists(w2v_path):
        models["w2v"] = w2v_module.load()
    lda_path = os.path.join(MODELS_DIR, "lda.pkl")
    if os.path.exists(lda_path):
        with open(lda_path, "rb") as f:
            models["lda"] = pickle.load(f)
    lda_vec_path = os.path.join(MODELS_DIR, "lda_vec.pkl")
    if os.path.exists(lda_vec_path):
        with open(lda_vec_path, "rb") as f:
            models["lda_vec"] = pickle.load(f)
    tfidf_path = os.path.join(MODELS_DIR, "tfidf.pkl")
    if os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            models["tfidf"] = pickle.load(f)
    kmeans_path = os.path.join(MODELS_DIR, "tfidf_kmeans.pkl")
    if os.path.exists(kmeans_path):
        with open(kmeans_path, "rb") as f:
            models["kmeans"] = pickle.load(f)
    bert_model_path = os.path.join(MODELS_DIR, "bert_model")
    if os.path.exists(bert_model_path) and bert_module.HAS_SENTENCE_TRANSFORMERS:
        try:
            models["bert"], models["bert_pca"] = bert_module.load_artifacts()
        except Exception:
            pass
    return models


def load_news(split):
    tf = ", ".join(f"'{t}'" for t in TICKERS)
    return pd.read_sql(f"""
        SELECT id, date, headline, summary, llm_summary, ticker, source
        FROM {split}_news
        WHERE headline IS NOT NULL AND LENGTH(headline) > 10
          AND (ticker IN ({tf}) OR ticker IS NULL OR ticker = '')
        ORDER BY date
    """, engine, parse_dates=["date"])


def load_social(split):
    tf = ", ".join(f"'{t}'" for t in TICKERS)
    return pd.read_sql(f"""
        SELECT id, date, headline, ticker, sentiment_score
        FROM {split}_social_sentiment
        WHERE headline IS NOT NULL AND LENGTH(headline) > 10
          AND (ticker IN ({tf}) OR ticker IS NULL OR ticker = '')
        ORDER BY date
    """, engine, parse_dates=["date"])


def prepare_corpus(df_news, df_social):
    texts = []
    for _, row in df_news.iterrows():
        parts = [row["headline"]]
        if pd.notna(row.get("llm_summary")) and row["llm_summary"]:
            parts.append(row["llm_summary"])
        elif pd.notna(row.get("summary")) and row["summary"]:
            parts.append(row["summary"])
        t = " ".join(parts)
        if len(t) > 10:
            texts.append(t)
    for _, row in df_social.iterrows():
        if len(row["headline"]) > 10:
            texts.append(row["headline"])
    return texts


def create_table(split, df, name):
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {name}"))
        cols = [f'"{c}" REAL' for c in df.columns if c not in ("ticker", "date")]
        sql = f'CREATE TABLE {name} (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, date DATE, {", ".join(cols)})'
        conn.execute(text(sql))
        df.to_sql(name, engine, if_exists="append", index=False)
        conn.commit()
    print(f"  -> {name}: {len(df):,} rows, {len(df.columns)} cols")


def process_split(split, models=None):
    print(f"\n{'='*50}")
    print(f"  SPLIT: {split.upper()}")
    print(f"{'='*50}")

    df_news = load_news(split)
    df_social = load_social(split)
    print(f"  News: {len(df_news):,} | Social: {len(df_social):,}")

    if df_news.empty:
        print("  No news data.")
        return models

    models = models or {}
    texts = _get_texts(df_news)
    all_texts = prepare_corpus(df_news, df_social)

    # 1. Sentiment
    print(f"  [1/11] Sentiment...")
    create_table(split, extract_sentiment(df_news), f"{split}_nlp_sentiment")

    # 2. Text stats
    print(f"  [2/11] Text stats...")
    create_table(split, extract_text_stats(df_news), f"{split}_nlp_text_stats")

    # 3. Keywords (Word2Vec-based)
    print(f"  [3/11] Keywords (Word2Vec)...")
    if split == "train":
        w2v_model = w2v_module.train(all_texts)
        w2v_module.save(w2v_model)
        models["w2v"] = w2v_model
    elif "w2v" not in models:
        models["w2v"] = w2v_module.load()
    w2v_model = models["w2v"]
    create_table(split, extract_keywords_w2v(df_news, w2v_model), f"{split}_nlp_kw_w2v")

    # 4. Keywords (BERT-based)
    if bert_module.HAS_SENTENCE_TRANSFORMERS:
        print(f"  [4/11] Keywords (BERT)...")
        if split == "train":
            if os.path.exists(os.path.join(MODELS_DIR, "bert_model")):
                bert_model, bert_pca = bert_module.load_artifacts()
            else:
                bert_model = bert_module.load_model()
                bert_pca = None
            models["bert"] = bert_model
            models["bert_pca"] = bert_pca
        elif "bert" not in models:
            try:
                models["bert"], models["bert_pca"] = bert_module.load_artifacts()
            except Exception:
                models["bert"] = bert_module.load_model()
                models["bert_pca"] = None
        create_table(split, extract_keywords_bert(df_news, models["bert"]), f"{split}_nlp_kw_bert")
    else:
        print(f"  [4/11] Keywords (BERT) skipped")

    # 5. LDA topics
    print(f"  [5/11] LDA topics...")
    if split == "train":
        topic_feats, lda, lda_vec = extract_topics(df_news)
        models["lda"] = lda
        models["lda_vec"] = lda_vec
    elif "lda" not in models:
        lda_path = os.path.join(MODELS_DIR, "lda.pkl")
        lda_vec_path = os.path.join(MODELS_DIR, "lda_vec.pkl")
        if os.path.exists(lda_path) and os.path.exists(lda_vec_path):
            with open(lda_path, "rb") as f:
                models["lda"] = pickle.load(f)
            with open(lda_vec_path, "rb") as f:
                models["lda_vec"] = pickle.load(f)
        else:
            topic_feats, lda, lda_vec = extract_topics(df_news)
            models["lda"] = lda
            models["lda_vec"] = lda_vec
            create_table(split, topic_feats, f"{split}_nlp_topics")
            return models
    topic_feats, _, _ = extract_topics(df_news, models["lda"], models["lda_vec"], texts)
    create_table(split, topic_feats, f"{split}_nlp_topics")

    # 6. Named entities
    print(f"  [6/11] Named entities...")
    create_table(split, extract_entities(df_news), f"{split}_nlp_entities")

    # 7. Word2Vec embeddings
    print(f"  [7/11] Word2Vec embeddings...")
    create_table(split, extract_word2vec(df_news, w2v_model), f"{split}_nlp_w2v")

    # 8. TF-IDF + K-Means
    print(f"  [8/11] TF-IDF + K-Means...")
    if split == "train":
        tfidf, kmeans = train_tfidf_kmeans(all_texts)
        models["tfidf"] = tfidf
        models["kmeans"] = kmeans
    elif "tfidf" not in models:
        tfidf_path = os.path.join(MODELS_DIR, "tfidf.pkl")
        kmeans_path = os.path.join(MODELS_DIR, "tfidf_kmeans.pkl")
        if os.path.exists(tfidf_path) and os.path.exists(kmeans_path):
            with open(tfidf_path, "rb") as f:
                models["tfidf"] = pickle.load(f)
            with open(kmeans_path, "rb") as f:
                models["kmeans"] = pickle.load(f)
        else:
            tfidf, kmeans = train_tfidf_kmeans(all_texts)
            models["tfidf"] = tfidf
            models["kmeans"] = kmeans
    create_table(split, extract_tfidf_clusters(df_news, models["tfidf"], models["kmeans"]), f"{split}_nlp_tfidf")

    # 9. BERT embeddings
    if bert_module.HAS_SENTENCE_TRANSFORMERS:
        print(f"  [9/11] BERT embeddings...")
        if split == "train":
            bert_feats, bert_model2, bert_pca2 = extract_bert(df_news)
            models["bert"] = bert_model2
            models["bert_pca"] = bert_pca2
        elif "bert" not in models:
            try:
                models["bert"], models["bert_pca"] = bert_module.load_artifacts()
            except Exception:
                models["bert"] = bert_module.load_model()
                models["bert_pca"] = None
            bert_feats, _, _ = extract_bert(df_news, models["bert"], models.get("bert_pca"), texts)
        else:
            bert_feats, _, _ = extract_bert(df_news, models["bert"], models.get("bert_pca"), texts)
        create_table(split, bert_feats, f"{split}_nlp_bert")
    else:
        print(f"  [9/11] BERT skipped")

    # 10. Article volume
    print(f"  [10/11] Article volume...")
    create_table(split, extract_volume(df_news), f"{split}_nlp_volume")

    # 11. Word vector similarity
    print(f"  [11/11] Word vector similarity...")
    create_table(split, extract_word_vector_similarity(df_news, w2v_model), f"{split}_nlp_wv_sim")

    # Save all NLP models after train split
    if split == "train":
        save_nlp_models(models)

    return models


def main():
    print("=" * 50)
    print("NLP FEATURE EXTRACTION")
    print(f"  Tickers: {TICKERS}")
    print(f"  BERT: {bert_module.HAS_SENTENCE_TRANSFORMERS}")
    print("=" * 50)

    # Load existing models from disk if available
    models = load_nlp_models()
    if models:
        print(f"  Loaded {len(models)} models from disk: {list(models.keys())}")

    for split in SPLITS:
        models = process_split(split, models)

    print(f"\n{'='*50}")
    print("DONE")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
