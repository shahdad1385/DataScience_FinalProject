"""
Clustering Orchestrator — combines all clustering methods.

Runs K-Means, DBSCAN, and Hierarchical on the same data,
compares them, and returns the best result.

This module is called by the pipeline to cluster:
1. Keyword vectors (from NLP feature extraction)
2. Social media sentiment (when pipeline connects everything)
"""

import os
import pickle
import numpy as np
import pandas as pd

from . import kmeans
from . import dbscan
from . import hierarchical
from .compare import evaluate, compare

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

# Clustering runs over the whole sector's news, matching nlp.extract. The old
# five-name list also excluded the industry sentinel, so a split made up of
# industry-wide articles (val: 40 of 40) produced "No keyword data for
# clustering" and contributed no cluster features at all.
from ..nlp.extract import TICKERS as _NEWS_TICKERS
from ..nlp.features import INDUSTRY_TICKER

TICKERS = list(_NEWS_TICKERS) + [INDUSTRY_TICKER]


def cluster_keyword_vectors(keyword_vectors, keyword_names, n_clusters=8):
    """
    Cluster keyword vectors by semantic similarity.
    This is what the NLP module saves for us — now we cluster them.

    Args:
        keyword_vectors: numpy array of shape (n_keywords, vector_dim)
        keyword_names: list of keyword strings
        n_clusters: number of clusters for K-Means

    Returns:
        dict with keyword -> cluster_id mapping and evaluation scores
    """
    if len(keyword_vectors) < n_clusters:
        n_clusters = max(2, len(keyword_vectors) // 2)

    # Run all methods
    results = {}

    # K-Means
    km = kmeans.create(n_clusters=n_clusters)
    km_labels = kmeans.fit_predict(km, keyword_vectors)
    results["kmeans"] = km_labels

    # DBSCAN (auto-tune eps)
    eps = np.median(np.linalg.norm(keyword_vectors - keyword_vectors.mean(axis=0), axis=1))
    db = dbscan.create(eps=eps, min_samples=2)
    db_labels = dbscan.predict(db, keyword_vectors)
    results["dbscan"] = db_labels

    # Hierarchical
    hc = hierarchical.create(n_clusters=n_clusters)
    hc_labels = hierarchical.fit_predict(hc, keyword_vectors)
    results["hierarchical"] = hc_labels

    # Compare and pick best
    best_method, scores = compare(keyword_vectors, results)

    # Build keyword -> cluster mapping
    keyword_clusters = {}
    best_labels = results[best_method]
    for name, label in zip(keyword_names, best_labels):
        keyword_clusters[name] = int(label)

    return {
        "keyword_clusters": keyword_clusters,
        "best_method": best_method,
        "scores": scores,
        "labels": results,
    }


def cluster_article_keywords(article_keywords_per_day, keyword_clusters):
    """
    Given article keywords (from NLP extraction) and keyword cluster assignments,
    count how many keywords from each cluster appear per (ticker, date).

    Args:
        article_keywords_per_day: DataFrame with columns [date, ticker, kw_1, kw_2, ...]
        keyword_clusters: dict mapping keyword string -> cluster_id

    Returns:
        DataFrame with [date, ticker, cluster_0_count, cluster_1_count, ...]
    """
    # Reverse mapping: keyword -> cluster
    kw_to_cluster = {}
    for kw, cluster_id in keyword_clusters.items():
        kw_to_cluster[kw.lower().replace(" ", "_")] = cluster_id

    # Count per cluster
    records = []
    for _, row in article_keywords_per_day.iterrows():
        cluster_counts = {}
        for col in row.index:
            if col.startswith("kw_"):
                kw_name = col[3:]  # remove "kw_" prefix
                if kw_name in kw_to_cluster:
                    cid = kw_to_cluster[kw_name]
                    cluster_counts[f"cluster_{cid}"] = cluster_counts.get(f"cluster_{cid}", 0) + row[col]

        record = {"date": row["date"], "ticker": row["ticker"]}
        record.update(cluster_counts)
        records.append(record)

    result = pd.DataFrame(records)
    cluster_cols = [c for c in result.columns if c.startswith("cluster_")]
    return result.groupby(["ticker", "date"])[cluster_cols].sum().reset_index()


def cluster_sentiment(sentiments, n_clusters=3):
    """
    Cluster sentiment scores into groups (bullish/neutral/bearish).
    Used for social media clustering in the pipeline.

    Args:
        sentiments: array of sentiment scores
        n_clusters: number of clusters

    Returns:
        dict with labels, method, scores
    """
    X = np.array(sentiments).reshape(-1, 1)

    results = {}

    # K-Means
    km = kmeans.create(n_clusters=n_clusters)
    km_labels = kmeans.fit_predict(km, X)
    results["kmeans"] = km_labels

    # DBSCAN
    db = dbscan.create(eps=0.3, min_samples=2)
    db_labels = dbscan.predict(db, X)
    results["dbscan"] = db_labels

    # Hierarchical
    hc = hierarchical.create(n_clusters=n_clusters)
    hc_labels = hierarchical.fit_predict(hc, X)
    results["hierarchical"] = hc_labels

    best_method, scores = compare(X, results)

    # Label clusters by mean sentiment
    best_labels = results[best_method]
    means = [X[best_labels == i].mean() for i in range(len(set(best_labels)) - (1 if -1 in best_labels else 0))]
    order = np.argsort(means)
    label_map = {}
    label_map[order[0]] = "bearish"
    if len(order) > 1:
        label_map[order[1]] = "neutral"
    if len(order) > 2:
        label_map[order[2]] = "bullish"

    labeled = np.array([label_map.get(l, "unknown") for l in best_labels])

    return {
        "labels": best_labels,
        "labeled": labeled,
        "best_method": best_method,
        "scores": scores,
    }


def save_clustering_models(models_dict):
    """Save clustering models to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in models_dict.items():
        path = os.path.join(MODELS_DIR, f"clustering_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
    print(f"  Clustering models saved to {MODELS_DIR}")


def load_clustering_models():
    """Load clustering models from disk."""
    models = {}
    for name in ["kmeans_keyword", "kmeans_sentiment", "best_keyword", "best_sentiment"]:
        path = os.path.join(MODELS_DIR, f"clustering_{name}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def cluster_keyword_vectors_from_db(split, w2v_model, engine):
    """
    Read keyword features from DB, cluster with K-Means/DBSCAN/Hierarchical,
    save cluster features to DB, and return the best clustering result.
    """
    from scripts.nlp.word2vec import tokenize

    tf = ", ".join(f"'{t}'" for t in TICKERS)
    df_kw = pd.read_sql(f"""
        SELECT * FROM {split}_nlp_kw_w2v
        WHERE ticker IN ({tf})
        ORDER BY date
    """, engine)

    if df_kw.empty:
        print(f"    No keyword data for clustering")
        return None, None

    kw_cols = [c for c in df_kw.columns if c.startswith("kw_")]
    if not kw_cols:
        print(f"    No keyword columns found")
        return None, None

    kw_vectors = df_kw[kw_cols].fillna(0).values
    kw_names = [c.replace("kw_", "").replace("_", " ") for c in kw_cols]

    if len(kw_vectors) < 3:
        print(f"    Insufficient keyword data for clustering ({len(kw_vectors)} rows)")
        return None, None

    n_clusters = min(8, max(3, len(kw_vectors) // 5))

    results = {}
    km = kmeans.create(n_clusters=n_clusters)
    km_labels = kmeans.fit_predict(km, kw_vectors)
    results["kmeans"] = km_labels

    eps = np.median(np.linalg.norm(kw_vectors - kw_vectors.mean(axis=0), axis=1))
    if eps > 0:
        db = dbscan.create(eps=eps, min_samples=2)
        db_labels = dbscan.predict(db, kw_vectors)
        results["dbscan"] = db_labels

    hc = hierarchical.create(n_clusters=n_clusters)
    hc_labels = hierarchical.fit_predict(hc, kw_vectors)
    results["hierarchical"] = hc_labels

    best_method, scores = compare(kw_vectors, results)
    best_labels = results[best_method]

    records = []
    for i in range(len(df_kw)):
        record = {"date": df_kw.iloc[i]["date"], "ticker": df_kw.iloc[i]["ticker"]}
        for c in range(n_clusters):
            record[f"kw_cluster_{c}"] = int(best_labels[i] == c) if i < len(best_labels) else 0
        record["kw_dominant_cluster"] = int(best_labels[i]) if i < len(best_labels) else 0
        records.append(record)

    result_df = pd.DataFrame(records)
    cluster_cols = [c for c in result_df.columns if c.startswith("kw_cluster_") or c == "kw_dominant_cluster"]
    agg = result_df.groupby(["ticker", "date"])[cluster_cols].sum().reset_index()
    agg["date"] = result_df.groupby(["ticker", "date"])["date"].first().values
    agg["ticker"] = result_df.groupby(["ticker", "date"])["ticker"].first().values

    return agg, {"best_method": best_method, "scores": scores, "n_clusters": n_clusters}


def cluster_sentiment_from_db(split, engine):
    """
    Read sentiment features from DB, cluster into bullish/neutral/bearish,
    save to DB, and return the result.
    """
    tf = ", ".join(f"'{t}'" for t in TICKERS)
    try:
        df_sent = pd.read_sql(f"""
            SELECT date, ticker, compound_mean FROM {split}_nlp_sentiment
            WHERE ticker IN ({tf})
            ORDER BY date
        """, engine)
    except Exception:
        print(f"    No sentiment table for clustering")
        return None, None

    if df_sent.empty or "compound_mean" not in df_sent.columns:
        print(f"    No sentiment data for clustering")
        return None, None

    sentiments = df_sent["compound_mean"].fillna(0).values.reshape(-1, 1)

    if len(sentiments) < 6:
        print(f"    Insufficient sentiment data for clustering ({len(sentiments)} rows)")
        return None, None

    results = {}
    n_clusters = 3

    km = kmeans.create(n_clusters=n_clusters)
    km_labels = kmeans.fit_predict(km, sentiments)
    results["kmeans"] = km_labels

    hc = hierarchical.create(n_clusters=n_clusters)
    hc_labels = hierarchical.fit_predict(hc, sentiments)
    results["hierarchical"] = hc_labels

    best_method, scores = compare(sentiments, results)
    best_labels = results[best_method]

    means = [sentiments[best_labels == i].mean() for i in range(n_clusters)]
    order = np.argsort(means)
    label_map = {order[0]: 0, order[1]: 1, order[2]: 2} if len(order) >= 3 else {0: 0, 1: 1}

    records = []
    for i in range(len(df_sent)):
        cluster_id = int(best_labels[i]) if i < len(best_labels) else 1
        records.append({
            "date": df_sent.iloc[i]["date"],
            "ticker": df_sent.iloc[i]["ticker"],
            "sent_cluster_bearish": 1 if cluster_id == label_map.get(0, 0) else 0,
            "sent_cluster_neutral": 1 if cluster_id == label_map.get(1, 1) else 0,
            "sent_cluster_bullish": 1 if cluster_id == label_map.get(2, 2) else 0,
            "sent_cluster_id": cluster_id,
        })

    result_df = pd.DataFrame(records)
    cluster_cols = [c for c in result_df.columns if c.startswith("sent_cluster_")]
    agg = result_df.groupby(["ticker", "date"])[cluster_cols].sum().reset_index()
    agg["date"] = result_df.groupby(["ticker", "date"])["date"].first().values
    agg["ticker"] = result_df.groupby(["ticker", "date"])["ticker"].first().values

    return agg, {"best_method": best_method, "scores": scores}
