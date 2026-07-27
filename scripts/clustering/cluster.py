"""
Clustering Orchestrator — combines all clustering methods.

Runs K-Means, DBSCAN, and Hierarchical on the same data,
compares them, and returns the best result.

This module is called by the pipeline to cluster:
1. Keyword vectors (from NLP feature extraction)
2. Social media sentiment (when pipeline connects everything)
"""

import numpy as np
import pandas as pd

from . import kmeans
from . import dbscan
from . import hierarchical
from .compare import evaluate, compare


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
