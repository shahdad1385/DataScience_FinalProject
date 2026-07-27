"""
Clustering comparison — evaluate and pick the best method.

Compares K-Means, DBSCAN, and Hierarchical using
Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index.
"""

import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def evaluate(X, labels):
    """
    Evaluate clustering quality.
    Returns dict with Silhouette, Davies-Bouldin, Calinski-Harabasz scores.
    """
    # Need at least 2 clusters and not all noise
    unique_labels = set(labels)
    unique_labels.discard(-1)  # remove noise label

    if len(unique_labels) < 2:
        return {"silhouette": -1, "davies_bouldin": float("inf"), "calinski_harabasz": 0}

    # Filter out noise points for evaluation
    mask = labels != -1
    if mask.sum() < 2:
        return {"silhouette": -1, "davies_bouldin": float("inf"), "calinski_harabasz": 0}

    X_filtered = X[mask]
    labels_filtered = labels[mask]

    sil = silhouette_score(X_filtered, labels_filtered)
    db = davies_bouldin_score(X_filtered, labels_filtered)
    ch = calinski_harabasz_score(X_filtered, labels_filtered)

    return {"silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch}


def compare(X, results_dict):
    """
    Compare multiple clustering results.
    results_dict: {method_name: labels_array}
    Returns the best method name and its scores.
    """
    scores = {}
    for name, labels in results_dict.items():
        scores[name] = evaluate(X, labels)

    # Pick best by silhouette (higher = better)
    best_method = max(scores, key=lambda k: scores[k]["silhouette"])
    return best_method, scores
