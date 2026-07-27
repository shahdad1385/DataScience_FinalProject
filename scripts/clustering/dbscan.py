"""
DBSCAN clustering model.

Good for arbitrary-shaped clusters and noise detection.
Used on sentiment scores to find natural groupings.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def create(eps=0.5, min_samples=5, metric="euclidean"):
    """Create a DBSCAN model."""
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric)


def fit(model, X):
    """Fit DBSCAN on data."""
    model.fit(X)
    return model


def predict(model, X):
    """Predict cluster assignments (labels)."""
    return model.fit_predict(X)


def get_labels(model):
    """Get cluster labels from fitted model."""
    return model.labels_


def get_n_clusters(labels):
    """Get number of clusters found (excluding noise label -1)."""
    return len(set(labels)) - (1 if -1 in labels else 0)


def get_noise_ratio(labels):
    """Get fraction of points classified as noise (-1)."""
    return np.sum(labels == -1) / len(labels) if len(labels) > 0 else 0
