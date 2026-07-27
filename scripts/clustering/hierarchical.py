"""
Hierarchical (Agglomerative) clustering model.

Good for dendrogram analysis and when cluster count is unknown.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def create(n_clusters=3, linkage="ward"):
    """Create a Hierarchical clustering model."""
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)


def fit_predict(model, X):
    """Fit and predict cluster assignments."""
    return model.fit_predict(X)


def get_labels(model):
    """Get cluster labels from fitted model."""
    return model.labels_
