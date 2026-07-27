"""
K-Means clustering model.

Handles both sentiment clustering (k=3: bullish/neutral/bearish)
and topic clustering (k=5-10 on keyword vectors).
"""

import numpy as np
from sklearn.cluster import KMeans


def create(n_clusters=3, random_state=42, n_init=10):
    """Create a K-Means model."""
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)


def fit(model, X):
    """Fit K-Means on data."""
    model.fit(X)
    return model


def predict(model, X):
    """Predict cluster assignments."""
    return model.predict(X)


def fit_predict(model, X):
    """Fit and predict in one step."""
    return model.fit_predict(X)


def get_cluster_centers(model):
    """Get cluster centers."""
    return model.cluster_centers_
