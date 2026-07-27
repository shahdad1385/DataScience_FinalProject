"""
BERT / sentence-transformers model — load, encode, save.

This file handles only the BERT model itself.
Feature extraction using BERT is in features.py.
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PCA_DIM = 50
DEFAULT_BATCH_SIZE = 64


def load_model(model_name=DEFAULT_MODEL):
    """Load sentence-transformers model."""
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers not installed")
    return SentenceTransformer(model_name)


def encode(model, texts, batch_size=DEFAULT_BATCH_SIZE):
    """Encode texts into embeddings."""
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


def fit_pca(embeddings, n_components=DEFAULT_PCA_DIM):
    """Fit PCA on training embeddings."""
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pca, reduced


def transform_pca(embeddings, pca):
    """Transform embeddings using fitted PCA."""
    return pca.transform(embeddings)


def save_artifacts(model, pca, name="bert"):
    """Save model and PCA to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, f"{name}_model"))
    with open(os.path.join(MODELS_DIR, f"{name}_pca.pkl"), "wb") as f:
        pickle.dump(pca, f)


def load_artifacts(name="bert"):
    """Load model and PCA from disk."""
    model = SentenceTransformer(os.path.join(MODELS_DIR, f"{name}_model"))
    with open(os.path.join(MODELS_DIR, f"{name}_pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    return model, pca
