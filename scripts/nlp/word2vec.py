"""
Word2Vec model — train, save, load.

This file handles only the Word2Vec model itself.
Feature extraction using Word2Vec is in features.py.
"""

import os
import numpy as np
from gensim.models import Word2Vec

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def tokenize(text):
    """Simple whitespace tokenizer."""
    import re
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def train(texts, vector_size=100, window=5, min_count=2, epochs=10):
    """Train Word2Vec on tokenized sentences."""
    sentences = [tokenize(t) for t in texts]
    sentences = [s for s in sentences if len(s) > 1]
    model = Word2Vec(sentences, vector_size=vector_size, window=window,
                     min_count=min_count, epochs=epochs, workers=4, seed=42)
    return model


def save(model, name="word2vec"):
    """Save model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, f"{name}.model"))


def load(name="word2vec"):
    """Load model from disk."""
    return Word2Vec.load(os.path.join(MODELS_DIR, f"{name}.model"))


def get_embedding(model, text):
    """Get average Word2Vec embedding for a single text."""
    tokens = tokenize(text)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


def get_embeddings(model, texts):
    """Get average Word2Vec embeddings for a list of texts."""
    return np.array([get_embedding(model, t) for t in texts])
