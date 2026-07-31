"""
Word2Vec model — train, save, load + keyword extraction.

This file handles the Word2Vec model itself and keyword extraction using it.
"""

import os
import re
import numpy as np
from gensim.models import Word2Vec

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")


def tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def train(texts, vector_size=100, window=5, min_count=2, epochs=10):
    sentences = [tokenize(t) for t in texts]
    sentences = [s for s in sentences if len(s) > 1]
    model = Word2Vec(sentences, vector_size=vector_size, window=window,
                     min_count=min_count, epochs=epochs, workers=4, seed=42)
    return model


def save(model, name="word2vec"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, f"{name}.model"))


def load(name="word2vec"):
    return Word2Vec.load(os.path.join(MODELS_DIR, f"{name}.model"))


def get_embedding(model, text):
    tokens = tokenize(text)
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


def get_embeddings(model, texts):
    return np.array([get_embedding(model, t) for t in texts])


def extract_keywords(model, text, top_n=5):
    """
    Extract keywords from text by comparing each word's vector
    to the document's average vector. Returns top_n keywords
    with highest cosine similarity to the document.
    """
    tokens = tokenize(text)
    vecs = []
    valid_tokens = []
    for t in tokens:
        if t in model.wv:
            vecs.append(model.wv[t])
            valid_tokens.append(t)

    if len(vecs) < 2:
        return valid_tokens[:top_n]

    doc_vec = np.mean(vecs, axis=0)
    sims = []
    for i, v in enumerate(vecs):
        sim = np.dot(doc_vec, v) / (np.linalg.norm(doc_vec) * np.linalg.norm(v) + 1e-8)
        sims.append((valid_tokens[i], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return [w for w, s in sims[:top_n]]
