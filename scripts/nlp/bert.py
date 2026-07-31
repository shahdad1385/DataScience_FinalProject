"""
BERT / sentence-transformers model — load, encode, save + keyword extraction.

This file handles the BERT model itself and keyword extraction using it.
"""

import os
import re
import pickle
import numpy as np
from sklearn.decomposition import PCA

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_PCA_DIM = 50
DEFAULT_BATCH_SIZE = 64


def load_model(model_name=DEFAULT_MODEL):
    if not HAS_SENTENCE_TRANSFORMERS:
        raise ImportError("sentence-transformers not installed")
    return SentenceTransformer(model_name, device="cpu")


def encode(model, texts, batch_size=DEFAULT_BATCH_SIZE):
    return model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)


def encode_single(model, text):
    return model.encode([text], convert_to_numpy=True)[0]


def fit_pca(embeddings, n_components=DEFAULT_PCA_DIM):
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    return pca, reduced


def transform_pca(embeddings, pca):
    return pca.transform(embeddings)


def save_artifacts(model, pca, name="bert"):
    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save(os.path.join(MODELS_DIR, f"{name}_model"))
    with open(os.path.join(MODELS_DIR, f"{name}_pca.pkl"), "wb") as f:
        pickle.dump(pca, f)


def load_artifacts(name="bert"):
    model = SentenceTransformer(os.path.join(MODELS_DIR, f"{name}_model"), device="cpu")
    with open(os.path.join(MODELS_DIR, f"{name}_pca.pkl"), "rb") as f:
        pca = pickle.load(f)
    return model, pca


def extract_keywords(model, text, top_n=5):
    """
    Extract keywords using BERT token embeddings.
    Encodes the full document, then encodes individual words,
    finds words whose embeddings are most similar to the document embedding.
    """
    doc_emb = encode_single(model, text)

    # Tokenize: simple split + lowercase, keep unique words
    tokens = re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()
    unique_tokens = list(dict.fromkeys(tokens))  # preserve order, remove dups

    if len(unique_tokens) < 2:
        return unique_tokens[:top_n]

    # Encode each unique word
    word_embs = model.encode(unique_tokens, convert_to_numpy=True)

    # Compute cosine similarity to document embedding
    doc_norm = np.linalg.norm(doc_emb)
    sims = []
    for i, w in enumerate(unique_tokens):
        w_norm = np.linalg.norm(word_embs[i])
        sim = np.dot(doc_emb, word_embs[i]) / (doc_norm * w_norm + 1e-8)
        sims.append((w, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in sims[:top_n]]
