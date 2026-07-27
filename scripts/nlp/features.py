"""
NLP Feature Extraction — ALL functions in one file.

Models (w2v_model, bert_model) are passed as parameters.
Clustering is NOT implemented — passed as parameter when needed later.
Keywords are extracted dynamically from data, not predefined.
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation

from . import word2vec as w2v_module
from . import bert as bert_module

warnings.filterwarnings("ignore")


# =============================================================================
# HELPERS
# =============================================================================

def _clean(text):
    if not isinstance(text, str) or pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text):
    return _clean(text).split()


def _get_texts(df):
    texts = []
    for _, row in df.iterrows():
        parts = [str(row.get("headline", ""))]
        if pd.notna(row.get("llm_summary")) and row["llm_summary"]:
            parts.append(str(row["llm_summary"]))
        elif pd.notna(row.get("summary")) and row["summary"]:
            parts.append(str(row["summary"]))
        texts.append(" ".join(parts))
    return texts


def _groupby(df, agg_dict):
    return df.groupby(["ticker", "date"]).agg(agg_dict).reset_index()


# =============================================================================
# 1. SENTIMENT
# =============================================================================

def extract_sentiment(df):
    """VADER sentiment: compound, pos, neg, neu — grouped by (ticker, date)."""
    analyzer = SentimentIntensityAnalyzer()
    records = []
    for _, row in df.iterrows():
        text = str(row.get("headline", ""))
        s = analyzer.polarity_scores(text) if text else {"compound": 0, "pos": 0, "neg": 0, "neu": 1}
        records.append({"date": row["date"], "ticker": row["ticker"],
                        "compound": s["compound"], "pos": s["pos"],
                        "neg": s["neg"], "neu": s["neu"]})
    result = pd.DataFrame(records)
    return _groupby(result, {
        "compound": ["mean", "std", "min", "max"],
        "pos": "mean", "neg": "mean", "neu": "mean",
    })


# =============================================================================
# 2. TEXT STATISTICS
# =============================================================================

def extract_text_stats(df):
    """Word count, sentence count, avg word length, etc. — grouped by (ticker, date)."""
    records = []
    for _, row in df.iterrows():
        text = str(row.get("headline", ""))
        words = text.split()
        wc = len(words)
        cc = len(text)
        sents = len(re.split(r'[.!?]+', text))
        records.append({
            "date": row["date"], "ticker": row["ticker"],
            "word_count": wc, "char_count": cc,
            "avg_word_length": np.mean([len(w) for w in words]) if words else 0,
            "sentence_count": sents,
            "avg_sentence_length": wc / sents if sents > 0 else 0,
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / cc if cc > 0 else 0,
            "number_ratio": sum(1 for c in text if c.isdigit()) / cc if cc > 0 else 0,
        })
    result = pd.DataFrame(records)
    return _groupby(result, {
        "word_count": ["mean", "std", "max"], "char_count": "mean",
        "avg_word_length": "mean", "sentence_count": ["mean", "max"],
        "avg_sentence_length": "mean", "uppercase_ratio": "mean", "number_ratio": "mean",
    })


# =============================================================================
# 3. KEYWORD EXTRACTION (dynamic — finds keywords from data)
# =============================================================================

def extract_keywords_w2v(df, w2v_model, top_n=5):
    """
    Extract keywords from each article using Word2Vec.
    Compares each word's vector to the document's average vector.
    Returns top_n keywords per article, then counts them per (ticker, date).
    """
    texts = _get_texts(df)
    all_keywords = []
    for text in texts:
        kws = w2v_module.extract_keywords(w2v_model, text, top_n=top_n)
        all_keywords.append(kws)

    # Count keyword occurrences per article
    records = []
    for i, kws in enumerate(all_keywords):
        row = {"date": df.iloc[i]["date"], "ticker": df.iloc[i]["ticker"]}
        for kw in kws:
            safe = kw.replace(" ", "_")
            row[f"kw_{safe}"] = row.get(f"kw_{safe}", 0) + 1
        records.append(row)

    result = pd.DataFrame(records)
    kw_cols = [c for c in result.columns if c.startswith("kw_")]
    if kw_cols:
        return _groupby(result, {col: "sum" for col in kw_cols})
    else:
        return df[["ticker", "date"]].drop_duplicates().reset_index(drop=True)


def extract_keywords_bert(df, bert_model, top_n=5):
    """
    Extract keywords from each article using BERT.
    Compares each word's BERT embedding to the document embedding.
    Returns top_n keywords per article, then counts them per (ticker, date).
    """
    texts = _get_texts(df)
    all_keywords = []
    for text in texts:
        kws = bert_module.extract_keywords(bert_model, text, top_n=top_n)
        all_keywords.append(kws)

    records = []
    for i, kws in enumerate(all_keywords):
        row = {"date": df.iloc[i]["date"], "ticker": df.iloc[i]["ticker"]}
        for kw in kws:
            safe = kw.replace(" ", "_")
            row[f"kw_{safe}"] = row.get(f"kw_{safe}", 0) + 1
        records.append(row)

    result = pd.DataFrame(records)
    kw_cols = [c for c in result.columns if c.startswith("kw_")]
    if kw_cols:
        return _groupby(result, {col: "sum" for col in kw_cols})
    else:
        return df[["ticker", "date"]].drop_duplicates().reset_index(drop=True)


# =============================================================================
# 4. LDA TOPICS
# =============================================================================

def train_lda(texts, n_topics=10):
    cleaned = [_clean(t) for t in texts if _clean(t)]
    vec = CountVectorizer(max_features=1000, stop_words="english", max_df=0.95, min_df=2)
    X = vec.fit_transform(cleaned)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=20)
    lda.fit(X)
    return lda, vec


def extract_topics(df, lda=None, vectorizer=None, texts=None):
    """LDA topic distributions — grouped by (ticker, date). Returns (DataFrame, lda, vectorizer)."""
    if texts is None:
        texts = _get_texts(df)
    cleaned = [_clean(t) for t in texts]

    if lda is None:
        lda, vectorizer = train_lda(texts)

    X = vectorizer.transform(cleaned)
    topic_dists = lda.transform(X)

    records = []
    for i in range(len(df)):
        row = {"date": df.iloc[i]["date"], "ticker": df.iloc[i]["ticker"]}
        for j in range(topic_dists.shape[1]):
            row[f"topic_{j}"] = topic_dists[i, j]
        row["dominant_topic"] = int(np.argmax(topic_dists[i]))
        records.append(row)

    result = pd.DataFrame(records)
    agg_dict = {f"topic_{j}": "mean" for j in range(topic_dists.shape[1])}
    agg_dict["dominant_topic"] = lambda x: x.mode()[0] if len(x.mode()) > 0 else 0
    return _groupby(result, agg_dict), lda, vectorizer


# =============================================================================
# 5. NAMED ENTITIES (dynamic — finds entities from data using word frequency)
# =============================================================================

def extract_entities(df):
    """
    Entity-like features: count of capitalized/unique words,
    vocabulary richness, and repeated terms.
    """
    texts = _get_texts(df)
    records = []
    for i, text in enumerate(texts):
        words = text.split()
        unique_words = list(set(words))
        tl = text.lower()
        records.append({
            "date": df.iloc[i]["date"], "ticker": df.iloc[i]["ticker"],
            "unique_word_count": len(unique_words),
            "vocabulary_richness": len(unique_words) / max(len(words), 1),
            "total_word_count": len(words),
            "avg_word_freq": np.mean([tl.count(w) for w in unique_words]) if unique_words else 0,
        })
    result = pd.DataFrame(records)
    return _groupby(result, {
        "unique_word_count": "mean", "vocabulary_richness": "mean",
        "total_word_count": "sum", "avg_word_freq": "mean",
    })


# =============================================================================
# 6. WORD2VEC EMBEDDINGS
# =============================================================================

def extract_word2vec(df, w2v_model):
    """Word2Vec embeddings — averaged per (ticker, date)."""
    texts = _get_texts(df)
    embeddings = w2v_module.get_embeddings(w2v_model, texts)

    feat_df = df[["date", "ticker"]].copy()
    for i in range(embeddings.shape[1]):
        feat_df[f"w2v_{i}"] = embeddings[:, i]

    agg_dict = {col: "mean" for col in feat_df.columns if col.startswith("w2v_")}
    agg = feat_df.groupby(["ticker", "date"]).agg(agg_dict).reset_index()
    agg["date"] = feat_df.groupby(["ticker", "date"])["date"].first().values
    agg["ticker"] = feat_df.groupby(["ticker", "date"])["ticker"].first().values
    return agg


# =============================================================================
# 7. TF-IDF + K-MEANS
# =============================================================================

def train_tfidf_kmeans(texts, max_features=500, n_clusters=8):
    cleaned = [_clean(t) for t in texts if _clean(t)]
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english",
                            ngram_range=(1, 2), min_df=2, max_df=0.95)
    X = tfidf.fit_transform(cleaned)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)
    return tfidf, kmeans


def extract_tfidf_clusters(df, tfidf, kmeans):
    """TF-IDF + K-Means cluster counts — grouped by (ticker, date)."""
    texts = _get_texts(df)
    cleaned = [_clean(t) for t in texts]
    X = tfidf.transform(cleaned)
    clusters = kmeans.predict(X)

    records = []
    for i in range(len(df)):
        row = {"date": df.iloc[i]["date"], "ticker": df.iloc[i]["ticker"]}
        for c in range(kmeans.n_clusters):
            row[f"tfidf_c{c}"] = 1 if clusters[i] == c else 0
        records.append(row)
    result = pd.DataFrame(records)
    cluster_cols = [c for c in result.columns if c.startswith("tfidf_c")]
    return _groupby(result, {col: "sum" for col in cluster_cols})


# =============================================================================
# 8. BERT EMBEDDINGS
# =============================================================================

def extract_bert(df, bert_model=None, pca=None, texts=None):
    """BERT embeddings — grouped by (ticker, date). Returns (DataFrame, bert_model, pca)."""
    if texts is None:
        texts = _get_texts(df)

    if bert_model is None:
        raw = bert_module.encode(bert_module.load_model(), texts)
        pca, reduced = bert_module.fit_pca(raw)
        bert_model = bert_module.load_model()
    else:
        raw = bert_module.encode(bert_model, texts, batch_size=64)
        reduced = bert_module.transform_pca(raw, pca)

    feat_df = df[["date", "ticker"]].copy()
    for i in range(reduced.shape[1]):
        feat_df[f"bert_{i}"] = reduced[:, i]

    agg_dict = {col: "mean" for col in feat_df.columns if col.startswith("bert_")}
    agg = feat_df.groupby(["ticker", "date"]).agg(agg_dict).reset_index()
    agg["date"] = feat_df.groupby(["ticker", "date"])["date"].first().values
    agg["ticker"] = feat_df.groupby(["ticker", "date"])["ticker"].first().values
    return agg, bert_model, pca


# =============================================================================
# 9. ARTICLE VOLUME
# =============================================================================

def extract_volume(df):
    """Article count + source count per (ticker, date)."""
    agg = df.groupby(["ticker", "date"]).agg({
        "headline": "count",
        "source": "nunique",
    }).reset_index()
    agg.columns = ["ticker", "date", "article_count", "source_count"]
    return agg


# =============================================================================
# 10. WORD VECTOR SIMILARITY
# =============================================================================

def extract_word_vector_similarity(df, w2v_model):
    """Word vector similarity features — consecutive word similarity, vocab coverage."""
    texts = _get_texts(df)
    records = []
    for i, text in enumerate(texts):
        tokens = _tokenize(text)
        vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        if len(vecs) < 2:
            avg_sim = max_sim = min_sim = 0
        else:
            sims = [np.dot(vecs[j], vecs[j+1]) / (np.linalg.norm(vecs[j]) * np.linalg.norm(vecs[j+1]) + 1e-8)
                    for j in range(len(vecs) - 1)]
            avg_sim, max_sim, min_sim = np.mean(sims), np.max(sims), np.min(sims)
        records.append({
            "date": df.iloc[i]["date"], "ticker": df.iloc[i]["ticker"],
            "wv_avg_sim": avg_sim, "wv_max_sim": max_sim, "wv_min_sim": min_sim,
            "wv_vocab_coverage": len(vecs) / max(len(tokens), 1),
        })
    result = pd.DataFrame(records)
    return _groupby(result, {
        "wv_avg_sim": "mean", "wv_max_sim": "mean",
        "wv_min_sim": "mean", "wv_vocab_coverage": "mean",
    })
