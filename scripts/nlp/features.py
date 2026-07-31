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

warnings.filterwarnings("ignore")


def _bert():
    """Import the BERT helper lazily.

    bert.py pulls in sentence_transformers -> transformers -> tensorflow. A
    broken install anywhere in that chain used to make this whole module
    unimportable, taking Word2Vec, TF-IDF, LDA and sentiment down with it even
    when BERT was not requested. Importing on demand keeps the non-BERT feature
    extractors usable and confines the failure to the BERT paths.
    """
    from . import bert as bert_module
    return bert_module

# Industry-level (non-ticker-specific) news is a real signal, not missing data.
# Grouping on a NaN ticker silently drops those rows, so they carry an explicit
# sentinel instead. data_assembly merges sentinel rows on date alone.
INDUSTRY_TICKER = "__INDUSTRY__"


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


def _flatten_columns(df):
    """Collapse a MultiIndex column axis into flat snake_case names.

    `df.groupby(...).agg({"compound": ["mean", "std"]})` produces a MultiIndex
    column axis. Left alone, those columns reach SQL as the literal text
    "('compound', 'mean')" and every downstream consumer that asks for
    `compound_mean` silently finds nothing. Flatten to `compound_mean`.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(part) for part in col if str(part) != "").strip("_")
            for col in df.columns
        ]
    return df


def _groupby(df, agg_dict):
    """Aggregate per (ticker, date), keeping industry-level rows.

    dropna=False is essential: pandas' default drops every group whose key
    contains NaN, which silently discarded all industry-level news (no ticker).
    Tickers are normalised to a sentinel upstream, but dropna=False keeps this
    correct even if a NaN slips through.
    """
    out = df.groupby(["ticker", "date"], dropna=False).agg(agg_dict).reset_index()
    return _flatten_columns(out)


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

def _keyword_counts_to_frame(df, all_keywords, max_keywords, vocab):
    """Turn per-article keyword lists into per-(ticker, date) count columns.

    `vocab` pins the column set. It is derived on train and then reused for val
    and test: each split otherwise selects its own top-N keywords from its own
    articles (train sees 2,587 articles, val only 40), so the same feature index
    meant different words in different splits and most train columns simply did
    not exist downstream. That was the source of the "58.6% of feature columns
    missing and zero-filled" warnings.

    Returns (frame, vocab_used).
    """
    records = []
    for i, kws in enumerate(all_keywords):
        row = {"date": df.iloc[i]["date"], "ticker": df.iloc[i]["ticker"]}
        for kw in kws:
            safe = f"kw_{kw.replace(' ', '_')}"
            row[safe] = row.get(safe, 0) + 1
        records.append(row)

    result = pd.DataFrame(records)
    if result.empty:
        result = df[["ticker", "date"]].copy()

    if vocab is None:
        # Training pass: choose the vocabulary and hand it back to be saved.
        kw_cols = [c for c in result.columns if c.startswith("kw_")]
        if kw_cols and len(kw_cols) > max_keywords:
            freq = result[kw_cols].sum().sort_values(ascending=False)
            kw_cols = list(freq.head(max_keywords).index)
            result = result.drop(columns=[c for c in result.columns
                                          if c.startswith("kw_") and c not in kw_cols])
        vocab = sorted(kw_cols)
    else:
        # Inference pass: force exactly the training columns, in the same order.
        for c in vocab:
            if c not in result.columns:
                result[c] = 0
        extra = [c for c in result.columns
                 if c.startswith("kw_") and c not in set(vocab)]
        if extra:
            result = result.drop(columns=extra)
        result = result[["ticker", "date"] + list(vocab)]

    agg_cols = [c for c in result.columns if c.startswith("kw_")]
    if not agg_cols:
        return df[["ticker", "date"]].drop_duplicates().reset_index(drop=True), vocab
    return _groupby(result, {c: "sum" for c in agg_cols}), vocab


def extract_keywords_w2v(df, w2v_model, top_n=5, max_keywords=500, vocab=None,
                         return_vocab=False):
    """
    Extract keywords from each article using Word2Vec.
    Compares each word's vector to the document's average vector.
    Returns top_n keywords per article, then counts them per (ticker, date).

    `vocab` pins the output columns to the training vocabulary (see
    _keyword_counts_to_frame). Pass return_vocab=True on the training split to
    receive (frame, vocab).
    """
    texts = _get_texts(df)
    all_keywords = [w2v_module.extract_keywords(w2v_model, t, top_n=top_n) for t in texts]
    frame, used = _keyword_counts_to_frame(df, all_keywords, max_keywords, vocab)
    return (frame, used) if return_vocab else frame


def extract_keywords_bert(df, bert_model, top_n=5, max_keywords=500, vocab=None,
                          return_vocab=False):
    """
    Extract keywords from each article using BERT.
    Compares each word's BERT embedding to the document embedding.
    Returns top_n keywords per article, then counts them per (ticker, date).

    Shares the vocabulary-pinning logic with the Word2Vec variant so val/test
    produce the same columns as train.
    """
    bert_module = _bert()
    texts = _get_texts(df)
    all_keywords = [bert_module.extract_keywords(bert_model, t, top_n=top_n) for t in texts]
    frame, used = _keyword_counts_to_frame(df, all_keywords, max_keywords, vocab)
    return (frame, used) if return_vocab else frame


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
    return _groupby(feat_df, agg_dict)


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

    bert_module = _bert()
    if bert_model is None:
        # Load once and reuse: the previous code loaded the model, encoded, then
        # loaded a second copy, paying the model-init cost twice per split.
        bert_model = bert_module.load_model()
        raw = bert_module.encode(bert_model, texts)
        pca, reduced = bert_module.fit_pca(raw)
    else:
        raw = bert_module.encode(bert_model, texts, batch_size=64)
        reduced = bert_module.transform_pca(raw, pca)

    feat_df = df[["date", "ticker"]].copy()
    for i in range(reduced.shape[1]):
        feat_df[f"bert_{i}"] = reduced[:, i]

    agg_dict = {col: "mean" for col in feat_df.columns if col.startswith("bert_")}
    agg = _groupby(feat_df, agg_dict)
    return agg, bert_model, pca


# =============================================================================
# 9. ARTICLE VOLUME
# =============================================================================

def extract_volume(df):
    """Article count + source count per (ticker, date)."""
    agg = df.groupby(["ticker", "date"], dropna=False).agg({
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
