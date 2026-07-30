"""
Data Assembly — merge all features, create sequences, define targets.

Reads from DB, joins NLP + clustering + stock + market indicators,
creates 30-day sequence windows, defines OHLC + direction targets.
Filtered to 5 tickers: NVDA, GOOGL, AVGO, AMD, TSM.
"""

import os
import sys
import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db import get_engine

TICKERS = ["NVDA", "GOOGL", "AVGO", "AMD", "TSM"]
SEQ_LEN = 30


def load_table(table_name):
    engine = get_engine()
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


def load_split_features(split):
    """Load all feature tables for a split and merge on (ticker, date)."""
    features = {}

    # NLP features
    nlp_tables = [
        f"{split}_nlp_sentiment",
        f"{split}_nlp_text_stats",
        f"{split}_nlp_kw_w2v",
        f"{split}_nlp_kw_bert",
        f"{split}_nlp_topics",
        f"{split}_nlp_entities",
        f"{split}_nlp_w2v",
        f"{split}_nlp_tfidf",
        f"{split}_nlp_bert",
        f"{split}_nlp_volume",
        f"{split}_nlp_wv_sim",
    ]

    for tbl in nlp_tables:
        try:
            df = load_table(tbl)
            if not df.empty:
                features[tbl] = df
        except Exception:
            pass  # table might not exist

    # Stock features
    stock = load_table(f"{split}_stock_prices")
    features[f"{split}_stock"] = stock

    # Market indicators
    indicators = load_table(f"{split}_market_indicators")
    features[f"{split}_indicators"] = indicators

    # Sentiment
    try:
        sentiment = load_table(f"{split}_news_sentiment")
        features[f"{split}_sentiment"] = sentiment
    except Exception:
        pass

    # Social
    try:
        social = load_table(f"{split}_social_sentiment")
        features[f"{split}_social"] = social
    except Exception:
        pass

    # Economic events
    try:
        events = load_table(f"{split}_economic_events")
        features[f"{split}_events"] = events
    except Exception:
        pass

    # Clustering features
    try:
        kw_cluster = load_table(f"{split}_cluster_keyword")
        features[f"{split}_kw_cluster"] = kw_cluster
    except Exception:
        pass

    try:
        sent_cluster = load_table(f"{split}_cluster_sentiment")
        features[f"{split}_sent_cluster"] = sent_cluster
    except Exception:
        pass

    return features


def merge_all_features(features, split):
    """Merge all feature tables into one DataFrame per (ticker, date)."""
    # Start with stock prices (the base)
    stock = features.get(f"{split}_stock")
    if stock is None or stock.empty:
        return pd.DataFrame()

    # Filter to 5 tickers
    stock = stock[stock["ticker"].isin(TICKERS)].copy()
    stock["date"] = pd.to_datetime(stock["date"])

    # Drop non-numeric columns from stock
    drop_cols = [c for c in stock.columns if c in ["id", "company", "day_name", "month_name"]]
    stock = stock.drop(columns=drop_cols, errors="ignore")

    # Merge each feature table
    for key, df in features.items():
        if key == f"{split}_stock":
            continue
        if df is None or df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

        # Filter to 5 tickers if ticker column exists
        if "ticker" in df.columns:
            df = df[df["ticker"].isin(TICKERS) | df["ticker"].isna()]

        # Drop id column and non-numeric columns
        df = df.drop(columns=["id"], errors="ignore")
        non_numeric = [c for c in df.columns if c not in ["ticker", "date"] and df[c].dtype not in ["float64", "float32", "int64", "int32", "bool"]]
        df = df.drop(columns=non_numeric, errors="ignore")

        # Merge on (ticker, date) if ticker exists, else on date only
        if "ticker" in df.columns and df["ticker"].notna().any():
            # Merge with stock on (ticker, date)
            merge_cols = ["ticker", "date"]
            # Only merge non-overlapping columns
            overlap = set(stock.columns) & set(df.columns) - {"ticker", "date"}
            df_merge = df.drop(columns=overlap, errors="ignore")
            stock = stock.merge(df_merge, on=merge_cols, how="left")
        else:
            # Industry-level: merge on date only
            overlap = set(stock.columns) & set(df.columns) - {"date"}
            df_merge = df.drop(columns=overlap, errors="ignore")
            if "ticker" in df_merge.columns:
                df_merge = df_merge.drop(columns=["ticker"])
            stock = stock.merge(df_merge, on="date", how="left")

    return stock


def add_targets(df):
    """
    Add target variables for each ticker.
    Regression: next-day OHLC for each ticker.
    Classification: next-day direction (up=1, down=0) for each ticker.
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    for ticker in TICKERS:
        mask = df["ticker"] == ticker
        ticker_df = df[mask].copy()

        # Next day OHLC (shift by -1)
        for col in ["open", "high", "low", "close"]:
            df.loc[mask, f"{ticker}_target_{col}"] = ticker_df[col].shift(-1).values

        # Direction: 1 if next close > today close, else 0
        df.loc[mask, f"{ticker}_target_direction"] = (
            ticker_df["close"].shift(-1) > ticker_df["close"]
        ).astype(int).values

    # Drop rows where this ticker's own targets are NaN (last day per ticker)
    own_target_cols = []
    for ticker in TICKERS:
        for suffix in ["target_open", "target_high", "target_low", "target_close", "target_direction"]:
            own_target_cols.append(f"{ticker}_{suffix}")

    def _row_has_own_nan(row):
        t = row["ticker"]
        return any(pd.isna(row.get(f"{t}_{s}")) for s in ["target_open", "target_high", "target_low", "target_close", "target_direction"])

    mask = df.apply(_row_has_own_nan, axis=1)
    df = df[~mask].reset_index(drop=True)

    return df


def split_shared_and_per_ticker_cols(df, feature_cols):
    """Classify feature columns as shared (industry-level) or ticker-specific.

    A column is 'shared' when, within a given date, it holds the same value for
    every ticker (market indicators, industry news/NLP features). Those are kept
    once. Everything else genuinely varies per ticker and is widened.
    """
    if df.empty:
        return [], list(feature_cols)

    # Sample dates that carry more than one ticker to keep this cheap.
    counts = df.groupby("date")["ticker"].nunique()
    multi_dates = counts[counts > 1].index
    if len(multi_dates) == 0:
        return list(feature_cols), []
    sample = df[df["date"].isin(multi_dates[-200:])]

    nun = sample.groupby("date")[list(feature_cols)].nunique(dropna=False)
    # Shared if never more than one distinct value per date.
    is_shared = (nun <= 1).all(axis=0)

    shared = [c for c in feature_cols if bool(is_shared.get(c, False))]
    per_ticker = [c for c in feature_cols if c not in set(shared)]
    return shared, per_ticker


def pivot_to_joint(df, feature_cols):
    """Collapse the long (ticker, date) frame into one row per date.

    One row per date holds every ticker's features (prefixed with the ticker)
    plus the shared industry features once, and all five tickers' targets are
    real values rather than NaN-filled zeros. This is the single joint model
    formulation: one model predicting all tickers at once.
    """
    shared_cols, per_ticker_cols = split_shared_and_per_ticker_cols(df, feature_cols)

    dates = pd.Index(sorted(df["date"].unique()), name="date")
    out = pd.DataFrame(index=dates)

    # Shared / industry-level features: take the first value seen per date.
    if shared_cols:
        shared = df.groupby("date")[shared_cols].first()
        out = out.join(shared)

    # Per-ticker features: widen to {TICKER}_{col}.
    for t in TICKERS:
        sub = df[df["ticker"] == t]
        if sub.empty:
            for c in per_ticker_cols:
                out[f"{t}_{c}"] = 0.0
            continue
        sub_feats = sub.set_index("date")[per_ticker_cols]
        sub_feats = sub_feats[~sub_feats.index.duplicated(keep="last")]
        sub_feats.columns = [f"{t}_{c}" for c in per_ticker_cols]
        out = out.join(sub_feats)

    # Targets: each ticker's own targets live on that ticker's rows.
    target_suffixes = ["target_open", "target_high", "target_low", "target_close", "target_direction"]
    for t in TICKERS:
        sub = df[df["ticker"] == t]
        cols = [f"{t}_{s}" for s in target_suffixes]
        present = [c for c in cols if c in df.columns]
        if sub.empty or not present:
            for c in cols:
                out[c] = np.nan
            continue
        tgt = sub.set_index("date")[present]
        tgt = tgt[~tgt.index.duplicated(keep="last")]
        out = out.join(tgt)

    out = out.sort_index().reset_index()

    # A joint row is only usable if every ticker has a real target.
    all_target_cols = [f"{t}_{s}" for t in TICKERS for s in target_suffixes]
    have = [c for c in all_target_cols if c in out.columns]
    if have:
        out = out.dropna(subset=have).reset_index(drop=True)

    feature_out = [c for c in out.columns
                   if c != "date" and not any(s in c for s in target_suffixes)]
    return out, feature_out


def create_sequences(X, y_reg, y_cls, seq_len=SEQ_LEN, groups=None):
    """Create sliding window sequences.

    When groups is given (e.g. a ticker array aligned to X), a window is only
    emitted if the whole window plus its label come from a single group, so
    sequences never straddle a boundary between two different series.
    """
    X_seq, y_reg_seq, y_cls_seq = [], [], []
    for i in range(len(X) - seq_len):
        if groups is not None:
            window = groups[i:i + seq_len + 1]
            if len(set(np.asarray(window).tolist())) > 1:
                continue
        X_seq.append(X[i:i + seq_len])
        y_reg_seq.append(y_reg[i + seq_len])
        y_cls_seq.append(y_cls[i + seq_len])

    if not X_seq:
        n_feat = X.shape[1] if X.ndim > 1 else 0
        return (np.empty((0, seq_len, n_feat), dtype=np.float32),
                np.empty((0, y_reg.shape[1]), dtype=np.float32),
                np.empty((0, y_cls.shape[1]), dtype=np.float32))

    return np.array(X_seq), np.array(y_reg_seq), np.array(y_cls_seq)



def prepare_data(split="train", feature_cols=None):
    """
    Full data preparation pipeline.
    If feature_cols is provided (for val/test), align to those columns.
    Returns: X, y_reg, y_cls, feature_cols, df_with_targets
    """
    print(f"\n  Loading {split} features...")
    features = load_split_features(split)

    print(f"  Merging features...")
    df = merge_all_features(features, split)

    if df.empty:
        print(f"  No data for {split}")
        return None, None, None, None, None

    print(f"  Merged shape: {df.shape}")

    # Add targets
    print(f"  Adding targets...")
    df = add_targets(df)

    if df.empty:
        print(f"  No data after adding targets")
        return None, None, None, None, None

    # Separate features and targets
    exclude_cols = {"id", "date", "ticker", "company", "day_name", "month_name"}
    target_prefixes = ("_target_",)
    long_feature_cols = [c for c in df.columns if c not in exclude_cols
                    and not any(p in c for p in target_prefixes)
                    and df[c].dtype in ["float64", "float32", "int64", "int32"]]

    # Collapse to one row per date so a single model predicts all tickers
    # jointly, and every ticker's targets on a row are real values.
    print(f"  Pivoting to joint rows (one per date)...")
    df, all_feature_cols = pivot_to_joint(df, long_feature_cols)
    print(f"  Joint shape: {df.shape}")

    if df.empty:
        print(f"  No data after pivoting")
        return None, None, None, None, None

    reg_target_cols = [f"{t}_target_{c}" for t in TICKERS for c in ["open", "high", "low", "close"]]
    cls_target_cols = [f"{t}_target_direction" for t in TICKERS]

    # Align to training feature columns if provided
    if feature_cols is not None:
        # Add missing columns as 0, drop extra columns
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
        final_cols = feature_cols
    else:
        final_cols = all_feature_cols

    # Fill NaN features with 0
    X = df[final_cols].fillna(0).values.astype(np.float32)
    y_reg = df[reg_target_cols].fillna(0).values.astype(np.float32)
    y_cls = df[cls_target_cols].fillna(0).values.astype(np.float32)

    print(f"  Features: {len(final_cols)} cols")
    print(f"  Regression targets: {y_reg.shape}")
    print(f"  Classification targets: {y_cls.shape}")

    return X, y_reg, y_cls, final_cols, df



def prepare_sequences(X_train, y_reg_train, y_cls_train,
                      X_val, y_reg_val, y_cls_val,
                      X_test, y_reg_test, y_cls_test):
    """Create sequences for all splits."""
    print("  Creating sequences...")
    X_tr, y_reg_tr, y_cls_tr = create_sequences(X_train, y_reg_train, y_cls_train)
    X_v, y_reg_v, y_cls_v = create_sequences(X_val, y_reg_val, y_cls_val)
    X_te, y_reg_te, y_cls_te = create_sequences(X_test, y_reg_test, y_cls_test)

    print(f"  Train: {X_tr.shape}")
    print(f"  Val: {X_v.shape}")
    print(f"  Test: {X_te.shape}")

    return X_tr, y_reg_tr, y_cls_tr, X_v, y_reg_v, y_cls_v, X_te, y_reg_te, y_cls_te


def flatten_sequences(X_seq):
    """Flatten sequences for tabular models (XGBoost, RF, etc.)."""
    n_samples = X_seq.shape[0]
    seq_len = X_seq.shape[1]
    n_features = X_seq.shape[2]
    return X_seq.reshape(n_samples, seq_len * n_features)
