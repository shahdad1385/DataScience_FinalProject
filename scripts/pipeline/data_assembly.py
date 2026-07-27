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

    # Drop non-numeric columns
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

        # Drop id column
        df = df.drop(columns=["id"], errors="ignore")

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

    # Drop rows with NaN targets (last day per ticker)
    target_cols = [c for c in df.columns if c.startswith("NVDA_target_") or
                   c.startswith("GOOGL_target_") or c.startswith("AVGO_target_") or
                   c.startswith("AMD_target_") or c.startswith("TSM_target_")]
    df = df.dropna(subset=target_cols)

    return df


def create_sequences(X, y_reg, y_cls, seq_len=SEQ_LEN):
    """Create sliding window sequences."""
    X_seq, y_reg_seq, y_cls_seq = [], [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i + seq_len])
        y_reg_seq.append(y_reg[i + seq_len])
        y_cls_seq.append(y_cls[i + seq_len])
    return np.array(X_seq), np.array(y_reg_seq), np.array(y_cls_seq)


def prepare_data(split="train"):
    """
    Full data preparation pipeline.
    Returns: X_seq, y_reg, y_cls, feature_names, df_with_targets
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
    feature_cols = [c for c in df.columns if not c.endswith("_target_open") and
                    not c.endswith("_target_high") and not c.endswith("_target_low") and
                    not c.endswith("_target_close") and not c.endswith("_target_direction") and
                    c not in ["id", "date", "ticker", "company", "day_name", "month_name"]]

    # Get one ticker's target columns
    reg_target_cols = [f"{t}_target_{c}" for t in TICKERS for c in ["open", "high", "low", "close"]]
    cls_target_cols = [f"{t}_target_direction" for t in TICKERS]

    # Fill NaN features with 0
    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y_reg = df[reg_target_cols].fillna(0).values.astype(np.float32)
    y_cls = df[cls_target_cols].fillna(0).values.astype(np.float32)

    print(f"  Features: {len(feature_cols)} cols")
    print(f"  Regression targets: {y_reg.shape}")
    print(f"  Classification targets: {y_cls.shape}")

    return X, y_reg, y_cls, feature_cols, df


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
