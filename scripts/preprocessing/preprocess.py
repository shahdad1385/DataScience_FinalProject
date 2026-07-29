import sys
import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db import get_engine

engine = get_engine()

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test gets the remaining 0.15


def compute_cutoff_dates(df):
    """Compute two cutoff dates for 70/15/15 split."""
    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)
    cutoff1_idx = int(n_dates * TRAIN_RATIO)
    cutoff2_idx = int(n_dates * (TRAIN_RATIO + VAL_RATIO))
    cutoff1 = pd.Timestamp(unique_dates[cutoff1_idx])
    cutoff2 = pd.Timestamp(unique_dates[cutoff2_idx])
    return cutoff1, cutoff2


def apply_split(df, cutoff1, cutoff2):
    """Temporal split into train/val/test."""
    train_df = df[df["date"] <= cutoff1].copy()
    val_df = df[(df["date"] > cutoff1) & (df["date"] <= cutoff2)].copy()
    test_df = df[df["date"] > cutoff2].copy()
    return train_df, val_df, test_df


def get_predictor_cols(train_df, target_col, numeric_cols):
    exclude = {"id", target_col, "date"}
    predictors = [c for c in numeric_cols if c not in exclude and train_df[c].notna().sum() > 100]
    return predictors


def train_lr_model(train_df, col, predictor_cols):
    train_mask = train_df[col].notna()
    X_train = train_df.loc[train_mask, predictor_cols].fillna(0)
    y_train = train_df.loc[train_mask, col]
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def fill_nulls_with_lr(train_df, val_df, test_df, col, predictor_cols, model):
    for split_df in [train_df, val_df, test_df]:
        null_mask = split_df[col].isna()
        if null_mask.any():
            X_pred = split_df.loc[null_mask, predictor_cols].fillna(0)
            split_df.loc[null_mask, col] = model.predict(X_pred)
    return train_df, val_df, test_df


def handle_nulls(train_df, val_df, test_df, numeric_cols, group_col=None):
    for col in numeric_cols:
        for df in [train_df, val_df, test_df]:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    null_cols = [c for c in numeric_cols if train_df[c].isnull().any()]

    if not null_cols:
        print("    No nulls found")
        return train_df, val_df, test_df

    print(f"    {len(null_cols)} columns with nulls in train:")
    for col in null_cols:
        n = train_df[col].isnull().sum()
        pct = n / len(train_df) * 100
        print(f"      {col}: {n} ({pct:.1f}%)")

    drop_cols = [c for c in null_cols if train_df[c].isnull().mean() > 0.50]
    if drop_cols:
        print(f"    Dropping {len(drop_cols)} columns (>50% nulls): {drop_cols}")
        train_df = train_df.drop(columns=drop_cols)
        val_df = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
        test_df = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        null_cols = [c for c in null_cols if c not in drop_cols]
        numeric_cols = [c for c in numeric_cols if c not in drop_cols]

    for col in null_cols:
        predictors = get_predictor_cols(train_df, col, numeric_cols)
        predictors = [c for c in predictors if c != col]

        if len(predictors) >= 2 and train_df[col].notna().sum() >= 50:
            model = train_lr_model(train_df, col, predictors)
            train_df, val_df, test_df = fill_nulls_with_lr(train_df, val_df, test_df, col, predictors, model)
            remaining = train_df[col].isnull().sum()
            if remaining > 0:
                fallback_predictors = [c for c in predictors if train_df[c].notna().all() and val_df[c].notna().all() and test_df[c].notna().all()]
                if len(fallback_predictors) >= 2:
                    model2 = train_lr_model(train_df, col, fallback_predictors)
                    train_df, val_df, test_df = fill_nulls_with_lr(train_df, val_df, test_df, col, fallback_predictors, model2)
                    remaining = train_df[col].isnull().sum()
                if remaining > 0:
                    for df in [train_df, val_df, test_df]:
                        if group_col and group_col in df.columns:
                            df[col] = df.groupby(group_col)[col].transform(
                                lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                            )
                        else:
                            df[col] = df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
                    print(f"    {col}: LR -> LR -> interpolation")
                else:
                    print(f"    {col}: LR ({len(predictors)} feats) -> LR ({len(fallback_predictors)} feats)")
            else:
                print(f"    {col}: LR ({len(predictors)} feats)")
        else:
            for df in [train_df, val_df, test_df]:
                if group_col and group_col in df.columns:
                    df[col] = df.groupby(group_col)[col].transform(
                        lambda x: x.interpolate(method="linear", limit_direction="both").ffill().bfill()
                    )
                else:
                    df[col] = df[col].interpolate(method="linear", limit_direction="both").ffill().bfill()
            print(f"    {col}: interpolation (insufficient predictors)")

    return train_df, val_df, test_df


def normalize_features(train_df, val_df, test_df, numeric_cols):
    """Fit scalers on train only, transform val and test."""
    robust_cols = [
        "open", "high", "low", "close", "volume",
        "volatility_7d", "volatility_14d", "volatility_30d",
        "close_ma_7", "close_ma_14", "close_ma_30",
        "volume_ma_7", "volume_ratio", "volume_change",
        "high_low_ratio", "close_open_ratio",
        "indicator_vol_7d", "indicator_ma_7", "indicator_ma_30",
    ]
    robust_cols = [c for c in robust_cols if c in train_df.columns]

    standard_cols = [
        "daily_return", "log_return", "price_range", "gap",
        "close_ma_ratio_7", "close_ma_ratio_14", "close_ma_ratio_30",
        "ewm_return_12", "ewm_return_26",
        "return_lag_1d", "return_lag_2d", "return_lag_3d",
        "return_lag_5d", "return_lag_10d",
        "indicator_return", "indicator_log_return",
        "indicator_lag_1d", "indicator_lag_5d", "indicator_lag_10d",
        "indicator_ma_ratio", "indicator_ewm_12",
        "headline_len", "daily_article_count", "daily_source_count",
        "headline_compound", "headline_pos", "headline_neg", "headline_neu",
        "summary_compound", "summary_pos", "summary_neg", "summary_neu",
        "overall_compound", "daily_avg_compound", "daily_sentiment_std",
        "ticker_daily_avg", "sentiment_momentum_3d", "sentiment_momentum_7d",
    ]
    standard_cols = [c for c in standard_cols if c in train_df.columns]

    applied = []
    if robust_cols and len(train_df) > 0:
        robust_scaler = RobustScaler()
        robust_scaler.fit(train_df[robust_cols])
        train_df[robust_cols] = robust_scaler.transform(train_df[robust_cols])
        if len(val_df) > 0:
            val_df[robust_cols] = robust_scaler.transform(val_df[robust_cols])
        if len(test_df) > 0:
            test_df[robust_cols] = robust_scaler.transform(test_df[robust_cols])
        applied.append(f"RobustScaler: {len(robust_cols)} cols")

    if standard_cols and len(train_df) > 0:
        standard_scaler = StandardScaler()
        standard_scaler.fit(train_df[standard_cols])
        train_df[standard_cols] = standard_scaler.transform(train_df[standard_cols])
        if len(val_df) > 0:
            val_df[standard_cols] = standard_scaler.transform(val_df[standard_cols])
        if len(test_df) > 0:
            test_df[standard_cols] = standard_scaler.transform(test_df[standard_cols])
        applied.append(f"StandardScaler: {len(standard_cols)} cols")

    for line in applied:
        print(f"    {line}")

    return train_df, val_df, test_df


def save_split(train_df, val_df, test_df, table_name):
    """Save train/val/test splits to DB."""
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        tbl = f"{split_name}_{table_name}"
        with engine.connect() as conn:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {tbl}"))
                conn.commit()
            except Exception:
                pass
        split_df.to_sql(tbl, engine, if_exists="replace", index=False)

        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
        print(f"    {tbl}: {count:,} rows")


def preprocess_stock_prices(cutoff1, cutoff2):
    print("\n--- stock_prices ---")
    df = pd.read_sql("SELECT * FROM stock_prices ORDER BY ticker, date", engine, parse_dates=["date"])
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, val_df, test_df = apply_split(df, cutoff1, cutoff2)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, val_df, test_df = handle_nulls(train_df, val_df, test_df, numeric_cols, group_col="ticker")

    print("  Normalizing:")
    train_df, val_df, test_df = normalize_features(train_df, val_df, test_df, numeric_cols)

    save_split(train_df, val_df, test_df, "stock_prices")


def preprocess_market_indicators(cutoff1, cutoff2):
    print("\n--- market_indicators ---")
    df = pd.read_sql("SELECT * FROM market_indicators ORDER BY indicator, date", engine, parse_dates=["date"])
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, val_df, test_df = apply_split(df, cutoff1, cutoff2)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, val_df, test_df = handle_nulls(train_df, val_df, test_df, numeric_cols, group_col="indicator")

    print("  Normalizing:")
    train_df, val_df, test_df = normalize_features(train_df, val_df, test_df, numeric_cols)

    save_split(train_df, val_df, test_df, "market_indicators")


def preprocess_news(cutoff1, cutoff2):
    print("\n--- news (merged) ---")
    try:
        df = pd.read_sql("SELECT * FROM news ORDER BY date", engine, parse_dates=["date"])
    except Exception:
        print("  Table not found, skipping.")
        return
    if df.empty:
        print("  No data, skipping.")
        return
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, val_df, test_df = apply_split(df, cutoff1, cutoff2)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, val_df, test_df = handle_nulls(train_df, val_df, test_df, numeric_cols)

    print("  Normalizing:")
    train_df, val_df, test_df = normalize_features(train_df, val_df, test_df, numeric_cols)

    save_split(train_df, val_df, test_df, "news")


def preprocess_social_sentiment(cutoff1, cutoff2):
    print("\n--- social_sentiment ---")
    try:
        df = pd.read_sql("SELECT * FROM social_sentiment ORDER BY date", engine, parse_dates=["date"])
    except Exception:
        print("  Table not found, skipping.")
        return
    if df.empty:
        print("  No data, skipping.")
        return
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, val_df, test_df = apply_split(df, cutoff1, cutoff2)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, val_df, test_df = handle_nulls(train_df, val_df, test_df, numeric_cols)

    print("  Normalizing:")
    train_df, val_df, test_df = normalize_features(train_df, val_df, test_df, numeric_cols)

    save_split(train_df, val_df, test_df, "social_sentiment")


def preprocess_news_sentiment(cutoff1, cutoff2):
    print("\n--- news_sentiment ---")
    try:
        df = pd.read_sql("SELECT * FROM news_sentiment ORDER BY date", engine, parse_dates=["date"])
    except Exception:
        print("  Table not found, skipping.")
        return
    if df.empty:
        print("  No data, skipping.")
        return
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, val_df, test_df = apply_split(df, cutoff1, cutoff2)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, val_df, test_df = handle_nulls(train_df, val_df, test_df, numeric_cols)

    print("  Normalizing:")
    train_df, val_df, test_df = normalize_features(train_df, val_df, test_df, numeric_cols)

    save_split(train_df, val_df, test_df, "news_sentiment")


def run_preprocessing():
    print("=" * 50)
    print("PREPROCESSING — ALL TABLES (70/15/15 split)")
    print("=" * 50)

    print("\nStep 1: Compute cutoff dates from news articles...")
    news_df = pd.read_sql("SELECT date FROM news ORDER BY date", engine, parse_dates=["date"])
    cutoff1, cutoff2 = compute_cutoff_dates(news_df)
    print(f"  Cutoff 1 (train/val): {cutoff1.date()}")
    print(f"  Cutoff 2 (val/test): {cutoff2.date()}")
    print(f"  (All tables will use these same cutoffs)")

    preprocess_stock_prices(cutoff1, cutoff2)
    preprocess_market_indicators(cutoff1, cutoff2)
    preprocess_news(cutoff1, cutoff2)
    preprocess_news_sentiment(cutoff1, cutoff2)
    preprocess_social_sentiment(cutoff1, cutoff2)
    preprocess_economic_events(cutoff1, cutoff2)

    print(f"\n{'=' * 50}")
    print("Preprocessing complete!")
    print(f"{'=' * 50}")


def preprocess_economic_events(cutoff1, cutoff2):
    print("\n--- economic_events ---")
    try:
        df = pd.read_sql("SELECT * FROM economic_events ORDER BY date", engine, parse_dates=["date"])
    except Exception:
        print("  Table not found, skipping.")
        return
    if df.empty:
        print("  No data, skipping.")
        return
    print(f"  Loaded: {len(df):,} rows, {df.shape[1]} cols")

    train_df, val_df, test_df = apply_split(df, cutoff1, cutoff2)
    print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    print("  Null handling:")
    train_df, val_df, test_df = handle_nulls(train_df, val_df, test_df, numeric_cols)

    print("  Normalizing:")
    train_df, val_df, test_df = normalize_features(train_df, val_df, test_df, numeric_cols)

    save_split(train_df, val_df, test_df, "economic_events")


if __name__ == "__main__":
    run_preprocessing()
