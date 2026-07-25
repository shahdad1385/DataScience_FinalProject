import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database_connection import get_engine

engine = get_engine()
BATCH_SIZE = 1000


# ---------------------------------------------------------------------------
# Utility: add columns to a table if they don't already exist
# ---------------------------------------------------------------------------
def add_columns_if_missing(table, columns):
    with engine.connect() as conn:
        existing = {row[1] for row in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}
        for col_name, col_type in columns:
            if col_name not in existing:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}"))
                print(f"    + {col_name} ({col_type})")
        conn.commit()


# ---------------------------------------------------------------------------
# Utility: batch UPDATE rows by id
# ---------------------------------------------------------------------------
def batch_update(table, df, id_col="id", columns=None):
    if columns is None:
        cols = [c for c in df.columns if c != id_col]
    else:
        cols = list(columns)

    with engine.connect() as conn:
        for start in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[start:start + BATCH_SIZE]
            for _, row in chunk.iterrows():
                sets = []
                vals = {id_col: int(row[id_col])}
                for c in cols:
                    v = row[c]
                    if pd.isna(v):
                        continue
                    sets.append(f"{c} = :{c}")
                    vals[c] = float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
                if sets:
                    conn.execute(text(f"UPDATE {table} SET {', '.join(sets)} WHERE {id_col} = :{id_col}"), vals)
        conn.commit()


# ---------------------------------------------------------------------------
# Stock price features (per ticker) + calendar + OHE + holidays
# ---------------------------------------------------------------------------
def add_stock_features():
    print("📈 Adding features to stock_prices...")

    features = [
        ("daily_return", "REAL"), ("log_return", "REAL"),
        ("price_range", "REAL"), ("gap", "REAL"),
        ("volume_change", "REAL"), ("volume_ma_7", "REAL"), ("volume_ratio", "REAL"),
        ("return_lag_1d", "REAL"), ("return_lag_2d", "REAL"), ("return_lag_3d", "REAL"),
        ("return_lag_5d", "REAL"), ("return_lag_10d", "REAL"),
        ("volatility_7d", "REAL"), ("volatility_14d", "REAL"), ("volatility_30d", "REAL"),
        ("close_ma_7", "REAL"), ("close_ma_14", "REAL"), ("close_ma_30", "REAL"),
        ("close_ma_ratio_7", "REAL"), ("close_ma_ratio_14", "REAL"), ("close_ma_ratio_30", "REAL"),
        ("ewm_return_12", "REAL"), ("ewm_return_26", "REAL"),
        ("high_low_ratio", "REAL"), ("close_open_ratio", "REAL"),
        ("year", "INTEGER"), ("quarter", "INTEGER"), ("month", "INTEGER"),
        ("week_of_year", "INTEGER"), ("day_of_week", "INTEGER"),
        ("is_weekend", "INTEGER"), ("is_month_start", "INTEGER"),
        ("is_month_end", "INTEGER"), ("is_quarter_end", "INTEGER"),
        ("day_name", "TEXT"), ("month_name", "TEXT"), ("week_number", "INTEGER"),
        ("is_holiday", "INTEGER"),
        ("day_name_monday", "INTEGER"), ("day_name_tuesday", "INTEGER"),
        ("day_name_wednesday", "INTEGER"), ("day_name_thursday", "INTEGER"),
        ("day_name_friday", "INTEGER"),
        ("month_name_january", "INTEGER"), ("month_name_february", "INTEGER"),
        ("month_name_march", "INTEGER"), ("month_name_april", "INTEGER"),
        ("month_name_may", "INTEGER"), ("month_name_june", "INTEGER"),
        ("month_name_july", "INTEGER"), ("month_name_august", "INTEGER"),
        ("month_name_september", "INTEGER"), ("month_name_october", "INTEGER"),
        ("month_name_november", "INTEGER"), ("month_name_december", "INTEGER"),
        ("quarter_1", "INTEGER"), ("quarter_2", "INTEGER"),
        ("quarter_3", "INTEGER"), ("quarter_4", "INTEGER"),
        ("day_of_month", "INTEGER"),
        ("day_of_week_sin", "REAL"), ("day_of_week_cos", "REAL"),
        ("month_sin", "REAL"), ("month_cos", "REAL"),
        ("week_of_year_sin", "REAL"), ("week_of_year_cos", "REAL"),
        ("day_of_month_sin", "REAL"), ("day_of_month_cos", "REAL"),
    ]
    add_columns_if_missing("stock_prices", features)

    df = pd.read_sql(
        "SELECT id, ticker, date, open, high, low, close, volume FROM stock_prices ORDER BY ticker, date",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])

    result = []
    for ticker, group in df.groupby("ticker"):
        g = group.sort_values("date").copy()
        idx = g.index

        g["daily_return"] = g["close"].pct_change()
        g["log_return"] = np.log(g["close"] / g["close"].shift(1))
        g["price_range"] = (g["high"] - g["low"]) / g["close"]
        g["gap"] = (g["open"] - g["close"].shift(1)) / g["close"].shift(1)

        g["volume_change"] = g["volume"].pct_change()
        g["volume_ma_7"] = g["volume"].shift(1).rolling(7).mean()
        g["volume_ratio"] = g["volume"] / (g["volume_ma_7"] + 1)

        for lag in [1, 2, 3, 5, 10]:
            g[f"return_lag_{lag}d"] = g["daily_return"].shift(lag)

        for w in [7, 14, 30]:
            shifted = g["daily_return"].shift(1)
            g[f"volatility_{w}d"] = shifted.rolling(w).std()
            g[f"close_ma_{w}"] = g["close"].shift(1).rolling(w).mean()
            g[f"close_ma_ratio_{w}"] = g["close"] / (g[f"close_ma_{w}"] + 1e-8)

        shifted = g["daily_return"].shift(1)
        g["ewm_return_12"] = shifted.ewm(span=12).mean()
        g["ewm_return_26"] = shifted.ewm(span=26).mean()

        g["high_low_ratio"] = g["high"] / (g["low"] + 1e-8)
        g["close_open_ratio"] = g["close"] / (g["open"] + 1e-8)

        result.append(g)

    updates = pd.concat(result, ignore_index=True)

    d = updates["date"]
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday',
                 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                   5: 'May', 6: 'June', 7: 'July', 8: 'August',
                   9: 'September', 10: 'October', 11: 'November', 12: 'December'}

    updates["year"] = d.dt.year
    updates["quarter"] = d.dt.quarter
    updates["month"] = d.dt.month
    updates["week_of_year"] = d.dt.isocalendar().week.astype(int)
    updates["day_of_week"] = d.dt.dayofweek
    updates["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)
    updates["is_month_start"] = d.dt.is_month_start.astype(int)
    updates["is_month_end"] = d.dt.is_month_end.astype(int)
    updates["is_quarter_end"] = d.dt.is_quarter_end.astype(int)
    updates["day_name"] = d.dt.dayofweek.map(day_names)
    updates["month_name"] = d.dt.month.map(month_names)
    updates["week_number"] = d.dt.isocalendar().week.astype(int)
    updates["day_of_month"] = d.dt.day

    updates["day_of_week_sin"] = np.sin(2 * np.pi * updates["day_of_week"] / 7)
    updates["day_of_week_cos"] = np.cos(2 * np.pi * updates["day_of_week"] / 7)
    updates["month_sin"] = np.sin(2 * np.pi * updates["month"] / 12)
    updates["month_cos"] = np.cos(2 * np.pi * updates["month"] / 12)
    updates["week_of_year_sin"] = np.sin(2 * np.pi * updates["week_of_year"] / 52)
    updates["week_of_year_cos"] = np.cos(2 * np.pi * updates["week_of_year"] / 52)
    updates["day_of_month_sin"] = np.sin(2 * np.pi * updates["day_of_month"] / 31)
    updates["day_of_month_cos"] = np.cos(2 * np.pi * updates["day_of_month"] / 31)

    from sklearn.preprocessing import OneHotEncoder as OHE
    cat_df = pd.DataFrame({
        "day_name": updates["day_name"],
        "month_name": updates["month_name"],
        "quarter": updates["quarter"],
    })
    enc = OHE(sparse_output=False, handle_unknown="ignore")
    enc.fit(cat_df.fillna("unknown"))
    encoded = enc.transform(cat_df.fillna("unknown"))
    for i, col in enumerate(enc.get_feature_names_out()):
        updates[col.replace(" ", "_").lower()] = encoded[:, i].astype(int)

    import holidays as hol
    us_holidays = hol.US(years=range(2018, 2027))
    updates["is_holiday"] = updates["date"].apply(
        lambda x: 1 if pd.to_datetime(x).date() in us_holidays else 0
    )

    stock_cols = ["daily_return", "log_return", "price_range", "gap",
                  "volume_change", "volume_ma_7", "volume_ratio",
                  "return_lag_1d", "return_lag_2d", "return_lag_3d", "return_lag_5d", "return_lag_10d",
                  "volatility_7d", "volatility_14d", "volatility_30d",
                  "close_ma_7", "close_ma_14", "close_ma_30",
                  "close_ma_ratio_7", "close_ma_ratio_14", "close_ma_ratio_30",
                  "ewm_return_12", "ewm_return_26", "high_low_ratio", "close_open_ratio",
                  "year", "quarter", "month", "week_of_year", "day_of_week", "day_of_month",
                  "is_weekend", "is_month_start", "is_month_end", "is_quarter_end",
                  "day_name", "month_name", "week_number", "is_holiday",
                  "day_of_week_sin", "day_of_week_cos",
                  "month_sin", "month_cos",
                  "week_of_year_sin", "week_of_year_cos",
                  "day_of_month_sin", "day_of_month_cos"]
    ohe_cols = [c for c in updates.columns if c.startswith("day_name_") or c.startswith("month_name_") or c.startswith("quarter_")]
    update_cols = stock_cols + ohe_cols
    updates = updates.where(pd.notnull(updates), None)
    batch_update("stock_prices", updates, columns=update_cols)

    print(f"    ✅ Updated {len(updates):,} rows")


# ---------------------------------------------------------------------------
# Market indicator features (per indicator) — writes back to DB
# ---------------------------------------------------------------------------
def add_indicator_features():
    print("📊 Adding features to market_indicators...")

    features = [
        ("indicator_return", "REAL"), ("indicator_log_return", "REAL"),
        ("indicator_lag_1d", "REAL"), ("indicator_lag_5d", "REAL"), ("indicator_lag_10d", "REAL"),
        ("indicator_vol_7d", "REAL"),
        ("indicator_ma_7", "REAL"), ("indicator_ma_30", "REAL"), ("indicator_ma_ratio", "REAL"),
        ("indicator_ewm_12", "REAL"),
    ]
    add_columns_if_missing("market_indicators", features)

    df = pd.read_sql(
        "SELECT id, date, close, indicator FROM market_indicators ORDER BY indicator, date",
        engine,
    )
    df["date"] = pd.to_datetime(df["date"])

    updates = pd.DataFrame(index=df.index)
    updates["id"] = df["id"]

    updates["indicator_return"] = df.groupby("indicator")["close"].pct_change()
    updates["indicator_log_return"] = df.groupby("indicator").apply(
        lambda g: np.log(g["close"] / g["close"].shift(1)), include_groups=False
    ).reset_index(level=0, drop=True)

    for lag in [1, 5, 10]:
        updates[f"indicator_lag_{lag}d"] = df.groupby("indicator")["close"].shift(lag)

    shifted = updates.groupby(df["indicator"])["indicator_return"].shift(1)
    updates["indicator_vol_7d"] = shifted.rolling(7).mean()

    updates["indicator_ma_7"] = df.groupby("indicator")["close"].shift(1).rolling(7).mean()
    updates["indicator_ma_30"] = df.groupby("indicator")["close"].shift(1).rolling(30).mean()
    updates["indicator_ma_ratio"] = df["close"] / (updates["indicator_ma_7"] + 1e-8)
    updates["indicator_ewm_12"] = shifted.ewm(span=12).mean()

    updates = updates.where(pd.notnull(updates), None)
    batch_update("market_indicators", updates)

    print(f"    ✅ Updated {len(updates):,} rows")


# ---------------------------------------------------------------------------
# News features — per-row + daily aggregates, writes back to DB
# ---------------------------------------------------------------------------
def add_news_features():
    print("📰 Adding features to news_articles...")

    features = [
        ("headline_len", "INTEGER"),
        ("daily_article_count", "INTEGER"),
        ("daily_source_count", "INTEGER"),
    ]
    add_columns_if_missing("news_articles", features)

    df = pd.read_sql("SELECT id, date, headline, source FROM news_articles", engine)
    df["date"] = pd.to_datetime(df["date"])
    df["headline_len"] = df["headline"].str.len()

    daily_counts = df.groupby("date").size().rename("daily_article_count")
    daily_sources = df.groupby("date")["source"].nunique().rename("daily_source_count")
    df = df.join(daily_counts, on="date").join(daily_sources, on="date")

    updates = df[["id", "headline_len", "daily_article_count", "daily_source_count"]].copy()
    updates = updates.where(pd.notnull(updates), None)
    batch_update("news_articles", updates)

    print(f"    ✅ Updated {len(df):,} rows")


# ---------------------------------------------------------------------------
# HF news features — per-row + daily counts, writes back to DB
# ---------------------------------------------------------------------------
def add_hf_features():
    print("📚 Adding features to hf_news...")

    features = [
        ("headline_len", "INTEGER"),
        ("daily_article_count", "INTEGER"),
    ]
    add_columns_if_missing("hf_news", features)

    df = pd.read_sql("SELECT id, date, headline FROM hf_news", engine)
    df["date"] = pd.to_datetime(df["date"])
    df["headline_len"] = df["headline"].str.len()

    daily_counts = df.groupby("date").size().rename("daily_article_count")
    df = df.join(daily_counts, on="date")

    updates = df[["id", "headline_len", "daily_article_count"]].copy()
    updates = updates.where(pd.notnull(updates), None)
    batch_update("hf_news", updates)

    print(f"    ✅ Updated {len(df):,} rows")


# ---------------------------------------------------------------------------
# Sentiment features — rolling avg, momentum, daily aggregates
# ---------------------------------------------------------------------------
def add_sentiment_features():
    print("💬 Adding features to news_sentiment...")

    features = [
        ("daily_avg_compound", "REAL"),
        ("daily_sentiment_std", "REAL"),
        ("daily_article_count", "INTEGER"),
        ("ticker_daily_avg", "REAL"),
        ("sentiment_momentum_3d", "REAL"),
        ("sentiment_momentum_7d", "REAL"),
    ]
    add_columns_if_missing("news_sentiment", features)

    df = pd.read_sql("SELECT id, date, ticker, overall_compound FROM news_sentiment", engine)
    if df.empty:
        print("    No sentiment data found, skipping.")
        return
    df["date"] = pd.to_datetime(df["date"])

    # Daily aggregates (across all tickers)
    daily_stats = df.groupby("date").agg(
        daily_avg_compound=("overall_compound", "mean"),
        daily_sentiment_std=("overall_compound", "std"),
        daily_article_count=("overall_compound", "count"),
    ).reset_index()
    df = df.merge(daily_stats, on="date", how="left")

    # Per-ticker daily average
    ticker_daily = df.groupby(["date", "ticker"])["overall_compound"].mean().rename("ticker_daily_avg").reset_index()
    df = df.merge(ticker_daily, on=["date", "ticker"], how="left")

    # Sentiment momentum: rolling mean change over 3 and 7 days
    df.sort_values(["ticker", "date"], inplace=True)
    for ticker, group in df.groupby("ticker"):
        if pd.isna(ticker):
            continue
        g_idx = group.index
        rolled_3 = group["overall_compound"].rolling(3, min_periods=1).mean()
        rolled_7 = group["overall_compound"].rolling(7, min_periods=1).mean()
        df.loc[g_idx, "sentiment_momentum_3d"] = rolled_3.diff(1)
        df.loc[g_idx, "sentiment_momentum_7d"] = rolled_7.diff(1)

    updates = df[["id", "daily_avg_compound", "daily_sentiment_std", "daily_article_count",
                   "ticker_daily_avg", "sentiment_momentum_3d", "sentiment_momentum_7d"]].copy()
    updates = updates.where(pd.notnull(updates), None)
    batch_update("news_sentiment", updates)

    print(f"    Updated {len(df):,} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Adding features to database tables...\n")
    add_stock_features()
    add_indicator_features()
    add_news_features()
    add_hf_features()
    add_sentiment_features()
    print(f"\n{'=' * 50}")
    print("All features added to database!")
    print(f"{'=' * 50}")
