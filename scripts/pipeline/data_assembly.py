"""
Data Assembly — merge all features, create sequences, define targets.

Reads from DB, joins NLP + clustering + stock + market indicators,
creates 30-day sequence windows, defines OHLC + direction targets.
Filtered to 5 tickers: NVDA, GOOGL, AVGO, AMD, TSM.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db import get_engine
from scripts.preprocessing.preprocess import inverse_transform_column
from scripts.nlp.features import INDUSTRY_TICKER

# Single-target formulation.
#
# The joint 5-ticker model had to fit ~1300 features on 1474 training rows while
# predicting 25 outputs, and it never beat a no-change baseline. Predicting one
# ticker cuts the output count from 25 to 5 and lets every feature be about NVDA
# or about NVDA's context, rather than being split across five names.
#
# Peer companies are NOT dropped: their prices and news are folded in as sector
# context (aggregates), which is where most of the text signal actually lives.
TARGET_TICKER = "NVDA"

# Kept as a list because the rest of the pipeline derives n_tickers, target
# column names and output widths from it. With one entry the regression head is
# 4 outputs (OHLC returns) and the classification head is 1 (direction).
TICKERS = [TARGET_TICKER]

# Peers used only for sector aggregates, never as prediction targets.
PEER_TICKERS = [
    "GOOGL", "AVGO", "AMD", "TSM", "INTC", "MU", "QCOM", "AMAT", "ASML",
    "KLAC", "LRCX", "MRVL", "NXPI", "ON", "META", "SMCI", "VRT", "ARM", "MOD",
]

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


KW_SELECTION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "kw_selection.pkl"
)
# Keyword columns kept per source table. The raw tables carry ~500 each, and the
# news tiering (nvda / industry / peer) would triple that to ~3,000 columns
# against ~1,500 training rows — the same p >> n problem that made the previous
# model unable to beat a baseline. The most frequent keywords carry nearly all
# the signal; the long tail is mostly zeros. Cut hard: at 40-per-table the joined
# matrix still reached ~695 columns, which is why no classifier could beat the
# no-change baseline. 15 keeps the head of the distribution only.
MAX_KEYWORD_COLS = 15


def _load_kw_selection():
    if not os.path.exists(KW_SELECTION_PATH):
        return {}
    with open(KW_SELECTION_PATH, "rb") as f:
        return pickle.load(f)


def _save_kw_selection(sel):
    os.makedirs(os.path.dirname(KW_SELECTION_PATH), exist_ok=True)
    with open(KW_SELECTION_PATH, "wb") as f:
        pickle.dump(sel, f)


def reduce_keyword_columns(df, table_key, split, selection):
    """Keep only the most frequent keyword columns, chosen on train.

    The selection is persisted and reused for val/test so all splits present the
    same keyword columns, and it also adds two aggregate columns (total keyword
    volume and distinct-keyword count) so information from the dropped tail is
    not lost entirely.
    """
    kw_cols = [c for c in df.columns if c.startswith("kw_")]
    if not kw_cols:
        return df, selection

    # Aggregates computed over ALL keyword columns, before any are dropped.
    totals = df[kw_cols].sum(axis=1)
    distinct = (df[kw_cols] > 0).sum(axis=1)

    if split == "train" and table_key not in selection:
        freq = df[kw_cols].sum().sort_values(ascending=False)
        selection[table_key] = list(freq.head(MAX_KEYWORD_COLS).index)

    keep = [c for c in selection.get(table_key, kw_cols[:MAX_KEYWORD_COLS])
            if c in df.columns]
    df = df.drop(columns=[c for c in kw_cols if c not in set(keep)], errors="ignore")
    df["kwagg_total"] = totals
    df["kwagg_distinct"] = distinct
    return df, selection


def _numeric_only(df, keep=("ticker", "date")):
    """Drop id/non-numeric columns, keeping the join keys."""
    df = df.drop(columns=["id"], errors="ignore")
    drop = [c for c in df.columns
            if c not in keep
            and df[c].dtype not in ["float64", "float32", "int64", "int32", "bool"]]
    return df.drop(columns=drop, errors="ignore")


def _dedupe_by_date(df, prefix=None):
    """Collapse to one row per date, optionally prefixing the value columns."""
    if df is None or df.empty:
        return None
    df = df.drop(columns=["ticker"], errors="ignore")
    df = df.groupby("date", as_index=False).mean(numeric_only=True)
    if prefix:
        df = df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != "date"})
    return df


def build_market_indicator_features(df):
    """One row per date, one column per (indicator, metric).

    market_indicators holds macro series — VIX, SP500, NASDAQ, SOX, Gold,
    Crude_Oil, treasuries — identified by `indicator`, with `ticker` holding the
    market symbol (^VIX, GC=F, SMH...). The old merge filtered these rows with
    `ticker.isin(TICKERS)`, which matched none of them, so all 24,776 rows were
    dropped and every macro column arrived as NaN. Pivoting on `indicator`
    instead keeps them, which is exactly the sector/market context a
    single-ticker model needs.
    """
    if df is None or df.empty or "indicator" not in df.columns:
        return None

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Keep only STATIONARY metrics. Absolute levels (open/high/low/close/volume,
    # ma_7, ma_30, the lags) drift far outside their training range on a macro
    # series just as the price target did, so they would reintroduce exactly the
    # extrapolation problem that made the OHLC-level model unusable. Ratios,
    # returns and volatilities are comparable across splits.
    #
    # This also cuts 15 metrics x 16 indicators = 240 columns down to 80, which
    # matters when only ~1,500 training rows are available.
    stationary = ["indicator_return", "indicator_log_return", "indicator_vol_7d",
                  "indicator_ma_ratio", "indicator_ewm_12"]
    value_cols = [c for c in stationary if c in df.columns
                  and df[c].dtype in ["float64", "float32", "int64", "int32", "bool"]]
    if not value_cols:
        return None

    df["indicator"] = (df["indicator"].astype(str)
                       .str.strip().str.lower()
                       .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))

    wide = df.pivot_table(index="date", columns="indicator", values=value_cols,
                          aggfunc="mean")
    wide.columns = [f"ind_{ind}_{metric}" for metric, ind in wide.columns]
    return wide.reset_index()


def build_peer_stock_features(stock_all, target=TARGET_TICKER):
    """Sector aggregates from the peer companies, one row per date.

    Peers are not prediction targets, but semiconductor names move together, so
    their cross-section is informative: the sector's mean/median return, its
    dispersion, the fraction of names up on the day (breadth), and the target's
    return minus the sector mean (relative strength). This is how the other 19
    tickers keep contributing after the switch to a single target.
    """
    if stock_all is None or stock_all.empty:
        return None

    df = stock_all.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "daily_return" not in df.columns:
        return None

    peers = df[df["ticker"].isin(PEER_TICKERS)]
    if peers.empty:
        return None

    grp = peers.groupby("date")["daily_return"]
    out = pd.DataFrame({
        "peer_ret_mean": grp.mean(),
        "peer_ret_median": grp.median(),
        "peer_ret_std": grp.std(),
        "peer_ret_min": grp.min(),
        "peer_ret_max": grp.max(),
        # NOTE: daily_return arrives StandardScaler-scaled, so ">0" means "above
        # the train-period mean return", not literally positive. Still a valid
        # breadth signal, but named for what it actually measures.
        "peer_breadth_above_mean": grp.apply(lambda s: (s > 0).mean()),
        "peer_count": grp.count(),
    })

    if "volume" in peers.columns:
        out["peer_volume_mean"] = peers.groupby("date")["volume"].mean()

    tgt = df[df["ticker"] == target].set_index("date")["daily_return"]
    out["target_rel_strength"] = tgt.reindex(out.index) - out["peer_ret_mean"]

    return out.reset_index()


def merge_all_features(features, split):
    """Assemble one row per date for the single target ticker.

    Layout:
      - base: the target ticker's own price features
      - macro indicators: one column per (indicator, metric)
      - peer aggregates: sector context from the other companies
      - NLP/cluster tables: split into NVDA-specific, industry-level and
        peer-level tiers, each merged on date
    """
    stock_all = features.get(f"{split}_stock")
    if stock_all is None or stock_all.empty:
        return pd.DataFrame()

    stock_all = stock_all.copy()
    stock_all["date"] = pd.to_datetime(stock_all["date"])

    base = stock_all[stock_all["ticker"] == TARGET_TICKER].copy()
    if base.empty:
        print(f"  No {TARGET_TICKER} rows in {split}_stock_prices")
        return pd.DataFrame()

    base = base.drop(columns=["company", "day_name", "month_name"], errors="ignore")
    base = _numeric_only(base)
    base = base.groupby("date", as_index=False).last()
    base["ticker"] = TARGET_TICKER

    # --- macro indicators (previously discarded entirely) ---
    ind = build_market_indicator_features(features.get(f"{split}_indicators"))
    if ind is not None:
        base = base.merge(ind, on="date", how="left")

    # --- peer sector aggregates ---
    peer = build_peer_stock_features(stock_all)
    if peer is not None:
        base = base.merge(peer, on="date", how="left")

    # --- NLP / sentiment / cluster tables, split into tiers ---
    #
    # Each ticker-aware table becomes three date-indexed blocks:
    #   nvda_*     : articles tagged NVDA (sparse in train — see note below)
    #   indnews_*  : industry-wide articles, the ticker sentinel from load_news
    #   peernews_* : all peer companies, averaged into one sector view
    # This is how "feed all the news to it" is realised while still predicting a
    # single ticker: peer and industry coverage becomes context rather than
    # separate prediction targets.
    kw_selection = _load_kw_selection()
    skip = {f"{split}_stock", f"{split}_indicators"}
    for key, df in features.items():
        if key in skip or df is None or df.empty:
            continue

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = _numeric_only(df)

        # Cap keyword width before tiering, or ~500 columns become ~1,500.
        df, kw_selection = reduce_keyword_columns(df, key.replace(f"{split}_", ""),
                                                  split, kw_selection)

        if "ticker" not in df.columns:
            merged = _dedupe_by_date(df, prefix="mkt_")
            if merged is not None:
                base = _safe_merge(base, merged)
            continue

        ticker = df["ticker"].astype(str)

        nvda_part = _dedupe_by_date(df[ticker == TARGET_TICKER], prefix="nvda_")
        ind_mask = ticker.isin([INDUSTRY_TICKER, "None", "nan", ""]) | df["ticker"].isna()
        industry_part = _dedupe_by_date(df[ind_mask], prefix="indnews_")
        peer_part = _dedupe_by_date(
            df[~ind_mask & (ticker != TARGET_TICKER)], prefix="peernews_")

        for part in (nvda_part, industry_part, peer_part):
            if part is not None:
                base = _safe_merge(base, part)

    if split == "train":
        _save_kw_selection(kw_selection)

    # News coverage is not uniform across dates: a date with no articles yields
    # NaN for every news column. Zero is the correct value for counts, but for
    # sentiment "no news" is not "neutral sentiment", so the model is told which
    # tier was actually observed via explicit coverage flags.
    for prefix in ("nvda_", "indnews_", "peernews_"):
        cols = [c for c in base.columns if c.startswith(prefix)]
        if cols:
            base[f"has_{prefix.rstrip('_')}_news"] = base[cols].notna().any(axis=1).astype(int)

    return base


def _safe_merge(left, right, on="date"):
    """Left-merge, dropping columns that would collide."""
    overlap = (set(left.columns) & set(right.columns)) - {on}
    if overlap:
        right = right.drop(columns=list(overlap), errors="ignore")
    return left.merge(right, on=on, how="left")


OHLC_COLS = ["open", "high", "low", "close"]


def _load_raw_ohlc(df):
    """Real (unscaled) OHLC prices for each (ticker, date) row of `df`.

    Reads the source `stock_prices` table, which is never written to by the
    pipeline. Returns a frame aligned to df.index; rows with no match are NaN so
    the caller drops them instead of computing a bogus return.
    """
    engine = get_engine()
    cols = ", ".join(OHLC_COLS)
    try:
        raw = pd.read_sql(
            f"SELECT ticker, date, {cols} FROM stock_prices", engine
        )
    except Exception as e:
        raise RuntimeError(
            "Cannot read raw prices from stock_prices; return targets would be "
            f"computed from scaled values and be meaningless. Original error: {e}"
        )

    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[~raw.duplicated(subset=["ticker", "date"], keep="last")]

    keys = pd.DataFrame({
        "ticker": df["ticker"].astype(str).values,
        "date": pd.to_datetime(df["date"]).values,
    })
    merged = keys.merge(raw, on=["ticker", "date"], how="left")
    merged.index = df.index
    return merged[OHLC_COLS]


TARGET_SUFFIXES = [f"target_ret_{c}" for c in OHLC_COLS] + ["target_direction"]
# Non-feature helper column: today's real close, needed to turn a predicted
# return back into a dollar price.
REF_CLOSE_SUFFIX = "refclose"

# Days where |next-day return| is below this carry no directional information:
# the move is within bid/ask noise and slippage, so labeling them UP/DOWN forces
# the classifier to fit coin-flips. In the last run the best classifier still
# needed every day labeled UP just to reach 52.8%. Labeling only meaningful
# moves and dropping the flat days from the classification target makes the
# UP/DOWN classes real; regression targets are deliberately untouched so price
# reconstruction keeps its full density.
DIRECTION_DEAD_ZONE = 0.003


def add_targets(df):
    """Add next-day RETURN targets for each ticker.

    Why returns rather than absolute price levels: the price columns are scaled
    with a RobustScaler fitted on train only, and these are non-stationary
    series in a strong uptrend. Test-set scaled closes reach 5-7 while train
    never exceeds ~1.6 (NVDA's test max is ~66x its train max), so a model asked
    to output price levels must extrapolate far outside anything it saw while
    training. That is what produced val losses stuck near 0.7 with train at 0.09,
    and R2 around -9 on test.

    Returns are stationary and comparable across both splits and tickers, so the
    same model output range is valid everywhere. Dollar prices are recovered
    downstream as ref_close * (1 + predicted_return).

    Regression targets: (next_day_{O,H,L,C} / today_close) - 1
    Classification target: 1 if next close > today close else 0
    """
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Returns are computed from the ORIGINAL unscaled prices, read straight from
    # the source stock_prices table (read-only; nothing there is modified).
    #
    # The previous version inverse-transformed the scaled split columns instead.
    # That silently depends on the scaler artifacts existing: when they are
    # absent, inverse_transform_column returns its input unchanged, so scaled
    # values (which are mostly negative) flow into the return formula. The
    # `close > 0` guard below then discarded 1,478 of 1,506 training rows,
    # leaving 27. Reading the raw prices removes the dependency entirely and is
    # exact rather than a reconstruction.
    real_df = _load_raw_ohlc(df)

    for ticker in TICKERS:
        mask = (df["ticker"] == ticker).values
        if not mask.any():
            continue
        today_close = pd.Series(real_df.loc[mask, "close"].values)
        # Guard against non-positive prices before dividing.
        denom = today_close.where(today_close > 0)

        for col in OHLC_COLS:
            next_price = pd.Series(real_df.loc[mask, col].values).shift(-1)
            df.loc[mask, f"{ticker}_target_ret_{col}"] = (next_price / denom - 1.0).values

        next_close = today_close.shift(-1)
        direction = (next_close > today_close)
        # Keep NaN where the next day is unknown so the row is dropped below,
        # rather than silently becoming a "down" label.
        direction = direction.where(next_close.notna() & denom.notna())

        # Dead-zone: near-flat days carry no directional information. Days whose
        # |next-day close return| is below DIRECTION_DEAD_ZONE are masked to NaN
        # for the direction label. The downstream prepare_data treats a missing
        # direction as "exclude this row", so these noisy days are removed from
        # training and evaluation entirely instead of forcing the classifier to
        # fit coin-flips.
        ret = (next_close / denom - 1.0).abs()
        direction = direction.where(ret >= DIRECTION_DEAD_ZONE)
        df.loc[mask, f"{ticker}_target_direction"] = direction.values

        # Today's real close, used to reconstruct dollar prices from returns.
        df.loc[mask, f"{ticker}_{REF_CLOSE_SUFFIX}"] = today_close.values

    # Drop rows whose own ticker's targets are unknown (last day per ticker).
    own_cols = [f"{t}_{s}" for t in TICKERS for s in TARGET_SUFFIXES]
    have = [c for c in own_cols if c in df.columns]
    if have:
        ticker_arr = df["ticker"].values
        bad = np.zeros(len(df), dtype=bool)
        for t in TICKERS:
            m = (ticker_arr == t)
            if not m.any():
                continue
            cols = [f"{t}_{s}" for s in TARGET_SUFFIXES if f"{t}_{s}" in df.columns]
            if cols:
                bad |= m & df[cols].isna().any(axis=1).values
        df = df[~bad].reset_index(drop=True)

    return df


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "joint_schema.pkl"
)


def save_joint_schema(shared_cols, per_ticker_cols):
    """Persist the train-derived shared/per-ticker split."""
    os.makedirs(os.path.dirname(SCHEMA_PATH), exist_ok=True)
    with open(SCHEMA_PATH, "wb") as f:
        pickle.dump({"shared": list(shared_cols), "per_ticker": list(per_ticker_cols)}, f)


def load_joint_schema():
    """Load the train-derived schema, or None when absent."""
    if not os.path.exists(SCHEMA_PATH):
        return None
    with open(SCHEMA_PATH, "rb") as f:
        d = pickle.load(f)
    return d.get("shared"), d.get("per_ticker")


def split_shared_and_per_ticker_cols(df, feature_cols, schema=None):
    """Classify feature columns as shared (industry-level) or ticker-specific.

    A column is 'shared' when, within a given date, it holds the same value for
    every ticker (market indicators, industry news/NLP features). Those are kept
    once. Everything else genuinely varies per ticker and is widened.

    When `schema` is given (always, for val/test) the train-time classification
    is reused. Re-deriving it per split was a correctness bug: a column that
    happens to be constant within a smaller split looks 'shared' there but
    'per-ticker' in train, so the same feature landed in different column names
    across splits. That produced 1351/666/5612-column frames for train/val/test
    and forced prepare_data to invent ~1000 all-zero columns for val.
    """
    if schema is not None:
        train_shared, train_per_ticker = schema
        present = set(feature_cols)
        shared = [c for c in train_shared if c in present]
        per_ticker = [c for c in train_per_ticker if c in present]
        return shared, per_ticker

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


def pivot_to_joint(df, feature_cols, schema=None):
    """Normalise the assembled frame to exactly one row per date.

    merge_all_features already emits one row per date for the single target
    ticker, so the old shared/per-ticker widening is no longer needed: there is
    nothing to widen when there is one ticker. This now just enforces
    one-row-per-date, drops rows whose targets are unknown, and reports which
    columns are features.

    `schema`, when provided, pins the feature column list to the training set so
    val and test present identical columns in identical order.
    """
    target_suffixes = list(TARGET_SUFFIXES) + [REF_CLOSE_SUFFIX]

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date")
    # Defensive: a duplicate date would silently double-count a day.
    out = out[~out["date"].duplicated(keep="last")].reset_index(drop=True)

    # A row is usable only when the target's own targets are present.
    target_cols = [f"{TARGET_TICKER}_{s}" for s in target_suffixes]
    have = [c for c in target_cols if c in out.columns]
    if have:
        out = out.dropna(subset=have).reset_index(drop=True)

    excluded = tuple(target_suffixes)
    present_features = [c for c in out.columns
                        if c not in ("date", "ticker")
                        and not any(c.endswith(s) for s in excluded)]

    if schema is not None:
        # schema is (shared, per_ticker) for backwards compatibility; with a
        # single ticker everything is "shared", so just take the union in order.
        pinned = list(schema[0]) + [c for c in schema[1] if c not in set(schema[0])]
        feature_out = [c for c in pinned if c in set(present_features)]
    else:
        feature_out = present_features

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



FEATURE_SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "feature_scaler.pkl"
)
TARGET_SCALER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "target_scaler.pkl"
)


def fit_target_scaler(y_reg):
    """Standardise the return targets (train only), and persist the scaler.

    Raw daily returns are ~0.02, so a Huber/MSE term on them lands around 1e-4
    while the direction BCE term sits near 0.69. Combined 50/50 that leaves the
    regression head contributing almost nothing to the gradient, and both early
    stopping and best-model selection end up driven purely by classification.
    Scaling the targets to unit variance puts the two terms on a comparable
    footing. Predictions are mapped back with inverse_target_scale before any
    price reconstruction, so reported returns stay in real units.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.asarray(y_reg, dtype=np.float64))
    os.makedirs(os.path.dirname(TARGET_SCALER_PATH), exist_ok=True)
    with open(TARGET_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def load_target_scaler():
    if not os.path.exists(TARGET_SCALER_PATH):
        return None
    with open(TARGET_SCALER_PATH, "rb") as f:
        return pickle.load(f)


def apply_target_scaler(y_reg, scaler):
    if scaler is None or y_reg is None:
        return y_reg
    out = scaler.transform(np.asarray(y_reg, dtype=np.float64))
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def inverse_target_scale(y_scaled, scaler=None):
    """Map scaled model outputs back to real return units."""
    if y_scaled is None:
        return None
    if scaler is None:
        scaler = load_target_scaler()
    if scaler is None:
        return np.asarray(y_scaled, dtype=float)
    return scaler.inverse_transform(np.asarray(y_scaled, dtype=np.float64))


def fit_feature_scaler(X):
    """Fit a StandardScaler over the assembled joint feature matrix (train only).

    preprocess.py scales only named price/return/sentiment columns, so the NLP
    and embedding features (Word2Vec, BERT, TF-IDF counts, keyword counts) reach
    the models unscaled and sit orders of magnitude away from the scaled ones.
    That mismatch is what stopped logistic regression from converging and it
    also hurts every other distance- and gradient-based model.
    """
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(np.asarray(X, dtype=np.float64))
    os.makedirs(os.path.dirname(FEATURE_SCALER_PATH), exist_ok=True)
    with open(FEATURE_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    return scaler


def load_feature_scaler():
    if not os.path.exists(FEATURE_SCALER_PATH):
        return None
    with open(FEATURE_SCALER_PATH, "rb") as f:
        return pickle.load(f)


def apply_feature_scaler(X, scaler):
    """Apply the train-fitted scaler; returns float32 for torch."""
    if scaler is None or X is None:
        return X
    Xs = scaler.transform(np.asarray(X, dtype=np.float64))
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def create_sequences_with_context(X, y_reg, y_cls, context=None, seq_len=SEQ_LEN):
    """Emit one sequence per row of X, using history from the previous split.

    Without context the first `seq_len` rows of a split cannot form a window, so
    val/test lost their first 30 days (322 rows -> 292 sequences) and no longer
    lined up with their own targets.

    `context` is the tail of the chronologically preceding split. A window for
    label row j spans the `seq_len` rows immediately *before* j, exactly as in
    training, so the model never sees the day it is predicting. Borrowing prior
    rows only supplies past inputs and introduces no leakage.

    Returns (X_seq, y_reg_seq, y_cls_seq, label_idx) where label_idx gives the
    row of X each sequence predicts, for aligning dates and metadata.
    """
    X = np.asarray(X, dtype=np.float32)
    n_feat = X.shape[1] if X.ndim > 1 else 0

    if context is not None and len(context) > 0:
        ctx = np.asarray(context, dtype=np.float32)[-seq_len:]
        if ctx.shape[1] != n_feat:
            raise ValueError(
                f"context has {ctx.shape[1]} features but split has {n_feat}; "
                "both splits must be aligned to the same feature schema."
            )
    else:
        ctx = np.empty((0, n_feat), dtype=np.float32)

    X_full = np.vstack([ctx, X]) if len(ctx) else X
    offset = len(ctx)

    X_seq, y_reg_seq, y_cls_seq, label_idx = [], [], [], []
    for j in range(len(X)):
        end = offset + j          # exclusive: window stops before the label row
        start = end - seq_len
        if start < 0:
            continue              # insufficient history even with context
        X_seq.append(X_full[start:end])
        y_reg_seq.append(y_reg[j])
        y_cls_seq.append(y_cls[j])
        label_idx.append(j)

    if not X_seq:
        return (np.empty((0, seq_len, n_feat), dtype=np.float32),
                np.empty((0, y_reg.shape[1]), dtype=np.float32),
                np.empty((0, y_cls.shape[1]), dtype=np.float32),
                np.empty((0,), dtype=int))

    return (np.asarray(X_seq, dtype=np.float32),
            np.asarray(y_reg_seq, dtype=np.float32),
            np.asarray(y_cls_seq, dtype=np.float32),
            np.asarray(label_idx, dtype=int))


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

    # Separate features and targets. `_refclose` is a helper for reconstructing
    # dollar prices from returns, not a feature — excluding it also prevents the
    # model from seeing today's unscaled close as an input.
    exclude_cols = {"id", "date", "ticker", "company", "day_name", "month_name"}
    target_markers = ("_target_", f"_{REF_CLOSE_SUFFIX}")
    long_feature_cols = [c for c in df.columns if c not in exclude_cols
                    and not any(p in c for p in target_markers)
                    and df[c].dtype in ["float64", "float32", "int64", "int32"]]

    # Normalise to one row per date. With a single target ticker there is no
    # shared/per-ticker distinction left to derive: every column already
    # describes either NVDA or NVDA's market context. The schema is therefore
    # just the ordered feature list from train, persisted so val/test present
    # identical columns.
    print(f"  Normalising to one row per date...")
    is_train = (split == "train") and (feature_cols is None)
    if is_train:
        # Drop columns that carry no information in train.
        #
        # This matters concretely here: NVDA-tagged articles exist only from 2025
        # onward, so every nvda_* news column is entirely empty in train but
        # populated in test. Keeping them would train the model on constant zeros
        # and then feed it real values at test time — a train/serve mismatch that
        # degrades predictions silently. Dropping them on train keeps the feature
        # set honest; the industry and peer news tiers still carry the text signal.
        useful, dropped = [], []
        n_train = len(df)
        for c in long_feature_cols:
            col = df[c]
            if col.notna().sum() == 0 or col.nunique(dropna=True) <= 1:
                dropped.append(c)
                continue
            # Mostly-empty columns are noise for the tree/linear models and a
            # train/serve hazard: they are constant across train but light up at
            # test time (the 41.7% zero-fill warning). Require some coverage.
            if col.notna().mean() < 0.20:
                dropped.append(c)
                continue
            # Near-zero-variance columns add dimensionality without signal and
            # inflate the p >> n problem.
            if float(col.std(skipna=True) or 0.0) < 1e-8:
                dropped.append(c)
                continue
            useful.append(c)
        if dropped:
            print(f"  Dropped {len(dropped)} constant/sparse/low-variance feature "
                  f"cols in train (e.g. {dropped[:4]})")
        long_feature_cols = useful

        save_joint_schema(long_feature_cols, [])
        schema = (long_feature_cols, [])
        print(f"  Schema: {len(long_feature_cols)} feature cols (saved)")
    else:
        schema = load_joint_schema()
        if schema is None:
            print("  WARNING: no saved joint schema; deriving from this split "
                  "(column names may not match train)")

    df, all_feature_cols = pivot_to_joint(df, long_feature_cols, schema=schema)
    print(f"  Joint shape: {df.shape}")

    if df.empty:
        print(f"  No data after pivoting")
        return None, None, None, None, None

    # Regression targets are now next-day RETURNS per OHLC field, per ticker.
    reg_target_cols = [f"{t}_target_ret_{c}" for t in TICKERS for c in OHLC_COLS]
    cls_target_cols = [f"{t}_target_direction" for t in TICKERS]

    # Align to training feature columns if provided
    if feature_cols is not None:
        missing = [c for c in feature_cols if c not in df.columns]
        for c in missing:
            df[c] = 0.0
        final_cols = feature_cols
        if missing:
            # Loud, because zero-filled columns are silent model degradation:
            # the split is being scored on features it does not actually have.
            pct = 100.0 * len(missing) / max(len(feature_cols), 1)
            print(f"  WARNING: {len(missing)} of {len(feature_cols)} feature columns "
                  f"({pct:.1f}%) missing in '{split}' and zero-filled.")
            if pct > 25:
                print(f"           Examples: {missing[:5]}")
    else:
        final_cols = all_feature_cols

    # Exclude dead-zone rows: a missing direction label marks a near-flat day,
    # which carries no UP/DOWN signal. Dropping the whole row (rather than
    # NaN->0) removes them from training and evaluation; regression keeps the
    # same rows so the two tasks stay aligned per date. Without this the NaN
    # would silently filldown to DOWN and re-inject the noise we just removed.
    present_cls = [c for c in cls_target_cols if c in df.columns]
    if present_cls:
        before = len(df)
        df = df.dropna(subset=present_cls).reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            print(f"  Dead-zone: dropped {dropped} near-flat rows "
                  f"({100.0*dropped/max(before,1):.1f}%) with |next return| < "
                  f"{DIRECTION_DEAD_ZONE}")

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
                      X_test, y_reg_test, y_cls_test,
                      return_index=False):
    """Create sequences for all splits.

    Val and test are seeded with the tail of the chronologically preceding
    split, so every one of their rows gets a prediction instead of losing the
    first SEQ_LEN. The borrowed rows are inputs only: each window still ends the
    day before the row being predicted.
    """
    print("  Creating sequences...")
    X_tr, y_reg_tr, y_cls_tr, idx_tr = create_sequences_with_context(
        X_train, y_reg_train, y_cls_train, context=None)
    X_v, y_reg_v, y_cls_v, idx_v = create_sequences_with_context(
        X_val, y_reg_val, y_cls_val, context=X_train)
    X_te, y_reg_te, y_cls_te, idx_te = create_sequences_with_context(
        X_test, y_reg_test, y_cls_test, context=X_val)

    print(f"  Train: {X_tr.shape}")
    print(f"  Val: {X_v.shape}  (all {len(X_val)} rows covered via train context)")
    print(f"  Test: {X_te.shape}  (all {len(X_test)} rows covered via val context)")

    if return_index:
        return (X_tr, y_reg_tr, y_cls_tr, X_v, y_reg_v, y_cls_v,
                X_te, y_reg_te, y_cls_te, idx_tr, idx_v, idx_te)
    return X_tr, y_reg_tr, y_cls_tr, X_v, y_reg_v, y_cls_v, X_te, y_reg_te, y_cls_te


def flatten_sequences(X_seq):
    """Flatten sequences for tabular models (XGBoost, RF, etc.)."""
    n_samples = X_seq.shape[0]
    seq_len = X_seq.shape[1]
    n_features = X_seq.shape[2]
    return X_seq.reshape(n_samples, seq_len * n_features)
