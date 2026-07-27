import os
import sys
import pandas as pd
import numpy as np
import holidays
from sqlalchemy import text
from sklearn.preprocessing import OneHotEncoder

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db import get_engine

engine = get_engine()
US_HOLIDAYS = holidays.US(years=range(2018, 2027))


def drop_old_ohe_columns():
    print("🗑️  Dropping old OneHotEncoded columns...")
    old_cols = [
        "is_holiday", "is_monday", "is_tuesday", "is_wednesday", "is_thursday", "is_friday",
        "is_january", "is_february", "is_march", "is_april", "is_may", "is_june",
        "is_july", "is_august", "is_september", "is_october", "is_november", "is_december",
        "is_q1", "is_q2", "is_q3", "is_q4",
    ]
    with engine.connect() as conn:
        for col in old_cols:
            try:
                conn.execute(text(f"ALTER TABLE stock_prices DROP COLUMN {col}"))
                print(f"    - {col}")
            except Exception:
                pass
        conn.commit()


def add_holiday_column():
    print("📅 Adding is_holiday column...")
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE stock_prices ADD COLUMN is_holiday INTEGER DEFAULT 0"))
        except Exception:
            pass

        df = pd.read_sql("SELECT id, date FROM stock_prices", engine)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["is_holiday"] = df["date"].apply(lambda d: 1 if d in US_HOLIDAYS else 0)

        for idx in df.index:
            conn.execute(text("UPDATE stock_prices SET is_holiday = :val WHERE id = :id"),
                         {"val": int(df.loc[idx, "is_holiday"]), "id": int(df.loc[idx, "id"])})
        conn.commit()
    print(f"    ✅ {df['is_holiday'].sum()} holiday rows found")


def one_hot_encode_sklearn():
    print("🔤 OneHotEncoding with sklearn...")

    df = pd.read_sql("SELECT id, day_name, month_name, quarter FROM stock_prices", engine)
    print(f"    Unique day_name: {sorted(df['day_name'].dropna().unique())}")
    print(f"    Unique month_name: {sorted(df['month_name'].dropna().unique())}")
    print(f"    Unique quarter: {sorted(df['quarter'].dropna().unique())}")

    cat_cols = ["day_name", "month_name", "quarter"]
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", drop=None)
    encoded = encoder.fit_transform(df[cat_cols].fillna("unknown"))
    feature_names = encoder.get_feature_names_out(cat_cols)
    print(f"    Generated {len(feature_names)} OneHotEncoded columns")

    ohe_df = pd.DataFrame(encoded.astype(int), columns=feature_names)
    ohe_df["id"] = df["id"].values

    with engine.connect() as conn:
        for col in feature_names:
            safe_col = col.replace(" ", "_").lower()
            try:
                conn.execute(text(f"ALTER TABLE stock_prices ADD COLUMN {safe_col} INTEGER DEFAULT 0"))
                print(f"    + {safe_col}")
            except Exception:
                pass

        for idx in ohe_df.index:
            sets = []
            vals = {}
            for col in feature_names:
                safe_col = col.replace(" ", "_").lower()
                v = int(ohe_df.loc[idx, col])
                sets.append(f"{safe_col} = :{safe_col}")
                vals[safe_col] = v
            vals["id"] = int(ohe_df.loc[idx, "id"])
            conn.execute(text(f"UPDATE stock_prices SET {', '.join(sets)} WHERE id = :id"), vals)
        conn.commit()

    print(f"    ✅ {len(df):,} rows updated")


if __name__ == "__main__":
    print("🔧 Rebuilding OneHotEncoded features with sklearn...\n")
    drop_old_ohe_columns()
    add_holiday_column()
    one_hot_encode_sklearn()
    print(f"\n{'='*50}")
    print("✅ OneHotEncoding complete!")
    print("{'='*50}")
