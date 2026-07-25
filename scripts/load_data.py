import os
import pandas as pd
from sqlalchemy import Column, Integer, Float, String, Text, DateTime, Date
from sqlalchemy.orm import Session

from database_connection import Base, get_engine, get_session, get_db_path


# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------
class StockPrice(Base):
    __tablename__ = "stock_prices"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    company = Column(String(50))
    ticker = Column(String(10))


class MarketIndicator(Base):
    __tablename__ = "market_indicators"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    indicator = Column(String(50))
    ticker = Column(String(20))


class NewsArticle(Base):
    __tablename__ = "news_articles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    headline = Column(Text)
    summary = Column(Text)
    url = Column(Text)
    source = Column(String(100))
    source_categories = Column(String(200))
    tags = Column(Text)


class HFNews(Base):
    __tablename__ = "hf_news"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    headline = Column(Text)
    source = Column(String(100))
    ticker = Column(String(10))
    url = Column(Text)


class NewsSentiment(Base):
    __tablename__ = "news_sentiment"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date)
    headline = Column(Text)
    source = Column(String(100))
    ticker = Column(String(10))
    url = Column(Text)
    headline_compound = Column(Float)
    headline_pos = Column(Float)
    headline_neg = Column(Float)
    headline_neu = Column(Float)
    summary_compound = Column(Float)
    summary_pos = Column(Float)
    summary_neg = Column(Float)
    summary_neu = Column(Float)
    overall_compound = Column(Float)


class Company(Base):
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(10), unique=True, nullable=False)
    name = Column(String(100))
    sector = Column(String(50))


# ---------------------------------------------------------------------------
# Company lookup data
# ---------------------------------------------------------------------------
COMPANIES = [
    {"ticker": "NVDA", "name": "NVIDIA", "sector": "Semiconductor / AI Compute"},
    {"ticker": "META", "name": "Meta Platforms", "sector": "Social Media / AI"},
    {"ticker": "GOOGL", "name": "Alphabet (Google)", "sector": "Technology / AI"},
    {"ticker": "TSM", "name": "Taiwan Semiconductor", "sector": "Semiconductor Foundry"},
    {"ticker": "VRT", "name": "Vertiv Holdings", "sector": "Data Center Infrastructure"},
    {"ticker": "MOD", "name": "Modine Manufacturing", "sector": "Cooling Systems"},
    {"ticker": "SMCI", "name": "Super Micro Computer", "sector": "Server / AI Hardware"},
    {"ticker": "AMD", "name": "Advanced Micro Devices", "sector": "Semiconductor"},
    {"ticker": "INTC", "name": "Intel Corporation", "sector": "Semiconductor"},
    {"ticker": "AVGO", "name": "Broadcom", "sector": "Semiconductor / Networking"},
    {"ticker": "QCOM", "name": "Qualcomm", "sector": "Semiconductor"},
    {"ticker": "MRVL", "name": "Marvell Technology", "sector": "Semiconductor"},
    {"ticker": "AMAT", "name": "Applied Materials", "sector": "Semiconductor Equipment"},
    {"ticker": "MU", "name": "Micron Technology", "sector": "Memory Semiconductor"},
    {"ticker": "LRCX", "name": "Lam Research", "sector": "Semiconductor Equipment"},
    {"ticker": "KLAC", "name": "KLA Corporation", "sector": "Semiconductor Equipment"},
    {"ticker": "ASML", "name": "ASML Holding", "sector": "Semiconductor Equipment"},
    {"ticker": "ARM", "name": "ARM Holdings", "sector": "Semiconductor Design"},
    {"ticker": "NXPI", "name": "NXP Semiconductors", "sector": "Semiconductor"},
    {"ticker": "ON", "name": "ON Semiconductor", "sector": "Semiconductor"},
]


# ---------------------------------------------------------------------------
# CSV-to-table mapping
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_samples")

CSV_MAP = [
    {
        "file": "ai_infrastructure_stock_data.csv",
        "model": StockPrice,
        "rename": {"Date": "date", "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume", "Company": "company", "Ticker": "ticker"},
        "date_col": "date",
    },
    {
        "file": "market_indicators.csv",
        "model": MarketIndicator,
        "rename": {"Date": "date", "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Volume": "volume", "Indicator": "indicator", "Ticker": "ticker"},
        "date_col": "date",
    },
    {
        "file": "ai_infrastructure_news.csv",
        "model": NewsArticle,
        "rename": None,
        "date_col": "date",
    },
    {
        "file": "hf_financial_news.csv",
        "model": HFNews,
        "rename": None,
        "date_col": "date",
    },
    {
        "file": "news_sentiment.csv",
        "model": NewsSentiment,
        "rename": None,
        "date_col": "date",
    },
]


def load_csv_to_table(csv_info, session):
    filepath = os.path.join(DATA_DIR, csv_info["file"])
    if not os.path.exists(filepath):
        print(f"   File not found: {filepath}, skipping.")
        return 0

    df = pd.read_csv(filepath)

    if csv_info["rename"]:
        df = df.rename(columns=csv_info["rename"])

    if csv_info["date_col"] and csv_info["date_col"] in df.columns:
        df[csv_info["date_col"]] = pd.to_datetime(df[csv_info["date_col"]], errors="coerce").dt.date

    df = df.where(pd.notnull(df), None)

    records = df.to_dict(orient="records")
    model = csv_info["model"]

    for record in records:
        session.add(model(**record))

    session.commit()
    return len(records)


def create_database():
    print("  Creating SQLite database...\n")

    engine = get_engine()
    Base.metadata.create_all(engine)
    print(f"   Database: {get_db_path()}")

    tables = [t.name for t in Base.metadata.sorted_tables]
    print(f"   Tables created: {', '.join(tables)}\n")

    session = get_session()

    # Load companies
    print("Loading companies...")
    for comp in COMPANIES:
        existing = session.query(Company).filter_by(ticker=comp["ticker"]).first()
        if not existing:
            session.add(Company(**comp))
    session.commit()
    print(f"   {len(COMPANIES)} companies loaded")

    # Load CSVs
    total = 0
    for csv_info in CSV_MAP:
        print(f"Loading {csv_info['file']}...")
        count = load_csv_to_table(csv_info, session)
        total += count
        print(f"   {count:,} rows inserted")

    session.close()

    print(f"\n{'='*50}")
    print(f"Database created successfully!")
    print(f"Total rows inserted: {total:,}")
    print(f"File: {get_db_path()}")
    print(f"{'='*50}")

    return get_db_path()


if __name__ == "__main__":
    create_database()
