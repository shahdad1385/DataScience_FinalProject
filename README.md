# Multimodal AI Infrastructure & Hardware Supply Chain Pipeline

### Phase 1: High-Performance Alternative Data Acquisition Engine

### 📋 Executive Overview

Predicting market valuation shifts for Big Tech AI market leaders requires data beyond traditional daily closing prices. This project implements a robust, multimodal ingestion pipeline designed to align standard financial time-series indicators with structured, alternative textual signals scraped from critical hardware supply chain and grid infrastructure networks.

The core objective is to map lag-correlations between physical constraints—such as grid capacity, high-velocity liquid cooling breakthroughs, and chip-foundry expansions—and the downstream valuation fluctuations of market cap leaders.

### 🛠️ System Architecture & Data Flow

```text
┌────────────────────────────────────────────────────────────────────┐
│                     Data Collection Pipeline                       │
├──────────────────────┬──────────────────────┬──────────────────────┤
│   Stock Data         │   News Sources       │   Market Context     │
│   (yfinance)         │   (RSS + HTML + API) │   (yfinance)         │
├──────────────────────┼──────────────────────┼──────────────────────┤
│ • 7 AI/semi stocks   │ • Reuters (200+)     │ • VIX volatility     │
│ • 2018–2026 daily    │ • TechCrunch (20)    │ • Treasury yields    │
│ • OHLCV format       │ • SemiEngineering    │ • SMH/XLK ETFs      │
│                      │ • The Register       │ • Bitcoin            │
│                      │ • Tom's Hardware     │ • USD Index          │
│                      │ • Reddit (5 subs)    │                      │
│                      │ • HuggingFace (6.8K) │                      │
├──────────────────────┴──────────────────────┴──────────────────────┤
│                          Output: /data_samples                     │
│  • ai_infrastructure_stock_data.csv  (14,630 rows)                 │
│  • ai_infrastructure_news.csv        (390 filtered)                │
│  • hf_financial_news.csv             (6,798 from HuggingFace)      │
│  • market_indicators.csv             (17,966 rows)                 │
│  • all_news_raw.csv                  (462 unfiltered)              │
│  • charts/                           (5 visualization PNGs)        │
└────────────────────────────────────────────────────────────────────┘
```

### 📂 Scripts Overview

| Script | Purpose | Output |
|---|---|---|
| `scrapers/AI_Stock.py` | Fetch historical stock data (7 tickers) | `ai_infrastructure_stock_data.csv` |
| `scrapers/collect_infrastructure_news.py` | Multi-source news + Reddit | `ai_infrastructure_news.csv`, `all_news_raw.csv` |
| `scrapers/download_datasets.py` | Download HuggingFace financial news | `hf_financial_news.csv` |
| `scrapers/market_indicators.py` | Fetch VIX, Treasuries, ETFs, BTC | `market_indicators.csv` |
| `scrapers/visualize_data.py` | Generate charts from all data | `charts/*.png` |

### 🚀 Execution Manual

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Fetch stock data
python3 scrapers/AI_Stock.py

# Step 3: Fetch market indicators
python3 scrapers/market_indicators.py

# Step 4: Collect news from all sources (RSS + Reddit)
python3 scrapers/collect_infrastructure_news.py

# Step 5: Download HuggingFace financial news dataset
python3 scrapers/download_datasets.py

# Step 6: Generate visualizations
python3 scrapers/visualize_data.py
```

### 📊 Datasets

#### Stock Data (`ai_infrastructure_stock_data.csv`)

| Feature | Type | Description |
|---|---|---|
| Date | Timestamp | Trading date (YYYY-MM-DD) |
| Open/High/Low/Close | Float | Price data |
| Volume | Int | Shares traded |
| Company/Ticker | String | Company identifier |

**Tickers:** NVDA, META, GOOGL, TSM, VRT, MOD, SMCI

#### Live News (`ai_infrastructure_news.csv`)

Collected from 12 sources via RSS, HTML scraping, and Google News RSS.

| Source | Method | Articles |
|---|---|---|
| Reuters | Google News RSS | ~200 |
| Reddit r/wallstreetbets | RSS | ~23 |
| Tom's Hardware | RSS | ~38 |
| The Register | Atom RSS | ~30 |
| DataCenterDynamics | HTML | ~21 |
| TechCrunch AI | RSS | ~20 |
| + 5 more Reddit subs | RSS | varies |

#### HuggingFace News (`hf_financial_news.csv`)

Historical financial news from ashraq/financial-news (2018–2020).

| Feature | Type | Description |
|---|---|---|
| date | Timestamp | Publication date |
| headline | String | Article headline |
| source | String | Publisher name |
| ticker | String | Stock ticker symbol |
| url | String | Article URL |

**Coverage:** 6,798 articles across 13 semiconductor/AI tickers.

#### Market Indicators (`market_indicators.csv`)

| Indicator | Ticker | Rows | Purpose |
|---|---|---|---|
| VIX | ^VIX | 2,127 | Market volatility |
| 10Y Treasury | ^TNX | 2,123 | Interest rates |
| 5Y Treasury | ^FVX | 2,123 | Interest rates |
| 30Y Treasury | ^TYX | 2,123 | Interest rates |
| Semiconductor ETF | SMH | 2,126 | Sector benchmark |
| Tech ETF | XLK | 2,126 | Tech sentiment |
| Bitcoin | BTC-USD | 3,090 | Risk sentiment |
| US Dollar Index | DX-Y.NYB | 2,128 | Currency strength |

---

### 🔮 Downstream Pipeline Context (Phase 2 & Phase 3)

The collected data provides the foundation for:

* **Sentiment Analysis (Phase 2):** FinBERT on news headlines → quantitative sentiment scores
* **Feature Engineering:** Lag features, rolling averages, keyword frequencies
* **Correlation Analysis:** News sentiment vs stock price movements
* **Predictive Modeling (Phase 3):** Time-series forecasting with sentiment + market indicators

---

## Phase 2: Data Pipeline

### Project Structure

```
CAF/
├── pipeline.py                    # Main pipeline orchestrator
├── requirements.txt               # Python dependencies
├── data_samples/
│   ├── caf_database.db            # SQLite database
│   ├── train_stock_prices.csv     # Train split
│   └── test_stock_prices.csv      # Test split
├── scripts/
│   ├── database_connection.py     # DB connection helper
│   ├── load_data.py               # CSV → DB ingestion
│   ├── feature_engineering.py     # Feature computation (writes to DB)
│   ├── preprocess.py              # Split, null handling, normalization
│   ├── one_hot_encode.py          # OHE encoding
│   └── preprocess.py              # Preprocessing pipeline
└── scrapers/                      # Data collection scripts
```

### How to Run the Pipeline

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the full pipeline
python pipeline.py
```

### Evaluation Flags (Phase 2+)

The evaluation mode supports extended analysis flags:

| Flag | Description |
|------|-------------|
| `--eval-detailed` | Per-ticker & per-output metrics breakdown (RMSE, MAE, R², F1, AUC per ticker/output) |
| `--eval-confusion` | Save confusion matrices as PNG heatmaps + CSV per ticker |
| `--eval-trading` | Trading simulation: Sharpe, Sortino, Max Drawdown, Hit Rate, Profit Factor |
| `--eval-feature-imp` | Feature importance for tree models (XGBoost, LightGBM, RF); permutation for others |
| `--eval-statistical` | Diebold-Mariano test vs naive baseline; permutation importance |
| `--eval-calibration` | Reliability diagrams / calibration curves for classifiers |
| `--eval-save-preds` | Save raw predictions + probabilities to CSV |
| `--eval-save-plots` | Save all evaluation plots to `models/eval_plots/` |
| `--eval-ensemble` | Evaluate ensemble predictions if available |

**Examples:**
```bash
# Full detailed evaluation with confusion matrices and trading metrics
python pipeline.py --mode evaluate --eval-detailed --eval-confusion --eval-trading

# Evaluate only XGBoost with feature importance
python pipeline.py --mode evaluate --model xgboost --eval-feature-imp --eval-confusion

# Save all predictions and plots
python pipeline.py --mode evaluate --eval-save-preds --eval-save-plots --eval-detailed
```

### Pipeline Usage Examples

The pipeline executes in this order:

| Step | Script | Description |
|---|---|---|
| 1 | `scripts/load_data.py` | Creates SQLite DB, loads raw CSVs into tables |
| 2 | `scripts/feature_engineering.py` | Computes technical indicators, calendar features, cyclical encoding, OHE — writes back to DB |
| 3 | `scripts/preprocess.py` | Temporal train/test split (85/15), null handling via Linear Regression, RobustScaler + StandardScaler normalization |

### Database Tables

| Table | Rows | Description |
|---|---|---|
| `stock_prices` | 14,630 | Daily OHLCV + 70 engineered features |
| `market_indicators` | 17,966 | VIX, Treasuries, ETFs, Bitcoin, USD Index |
| `news_articles` | 390 | Industry news headlines + TF-IDF features |
| `hf_news` | 6,798 | HuggingFace financial news (2018-2020) |
| `companies` | 20 | Ticker → company name → sector mapping |

### Preprocessing Details

- **Temporal Split**: 85% train / 15% test, split by date cutoff (no random shuffling)
- **Null Handling**: Linear Regression trained on train set using all available features; interpolation as fallback
- **Normalization**: RobustScaler for price/volume/volatility; StandardScaler for returns/ratios — all fit on train only
- **No Data Leakage**: All statistics computed from train data only
