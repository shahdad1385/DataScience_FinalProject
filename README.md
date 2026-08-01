# CAF — Multimodal AI Infrastructure & Hardware Supply Chain Pipeline

Predicting market valuation shifts for Big Tech AI leaders by aligning financial time-series with alternative textual signals from hardware supply chain and grid infrastructure networks.

## Architecture

```text
┌────────────────────────────────────────────────────────────────────┐
│                     Data Collection (scrapers/)                    │
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
│              ML Pipeline (scripts/ + pipeline.py)                  │
│  NLP (BERT, Word2Vec, TF-IDF, VADER) → Clustering (K-Means,      │
│  DBSCAN, Hierarchical) → Feature Assembly → Model Training         │
│  (LSTM, GRU, Transformer, BiLSTM, TCN, XGBoost, LightGBM,        │
│  Random Forest, MLP, Ridge, Logistic) → Ensemble → Evaluation     │
└────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
CAF/
├── pipeline.py                          # CLI entry point (train/finetune/evaluate/full)
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # Container image (python:3.12-slim)
├── k8s-deployment.yaml                  # Kubernetes Job manifest
├── .github/workflows/pipeline.yml       # CI: install + evaluate on push/PR
├── scrapers/                            # Data collection
│   ├── AI_Stock.py                      #   OHLCV stock data via yfinance
│   ├── collect_infrastructure_news.py   #   Multi-source news scraping
│   ├── download_datasets.py             #   HuggingFace financial news
│   ├── market_indicators.py             #   VIX, treasuries, ETFs, BTC, DXY
│   ├── economic_calendar.py             #   Earnings dates + macro indicators
│   ├── sentiment_analysis.py            #   VADER sentiment on news headlines
│   ├── social_sentiment.py              #   Ticker-specific social media sentiment
│   └── visualize_data.py                #   Matplotlib/seaborn chart generation
├── scripts/
│   ├── activations.py                   #   Shared activation layer factory
│   ├── one_hot_encode.py                #   Categorical OHE encoding
│   ├── generate_summaries.py            #   LLM summaries via NVIDIA NIM API
│   ├── export_llm_summaries.py          #   Export LLM summaries to CSV cache
│   ├── db/database_connection.py        #   SQLAlchemy engine + session factory
│   ├── preprocessing/
│   │   ├── load_data.py                 #     CSV → SQLite ingestion
│   │   ├── feature_engineering.py       #     Technical indicators, returns, lags
│   │   └── preprocess.py               #     Temporal split, scaling, detrending
│   ├── nlp/
│   │   ├── extract.py                   #     NLP orchestrator per split
│   │   ├── features.py                  #     TF-IDF, Word2Vec, BERT, sentiment
│   │   ├── bert.py                      #     Sentence-transformers BERT
│   │   └── word2vec.py                  #     Gensim Word2Vec training
│   ├── clustering/
│   │   ├── cluster.py                   #     Orchestrator: best-of-3 selection
│   │   ├── compare.py                   #     Silhouette/DB/CH metric comparison
│   │   ├── kmeans.py                    #     K-Means clustering
│   │   ├── dbscan.py                    #     DBSCAN clustering
│   │   └── hierarchical.py             #     Agglomerative hierarchical
│   ├── timeseries/
│   │   ├── train.py                     #     Shared GPU training loop
│   │   ├── lstm.py                      #     LSTM model
│   │   ├── gru.py                       #     GRU model
│   │   ├── transformer.py               #     Transformer with positional encoding
│   │   ├── bilstm.py                    #     Bidirectional LSTM
│   │   └── tcn.py                       #     Temporal Convolutional Network
│   ├── tabular/
│   │   ├── pipe.py                      #     Tabular model orchestrator
│   │   ├── xgboost_model.py             #     XGBoost reg + clf
│   │   ├── lightgbm_model.py            #     LightGBM reg + clf
│   │   ├── random_forest.py             #     Random Forest reg + clf
│   │   ├── mlp.py                       #     PyTorch MLP
│   │   ├── logistic.py                  #     Logistic Regression + PCA
│   │   ├── ridge.py                     #     Ridge Regression baseline
│   │   ├── regress.py                   #     Shared regression metrics
│   │   └── classify.py                  #     Shared classification metrics
│   └── pipeline/
│       ├── run.py                       #     Full pipeline orchestrator
│       ├── train.py                     #     Train all models + ensemble
│       ├── predict.py                   #     Test-set predictions
│       ├── evaluate.py                  #     Detailed evaluation + plots
│       ├── finetune.py                  #     Optuna hyperparameter optimization
│       └── data_assembly.py             #     Merge features into sequence windows
└── data_samples/                        # Generated data outputs
    ├── caf_database.db                  #   SQLite database (all tables)
    ├── ai_infrastructure_stock_data.csv #   14,630 rows OHLCV
    ├── ai_infrastructure_news.csv       #   390 filtered news articles
    ├── hf_financial_news.csv            #   6,798 HuggingFace articles
    ├── market_indicators.csv            #   17,966 rows indicators
    ├── news_sentiment.csv               #   VADER sentiment scores
    ├── social_sentiment.csv             #   Social media sentiment
    ├── economic_events.csv              #   Earnings + macro events
    ├── train.csv / test.csv             #   Preprocessed splits
    └── charts/                          #   Visualization PNGs
```

## Quick Start

```bash
pip install -r requirements.txt

# Fetch all raw data
python3 scrapers/AI_Stock.py
python3 scrapers/market_indicators.py
python3 scrapers/collect_infrastructure_news.py
python3 scrapers/download_datasets.py
python3 scrapers/sentiment_analysis.py
python3 scrapers/social_sentiment.py
python3 scrapers/economic_calendar.py
python3 scrapers/visualize_data.py

# Run the full ML pipeline (preprocess + train + evaluate)
python pipeline.py --mode full
```

## Pipeline Modes

```bash
python pipeline.py --mode train              # Train all models from scratch
python pipeline.py --mode finetune           # Optuna hyperparameter tuning
python pipeline.py --mode evaluate           # Evaluate saved models on test set
python pipeline.py --mode full               # Preprocess + train + evaluate
python pipeline.py --mode train --model lstm # Train only LSTM
python pipeline.py --mode finetune --trials 100  # 100 Optuna trials
```

### CLI Flags

| Flag | Description |
|------|-------------|
| `--model` | Single model: lstm, gru, transformer, bilstm, tcn, xgboost, lightgbm, random_forest, mlp, ridge, logistic |
| `--epochs` | Max training epochs (default: 100) |
| `--lr` | Learning rate (default: 1e-3) |
| `--patience` | Early stopping patience (default: 30) |
| `--activation` | Neural activation: leaky_relu, relu, gelu, silu, elu (default: leaky_relu) |
| `--reg-loss` | Regression loss: huber, mse, mae (default: huber) |
| `--trials` | Optuna trials for finetune (default: 50) |
| `--skip-nlp` | Skip NLP feature extraction |
| `--skip-clustering` | Skip clustering step |

### Evaluation Flags

| Flag | Description |
|------|-------------|
| `--eval-detailed` | Per-ticker & per-output metrics breakdown |
| `--eval-confusion` | Confusion matrices as PNG + CSV |
| `--eval-trading` | Sharpe, Sortino, Max Drawdown, Hit Rate, Profit Factor |
| `--eval-feature-imp` | Feature importance (tree models) |
| `--eval-statistical` | Diebold-Mariano test, permutation importance |
| `--eval-calibration` | Reliability diagrams for classifiers |
| `--eval-save-preds` | Save raw predictions to CSV |
| `--eval-save-plots` | Save all plots to `models/eval_plots/` |
| `--eval-ensemble` | Evaluate ensemble predictions |

## Database Tables

| Table | Rows | Description |
|---|---|---|
| `stock_prices` | 14,630 | Daily OHLCV + engineered features |
| `market_indicators` | 17,966 | VIX, Treasuries, ETFs, BTC, USD Index |
| `news_articles` | 390 | Industry news + TF-IDF features |
| `hf_news` | 6,798 | HuggingFace financial news (2018-2020) |
| `companies` | 20 | Ticker → company → sector mapping |

## Preprocessing

- **Temporal Split**: Train/val/test by date cutoff (no shuffling)
- **Null Handling**: Linear Regression imputation on train features; interpolation fallback
- **Normalization**: RobustScaler for price/volume/volatility; StandardScaler for returns/ratios — fit on train only
- **No Data Leakage**: All statistics computed from train data only

## Models

**Time-Series:** LSTM, GRU, Transformer, BiLSTM, TCN — multi-output OHLC regression + direction classification

**Tabular:** XGBoost, LightGBM, Random Forest, MLP, Ridge, Logistic Regression — flattened feature vectors

**Ensemble:** Best-model selection across all architectures

## Deployment

```bash
# Docker
docker build -t caf-pipeline .
docker run caf-pipeline --mode evaluate

# Kubernetes
kubectl apply -f k8s-deployment.yaml
```

## CI

GitHub Actions runs `pipeline.py --mode evaluate` on every push/PR to `main`.

