"""
Pipeline — prediction and output.

Loads trained models, makes predictions on test set, inverse-transforms them
back into real dollar prices, and saves them to the DB.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sqlalchemy import text
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.db import get_engine
from scripts.preprocessing.preprocess import inverse_transform_column

from .data_assembly import prepare_data, flatten_sequences, TICKERS, SEQ_LEN

OHLC = ["open", "high", "low", "close"]


def load_models():
    """Load all trained models from disk."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    models = {}

    # Time series
    from ..timeseries import lstm, gru, transformer, bilstm, tcn
    for name, mod in [("lstm", lstm), ("gru", gru), ("transformer", transformer),
                      ("bilstm", bilstm), ("tcn", tcn)]:
        path = os.path.join(models_dir, f"timeseries_{name}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu")
            model = mod.create_model(ckpt["input_size"], n_tickers=5)
            model.load_state_dict(ckpt["model_state_dict"])
            models[f"ts_{name}"] = model

    # Tabular
    for name in ["xgboost", "random_forest", "lightgbm"]:
        for task in ["reg", "cls"]:
            path = os.path.join(models_dir, f"tabular_{name}_{task}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    models[f"tab_{name}_{task}"] = pickle.load(f)

    # MLP
    from ..tabular import mlp
    for task in ["reg", "cls"]:
        path = os.path.join(models_dir, f"tabular_mlp_{task}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu")
            if ckpt["is_classifier"]:
                model = mlp.MLPClassifier(ckpt["input_size"], ckpt["hidden_sizes"], output_size=ckpt["output_size"])
            else:
                model = mlp.MLPRegressor(ckpt["input_size"], ckpt["hidden_sizes"], output_size=ckpt["output_size"])
            model.load_state_dict(ckpt["model_state_dict"])
            models[f"tab_mlp_{task}"] = model

    # Ridge / Logistic
    for name, task in [("ridge", "reg"), ("logistic", "cls")]:
        path = os.path.join(models_dir, f"tabular_{name}_{task}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[f"tab_{name}_{task}"] = pickle.load(f)

    # Feature names
    fn_path = os.path.join(models_dir, "feature_names.pkl")
    if os.path.exists(fn_path):
        with open(fn_path, "rb") as f:
            models["feature_names"] = pickle.load(f)

    return models


def predict_all(models):
    """Run predictions on test set using all models."""
    from ..timeseries.train import predict as ts_predict
    from ..tabular.mlp import predict_regressor as mlp_reg, predict_classifier as mlp_cls
    from ..tabular.xgboost_model import predict_regressor as xgb_reg, predict_classifier as xgb_cls
    from ..tabular.random_forest import predict_regressor as rf_reg, predict_classifier as rf_cls
    from ..tabular.ridge import predict as ridge_pred
    from ..tabular import logistic

    # Load test data (align to training features)
    feature_names = models.get("feature_names")
    if feature_names is None:
        _, _, _, feature_names, _ = prepare_data("train")
    X_test, y_reg_test, y_cls_test, _, df_test = prepare_data("test", feature_cols=feature_names)
    if X_test is None:
        return None

    # Create sequences for time series
    from .data_assembly import create_sequences
    X_seq, y_reg_seq, y_cls_seq = create_sequences(X_test, y_reg_test, y_cls_test)
    X_flat = flatten_sequences(X_seq)

    if len(X_seq) == 0:
        print("  No test sequences available (test split shorter than SEQ_LEN)")
        return None

    # Rows of df_test that each sequence label corresponds to
    label_index = np.arange(SEQ_LEN, SEQ_LEN + len(X_seq))
    df_labels = df_test.iloc[label_index].reset_index(drop=True)

    results = {}

    # Time series predictions
    for name in ["lstm", "gru", "transformer", "bilstm", "tcn"]:
        key = f"ts_{name}"
        if key in models:
            loader = _make_loader(X_seq)
            reg_pred, cls_pred = ts_predict(models[key], loader)
            results[key] = {"reg": reg_pred, "cls": cls_pred}

    # Tabular predictions
    for name in ["xgboost", "random_forest"]:
        reg_key = f"tab_{name}_reg"
        cls_key = f"tab_{name}_cls"
        if reg_key in models:
            results[reg_key] = {"reg": xgb_reg(models[reg_key], X_flat) if name == "xgboost" else rf_reg(models[reg_key], X_flat)}
        if cls_key in models:
            pred, prob = xgb_cls(models[cls_key], X_flat) if name == "xgboost" else rf_cls(models[cls_key], X_flat)
            results[cls_key] = {"cls": pred, "prob": prob}

    # MLP
    if "tab_mlp_reg" in models:
        results["tab_mlp_reg"] = {"reg": mlp_reg(models["tab_mlp_reg"], X_flat)}
    if "tab_mlp_cls" in models:
        pred, prob = mlp_cls(models["tab_mlp_cls"], X_flat)
        results["tab_mlp_cls"] = {"cls": pred, "prob": prob}

    # Ridge
    if "tab_ridge_reg" in models:
        results["tab_ridge_reg"] = {"reg": ridge_pred(models["tab_ridge_reg"], X_flat)}

    # Logistic (per ticker)
    if "tab_logistic_cls" in models:
        n_t = len(models["tab_logistic_cls"])
        cls_pred = np.zeros((X_flat.shape[0], n_t))
        cls_prob = np.zeros((X_flat.shape[0], n_t))
        for t, lr_m in models["tab_logistic_cls"].items():
            cls_pred[:, t], cls_prob[:, t] = logistic.predict(lr_m, X_flat)
        results["tab_logistic_cls"] = {"cls": cls_pred, "prob": cls_prob}

    results["_actuals"] = {"reg": y_reg_seq, "cls": y_cls_seq}

    return results, df_labels


def build_ensemble(predictions):
    """Blend model outputs into one regression and one classification result.

    Regression: 0.6 * mean(time-series models) + 0.4 * mean(tabular models),
    falling back to whichever family is present.
    Classification: mean probability across models, thresholded at 0.5.
    """
    ts_reg = [v["reg"] for k, v in predictions.items()
              if k.startswith("ts_") and v.get("reg") is not None]
    tab_reg = [v["reg"] for k, v in predictions.items()
               if k.startswith("tab_") and v.get("reg") is not None]

    reg = None
    if ts_reg and tab_reg:
        reg = 0.6 * np.mean(ts_reg, axis=0) + 0.4 * np.mean(tab_reg, axis=0)
    elif ts_reg:
        reg = np.mean(ts_reg, axis=0)
    elif tab_reg:
        reg = np.mean(tab_reg, axis=0)

    # Probabilities: time series cls heads are already 0-1, tabular give prob
    probs = []
    for k, v in predictions.items():
        if k == "_actuals":
            continue
        if v.get("prob") is not None:
            probs.append(np.asarray(v["prob"], dtype=float))
        elif k.startswith("ts_") and v.get("cls") is not None:
            probs.append(np.asarray(v["cls"], dtype=float))

    prob = np.mean(probs, axis=0) if probs else None
    cls = (prob > 0.5).astype(int) if prob is not None else None

    return reg, cls, prob


def to_real_prices(scaled_matrix):
    """Inverse-transform a (n, n_tickers*4) OHLC matrix into dollar prices.

    Targets were derived from the RobustScaler-scaled stock_prices columns, so
    each OHLC column is mapped back through that same column's scaler.
    """
    if scaled_matrix is None:
        return None
    scaled_matrix = np.asarray(scaled_matrix, dtype=float)
    out = np.empty_like(scaled_matrix)
    for t_i in range(len(TICKERS)):
        for c_i, col in enumerate(OHLC):
            j = t_i * 4 + c_i
            if j < scaled_matrix.shape[1]:
                out[:, j] = inverse_transform_column("stock_prices", col, scaled_matrix[:, j])
    return out


def save_predictions_to_db(predictions, df_test):
    """Save ensemble predictions to the predictions table in DB.

    One row per (date, ticker) with predicted and actual OHLC in real dollars,
    plus predicted direction and its confidence.
    """
    engine = get_engine()

    reg, cls, prob = build_ensemble(predictions)
    actuals = predictions.get("_actuals", {})
    y_reg = actuals.get("reg")
    y_cls = actuals.get("cls")

    if reg is None and cls is None:
        print("  No predictions to save")
        return

    n_rows = len(reg) if reg is not None else len(cls)

    reg_real = to_real_prices(reg)
    actual_real = to_real_prices(y_reg)

    dates = pd.to_datetime(df_test["date"]).dt.strftime("%Y-%m-%d").tolist() \
        if "date" in df_test.columns else [None] * n_rows

    rows = []
    for i in range(n_rows):
        for t_i, ticker in enumerate(TICKERS):
            row = {
                "date": dates[i] if i < len(dates) else None,
                "ticker": ticker,
                "model_name": "ensemble",
            }
            for c_i, col in enumerate(OHLC):
                j = t_i * 4 + c_i
                row[f"predicted_{col}"] = (
                    float(reg_real[i, j]) if reg_real is not None and j < reg_real.shape[1] else None
                )
                row[f"actual_{col}"] = (
                    float(actual_real[i, j]) if actual_real is not None and j < actual_real.shape[1] else None
                )

            row["direction"] = (
                int(cls[i, t_i]) if cls is not None and t_i < cls.shape[1] else None
            )
            if prob is not None and t_i < prob.shape[1]:
                p = float(prob[i, t_i])
                # Distance from the decision boundary, scaled to 0-1
                row["confidence"] = abs(p - 0.5) * 2
            else:
                row["confidence"] = None
            row["actual_direction"] = (
                int(y_cls[i, t_i]) if y_cls is not None and t_i < y_cls.shape[1] else None
            )
            rows.append(row)

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS predictions"))
        conn.execute(text("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                ticker TEXT,
                predicted_open FLOAT,
                predicted_high FLOAT,
                predicted_low FLOAT,
                predicted_close FLOAT,
                direction INTEGER,
                confidence FLOAT,
                model_name TEXT,
                actual_open FLOAT,
                actual_high FLOAT,
                actual_low FLOAT,
                actual_close FLOAT,
                actual_direction INTEGER
            )
        """))
        conn.commit()

    out_df = pd.DataFrame(rows)
    out_df.to_sql("predictions", engine, if_exists="append", index=False)

    print(f"  Predictions saved to DB: {len(out_df):,} rows "
          f"({n_rows:,} dates × {len(TICKERS)} tickers)")


def _make_loader(X_seq):
    """Create DataLoader from sequence data."""

    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.FloatTensor(X_seq), torch.zeros(len(X_seq)), torch.zeros(len(X_seq)))
    return DataLoader(ds, batch_size=64, shuffle=False)
