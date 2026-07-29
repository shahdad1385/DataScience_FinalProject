"""
Pipeline — prediction and output.

Loads trained models, makes predictions on test set, saves to DB.
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

from .data_assembly import prepare_data, flatten_sequences, TICKERS, SEQ_LEN


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
    import torch
    from ..timeseries.train import predict as ts_predict
    from ..tabular.mlp import predict_regressor as mlp_reg, predict_classifier as mlp_cls
    from ..tabular.xgboost_model import predict_regressor as xgb_reg, predict_classifier as xgb_cls
    from ..tabular.random_forest import predict_regressor as rf_reg, predict_classifier as rf_cls
    from ..tabular.ridge import predict as ridge_pred
    from ..tabular import logistic

    # Load test data (align to training features)
    _, _, _, feature_names, _ = prepare_data("train")
    X_test, y_reg_test, y_cls_test, _, df_test = prepare_data("test", feature_cols=feature_names)
    if X_test is None:
        return None

    # Create sequences for time series
    from .data_assembly import create_sequences
    X_seq, y_reg_seq, y_cls_seq = create_sequences(X_test, y_reg_test, y_cls_test)
    X_flat = flatten_sequences(X_seq)

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

    return results, df_test


def save_predictions_to_db(predictions, df_test):
    """Save predictions to the predictions table in DB."""
    engine = get_engine()

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

        # Use ensemble predictions
        # ... (implementation depends on ensemble results)

        conn.commit()

    print("  Predictions saved to DB")


def _make_loader(X_seq):
    """Create DataLoader from sequence data."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.FloatTensor(X_seq), torch.zeros(len(X_seq)), torch.zeros(len(X_seq)))
    return DataLoader(ds, batch_size=64, shuffle=False)
