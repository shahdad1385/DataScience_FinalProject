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
from .data_assembly import (
    prepare_data, flatten_sequences, TICKERS, SEQ_LEN, REF_CLOSE_SUFFIX,
)

OHLC = ["open", "high", "low", "close"]


def load_models(n_features=None):
    """Load all trained models from disk.

    Args:
        n_features: if provided, skip models whose saved input dimension
            doesn't match this count (stale weights from a different schema).
    """
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")

    models = {}

    # Time series.
    #
    # Loading goes through timeseries.train.load_model so the architecture is
    # rebuilt from the kwargs stored in the checkpoint. Calling
    # create_model(input_size, n_tickers=5) directly, as before, silently used
    # the default hidden_size=128 even when the model was trained with 64, so
    # load_state_dict failed with a size mismatch on every run.
    from ..timeseries import lstm, gru, transformer, bilstm, tcn
    from ..timeseries.train import load_model as load_ts_model
    for name, mod in [("lstm", lstm), ("gru", gru), ("transformer", transformer),
                      ("bilstm", bilstm), ("tcn", tcn)]:
        path = os.path.join(models_dir, f"timeseries_{name}.pt")
        if os.path.exists(path):
            model = load_ts_model(
                mod.create_model, f"timeseries_{name}",
                default_kwargs={"n_tickers": len(TICKERS)},
            )
            if n_features is not None:
                saved_n = None
                for p in model.parameters():
                    saved_n = p.shape[-1]
                    break
                if saved_n is not None and saved_n != n_features:
                    print(f"  Skipping ts_{name}: input_size mismatch ({saved_n} vs {n_features})")
                    continue
            models[f"ts_{name}"] = model

    # Tabular
    for name in ["xgboost", "random_forest", "lightgbm"]:
        for task in ["reg", "cls"]:
            path = os.path.join(models_dir, f"tabular_{name}_{task}.pkl")
            if os.path.exists(path):
                with open(path, "rb") as f:
                    m = pickle.load(f)
                if n_features is not None:
                    saved_n = getattr(m, "n_features_in_", None)
                    if saved_n is not None and saved_n != n_features:
                        print(f"  Skipping tab_{name}_{task}: input_size mismatch ({saved_n} vs {n_features})")
                        continue
                models[f"tab_{name}_{task}"] = m

    # MLP.
    #
    # Rebuilt via mlp.load_model, which reads the stored build_config. Building
    # it here from ckpt["hidden_sizes"] was the direct cause of the
    # "Missing key(s) net.9.*/net.12.*" crash, because that list included the
    # output layer and produced a deeper network than the one that was trained.
    from ..tabular import mlp
    for task in ["reg", "cls"]:
        path = os.path.join(models_dir, f"tabular_mlp_{task}.pt")
        if os.path.exists(path):
            m = mlp.load_model(f"tabular_mlp_{task}")
            if n_features is not None:
                cfg = getattr(m, "build_config", None) or {}
                saved_n = cfg.get("input_size") or getattr(m, "input_size", None)
                if saved_n is not None and saved_n != n_features:
                    print(f"  Skipping tab_mlp_{task}: input_size mismatch ({saved_n} vs {n_features})")
                    continue
            models[f"tab_mlp_{task}"] = m

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

    # Seed the sequence windows with the tail of the validation split so every
    # test date is predicted, including the first 30. Each window still ends the
    # day before its label, so no future information reaches the model.
    from .data_assembly import (create_sequences_with_context, load_feature_scaler,
                                apply_feature_scaler)
    X_val, _, _, _, _ = prepare_data("val", feature_cols=feature_names)

    # Same train-fitted scaler used during training, or predictions are garbage.
    scaler = load_feature_scaler()
    if scaler is not None:
        X_test = apply_feature_scaler(X_test, scaler)
        X_val = apply_feature_scaler(X_val, scaler)

    # y_reg_test stays in REAL return units here: model outputs are inverse
    # transformed below, so actuals must not be scaled.
    X_seq, y_reg_seq, y_cls_seq, label_index = create_sequences_with_context(
        X_test, y_reg_test, y_cls_test, context=X_val)
    X_flat = flatten_sequences(X_seq)

    if len(X_seq) == 0:
        print("  No test sequences available (test split shorter than SEQ_LEN)")
        return None

    # Rows of df_test that each sequence label corresponds to.
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


def to_real_prices(return_matrix, df_labels):
    """Convert predicted next-day RETURNS into dollar prices.

    The models now predict returns relative to today's close, so a price is
    recovered as ref_close * (1 + predicted_return). ref_close is today's real
    (unscaled) close, carried alongside the targets as {TICKER}_refclose.

    This replaces the old inverse-scaler path, which applied only when the
    targets were scaled absolute price levels.
    """
    if return_matrix is None:
        return None
    return_matrix = np.asarray(return_matrix, dtype=float)
    out = np.full_like(return_matrix, np.nan)

    for t_i, ticker in enumerate(TICKERS):
        ref_col = f"{ticker}_{REF_CLOSE_SUFFIX}"
        if ref_col not in df_labels.columns:
            continue
        ref = pd.to_numeric(df_labels[ref_col], errors="coerce").values
        n = min(len(ref), return_matrix.shape[0])
        for c_i in range(len(OHLC)):
            j = t_i * 4 + c_i
            if j < return_matrix.shape[1]:
                out[:n, j] = ref[:n] * (1.0 + return_matrix[:n, j])
    return out


def save_predictions_to_db(predictions, df_test):
    """Save ensemble predictions to the predictions table in DB.

    One row per (date, ticker) with predicted and actual OHLC in real dollars,
    plus predicted direction and its confidence.
    """
    engine = get_engine()

    reg, cls, prob = build_ensemble(predictions)
    # Models were trained on standardised returns, so their outputs must be
    # mapped back to real return units before any price reconstruction.
    from .data_assembly import inverse_target_scale, load_target_scaler
    target_scaler = load_target_scaler()
    if target_scaler is not None and reg is not None:
        reg = inverse_target_scale(reg, target_scaler)
    actuals = predictions.get("_actuals", {})
    y_reg = actuals.get("reg")
    y_cls = actuals.get("cls")

    if reg is None and cls is None:
        print("  No predictions to save")
        return

    n_rows = len(reg) if reg is not None else len(cls)

    # Predicted/actual returns -> dollar prices via today's real close.
    reg_real = to_real_prices(reg, df_test)
    actual_real = to_real_prices(y_reg, df_test)

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
                    float(reg_real[i, j]) if reg_real is not None and j < reg_real.shape[1]
                    and np.isfinite(reg_real[i, j]) else None
                )
                row[f"actual_{col}"] = (
                    float(actual_real[i, j]) if actual_real is not None and j < actual_real.shape[1]
                    and np.isfinite(actual_real[i, j]) else None
                )
                # Keep the raw model output too: returns are what was actually
                # predicted, prices are a reconstruction on top of them.
                row[f"predicted_return_{col}"] = (
                    float(reg[i, j]) if reg is not None and j < reg.shape[1] else None
                )
                row[f"actual_return_{col}"] = (
                    float(y_reg[i, j]) if y_reg is not None and j < y_reg.shape[1] else None
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
                predicted_return_open FLOAT,
                predicted_return_high FLOAT,
                predicted_return_low FLOAT,
                predicted_return_close FLOAT,
                direction INTEGER,
                confidence FLOAT,
                model_name TEXT,
                actual_open FLOAT,
                actual_high FLOAT,
                actual_low FLOAT,
                actual_close FLOAT,
                actual_return_open FLOAT,
                actual_return_high FLOAT,
                actual_return_low FLOAT,
                actual_return_close FLOAT,
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
