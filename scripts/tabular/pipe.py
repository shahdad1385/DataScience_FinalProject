"""
Tabular Models Orchestrator — combines all models.

Runs XGBoost, Random Forest, LightGBM, MLP, Logistic Regression,
and Ridge Regression on the same data, compares them, and returns the best.

This module is called by the pipeline after NLP, clustering, and time series
features are ready (flattened features).
"""

import numpy as np

from . import xgboost_model
from . import random_forest
from . import lightgbm_model
from . import mlp
from . import logistic
from . import ridge
from .regress import evaluate_regression, evaluate_per_output
from .classify import evaluate_classification, evaluate_per_ticker, confidence_from_probs

TICKERS = ["NVDA", "GOOGL", "AVGO", "AMD", "TSM"]
OHLC_COLS = ["open", "high", "low", "close"]


def get_ohlc_output_names(n_tickers=5):
    """Generate output names like NVDA_open, NVDA_high, etc."""
    names = []
    for t in TICKERS[:n_tickers]:
        for c in OHLC_COLS:
            names.append(f"{t}_{c}")
    return names


def train_all_regression(X_train, y_reg_train, X_val, y_reg_val,
                         feature_names=None, verbose=False,
                         model_filter=None, hyperparams=None):
    """
    Train all regression models and compare.
    y_reg_train/val: shape (n_samples, n_tickers * 4) — OHLC for each ticker.

    Args:
        model_filter: list of model names to train (default: all)
        hyperparams: dict of best hyperparameters from finetune

    Returns:
        dict with {model_name: (model, metrics)}
    """
    hyperparams = hyperparams or {}
    output_names = get_ohlc_output_names(y_reg_train.shape[1] // 4)
    results = {}
    all_models = ["xgboost", "random_forest", "lightgbm", "mlp", "ridge"]
    models_to_train = model_filter if model_filter else all_models

    # XGBoost
    if "xgboost" in models_to_train:
        hp = hyperparams.get("tab_reg_xgboost", {})
        print(f"  XGBoost Regression (params: {hp})")
        xgb_reg = xgboost_model.train_regressor(
            X_train, y_reg_train, X_val, y_reg_val, verbose=verbose, **hp,
        )
        xgb_pred = xgboost_model.predict_regressor(xgb_reg, X_val)
        xgb_metrics = evaluate_per_output(y_reg_val, xgb_pred, output_names)
        xgb_overall = evaluate_regression(y_reg_val, xgb_pred)
        results["xgboost"] = {"model": xgb_reg, "metrics": xgb_metrics, "overall": xgb_overall}

    # Random Forest
    if "random_forest" in models_to_train:
        hp = hyperparams.get("tab_reg_random_forest", {})
        print(f"  Random Forest Regression (params: {hp})")
        rf_reg = random_forest.train_regressor(X_train, y_reg_train, **hp)
        rf_pred = random_forest.predict_regressor(rf_reg, X_val)
        rf_metrics = evaluate_per_output(y_reg_val, rf_pred, output_names)
        rf_overall = evaluate_regression(y_reg_val, rf_pred)
        results["random_forest"] = {"model": rf_reg, "metrics": rf_metrics, "overall": rf_overall}

    # LightGBM
    if "lightgbm" in models_to_train and lightgbm_model.HAS_LIGHTGBM:
        hp = hyperparams.get("tab_reg_lightgbm", {})
        print(f"  LightGBM Regression (params: {hp})")
        lgb_reg = lightgbm_model.train_regressor(X_train, y_reg_train, X_val, y_reg_val, **hp)
        lgb_pred = lightgbm_model.predict_regressor(lgb_reg, X_val)
        lgb_metrics = evaluate_per_output(y_reg_val, lgb_pred, output_names)
        lgb_overall = evaluate_regression(y_reg_val, lgb_pred)
        results["lightgbm"] = {"model": lgb_reg, "metrics": lgb_metrics, "overall": lgb_overall}

    # MLP
    if "mlp" in models_to_train:
        hp = hyperparams.get("tab_reg_mlp", {})
        print(f"  MLP Regression (params: {hp})")
        mlp_reg = mlp.train_regressor(X_train, y_reg_train, X_val, y_reg_val, verbose=verbose, **hp)
        mlp_pred = mlp.predict_regressor(mlp_reg, X_val)
        mlp_metrics = evaluate_per_output(y_reg_val, mlp_pred, output_names)
        mlp_overall = evaluate_regression(y_reg_val, mlp_pred)
        results["mlp"] = {"model": mlp_reg, "metrics": mlp_metrics, "overall": mlp_overall}

    # Ridge
    if "ridge" in models_to_train:
        hp = hyperparams.get("tab_reg_ridge", {})
        print(f"  Ridge Regression (params: {hp})")
        ridge_reg = ridge.train(X_train, y_reg_train, **hp)
        ridge_pred = ridge.predict(ridge_reg, X_val)
        ridge_metrics = evaluate_per_output(y_reg_val, ridge_pred, output_names)
        ridge_overall = evaluate_regression(y_reg_val, ridge_pred)
        results["ridge"] = {"model": ridge_reg, "metrics": ridge_metrics, "overall": ridge_overall}

    if not results:
        return results

    best_name = min(results, key=lambda k: results[k]["overall"]["rmse"])
    results["_best"] = best_name

    return results


def train_all_classification(X_train, y_cls_train, X_val, y_cls_val,
                             feature_names=None, verbose=False,
                             model_filter=None, hyperparams=None):
    """
    Train all classification models and compare.
    y_cls_train/val: shape (n_samples, n_tickers) — direction for each ticker.

    Args:
        model_filter: list of model names to train (default: all)
        hyperparams: dict of best hyperparameters from finetune

    Returns:
        dict with {model_name: (model, metrics)}
    """
    hyperparams = hyperparams or {}
    n_tickers = y_cls_train.shape[1]
    results = {}
    all_models = ["xgboost", "random_forest", "lightgbm", "mlp", "logistic"]
    models_to_train = model_filter if model_filter else all_models

    # XGBoost
    if "xgboost" in models_to_train:
        hp = hyperparams.get("tab_cls_xgboost", {})
        print(f"  XGBoost Classification (params: {hp})")
        xgb_cls = xgboost_model.train_classifier(X_train, y_cls_train, X_val, y_cls_val, verbose=verbose, **hp)
        xgb_pred, xgb_prob = xgboost_model.predict_classifier(xgb_cls, X_val)
        xgb_metrics = evaluate_per_ticker(y_cls_val, xgb_pred, xgb_prob, TICKERS[:n_tickers])
        xgb_overall = evaluate_classification(y_cls_val, xgb_pred, xgb_prob)
        results["xgboost"] = {"model": xgb_cls, "metrics": xgb_metrics, "overall": xgb_overall}

    # Random Forest
    if "random_forest" in models_to_train:
        hp = hyperparams.get("tab_cls_random_forest", {})
        print(f"  Random Forest Classification (params: {hp})")
        rf_cls = random_forest.train_classifier(X_train, y_cls_train, **hp)
        rf_pred, rf_prob = random_forest.predict_classifier(rf_cls, X_val)
        rf_metrics = evaluate_per_ticker(y_cls_val, rf_pred, rf_prob, TICKERS[:n_tickers])
        rf_overall = evaluate_classification(y_cls_val, rf_pred, rf_prob)
        results["random_forest"] = {"model": rf_cls, "metrics": rf_metrics, "overall": rf_overall}

    # LightGBM
    if "lightgbm" in models_to_train and lightgbm_model.HAS_LIGHTGBM:
        hp = hyperparams.get("tab_cls_lightgbm", {})
        print(f"  LightGBM Classification (params: {hp})")
        lgb_cls = lightgbm_model.train_classifier(X_train, y_cls_train, X_val, y_cls_val, **hp)
        lgb_pred, lgb_prob = lightgbm_model.predict_classifier(lgb_cls, X_val)
        lgb_metrics = evaluate_per_ticker(y_cls_val, lgb_pred, lgb_prob, TICKERS[:n_tickers])
        lgb_overall = evaluate_classification(y_cls_val, lgb_pred, lgb_prob)
        results["lightgbm"] = {"model": lgb_cls, "metrics": lgb_metrics, "overall": lgb_overall}

    # MLP
    if "mlp" in models_to_train:
        hp = hyperparams.get("tab_cls_mlp", {})
        print(f"  MLP Classification (params: {hp})")
        mlp_cls = mlp.train_classifier(X_train, y_cls_train, X_val, y_cls_val, verbose=verbose, **hp)
        mlp_pred, mlp_prob = mlp.predict_classifier(mlp_cls, X_val)
        mlp_metrics = evaluate_per_ticker(y_cls_val, mlp_pred, mlp_prob, TICKERS[:n_tickers])
        mlp_overall = evaluate_classification(y_cls_val, mlp_pred, mlp_prob)
        results["mlp"] = {"model": mlp_cls, "metrics": mlp_metrics, "overall": mlp_overall}

    # Logistic Regression
    if "logistic" in models_to_train:
        hp = hyperparams.get("tab_cls_logistic", {})
        C = hp.get("C", 1.0)
        print(f"  Logistic Regression (C={C})")
        lr_models = {}
        lr_preds = np.zeros_like(y_cls_val)
        lr_probs = np.zeros_like(y_cls_val, dtype=float)
        for t in range(n_tickers):
            lr_model = logistic.train(X_train, y_cls_train[:, t], C=C)
            lr_models[t] = lr_model
            lr_preds[:, t], lr_probs[:, t] = logistic.predict(lr_model, X_val)
        lr_metrics = evaluate_per_ticker(y_cls_val, lr_preds, lr_probs, TICKERS[:n_tickers])
        lr_overall = evaluate_classification(y_cls_val, lr_preds, lr_probs)
        results["logistic"] = {"model": lr_models, "metrics": lr_metrics, "overall": lr_overall}

    if not results:
        return results

    best_name = max(results, key=lambda k: results[k]["overall"].get("f1", 0))
    results["_best"] = best_name

    return results


def predict_regression(model_name, model, X):
    """Make regression predictions using the specified model."""
    if model_name == "mlp":
        return mlp.predict_regressor(model, X)
    elif model_name in ("xgboost", "lightgbm", "random_forest"):
        return {"xgboost": xgboost_model, "lightgbm": lightgbm_model, "random_forest": random_forest}[model_name].predict_regressor(model, X)
    elif model_name == "ridge":
        return ridge.predict(model, X)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_classification(model_name, model, X):
    """Make classification predictions using the specified model."""
    if model_name == "mlp":
        return mlp.predict_classifier(model, X)
    elif model_name in ("xgboost", "lightgbm", "random_forest"):
        return {"xgboost": xgboost_model, "lightgbm": lightgbm_model, "random_forest": random_forest}[model_name].predict_classifier(model, X)
    elif model_name == "logistic":
        n_tickers = len(model)
        y_pred = np.zeros((X.shape[0], n_tickers))
        y_prob = np.zeros((X.shape[0], n_tickers))
        for t, lr_model in model.items():
            y_pred[:, t], y_prob[:, t] = logistic.predict(lr_model, X)
        return y_pred, y_prob
    else:
        raise ValueError(f"Unknown model: {model_name}")
