"""
LightGBM model for stock price prediction.

Both regression (OHLC) and classification (direction).
Faster than XGBoost, similar quality.
"""

import numpy as np

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from .regress import evaluate_regression
from .classify import evaluate_classification, normalize_binary_outputs


def train_regressor(X_train, y_train, X_val=None, y_val=None,
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                    num_leaves=31, early_stopping_rounds=20, verbose=-1):
    """Train LightGBM regressor for OHLC prediction.

    Trains one model per target column (LightGBM doesn't support multi-output).
    Returns a list of models (one per column).
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    n_targets = y_train.shape[1]
    if n_targets == 1:
        y_train = y_train.ravel()
        if y_val is not None:
            y_val = y_val.ravel()

    models = []
    for col in range(n_targets):
        y_col = y_train[:, col] if n_targets > 1 else y_train
        y_val_col = y_val[:, col] if (y_val is not None and n_targets > 1) else y_val

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": verbose,
        }
        model = lgb.LGBMRegressor(**params)

        fit_params = {}
        if X_val is not None:
            fit_params["eval_set"] = [(X_val, y_val_col)]
            fit_params["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model.fit(X_train, y_col, **fit_params)
        models.append(model)

    return models


def train_classifier(X_train, y_train, X_val=None, y_val=None,
                     n_estimators=300, max_depth=3, learning_rate=0.03,
                     num_leaves=15, subsample=0.7, colsample_bytree=0.5,
                     min_child_samples=30, reg_lambda=2.0, reg_alpha=0.5,
                     early_stopping_rounds=40, verbose=-1):
    """Train LightGBM classifier for direction prediction.

    Trains one model per target column (LightGBM doesn't support multi-output).
    Returns a list of models (one per column).

    Defaults are regularized for ~1,500 noisy rows: shallow trees, fewer leaves,
    heavier min_child_samples, row+column subsampling, and L1/L2, otherwise the
    model memorizes noise within the first few iterations. class_weight balances
    up/down days so the model cannot collapse to always-up (which the run showed
    scoring ~54% while learning nothing).
    """
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    n_targets = y_train.shape[1]
    if n_targets == 1:
        y_train = y_train.ravel()
        if y_val is not None:
            y_val = y_val.ravel()

    models = []
    for col in range(n_targets):
        y_col = y_train[:, col] if n_targets > 1 else y_train
        y_val_col = y_val[:, col] if (y_val is not None and n_targets > 1) else y_val

        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_samples": min_child_samples,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": verbose,
            "objective": "binary",
            "eval_metric": "logloss",
        }
        model = lgb.LGBMClassifier(**params)

        fit_params = {}
        if X_val is not None:
            fit_params["eval_set"] = [(X_val, y_val_col)]
            fit_params["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model.fit(X_train, y_col, **fit_params)
        models.append(model)

    return models


def predict_regressor(model, X):
    """Predict OHLC values. model is a list of per-column models."""
    if isinstance(model, list):
        preds = [m.predict(X).reshape(-1, 1) for m in model]
        return np.hstack(preds)
    return model.predict(X)


def predict_classifier(model, X):
    """Predict direction and probabilities, always shaped (n_samples, n_outputs).

    LightGBM is trained as one model per target column, so the list branch
    already produced 2-D output. The single-model branch did not, which would
    break the [:, i] indexing every consumer relies on.
    """
    if isinstance(model, list):
        preds, probs = [], []
        for m in model:
            p = np.asarray(m.predict(X)).reshape(-1, 1)
            pr = np.asarray(m.predict_proba(X))
            pr = pr[:, [1]] if pr.ndim == 2 and pr.shape[1] > 1 else pr.reshape(-1, 1)
            preds.append(p)
            probs.append(pr)
        return np.hstack(preds).astype(int), np.hstack(probs)

    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    return normalize_binary_outputs(y_pred, proba)


def evaluate_regressor(model, X_test, y_test):
    """Evaluate regression model."""
    y_pred = predict_regressor(model, X_test)
    return evaluate_regression(y_test, y_pred)


def evaluate_classifier(model, X_test, y_test):
    """Evaluate classification model."""
    y_pred, y_prob = predict_classifier(model, X_test)
    return evaluate_classification(y_test, y_pred, y_prob)


def feature_importance(model, feature_names=None):
    """Get feature importance scores."""
    importance = model.feature_importances_
    if feature_names is not None:
        return dict(zip(feature_names, importance))
    return importance
