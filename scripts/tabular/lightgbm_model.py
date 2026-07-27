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
from .classify import evaluate_classification


def train_regressor(X_train, y_train, X_val=None, y_val=None,
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    num_leaves=31, verbose=-1):
    """Train LightGBM regressor for OHLC prediction."""
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

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

    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
    )
    return model


def train_classifier(X_train, y_train, X_val=None, y_val=None,
                     n_estimators=300, max_depth=5, learning_rate=0.05,
                     num_leaves=31, verbose=-1):
    """Train LightGBM classifier for direction prediction."""
    if not HAS_LIGHTGBM:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

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
    model = lgb.LGBMClassifier(**params)

    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
    )
    return model


def predict_regressor(model, X):
    """Predict OHLC values."""
    return model.predict(X)


def predict_classifier(model, X):
    """Predict direction and probabilities."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


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
