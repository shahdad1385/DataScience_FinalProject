"""
XGBoost model for stock price prediction.

Both regression (OHLC) and classification (direction).
Uses flattened features (no sequence).
"""

import numpy as np
import xgboost as xgb

from .regress import evaluate_regression
from .classify import evaluate_classification


def train_regressor(X_train, y_train, X_val=None, y_val=None,
                    n_estimators=300, max_depth=5, learning_rate=0.05,
                    early_stopping_rounds=20, verbose=False):
    """Train XGBoost regressor for OHLC prediction."""
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = xgb.XGBRegressor(**params)

    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=verbose,
    )
    return model


def train_classifier(X_train, y_train, X_val=None, y_val=None,
                     n_estimators=300, max_depth=5, learning_rate=0.05,
                     early_stopping_rounds=20, verbose=False):
    """Train XGBoost classifier for direction prediction."""
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    model = xgb.XGBClassifier(**params)

    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=verbose,
    )
    return model


def predict_regressor(model, X):
    """Predict OHLC values."""
    return model.predict(X)


def predict_classifier(model, X):
    """Predict direction and probabilities."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # probability of class 1 (up)
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
