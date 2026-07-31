"""
XGBoost model for stock price prediction.

Both regression (OHLC) and classification (direction).
Uses flattened features (no sequence).
"""

import numpy as np
import xgboost as xgb

from .regress import evaluate_regression
from .classify import evaluate_classification, normalize_binary_outputs


def train_regressor(X_train, y_train, X_val=None, y_val=None,
                    n_estimators=40, max_depth=5, learning_rate=0.05,
                    early_stopping_rounds=20, verbose=False):
    """Train XGBoost regressor for OHLC prediction."""
    model = xgb.XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
    )
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)
    return model


def train_classifier(X_train, y_train, X_val=None, y_val=None,
                     n_estimators=40, max_depth=5, learning_rate=0.05,
                     early_stopping_rounds=20, verbose=False):
    """Train XGBoost classifier for direction prediction."""
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1,
        eval_metric="logloss",
    )
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    else:
        model.fit(X_train, y_train)
    return model


def predict_regressor(model, X):
    """Predict OHLC values."""
    return model.predict(X)


def predict_classifier(model, X):
    """Predict direction and probabilities, always shaped (n_samples, n_outputs).

    The previous shape logic read `y_pred.shape[1]`, which does not exist when
    there is a single target: XGBoost then treats the task as plain binary
    classification and returns a 1-D prediction array. That raised IndexError and
    aborted training right after the classifier finished fitting.
    """
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
