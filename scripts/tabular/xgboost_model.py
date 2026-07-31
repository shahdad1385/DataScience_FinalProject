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
                     n_estimators=300, max_depth=3, learning_rate=0.03,
                     subsample=0.7, colsample_bytree=0.5, min_child_weight=10,
                     reg_lambda=2.0, reg_alpha=0.5, early_stopping_rounds=40,
                     verbose=False):
    """Train XGBoost classifier for direction prediction.

    The previous defaults (depth 5, 40 rounds, lr 0.05, no weighting) overfit
    immediately on ~1,500 noisy rows: the run log shows validation logloss
    climbing from the very first round. Shallower trees trained longer at a lower
    learning rate, with subsampling on both rows and the ~300 feature columns,
    plus L1/L2 and min_child_weight, regularize toward the majority structure
    rather than memorizing noise. Class weighting mirrors the deep pos_weight so
    the classifier cannot collapse to always-up.
    """
    y = np.asarray(y_train)
    single = y.ndim == 1 or y.shape[1] == 1
    flat = y.ravel()
    neg, pos = float((flat == 0).sum()), float((flat == 1).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=subsample,
        colsample_bytree=colsample_bytree, min_child_weight=min_child_weight,
        reg_lambda=reg_lambda, reg_alpha=reg_alpha,
        scale_pos_weight=scale_pos_weight,
        random_state=42, n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
    )
    y_tr = flat if single else y
    if X_val is not None:
        yv = np.asarray(y_val).ravel() if single else np.asarray(y_val)
        model.fit(X_train, y_tr, eval_set=[(X_val, yv)], verbose=verbose)
    else:
        model.fit(X_train, y_tr)
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
