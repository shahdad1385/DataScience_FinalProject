"""
Random Forest model for stock price prediction.

Both regression (OHLC) and classification (direction).
Robust baseline, less prone to overfitting.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .regress import evaluate_regression
from .classify import evaluate_classification, normalize_binary_outputs


def train_regressor(X_train, y_train, n_estimators=50, max_depth=8,
                    min_samples_split=5, random_state=42, n_jobs=-1):
    """Train Random Forest regressor for OHLC prediction."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def train_classifier(X_train, y_train, n_estimators=50, max_depth=8,
                     min_samples_split=5, random_state=42, n_jobs=-1):
    """Train Random Forest classifier for direction prediction."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train)
    return model


def predict_regressor(model, X):
    """Predict OHLC values."""
    return model.predict(X)


def predict_classifier(model, X):
    """Predict direction and probabilities.

    Handles both single-output and multi-output classifiers. For a
    multi-output RandomForestClassifier, predict_proba returns a LIST of
    (n_samples, n_classes) arrays — one per output — so we take the
    positive-class column of each and stack them into (n_samples, n_outputs).
    """
    y_pred = model.predict(X)
    proba = model.predict_proba(X)
    # Shared normaliser: also collapses the single-target case, where predict()
    # returns 1-D and predict_proba() returns (n, 2) class probabilities rather
    # than one column per output.
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
