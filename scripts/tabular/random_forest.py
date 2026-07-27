"""
Random Forest model for stock price prediction.

Both regression (OHLC) and classification (direction).
Robust baseline, less prone to overfitting.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .regress import evaluate_regression
from .classify import evaluate_classification


def train_regressor(X_train, y_train, n_estimators=500, max_depth=10,
                    min_samples_split=5, random_state=42):
    """Train Random Forest regressor for OHLC prediction."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_classifier(X_train, y_train, n_estimators=500, max_depth=10,
                     min_samples_split=5, random_state=42):
    """Train Random Forest classifier for direction prediction."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
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
