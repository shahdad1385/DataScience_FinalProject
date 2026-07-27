"""
Ridge Regression for stock price OHLC prediction.

Regression only — linear baseline with L2 regularization.
"""

import numpy as np
from sklearn.linear_model import Ridge

from .regress import evaluate_regression


def train(X_train, y_train, alpha=1.0):
    """Train Ridge regressor."""
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    """Predict OHLC values."""
    return model.predict(X)


def evaluate(model, X_test, y_test):
    """Evaluate regression model."""
    y_pred = predict(model, X_test)
    return evaluate_regression(y_test, y_pred)
