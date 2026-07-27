"""
Logistic Regression for stock price direction prediction.

Classification only — linear baseline.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .classify import evaluate_classification


def train(X_train, y_train, C=1.0, max_iter=1000, random_state=42):
    """Train Logistic Regression classifier."""
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    """Predict direction and probabilities."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return y_pred, y_prob


def evaluate(model, X_test, y_test):
    """Evaluate classification model."""
    y_pred, y_prob = predict(model, X_test)
    return evaluate_classification(y_test, y_pred, y_prob)
