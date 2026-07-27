"""
Shared classification utilities — metrics, evaluation.

Used by all tabular classification models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Full classification evaluation.
    Returns dict with accuracy, precision, recall, F1, AUC-ROC.
    """
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        try:
            results["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            results["auc_roc"] = 0.0
    return results


def evaluate_per_ticker(y_true, y_pred, y_prob=None, ticker_names=None):
    """
    Evaluate classification per ticker.
    y_true, y_pred: shape (n_samples, n_tickers)
    Returns dict with metrics for each ticker.
    """
    n_tickers = y_true.shape[1]
    if ticker_names is None:
        ticker_names = [f"ticker_{i}" for i in range(n_tickers)]

    results = {}
    for i, name in enumerate(ticker_names):
        results[name] = evaluate_classification(
            y_true[:, i], y_pred[:, i],
            y_prob[:, i] if y_prob is not None else None,
        )
    return results


def confidence_from_probs(y_prob):
    """
    Convert sigmoid probabilities to confidence scores.
    confidence = |probability - 0.5| * 2 (ranges 0 to 1)
    """
    return np.abs(y_prob - 0.5) * 2
