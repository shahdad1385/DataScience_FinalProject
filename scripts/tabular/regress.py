"""
Shared regression utilities — metrics, evaluation, plotting.

Used by all tabular regression models.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100


def evaluate_regression(y_true, y_pred):
    """
    Full regression evaluation.
    Returns dict with MAE, MSE, RMSE, R², MAPE, sMAPE.
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }


def evaluate_per_output(y_true, y_pred, output_names=None):
    """
    Evaluate regression per output (e.g., per OHLC per ticker).
    Returns dict with metrics for each output.
    """
    if output_names is None:
        output_names = [f"output_{i}" for i in range(y_true.shape[1])]

    results = {}
    for i, name in enumerate(output_names):
        results[name] = evaluate_regression(y_true[:, i], y_pred[:, i])
    return results
