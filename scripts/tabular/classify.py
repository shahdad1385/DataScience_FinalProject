"""
Shared classification utilities — metrics, evaluation.

Used by all tabular classification models.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def evaluate_classification(y_true, y_pred, y_prob=None):
    """
    Full classification evaluation.

    For multi-output (multi-label) input, `accuracy` is reported as the mean
    per-ticker accuracy — the directional hit rate, which is the metric that
    actually matters here.

    sklearn's accuracy_score on a 2D array computes SUBSET accuracy: a row counts
    as correct only if all 5 tickers are simultaneously right. That made a model
    with ~69% per-ticker skill report 0.12 accuracy next to an F1 of 0.66, which
    reads as "worse than chance" when random subset accuracy is only 0.5^5 =
    0.03. Both are still reported, under unambiguous names.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    results = {
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    if y_true.ndim > 1 and y_true.shape[1] > 1:
        per_output = [
            accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])
        ]
        results["accuracy"] = float(np.mean(per_output))       # directional hit rate
        results["accuracy_per_output"] = [float(a) for a in per_output]
        results["subset_accuracy"] = float(accuracy_score(y_true, y_pred))
        results["hamming_accuracy"] = float((y_true == y_pred).mean())
    else:
        results["accuracy"] = float(accuracy_score(y_true, y_pred))

    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if y_true.ndim > 1 and y_true.shape[1] > 1:
                # Average per-output AUC; a column with a single observed class
                # has no defined AUC and is skipped rather than scored as 0.
                aucs = []
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) < 2:
                        continue
                    aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
                results["auc_roc"] = float(np.mean(aucs)) if aucs else 0.0
            else:
                results["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except (ValueError, IndexError):
            results["auc_roc"] = 0.0
    return results


def evaluate_per_ticker(y_true, y_pred, y_prob=None, ticker_names=None):
    """
    Evaluate classification per ticker.
    y_true, y_pred: shape (n_samples, n_tickers)
    Returns dict with metrics for each ticker.

    Single-target (n_tickers == 1) inputs legitimately arrive 1-D from some
    classifiers; coerce everything to 2-D so the [:, i] indexing below never
    sees a 1-dimensional array (that was the IndexError on random_forest).
    """
    y_true = np.asarray(y_true)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        if y_prob.ndim == 1:
            y_prob = y_prob.reshape(-1, 1)

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


def normalize_binary_outputs(y_pred, proba, n_outputs=None):
    """Force classifier outputs to (n_samples, n_outputs).

    Single-target models break the 2-D contract the rest of the pipeline relies
    on. With y of shape (n, 1), scikit-learn and XGBoost treat the problem as
    ordinary binary classification, so:
      - predict() returns a 1-D (n,) array, not (n, 1)
      - predict_proba() returns (n, 2) CLASS probabilities [P(down), P(up)],
        which must not be confused with one column per output

    Every consumer here — evaluate_per_ticker, the per-ticker metric loops, the
    stored cls_pred/cls_prob arrays and the DB writer — indexes [:, i]. Passing a
    1-D array through made `y_pred.shape[1]` raise IndexError, which is what
    aborted training right after the XGBoost classifier finished fitting.

    Returns (y_pred_2d, y_prob_2d), where y_prob holds P(up) per output.
    """
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if n_outputs is None:
        n_outputs = y_pred.shape[1]

    # Multi-output estimators (e.g. RandomForest) hand back one array per output.
    if isinstance(proba, (list, tuple)):
        cols = []
        for p in proba:
            p = np.asarray(p)
            # A degenerate output saw a single class in training, so there is no
            # positive-class column to read.
            cols.append(p[:, 1] if p.ndim == 2 and p.shape[1] > 1 else p.ravel())
        y_prob = np.column_stack(cols)
    else:
        proba = np.asarray(proba)
        if proba.ndim == 3:
            # (n_outputs, n_samples, n_classes) or (n_samples, n_outputs, n_classes)
            y_prob = proba[:, :, 1] if proba.shape[0] == y_pred.shape[0] else proba[:, :, 1].T
        elif proba.ndim == 1:
            y_prob = proba.reshape(-1, 1)
        elif n_outputs == 1 and proba.shape[1] == 2:
            # Binary two-column output: keep P(positive) only. Taking the whole
            # array here would silently feed P(down) in as a second "ticker".
            y_prob = proba[:, [1]]
        elif proba.shape[1] == n_outputs:
            y_prob = proba
        else:
            y_prob = proba[:, [1]] if proba.shape[1] > 1 else proba

    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        y_prob = y_prob.reshape(-1, 1)
    return y_pred, y_prob
