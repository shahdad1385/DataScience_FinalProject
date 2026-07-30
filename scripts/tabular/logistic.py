"""
Logistic Regression for stock price direction prediction.

Classification only — linear baseline.

Note on convergence: this model receives the flattened sequence window
(SEQ_LEN * n_features ≈ 39k columns for ~1.5k samples), and the NLP/embedding
columns are not covered by the preprocessing scalers, so raw feature scales
differ by orders of magnitude. lbfgs cannot converge on that within any sane
iteration budget. The estimator is therefore wrapped in a StandardScaler and
uses a solver suited to the p >> n regime.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classify import evaluate_classification


def train(X_train, y_train, C=1.0, max_iter=2000, random_state=42, tol=1e-3):
    """Train a Logistic Regression classifier.

    Args:
        C: inverse regularization strength — smaller means stronger L2
           shrinkage. Kept at 1.0 by default, which is well suited here
           because features vastly outnumber samples.
        max_iter: iteration cap for the solver.
        tol: stopping tolerance. 1e-3 is ample for a direction baseline and
             avoids burning iterations chasing negligible loss changes.

    Returns:
        A fitted Pipeline (scaler + classifier). predict/predict_proba work
        exactly as on a bare LogisticRegression.
    """
    # Degenerate label case: a ticker with a single observed class in this
    # split would make lbfgs fail outright.
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError(
            f"Logistic regression needs both direction classes, got only {classes}. "
            "The training split is degenerate for this ticker."
        )

    model = Pipeline([
        # with_mean=True is safe: X_flat is dense. Zero-variance columns get
        # scale_=1.0 from sklearn, so they pass through instead of dividing by 0.
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            tol=tol,
            # lbfgs stalls when p >> n. saga handles wide problems and,
            # unlike lbfgs, actually benefits from the scaling above.
            solver="saga",
            # n_jobs is dropped: it has no effect on a binary fit for these
            # solvers and only added overhead across the 5 per-ticker fits.
        )),
    ])
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
