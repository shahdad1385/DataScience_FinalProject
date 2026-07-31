"""
Logistic Regression for stock price direction prediction.

Classification only — linear baseline.

Note on dimensionality: this model receives the flattened sequence window
(SEQ_LEN * n_features ~= 39k columns) against only ~1.5k training samples. A
linear fit in that regime is both statistically meaningless and extremely slow —
it is what made a full pipeline run hang for hours in the `saga`/`lbfgs` solvers.

The estimator therefore reduces dimensionality before fitting:
  StandardScaler -> TruncatedSVD -> LogisticRegression
TruncatedSVD (rather than PCA) keeps the transform cheap on a wide dense matrix
and needs no covariance estimate. The projection is fitted on train only and
carried inside the Pipeline, so predict/predict_proba apply it automatically and
no leakage is possible.
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .classify import evaluate_classification

# Components retained from the flattened window. 64 keeps the fit fast (seconds
# per ticker) while preserving the dominant directions of variation.
DEFAULT_N_COMPONENTS = 64


def train(X_train, y_train, C=1.0, max_iter=1000, random_state=42, tol=1e-4,
          n_components=DEFAULT_N_COMPONENTS):
    """Train a Logistic Regression classifier on an SVD-reduced feature space.

    Args:
        C: inverse regularization strength — smaller means stronger L2 shrinkage.
        max_iter: iteration cap for the solver. Ample post-reduction.
        tol: stopping tolerance.
        n_components: SVD rank. Clipped to the data's actual limits.

    Returns:
        A fitted Pipeline. predict/predict_proba work as on a bare estimator.
    """
    X_train = np.asarray(X_train)

    # Degenerate label case: a ticker with a single observed class cannot be fit.
    classes = np.unique(y_train)
    if len(classes) < 2:
        raise ValueError(
            f"Logistic regression needs both direction classes, got only {classes}. "
            "The training split is degenerate for this ticker."
        )

    n_samples, n_features = X_train.shape
    # TruncatedSVD requires n_components < n_features, and more than
    # n_samples components carries no information.
    max_rank = max(1, min(n_features - 1, n_samples - 1))
    k = int(min(n_components, max_rank))

    steps = [
        # with_mean=True is safe: X_flat is dense. Zero-variance columns get
        # scale_=1.0 from sklearn, so they pass through instead of dividing by 0.
        ("scaler", StandardScaler()),
    ]
    if k < n_features:
        steps.append(("svd", TruncatedSVD(n_components=k, random_state=random_state)))
    steps.append((
        "clf",
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            tol=tol,
            # Balanced weighting keeps a rising period from being learned as
            # always-up, which is what the unweighted classifier collapsed to.
            class_weight="balanced",
            # After reduction the problem is well-conditioned and n > p, so
            # lbfgs converges quickly and reliably.
            solver="lbfgs",
        ),
    ))

    model = Pipeline(steps)
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
