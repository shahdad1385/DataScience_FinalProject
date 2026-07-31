"""
MLP (Multi-Layer Perceptron) neural network for stock price prediction.

Both regression (OHLC) and classification (direction).
Dense layers on flattened features — a simple neural network baseline.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from .regress import evaluate_regression
from .classify import evaluate_classification
from ..activations import get_activation, DEFAULT_ACTIVATION

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _build_stack(input_size, hidden_sizes, dropout, output_size, activation):
    """Linear/BatchNorm/activation/Dropout stack ending in a linear head."""
    layers = []
    prev_size = input_size
    for h in hidden_sizes:
        layers.extend([
            nn.Linear(prev_size, h),
            nn.BatchNorm1d(h),
            get_activation(activation),
            nn.Dropout(dropout),
        ])
        prev_size = h
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], dropout=0.2, output_size=20,
                 activation=DEFAULT_ACTIVATION):
        """
        Args:
            input_size: number of input features
            hidden_sizes: HIDDEN layer sizes only — must not include output_size
            dropout: dropout rate
            output_size: number of regression outputs (n_tickers * 4)
            activation: activation name (see scripts/activations.py)
        """
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.net = _build_stack(input_size, hidden_sizes, dropout, output_size, activation)
        # Record the exact build config so save_model never has to reverse
        # engineer it from the layer shapes.
        self.build_config = {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "output_size": output_size,
            "activation": activation,
        }

    def forward(self, x):
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], dropout=0.2, output_size=5,
                 activation=DEFAULT_ACTIVATION):
        """
        Args:
            input_size: number of input features
            hidden_sizes: HIDDEN layer sizes only — must not include output_size
            dropout: dropout rate
            output_size: number of classification outputs (n_tickers)
            activation: activation name (see scripts/activations.py)
        """
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.net = _build_stack(input_size, hidden_sizes, dropout, output_size, activation)
        self.build_config = {
            "input_size": input_size,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "output_size": output_size,
            "activation": activation,
        }

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def _train_loop(model, train_loader, val_loader, lr=1e-3, epochs=50,
                patience=30, use_amp=True, verbose=False):
    """Shared training loop for MLP models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            with autocast(enabled=use_amp and device.type == "cuda"):
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model.to("cpu"), best_val_loss


def train_regressor(X_train, y_train, X_val=None, y_val=None,
                    hidden_sizes=[256, 128], dropout=0.2, lr=1e-3,
                    epochs=50, batch_size=64, patience=30, verbose=False,
                    activation=DEFAULT_ACTIVATION):
    """Train MLP regressor for next-day return prediction."""
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MLPRegressor(input_size, hidden_sizes, dropout, output_size,
                         activation=activation)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if X_val is not None:
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size)
    else:
        val_loader = train_loader

    model, _ = _train_loop(model, train_loader, val_loader, lr, epochs, patience, verbose=verbose)
    return model


def train_classifier(X_train, y_train, X_val=None, y_val=None,
                     hidden_sizes=[256, 128], dropout=0.2, lr=1e-3,
                     epochs=50, batch_size=64, patience=30, verbose=False,
                     activation=DEFAULT_ACTIVATION):
    """Train MLP classifier for direction prediction."""
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MLPClassifier(input_size, hidden_sizes, dropout, output_size,
                          activation=activation)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    if X_val is not None:
        val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size)
    else:
        val_loader = train_loader

    model, _ = _train_loop(model, train_loader, val_loader, lr, epochs, patience, verbose=verbose)
    return model


def predict_regressor(model, X):
    """Predict OHLC values."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X).to(device)).cpu().numpy()


def predict_classifier(model, X):
    """Predict direction and probabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X).to(device)).cpu().numpy()
    y_pred = (probs > 0.5).astype(int)
    return y_pred, probs


def evaluate_regressor(model, X_test, y_test):
    """Evaluate regression model."""
    y_pred = predict_regressor(model, X_test)
    return evaluate_regression(y_test, y_pred)


def evaluate_classifier(model, X_test, y_test):
    """Evaluate classification model."""
    y_pred, y_prob = predict_classifier(model, X_test)
    return evaluate_classification(y_test, y_pred, y_prob)


def save_model(model, name, input_size=None, hidden_sizes=None, output_size=None,
               is_classifier=None):
    """Save a trained MLP using the config recorded at construction time.

    The explicit arguments are kept for backwards compatibility but are only
    fallbacks. Callers previously derived hidden_sizes from the layer shapes with
    `[m.out_features for m in model.net if hasattr(m, "out_features")]`, which
    also picks up BatchNorm layers and the final output layer — producing
    [256, 128, 20] instead of [256, 128]. Rebuilding from that list created four
    extra layers and load_state_dict failed on missing net.9.*/net.12.* keys.
    """
    cfg = dict(getattr(model, "build_config", {}) or {})
    if not cfg:
        cfg = {
            "input_size": input_size,
            "hidden_sizes": list(hidden_sizes or []),
            "output_size": output_size,
            "dropout": 0.2,
            "activation": DEFAULT_ACTIVATION,
        }
    if is_classifier is None:
        is_classifier = isinstance(model, MLPClassifier)

    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "build_config": cfg,
        # Flat copies kept so older readers still work.
        "input_size": cfg.get("input_size"),
        "hidden_sizes": cfg.get("hidden_sizes"),
        "output_size": cfg.get("output_size"),
        "is_classifier": bool(is_classifier),
    }, path)
    return path


def load_model(name):
    """Load a trained MLP, rebuilding from the stored build config."""
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No MLP checkpoint at {path}")
    ckpt = torch.load(path, map_location="cpu")

    cfg = ckpt.get("build_config") or {
        "input_size": ckpt.get("input_size"),
        "hidden_sizes": ckpt.get("hidden_sizes") or [],
        "output_size": ckpt.get("output_size"),
    }
    cls = MLPClassifier if ckpt.get("is_classifier") else MLPRegressor
    model = cls(
        cfg["input_size"],
        cfg.get("hidden_sizes") or [256, 128],
        dropout=cfg.get("dropout", 0.2),
        output_size=cfg["output_size"],
        activation=cfg.get("activation", DEFAULT_ACTIVATION),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model
