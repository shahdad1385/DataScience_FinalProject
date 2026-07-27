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

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class MLPRegressor(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], dropout=0.2, output_size=20):
        """
        Args:
            input_size: number of input features
            hidden_sizes: list of hidden layer sizes
            dropout: dropout rate
            output_size: number of regression outputs (n_tickers * 4)
        """
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128], dropout=0.2, output_size=5):
        """
        Args:
            input_size: number of input features
            hidden_sizes: list of hidden layer sizes
            dropout: dropout rate
            output_size: number of classification outputs (n_tickers)
        """
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def _train_loop(model, train_loader, val_loader, lr=1e-3, epochs=50,
                patience=10, use_amp=True, verbose=False):
    """Shared training loop for MLP models."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
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
                    epochs=50, batch_size=64, patience=10, verbose=False):
    """Train MLP regressor for OHLC prediction."""
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MLPRegressor(input_size, hidden_sizes, dropout, output_size)

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
                     epochs=50, batch_size=64, patience=10, verbose=False):
    """Train MLP classifier for direction prediction."""
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    model = MLPClassifier(input_size, hidden_sizes, dropout, output_size)

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
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy()


def predict_classifier(model, X):
    """Predict direction and probabilities."""
    model.eval()
    with torch.no_grad():
        probs = model(torch.FloatTensor(X)).numpy()
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


def save_model(model, name, input_size, hidden_sizes, output_size, is_classifier=False):
    """Save trained MLP model."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
        "is_classifier": is_classifier,
    }, path)
    return path


def load_model(name):
    """Load trained MLP model."""
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    ckpt = torch.load(path, map_location="cpu")
    if ckpt["is_classifier"]:
        model = MLPClassifier(ckpt["input_size"], ckpt["hidden_sizes"], output_size=ckpt["output_size"])
    else:
        model = MLPRegressor(ckpt["input_size"], ckpt["hidden_sizes"], output_size=ckpt["output_size"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model
