"""
Time Series Orchestrator — combines all models.

Runs LSTM, GRU, Transformer, BiLSTM, TCN, and Prophet,
compares them, and returns the best result.

This module is called by the pipeline after NLP and clustering features are ready.
"""

import torch
import numpy as np

from . import lstm
from . import gru
from . import transformer
from . import bilstm
from . import tcn
from . import prophet_model
from .train import (
    get_device, train_model, save_model, load_model, predict as ts_predict,
)
from .data import create_dataloaders


def train_all(X_train, y_reg_train, y_cls_train,
              X_val, y_reg_val, y_cls_val,
              seq_len=30, batch_size=64, lr=1e-3, epochs=100,
              patience=10, n_tickers=5, tickers=None, verbose=True):
    """
    Train all time series models and compare.

    Returns:
        dict with {model_name: (model, history, val_results)}
    """
    device = get_device()
    input_size = X_train.shape[1]

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        X_train, y_reg_train, y_cls_train,
        X_val, y_reg_val, y_cls_val,
        X_val, y_reg_val, y_cls_val,
        seq_len=seq_len, batch_size=batch_size,
    )

    results = {}

    # --- PyTorch models ---
    pytorch_models = {
        "lstm": lstm,
        "gru": gru,
        "transformer": transformer,
        "bilstm": bilstm,
        "tcn": tcn,
    }

    for name, mod in pytorch_models.items():
        if verbose:
            print(f"\n{'='*40}")
            print(f"  Training {name.upper()}")
            print(f"{'='*40}")
        model = mod.create_model(input_size, n_tickers=n_tickers)
        model, hist = train_model(
            model, train_loader, val_loader,
            lr=lr, epochs=epochs, patience=patience,
            device=device, verbose=verbose,
        )
        val_loss, val_preds = _validate(model, val_loader, device)
        results[name] = {"model": model, "history": hist, "val_loss": val_loss, "val_preds": val_preds}

    # --- Prophet (non-PyTorch) ---
    if prophet_model.HAS_PROPHET and tickers:
        if verbose:
            print(f"\n{'='*40}")
            print(f"  Training Prophet")
            print(f"{'='*40}")
        # Prophet needs raw data, not sequences — we skip it here
        # and handle it separately in the pipeline
        if verbose:
            print("  Prophet: handled separately in pipeline (needs raw OHLC data)")
    else:
        if verbose:
            print(f"\n  Prophet: skipped (not installed or no tickers)")

    return results


def _validate(model, loader, device):
    """Validate a single model."""
    from .train import validate as _val
    return _val(model, loader, device)


def train_single(model_name, X_train, y_reg_train, y_cls_train,
                 X_val, y_reg_val, y_cls_val,
                 seq_len=30, n_tickers=5, lr=1e-3, epochs=100,
                 patience=10, batch_size=64, hidden_size=64,
                 n_layers=2, dropout=0.2, verbose=True):
    """
    Train a single time series model with custom hyperparameters.

    Args:
        model_name: one of lstm, gru, transformer, bilstm, tcn
        hidden_size: hidden dimension size
        n_layers: number of layers
        dropout: dropout rate
    """
    device = get_device()
    input_size = X_train.shape[1]

    model_modules = {
        "lstm": lstm, "gru": gru, "transformer": transformer,
        "bilstm": bilstm, "tcn": tcn,
    }

    if model_name not in model_modules:
        raise ValueError(f"Unknown model: {model_name}")

    mod = model_modules[model_name]
    extra_kwargs = {}
    if model_name in ("transformer", "tcn"):
        extra_kwargs["nhead"] = 4

    model = mod.create_model(
        input_size, hidden_size=hidden_size, n_layers=n_layers,
        dropout=dropout, n_tickers=n_tickers, **extra_kwargs,
    )

    train_loader, val_loader, _ = create_dataloaders(
        X_train, y_reg_train, y_cls_train,
        X_val, y_reg_val, y_cls_val,
        X_val, y_reg_val, y_cls_val,
        seq_len=seq_len, batch_size=batch_size,
    )

    if verbose:
        print(f"\n{'='*40}")
        print(f"  Training {model_name.upper()}")
        print(f"  hidden={hidden_size}, layers={n_layers}, dropout={dropout}, lr={lr:.2e}")
        print(f"{'='*40}")

    model, hist = train_model(
        model, train_loader, val_loader,
        lr=lr, epochs=epochs, patience=patience,
        device=device, verbose=verbose,
    )

    val_loss, val_preds = _validate(model, val_loader, device)

    return {"model": model, "history": hist, "val_loss": val_loss, "val_preds": val_preds}


def select_best(results):
    """Pick the best model by validation loss."""
    best_name = min(results, key=lambda k: results[k]["val_loss"])
    return best_name, results[best_name]


def save_best(best_name, best_result, input_size, **kwargs):
    """Save the best model to disk."""
    model = best_result["model"]
    return save_model(model, f"timeseries_{best_name}", input_size, **kwargs)


def load_best(name, n_tickers=5):
    """Load a trained model from disk."""
    model_map = {
        "lstm": lstm, "gru": gru, "transformer": transformer,
        "bilstm": bilstm, "tcn": tcn,
    }
    if name in model_map:
        mod = model_map[name]
        return load_model(mod.create_model, f"timeseries_{name}", n_tickers=n_tickers)
    elif name == "prophet":
        raise ValueError("Prophet models are saved per-ticker, use prophet_model.load_per_ticker()")
    else:
        raise ValueError(f"Unknown model: {name}")
