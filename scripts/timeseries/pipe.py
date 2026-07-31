"""
Time Series Orchestrator — combines all models.

Runs LSTM, GRU, Transformer, BiLSTM, TCN, and Prophet,
compares them, and returns the best result.

This module is called by the pipeline after NLP and clustering features are ready.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from . import lstm
from . import gru
from . import transformer
from . import bilstm
from . import tcn
from . import prophet_model
from .train import (
    get_device, train_model, save_model, load_model, predict as ts_predict,
    compute_pos_weight, DEFAULT_REG_LOSS, DEFAULT_HUBER_DELTA,
)
from .data import create_dataloaders
from ..activations import DEFAULT_ACTIVATION


def _make_loader(X, y_reg, y_cls, batch_size, shuffle):
    """Create DataLoader. Handles both 2D (raw) and 3D (pre-sequenced) input."""
    X_t = torch.FloatTensor(X)
    y_reg_t = torch.FloatTensor(y_reg)
    y_cls_t = torch.FloatTensor(y_cls)
    ds = TensorDataset(X_t, y_reg_t, y_cls_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def make_loss_kwargs(y_cls_train, reg_loss=DEFAULT_REG_LOSS,
                     huber_delta=DEFAULT_HUBER_DELTA, balance_classes=True):
    """Build the combined_loss configuration shared by every sequence model."""
    kwargs = {"reg_loss_kind": reg_loss, "huber_delta": huber_delta}
    if balance_classes and y_cls_train is not None:
        kwargs["pos_weight"] = compute_pos_weight(y_cls_train)
    return kwargs


def train_all(X_train, y_reg_train, y_cls_train,
              X_val, y_reg_val, y_cls_val,
              seq_len=30, batch_size=64, lr=1e-3, epochs=200,
              patience=30, n_tickers=5, tickers=None, verbose=True,
              activation=DEFAULT_ACTIVATION, hidden_size=64, n_layers=2,
              dropout=0.2, loss_kwargs=None):
    """
    Train all time series models and compare.

    Returns:
        dict with {model_name: (model, history, val_results)}
    """
    device = get_device()
    input_size = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]

    train_loader = _make_loader(X_train, y_reg_train, y_cls_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_reg_val, y_cls_val, batch_size, shuffle=False)

    if loss_kwargs is None:
        loss_kwargs = make_loss_kwargs(y_cls_train)

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
            print(f"  Training {name.upper()}  (act={activation})")
            print(f"{'='*40}")
        # Per-architecture argument names, so every model honours the same
        # hidden_size/n_layers/activation settings.
        kwargs = build_kwargs(name, hidden_size, n_layers, dropout, n_tickers,
                              activation=activation)
        model = _build(mod, input_size, **kwargs)
        model, hist = train_model(
            model, train_loader, val_loader,
            lr=lr, epochs=epochs, patience=patience,
            device=device, verbose=verbose, loss_kwargs=loss_kwargs,
        )
        val_loss, val_preds = _validate(model, val_loader, device, loss_kwargs=loss_kwargs)
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


def _build(mod, input_size, **kwargs):
    """Create a model and record exactly how it was built.

    save_model needs the real constructor arguments to rebuild the model later.
    Recovering them after the fact (e.g. reading reg_head.in_features) yields the
    shared-layer width, not input_size, so the reloaded model had the wrong
    shape and load_state_dict failed. Stashing them here keeps the checkpoint
    faithful to the object that was trained.
    """
    model = mod.create_model(input_size, **kwargs)
    model._build_input_size = input_size
    model._build_kwargs = dict(kwargs)
    return model


def get_build_config(model, fallback_input_size=None, fallback_kwargs=None):
    """Return (input_size, kwargs) recorded by _build, with fallbacks."""
    input_size = getattr(model, "_build_input_size", None)
    if input_size is None:
        input_size = fallback_input_size
    kwargs = getattr(model, "_build_kwargs", None)
    if kwargs is None:
        kwargs = dict(fallback_kwargs or {})
    return input_size, dict(kwargs)


def _validate(model, loader, device, loss_kwargs=None):
    """Validate a single model using the same loss configuration as training."""
    from .train import validate as _val
    return _val(model, loader, device, loss_kwargs=loss_kwargs)


def build_kwargs(model_name, hidden_size, n_layers, dropout, n_tickers,
                 activation=DEFAULT_ACTIVATION, nhead=4, kernel_size=3):
    """Map generic hyperparameters onto each architecture's own argument names.

    lstm/gru/bilstm take hidden_size/num_layers, the transformer takes
    d_model/nhead/num_layers, and the TCN takes n_channels/n_layers/kernel_size.
    Centralising the mapping keeps train_single and the Optuna objective in sync.
    """
    kwargs = {"dropout": dropout, "n_tickers": n_tickers, "activation": activation}
    if model_name == "transformer":
        # nn.MultiheadAttention requires d_model % nhead == 0.
        d_model = int(np.ceil(hidden_size / nhead) * nhead)
        kwargs.update(d_model=d_model, nhead=nhead, num_layers=n_layers)
    elif model_name == "tcn":
        kwargs.update(n_channels=hidden_size, n_layers=n_layers, kernel_size=kernel_size)
    else:
        kwargs.update(hidden_size=hidden_size, num_layers=n_layers)
    return kwargs


def train_single(model_name, X_train, y_reg_train, y_cls_train,
                 X_val, y_reg_val, y_cls_val,
                 seq_len=30, n_tickers=5, lr=1e-3, epochs=200,
                 patience=30, batch_size=64, hidden_size=64,
                 n_layers=2, dropout=0.2, verbose=True,
                 activation=DEFAULT_ACTIVATION, loss_kwargs=None):
    """
    Train a single time series model with custom hyperparameters.

    Args:
        model_name: one of lstm, gru, transformer, bilstm, tcn
        hidden_size: hidden dimension size
        n_layers: number of layers
        dropout: dropout rate
        activation: activation name (see scripts/activations.py)
        loss_kwargs: forwarded to combined_loss
    """
    device = get_device()
    input_size = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]

    model_modules = {
        "lstm": lstm, "gru": gru, "transformer": transformer,
        "bilstm": bilstm, "tcn": tcn,
    }

    if model_name not in model_modules:
        raise ValueError(f"Unknown model: {model_name}")

    mod = model_modules[model_name]
    extra_kwargs = build_kwargs(model_name, hidden_size, n_layers, dropout, n_tickers,
                                activation=activation)
    model = _build(mod, input_size, **extra_kwargs)

    train_loader = _make_loader(X_train, y_reg_train, y_cls_train, batch_size, shuffle=True)
    val_loader = _make_loader(X_val, y_reg_val, y_cls_val, batch_size, shuffle=False)

    if verbose:
        print(f"\n{'='*40}")
        print(f"  Training {model_name.upper()}")
        print(f"  hidden={hidden_size}, layers={n_layers}, dropout={dropout}, "
              f"lr={lr:.2e}, act={activation}")
        print(f"{'='*40}")

    model, hist = train_model(
        model, train_loader, val_loader,
        lr=lr, epochs=epochs, patience=patience,
        device=device, verbose=verbose, loss_kwargs=loss_kwargs,
    )

    val_loss, val_preds = _validate(model, val_loader, device, loss_kwargs=loss_kwargs)

    return {"model": model, "history": hist, "val_loss": val_loss, "val_preds": val_preds}


def select_best(results):
    """Pick the best model by validation loss."""
    best_name = min(results, key=lambda k: results[k]["val_loss"])
    return best_name, results[best_name]


def save_best(best_name, best_result, input_size=None, **kwargs):
    """Save a trained model, preferring the config it was actually built with."""
    model = best_result["model"]
    real_input_size, real_kwargs = get_build_config(model, input_size, kwargs)
    if real_input_size is None:
        raise ValueError(
            f"Cannot save timeseries_{best_name}: input_size unknown. "
            "Build models through _build/train_single so the config is recorded."
        )
    return save_model(model, f"timeseries_{best_name}", real_input_size, **real_kwargs)


def load_best(name, n_tickers=5):
    """Load a trained model from disk.

    The checkpoint carries input_size and the constructor kwargs, so nothing is
    passed positionally here. Previously n_tickers was forwarded into
    load_model(), which does not accept it, raising TypeError on every load.
    """
    model_map = {
        "lstm": lstm, "gru": gru, "transformer": transformer,
        "bilstm": bilstm, "tcn": tcn,
    }
    if name in model_map:
        mod = model_map[name]
        return load_model(mod.create_model, f"timeseries_{name}", default_kwargs={"n_tickers": n_tickers})
    elif name == "prophet":
        raise ValueError("Prophet models are saved per-ticker, use prophet_model.load_per_ticker()")
    else:
        raise ValueError(f"Unknown model: {name}")
