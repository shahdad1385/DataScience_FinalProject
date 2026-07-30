"""
Shared training loop for all time series models.

Handles: mixed precision, combined loss, early stopping, hyperparameter search.
Works with LSTM, GRU, and Transformer — they all have the same forward() signature.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def combined_loss(reg_pred, cls_pred, reg_target, cls_target, reg_weight=0.5, cls_weight=0.5):
    """
    Combined loss: MSE for OHLC regression + BCE for direction classification.
    cls_pred should be raw logits (no sigmoid) — BCEWithLogitsLoss handles it.
    """
    reg_loss = nn.MSELoss()(reg_pred, reg_target)
    cls_loss = nn.BCEWithLogitsLoss()(cls_pred, cls_target)
    return reg_weight * reg_loss + cls_weight * cls_loss, reg_loss.item(), cls_loss.item()


def train_one_epoch(model, loader, optimizer, device, use_amp=True):
    """Train for one epoch. Returns average loss."""
    model.train()
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    total_loss = 0
    n_batches = 0

    for x, y_reg, y_cls in loader:
        x = x.to(device)
        y_reg = y_reg.to(device)
        y_cls = y_cls.to(device)

        optimizer.zero_grad()
        with autocast(enabled=use_amp and device.type == "cuda"):
            reg_pred, cls_pred = model(x)
            loss, _, _ = combined_loss(reg_pred, cls_pred, y_reg, y_cls)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, device):
    """Validate model. Returns average loss and predictions (cls predictions are sigmoid-applied)."""
    model = model.to(device)
    model.eval()
    total_loss = 0
    n_batches = 0
    all_reg_pred, all_cls_pred = [], []
    all_reg_true, all_cls_true = [], []

    with torch.no_grad():
        for x, y_reg, y_cls in loader:
            x = x.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)

            reg_pred, cls_logits = model(x)
            loss, _, _ = combined_loss(reg_pred, cls_logits, y_reg, y_cls)

            total_loss += loss.item()
            n_batches += 1

            all_reg_pred.append(reg_pred.cpu().numpy())
            all_cls_pred.append(torch.sigmoid(cls_logits).cpu().numpy())
            all_reg_true.append(y_reg.cpu().numpy())
            all_cls_true.append(y_cls.cpu().numpy())

    if n_batches == 0:
        raise ValueError(
            "Validation loader is empty — no sequences were produced for this split. "
            "Each split must span more than SEQ_LEN dates; check the train/val/test "
            "cutoffs in preprocessing."
        )

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, (
        np.concatenate(all_reg_pred),
        np.concatenate(all_cls_pred),
        np.concatenate(all_reg_true),
        np.concatenate(all_cls_true),
    )



def train_model(model, train_loader, val_loader, lr=1e-3, epochs=200,
                patience=30, device=None, use_amp=True, verbose=True):
    """
    Full training loop with early stopping.

    Args:
        model: any model with forward() returning (reg, cls)
        train_loader: training DataLoader
        val_loader: validation DataLoader
        lr: learning rate
        epochs: maximum epochs
        patience: early stopping patience
        device: torch device (auto-detected if None)
        use_amp: use mixed precision on GPU
        verbose: print progress

    Returns:
        best_model, history dict
    """
    if device is None:
        device = get_device()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    best_state = None
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, use_amp)
        val_loss, _ = validate(model, val_loader, device)
        scheduler.step(val_loss)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch:3d} | train: {train_loss:.6f} | val: {val_loss:.6f} | {dt:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def save_model(model, name, input_size, **kwargs):
    """Save trained model to disk.

    input_size must be the real per-timestep feature count and kwargs the real
    constructor arguments, otherwise the model cannot be rebuilt on load.
    """
    if not input_size or int(input_size) <= 0:
        raise ValueError(f"save_model({name}): invalid input_size={input_size!r}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_size": int(input_size),
        "kwargs": dict(kwargs),
    }, path)
    return path


def load_model(model_class, name, device="cpu", default_kwargs=None):
    """Load trained model from disk.

    Rebuilds using the input_size and kwargs stored at save time. Loads weights
    strictly: a silent shape mismatch here would produce a model that returns
    predictions from partially random weights.
    """
    path = os.path.join(MODELS_DIR, f"{name}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}. Train before evaluating.")
    checkpoint = torch.load(path, map_location=device)

    kwargs = dict(default_kwargs or {})
    kwargs.update(checkpoint.get("kwargs") or {})

    input_size = checkpoint.get("input_size")
    if not input_size:
        raise ValueError(
            f"{path} has no usable input_size ({input_size!r}); it was written by "
            "an older buggy save path. Retrain to regenerate the checkpoint."
        )

    model = model_class(input_size=input_size, **kwargs)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def predict(model, loader, device=None):
    """Run inference on a DataLoader. Returns (reg_pred, cls_prob)."""
    import torch
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    all_reg, all_cls = [], []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(device)
            reg, cls_logits = model(x)
            all_reg.append(reg.cpu().numpy())
            all_cls.append(torch.sigmoid(cls_logits).cpu().numpy())

    return np.concatenate(all_reg), np.concatenate(all_cls)
