"""
Sequence Dataset and DataLoader for time series models.

Handles creating 30-day sliding windows and PyTorch DataLoaders.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for time series sequences.
    Creates sliding windows of (input_seq, target_reg, target_cls).
    """

    def __init__(self, X, y_reg, y_cls, seq_len=30):
        """
        Args:
            X: feature array of shape (n_samples, n_features)
            y_reg: regression targets of shape (n_samples, n_tickers * 4) — OHLC
            y_cls: classification targets of shape (n_samples, n_tickers) — direction
            seq_len: number of days in each input sequence
        """
        self.X = torch.FloatTensor(X)
        self.y_reg = torch.FloatTensor(y_reg)
        self.y_cls = torch.FloatTensor(y_cls)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.X) - self.seq_len)

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        y_reg = self.y_reg[idx + self.seq_len]
        y_cls = self.y_cls[idx + self.seq_len]
        return x, y_reg, y_cls


def create_dataloaders(X_train, y_reg_train, y_cls_train,
                       X_val, y_reg_val, y_cls_val,
                       X_test, y_reg_test, y_cls_test,
                       seq_len=30, batch_size=64, num_workers=2):
    """
    Create train/val/test DataLoaders with sequence windows.
    Returns (train_loader, val_loader, test_loader).
    """
    train_ds = SequenceDataset(X_train, y_reg_train, y_cls_train, seq_len)
    val_ds = SequenceDataset(X_val, y_reg_val, y_cls_val, seq_len)
    test_ds = SequenceDataset(X_test, y_reg_test, y_cls_test, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
