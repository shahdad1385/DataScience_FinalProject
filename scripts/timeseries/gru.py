"""
GRU model for stock price prediction.

Multi-output: regression (OHLC × 5 tickers = 20) + classification (direction × 5 tickers = 5).
Lighter alternative to LSTM with similar performance.
"""

import torch
import torch.nn as nn


class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, n_tickers=5):
        """
        Args:
            input_size: number of input features per timestep
            hidden_size: GRU hidden dimension
            num_layers: number of GRU layers
            dropout: dropout between GRU layers
            n_tickers: number of tickers to predict
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.shared_fc = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Regression head: OHLC × n_tickers
        self.reg_head = nn.Linear(128, n_tickers * 4)
        # Classification head: direction × n_tickers
        self.cls_head = nn.Linear(128, n_tickers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            reg: (batch, n_tickers * 4) — OHLC predictions
            cls: (batch, n_tickers) — direction probabilities (0-1)
        """
        gru_out, _ = self.gru(x)
        last = gru_out[:, -1, :]
        shared = self.relu(self.shared_fc(last))
        shared = self.dropout(shared)
        reg = self.reg_head(shared)
        cls = self.cls_head(shared)
        return reg, cls


def create_model(input_size, hidden_size=128, num_layers=2, dropout=0.2, n_tickers=5):
    """Factory function to create GRU model."""
    return StockGRU(input_size, hidden_size, num_layers, dropout, n_tickers)
