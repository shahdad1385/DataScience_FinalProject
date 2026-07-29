"""
BiLSTM (Bidirectional LSTM) model for stock price prediction.

Multi-output: regression (OHLC × 5 tickers = 20) + classification (direction × 5 tickers = 5).
Captures both forward and backward temporal patterns.
"""

import torch
import torch.nn as nn


class StockBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2, n_tickers=5):
        """
        Args:
            input_size: number of input features per timestep
            hidden_size: LSTM hidden dimension (output is 2*hidden_size due to bidirectional)
            num_layers: number of LSTM layers
            dropout: dropout between LSTM layers
            n_tickers: number of tickers to predict
        """
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.shared_fc = nn.Linear(hidden_size * 2, 128)  # *2 for bidirectional
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(128, n_tickers * 4)
        self.cls_head = nn.Linear(128, n_tickers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            reg: (batch, n_tickers * 4) — OHLC predictions
            cls: (batch, n_tickers) — direction probabilities (0-1)
        """
        lstm_out, _ = self.bilstm(x)
        last = lstm_out[:, -1, :]  # take last timestep (both directions concatenated)
        shared = self.relu(self.shared_fc(last))
        shared = self.dropout(shared)
        reg = self.reg_head(shared)
        cls = self.cls_head(shared)
        return reg, cls


def create_model(input_size, hidden_size=128, num_layers=2, dropout=0.2, n_tickers=5):
    """Factory function to create BiLSTM model."""
    return StockBiLSTM(input_size, hidden_size, num_layers, dropout, n_tickers)
