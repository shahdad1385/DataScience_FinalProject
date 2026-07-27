"""
Transformer model for stock price prediction.

Multi-output: regression (OHLC × 5 tickers = 20) + classification (direction × 5 tickers = 5).
Main candidate model — captures long-range dependencies via self-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Adds positional encoding to input embeddings."""

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2,
                 dropout=0.2, n_tickers=5, max_len=500):
        """
        Args:
            input_size: number of input features per timestep
            d_model: model dimension (must be divisible by nhead)
            nhead: number of attention heads
            num_layers: number of Transformer encoder layers
            dropout: dropout rate
            n_tickers: number of tickers to predict
            max_len: maximum sequence length for positional encoding
        """
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.shared_fc = nn.Linear(d_model, 128)
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
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        shared = self.relu(self.shared_fc(last))
        shared = self.dropout(shared)
        reg = self.reg_head(shared)
        cls = torch.sigmoid(self.cls_head(shared))
        return reg, cls


def create_model(input_size, d_model=128, nhead=8, num_layers=2,
                 dropout=0.2, n_tickers=5):
    """Factory function to create Transformer model."""
    return StockTransformer(input_size, d_model, nhead, num_layers, dropout, n_tickers)
