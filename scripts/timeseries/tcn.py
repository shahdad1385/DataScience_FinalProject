"""
Temporal Convolutional Network (TCN) for stock price prediction.

Multi-output: regression (OHLC × 5 tickers = 20) + classification (direction × 5 tickers = 5).
Uses dilated causal convolutions for long-range dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Causal 1D convolution — no look-ahead into the future."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, channels, seq_len) — same length as input
        """
        out = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """Single TCN block: Conv → ReLU → Dropout → Conv → ReLU → Dropout + residual."""

    def __init__(self, n_inputs, n_outputs, kernel_size=3, dropout=0.1, dilation=1):
        super().__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, n_outputs, seq_len)
        """
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        residual = self.residual(x)
        return out + residual


class StockTCN(nn.Module):
    def __init__(self, input_size, n_channels=64, n_layers=4, kernel_size=3,
                 dropout=0.1, n_tickers=5):
        """
        Args:
            input_size: number of input features per timestep
            n_channels: number of channels in TCN layers
            n_layers: number of TCN blocks (receptive field = (kernel_size-1) * sum(dilations))
            kernel_size: convolution kernel size
            dropout: dropout rate
            n_tickers: number of tickers to predict
        """
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = input_size if i == 0 else n_channels
            dilation = 2 ** i  # exponential dilation
            layers.append(TCNBlock(in_ch, n_channels, kernel_size, dropout, dilation))
        self.tcn = nn.Sequential(*layers)

        self.shared_fc = nn.Linear(n_channels, 128)
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
        # TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        # Take last timestep
        last = out[:, :, -1]
        shared = self.relu(self.shared_fc(last))
        shared = self.dropout(shared)
        reg = self.reg_head(shared)
        cls = torch.sigmoid(self.cls_head(shared))
        return reg, cls


def create_model(input_size, n_channels=64, n_layers=4, kernel_size=3,
                 dropout=0.1, n_tickers=5):
    """Factory function to create TCN model."""
    return StockTCN(input_size, n_channels, n_layers, kernel_size, dropout, n_tickers)
