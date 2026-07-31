"""
Temporal Convolutional Network (TCN) for stock price prediction.

Multi-output: regression (OHLC × 5 tickers = 20) + classification (direction × 5 tickers = 5).
Uses dilated causal convolutions for long-range dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import get_activation, DEFAULT_ACTIVATION


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
    """Single TCN block: Conv → Norm → act → Dropout (x2) + residual.

    The two BatchNorm1d layers are the fix for the TCN's diverging validation
    loss (64-87 while every other model sat near 0.7, with train loss at 0.09).
    Without any normalization, each block's residual branch adds unnormalized
    activations on top of the previous block's output, so magnitudes compound
    across the stack. The first conv also projects ~1300 input channels down to
    64, which makes that first activation especially large. Training data stayed
    (barely) in range while unseen inputs pushed the accumulated activations far
    out, producing huge validation errors.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size=3, dropout=0.1, dilation=1,
                 activation=DEFAULT_ACTIVATION):
        super().__init__()
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(n_outputs)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act1 = get_activation(activation)
        self.act2 = get_activation(activation)
        self.residual = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (batch, channels, seq_len)
        Returns:
            (batch, n_outputs, seq_len)
        """
        out = self.dropout1(self.act1(self.norm1(self.conv1(x))))
        out = self.dropout2(self.act2(self.norm2(self.conv2(out))))
        return out + self.residual(x)


class StockTCN(nn.Module):
    def __init__(self, input_size, n_channels=64, n_layers=4, kernel_size=3,
                 dropout=0.1, n_tickers=5, activation=DEFAULT_ACTIVATION):
        """
        Args:
            input_size: number of input features per timestep
            n_channels: number of channels in TCN layers
            n_layers: number of TCN blocks (receptive field = (kernel_size-1) * sum(dilations))
            kernel_size: convolution kernel size
            dropout: dropout rate
            n_tickers: number of tickers to predict
            activation: activation used inside the blocks and the shared layer
        """
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = input_size if i == 0 else n_channels
            dilation = 2 ** i  # exponential dilation
            layers.append(TCNBlock(in_ch, n_channels, kernel_size, dropout, dilation,
                                   activation=activation))
        self.tcn = nn.Sequential(*layers)

        self.shared_fc = nn.Linear(n_channels, 128)
        self.norm = nn.LayerNorm(128)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(128, n_tickers * 4)
        self.cls_head = nn.Linear(128, n_tickers)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            reg: (batch, n_tickers * 4) — predicted next-day returns
            cls: (batch, n_tickers) — direction logits
        """
        # TCN expects (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        # Take last timestep
        last = out[:, :, -1]
        shared = self.dropout(self.act(self.norm(self.shared_fc(last))))
        reg = self.reg_head(shared)
        cls = self.cls_head(shared)
        return reg, cls


def create_model(input_size, n_channels=64, n_layers=4, kernel_size=3,
                 dropout=0.1, n_tickers=5, activation=DEFAULT_ACTIVATION):
    """Factory function to create TCN model."""
    return StockTCN(input_size, n_channels, n_layers, kernel_size, dropout, n_tickers,
                    activation)
