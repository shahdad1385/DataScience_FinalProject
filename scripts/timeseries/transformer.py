"""
Transformer model for stock price prediction.

Multi-output: regression (OHLC × 5 tickers = 20) + classification (direction × 5 tickers = 5).
Main candidate model — captures long-range dependencies via self-attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..activations import get_activation, DEFAULT_ACTIVATION


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
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class _EncoderLayer(nn.Module):
    """TransformerEncoderLayer built from scratch to avoid torch.empty() kwarg bug on Kaggle."""

    def __init__(self, d_model, nhead, dropout=0.1, batch_first=True,
                 activation=DEFAULT_ACTIVATION):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # Resolved through the shared factory so --activation also controls the
        # feed-forward block, not just the shared dense layer.
        self.activation = get_activation(activation)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        q = k = v = src
        src2, _ = self.self_attn(q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class StockTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=2,
                 dropout=0.2, n_tickers=5, max_len=500,
                 activation=DEFAULT_ACTIVATION):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})."
            )
        self.input_proj = nn.Linear(input_size, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        encoder_layer = _EncoderLayer(d_model, nhead, dropout, batch_first=True,
                                      activation=activation)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.shared_fc = nn.Linear(d_model, 128)
        self.norm = nn.LayerNorm(128)
        self.act = get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(128, n_tickers * 4)
        self.cls_head = nn.Linear(128, n_tickers)

    def forward(self, x):
        # Normalising the projection keeps the additive positional encoding on a
        # comparable scale to the embedded features; with ~1300 input features the
        # raw projection can otherwise dominate it entirely.
        x = self.input_norm(self.input_proj(x))
        x = self.pos_encoding(x)
        out = self.transformer(x)
        last = out[:, -1, :]
        shared = self.dropout(self.act(self.norm(self.shared_fc(last))))
        reg = self.reg_head(shared)
        cls = self.cls_head(shared)
        return reg, cls


def create_model(input_size, d_model=128, nhead=8, num_layers=2,
                 dropout=0.2, n_tickers=5, activation=DEFAULT_ACTIVATION):
    return StockTransformer(input_size, d_model, nhead, num_layers, dropout, n_tickers,
                            activation=activation)
