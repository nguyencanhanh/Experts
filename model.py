import math
import torch
import torch.nn as nn

from config import (
    CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT,
    TRANSFORMER_DIM, TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    DROP_PATH_RATE,
)
from utils import SwiGLU, DropPath


class LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding — adapts to data patterns."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        c1 = out_channels // 3
        c2 = out_channels // 3
        c3 = out_channels - c1 - c2
        self.conv3 = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.conv7 = nn.Conv1d(in_channels, c2, kernel_size=7, padding=3)
        self.conv15 = nn.Conv1d(in_channels, c3, kernel_size=15, padding=7)
        self.act = nn.GELU()
        self.norm = nn.BatchNorm1d(out_channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([self.conv3(x), self.conv7(x), self.conv15(x)], dim=1)
        return self.drop(self.norm(self.act(out)))


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(nn.Linear(channels, mid), nn.GELU(), nn.Linear(mid, channels), nn.Sigmoid())

    def forward(self, x):
        w = self.fc(x.mean(dim=2)).unsqueeze(2)
        return x * w


class SwiGLUTransformerLayer(nn.Module):
    """Transformer encoder layer with SwiGLU feedforward and DropPath."""

    def __init__(self, d_model: int, nhead: int, dropout: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * 4)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm attention
        x2 = self.norm1(x)
        x2, _ = self.attn(x2, x2, x2)
        x = x + self.drop_path(self.dropout(x2))

        # Pre-norm SwiGLU FFN
        x2 = self.norm2(x)
        x2 = self.ffn(x2)
        x = x + self.drop_path(self.dropout(x2))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(), nn.Linear(hidden_size // 2, 1))

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


class CNNBiLSTMTransformer(nn.Module):
    def __init__(self, input_size: int, num_classes: int = 3):
        super().__init__()

        # Multi-scale CNN
        self.conv = MultiScaleConv(input_size, CNN_CHANNELS, DROPOUT)
        self.se = SqueezeExcite(CNN_CHANNELS)
        self.conv2 = nn.Sequential(
            nn.Conv1d(CNN_CHANNELS, CNN_CHANNELS, kernel_size=3, padding=1),
            nn.GELU(), nn.BatchNorm1d(CNN_CHANNELS), nn.Dropout(DROPOUT),
        )

        # Projection + Learnable PE
        self.proj = nn.Linear(CNN_CHANNELS, TRANSFORMER_DIM)
        self.pos = LearnablePositionalEncoding(TRANSFORMER_DIM)
        self.ln_pre = nn.LayerNorm(TRANSFORMER_DIM)

        # SwiGLU Transformer with stochastic depth
        dp_rates = [DROP_PATH_RATE * i / max(TRANSFORMER_LAYERS - 1, 1) for i in range(TRANSFORMER_LAYERS)]
        self.transformer_layers = nn.ModuleList([
            SwiGLUTransformerLayer(TRANSFORMER_DIM, TRANSFORMER_HEADS, DROPOUT, dp_rates[i])
            for i in range(TRANSFORMER_LAYERS)
        ])

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=TRANSFORMER_DIM, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYERS,
            dropout=DROPOUT if LSTM_LAYERS > 1 else 0.0, batch_first=True, bidirectional=True,
        )

        lstm_out = LSTM_HIDDEN * 2
        self.attn_pool = AttentionPooling(lstm_out)
        self.skip_proj = nn.Linear(CNN_CHANNELS, lstm_out)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out), nn.Linear(lstm_out, LSTM_HIDDEN),
            nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(LSTM_HIDDEN, num_classes),
        )

    def forward(self, x):
        # CNN
        c = x.transpose(1, 2)
        c = self.conv(c)
        c = self.se(c)
        c = c + self.conv2(c)

        skip = self.skip_proj(c.mean(dim=2))

        # Transformer
        s = self.proj(c.transpose(1, 2))
        s = self.ln_pre(self.pos(s))
        for layer in self.transformer_layers:
            s = layer(s)

        # BiLSTM
        out, _ = self.bilstm(s)
        pooled = self.attn_pool(out) + skip
        return self.head(pooled)
