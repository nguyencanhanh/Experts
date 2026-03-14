import math
import torch
import torch.nn as nn

from config import (
    CNN_CHANNELS, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT,
    TRANSFORMER_DIM, TRANSFORMER_HEADS, TRANSFORMER_LAYERS,
    DROP_PATH_RATE,
)
from utils import SwiGLU, DropPath


def guess_transformer_heads(transformer_dim: int, preferred: int = None) -> int:
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend([4, 8, 6, 3, 2, 1])

    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if cand > 0 and transformer_dim % cand == 0:
            return cand
    return 1


def infer_model_arch_from_state_dict(state_dict: dict, input_size: int = None) -> dict:
    transformer_layers = sorted({
        int(key.split(".")[1])
        for key in state_dict
        if key.startswith("transformer_layers.") and key.split(".")[1].isdigit()
    })
    bilstm_layers = sorted({
        int(key.split("bilstm.weight_ih_l", 1)[1].split("_", 1)[0])
        for key in state_dict
        if key.startswith("bilstm.weight_ih_l")
    })

    inferred_input_size = input_size
    if inferred_input_size is None:
        inferred_input_size = int(state_dict["conv.conv3.weight"].shape[1])

    transformer_dim = int(state_dict["ln_pre.weight"].shape[0])
    return {
        "input_size": inferred_input_size,
        "num_classes": int(state_dict["head.4.weight"].shape[0]),
        "cnn_channels": int(state_dict["conv.norm.weight"].shape[0]),
        "lstm_hidden": int(state_dict["bilstm.weight_hh_l0"].shape[1]),
        "lstm_layers": (max(bilstm_layers) + 1) if bilstm_layers else LSTM_LAYERS,
        "dropout": DROPOUT,
        "transformer_dim": transformer_dim,
        "transformer_heads": guess_transformer_heads(transformer_dim, preferred=TRANSFORMER_HEADS),
        "transformer_layers": (max(transformer_layers) + 1) if transformer_layers else TRANSFORMER_LAYERS,
        "drop_path_rate": DROP_PATH_RATE,
    }


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
    def __init__(self, input_size: int, num_classes: int = 3,
                 cnn_channels: int = CNN_CHANNELS,
                 lstm_hidden: int = LSTM_HIDDEN,
                 lstm_layers: int = LSTM_LAYERS,
                 dropout: float = DROPOUT,
                 transformer_dim: int = TRANSFORMER_DIM,
                 transformer_heads: int = TRANSFORMER_HEADS,
                 transformer_layers: int = TRANSFORMER_LAYERS,
                 drop_path_rate: float = DROP_PATH_RATE):
        super().__init__()
        self.arch_config = {
            "input_size": input_size,
            "num_classes": num_classes,
            "cnn_channels": cnn_channels,
            "lstm_hidden": lstm_hidden,
            "lstm_layers": lstm_layers,
            "dropout": dropout,
            "transformer_dim": transformer_dim,
            "transformer_heads": transformer_heads,
            "transformer_layers": transformer_layers,
            "drop_path_rate": drop_path_rate,
        }

        # Multi-scale CNN
        self.conv = MultiScaleConv(input_size, cnn_channels, dropout)
        self.se = SqueezeExcite(cnn_channels)
        self.conv2 = nn.Sequential(
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.GELU(), nn.BatchNorm1d(cnn_channels), nn.Dropout(dropout),
        )

        # Projection + Learnable PE
        self.proj = nn.Linear(cnn_channels, transformer_dim)
        self.pos = LearnablePositionalEncoding(transformer_dim)
        self.ln_pre = nn.LayerNorm(transformer_dim)

        # SwiGLU Transformer with stochastic depth
        dp_rates = [drop_path_rate * i / max(transformer_layers - 1, 1) for i in range(transformer_layers)]
        self.transformer_layers = nn.ModuleList([
            SwiGLUTransformerLayer(transformer_dim, transformer_heads, dropout, dp_rates[i])
            for i in range(transformer_layers)
        ])

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=transformer_dim, hidden_size=lstm_hidden, num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0.0, batch_first=True, bidirectional=True,
        )

        lstm_out = lstm_hidden * 2
        self.attn_pool = AttentionPooling(lstm_out)
        self.skip_proj = nn.Linear(cnn_channels, lstm_out)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out), nn.Linear(lstm_out, lstm_hidden),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(lstm_hidden, num_classes),
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
