import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from config import SEED


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Scalers ──

class FeatureScaler:
    """Memory-efficient 2D feature scaler (fits on 2D, transforms 2D or 3D)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, features_2d: np.ndarray):
        if features_2d.ndim == 3:
            features_2d = features_2d.reshape(-1, features_2d.shape[-1])
        self.mean_ = features_2d.mean(axis=0).astype(np.float32)
        self.std_ = features_2d.std(axis=0).astype(np.float32)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


# Backward compatibility alias
StandardScaler3D = FeatureScaler


# ── Loss Functions ──

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = torch.nn.functional.cross_entropy(
            logits, targets, weight=self.alpha,
            reduction="none", label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


# ── Model Utilities ──

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])


class SwiGLU(nn.Module):
    """SwiGLU activation — state-of-the-art for financial time series."""

    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.w3 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.w3(nn.functional.silu(self.w1(x)) * self.w2(x))


class DropPath(nn.Module):
    """Stochastic depth — randomly drops entire residual branches."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device).floor_().clamp_(0, 1)
        mask = (torch.rand(shape, dtype=x.dtype, device=x.device) + keep).floor_()
        return x * mask / keep


# ── IO Helpers ──

def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_csv_row(path: str, row_dict: dict):
    ensure_parent_dir(path)
    df = pd.DataFrame([row_dict])
    if os.path.exists(path):
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)
