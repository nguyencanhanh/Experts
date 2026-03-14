import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from config import SEQ_LEN
from labels import build_labels_no_lookahead
from features import get_base_features


@dataclass
class SequenceBundle:
    """Memory-efficient bundle: stores 2D features + metadata (no 3D pre-allocation)."""
    features: np.ndarray      # (M, F) float32 — raw feature array
    targets: np.ndarray       # (N_seq,) int64 — one per sequence
    times: np.ndarray         # (N_seq,) — time at end of each window
    row_df: pd.DataFrame      # N_seq rows — metadata for each sequence
    seq_len: int

    @property
    def n_sequences(self) -> int:
        return len(self.targets)


class SequenceDataset(Dataset):
    """Lazy sequence dataset — creates (seq_len, F) windows on-the-fly.
    Memory: O(M*F) instead of O(N*seq_len*F) — ~90x reduction.
    """

    def __init__(self, scaled_features: np.ndarray, targets: np.ndarray,
                 seq_len: int, start: int, count: int):
        """
        Args:
            scaled_features: (M, F) full scaled feature array
            targets: (N_seq,) aligned targets
            seq_len: window length
            start: first sequence index
            count: number of sequences in this split
        """
        self.features = scaled_features
        self.targets = targets
        self.seq_len = seq_len
        self.start = start
        self.count = count

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx):
        seq_idx = self.start + idx
        x = self.features[seq_idx:seq_idx + self.seq_len].copy()
        y = int(self.targets[seq_idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    def get_all_targets(self) -> np.ndarray:
        return self.targets[self.start:self.start + self.count].copy()


def build_sequence_bundle(df: pd.DataFrame, seq_len: int = SEQ_LEN, point_size: float = None) -> SequenceBundle:
    df = df.copy()
    df["target"] = build_labels_no_lookahead(df, point_size=point_size)

    feat_df = get_base_features(df)
    row_meta_cols = [
        "time", "open", "high", "low", "close", "tick_volume", "spread", "target",
        # Backtest/live filters use these raw indicator columns directly.
        "M1_ema_20", "M1_atr_14", "M1_atr_pct",
        "M15_ema_spread_20_50", "H1_ema_spread_20_50",
        "M15_trend_up_20_50", "H1_trend_up_20_50",
        "buy_context_score", "sell_context_score",
    ]
    selected_cols = row_meta_cols + [c for c in feat_df.columns if c not in row_meta_cols]
    full_df = df[selected_cols].copy()
    full_df = full_df.dropna().reset_index(drop=True)

    feature_cols = list(feat_df.columns)
    arr = full_df[feature_cols].values.astype(np.float32)
    target_raw = full_df["target"].values.astype(np.int64)
    time_arr = full_df["time"].values

    m = len(arr)
    if m < seq_len:
        raise ValueError(f"Not enough data: {m} rows < {seq_len} seq_len")

    n_seq = m - seq_len + 1

    # Aligned targets: targets[i] = target of last row in window i
    targets = target_raw[seq_len - 1:]   # shape (n_seq,)
    times = time_arr[seq_len - 1:]       # shape (n_seq,)
    row_idx = list(range(seq_len - 1, m))

    return SequenceBundle(
        features=arr,
        targets=targets,
        times=times,
        row_df=full_df.iloc[row_idx].reset_index(drop=True).copy(),
        seq_len=seq_len,
    )
