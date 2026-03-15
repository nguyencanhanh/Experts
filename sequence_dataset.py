import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from config import HORIZON_BARS, SEQ_LEN
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


@dataclass
class SequenceSlice:
    start: int
    end: int

    @property
    def count(self) -> int:
        return max(self.end - self.start, 0)

    def shifted(self, offset: int) -> "SequenceSlice":
        return SequenceSlice(self.start + offset, self.end + offset)


@dataclass
class PurgedFourWaySplit:
    purge_gap: int
    train: SequenceSlice
    valid: SequenceSlice
    calib: SequenceSlice
    test: SequenceSlice


@dataclass
class PurgedTrainValidWindow:
    purge_gap: int
    train: SequenceSlice
    valid: SequenceSlice


def required_purge_gap(seq_len: int = SEQ_LEN, horizon: int = HORIZON_BARS) -> int:
    # Targets at the end of a sequence still consume `horizon` future bars, so
    # we purge this many sequence indices between train/valid/test segments.
    return max(seq_len + horizon - 1, 0)


def build_purged_four_way_split(
    n_sequences: int,
    train_ratio: float,
    valid_ratio: float,
    calib_ratio: float,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON_BARS,
) -> PurgedFourWaySplit:
    test_ratio = 1.0 - train_ratio - valid_ratio - calib_ratio
    if test_ratio <= 0:
        raise ValueError("Train/valid/calib ratios leave no room for test data")

    purge_gap = required_purge_gap(seq_len, horizon)
    usable = n_sequences - purge_gap * 3
    if usable < 4:
        raise ValueError(
            f"Not enough sequences ({n_sequences}) for a 4-way purged split with gap={purge_gap}"
        )

    train_count = int(usable * train_ratio)
    valid_count = int(usable * valid_ratio)
    calib_count = int(usable * calib_ratio)
    test_count = usable - train_count - valid_count - calib_count

    if min(train_count, valid_count, calib_count, test_count) <= 0:
        raise ValueError(
            "Purged split produced an empty segment. "
            f"counts={(train_count, valid_count, calib_count, test_count)}"
        )

    train = SequenceSlice(0, train_count)
    valid = SequenceSlice(train.end + purge_gap, train.end + purge_gap + valid_count)
    calib = SequenceSlice(valid.end + purge_gap, valid.end + purge_gap + calib_count)
    test = SequenceSlice(calib.end + purge_gap, calib.end + purge_gap + test_count)
    return PurgedFourWaySplit(purge_gap=purge_gap, train=train, valid=valid, calib=calib, test=test)


def build_purged_train_valid_window(
    window_size: int,
    valid_ratio: float = 0.15,
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON_BARS,
) -> PurgedTrainValidWindow:
    if not (0.0 < valid_ratio < 1.0):
        raise ValueError(f"valid_ratio must be between 0 and 1, got {valid_ratio}")

    purge_gap = required_purge_gap(seq_len, horizon)
    usable = window_size - purge_gap
    if usable < 2:
        raise ValueError(
            f"Not enough sequences ({window_size}) for a purged train/valid window with gap={purge_gap}"
        )

    valid_count = max(int(usable * valid_ratio), 1)
    train_count = usable - valid_count
    if train_count <= 0:
        raise ValueError(
            "Purged train/valid window produced no training samples. "
            f"window_size={window_size} gap={purge_gap} valid_count={valid_count}"
        )

    train = SequenceSlice(0, train_count)
    valid = SequenceSlice(train.end + purge_gap, train.end + purge_gap + valid_count)
    return PurgedTrainValidWindow(purge_gap=purge_gap, train=train, valid=valid)


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
