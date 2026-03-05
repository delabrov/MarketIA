from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class SplitConfig:
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train_end_date: Optional[str] = None
    val_end_date: Optional[str] = None


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def _date_to_pos(index: pd.DatetimeIndex, date_str: str) -> int:
    ts = pd.to_datetime(date_str)
    if index.tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
    pos = index.searchsorted(ts, side="right") - 1
    return int(max(pos, 0))


def compute_split_masks(index: pd.DatetimeIndex, cfg: SplitConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(index)
    if cfg.train_end_date or cfg.val_end_date:
        train_end = _date_to_pos(index, cfg.train_end_date) + 1 if cfg.train_end_date else int(n * cfg.train_ratio)
        val_end = _date_to_pos(index, cfg.val_end_date) + 1 if cfg.val_end_date else int(n * (cfg.train_ratio + cfg.val_ratio))
    else:
        train_end = int(n * cfg.train_ratio)
        val_end = int(n * (cfg.train_ratio + cfg.val_ratio))

    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    idx = np.arange(n)
    train_mask = idx < train_end
    val_mask = (idx >= train_end) & (idx < val_end)
    test_mask = idx >= val_end
    return train_mask, val_mask, test_mask


def scale_features(
    features: pd.DataFrame,
    train_mask: np.ndarray,
) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = features.values.astype(float)
    scaler.fit(X[train_mask])
    X_scaled = scaler.transform(X)
    return X_scaled, scaler


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(y)
    if n == 0:
        return np.empty((0, seq_len, X.shape[1])), np.array([]), np.array([]), np.array([])

    start = seq_len - 1
    idxs = np.arange(start, n)
    X_seq = np.stack([X[i - seq_len + 1 : i + 1] for i in idxs], axis=0)
    y_seq = y[idxs]
    dates_seq = dates[idxs]
    return X_seq, y_seq, dates_seq, idxs


def create_loaders(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    idxs: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_idx = idxs[train_mask[idxs]]
    val_idx = idxs[val_mask[idxs]]
    test_idx = idxs[test_mask[idxs]]

    X_train = X_seq[train_mask[idxs]]
    y_train = y_seq[train_mask[idxs]]
    X_val = X_seq[val_mask[idxs]]
    y_val = y_seq[val_mask[idxs]]
    X_test = X_seq[test_mask[idxs]]
    y_test = y_seq[test_mask[idxs]]

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val)
    test_ds = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
