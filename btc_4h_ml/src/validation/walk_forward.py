from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int
    val_size: int
    test_size: int
    step_size: Optional[int] = None  # if None, defaults to test_size
    embargo: int = 0                 # number of samples to drop between train and val/test


def walk_forward_splits(
    df: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> Iterator[Tuple[pd.Index, pd.Index, pd.Index]]:
    """
    Generate walk-forward splits using integer positions, returning index slices:
      (train_index, val_index, test_index)

    The split is:
      train: [t0, t0+train_size)
      embargo: [.. + embargo)
      val:   next val_size
      embargo
      test:  next test_size

    Then slide forward by step_size.
    """
    n = len(df)
    step = cfg.step_size if cfg.step_size is not None else cfg.test_size

    start = 0
    while True:
        train_start = start
        train_end = train_start + cfg.train_size

        val_start = train_end + cfg.embargo
        val_end = val_start + cfg.val_size

        test_start = val_end + cfg.embargo
        test_end = test_start + cfg.test_size

        if test_end > n:
            break

        train_idx = df.index[train_start:train_end]
        val_idx = df.index[val_start:val_end]
        test_idx = df.index[test_start:test_end]

        yield train_idx, val_idx, test_idx

        start += step
