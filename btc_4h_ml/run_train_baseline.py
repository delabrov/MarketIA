from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.dataset import build_dataset
from src.validation.walk_forward import WalkForwardConfig, walk_forward_splits
from src.models.baseline import make_logistic_baseline
from src.models.training import run_walk_forward_classification


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

parquet_path = DATA_DIR / "btc_4h.parquet"
csv_path = DATA_DIR / "btc_4h.csv"

if parquet_path.exists():
    DATA_PATH = parquet_path
elif csv_path.exists():
    DATA_PATH = csv_path
else:
    raise FileNotFoundError(f"No data file found in {DATA_DIR}. Run: python run_download.py")


def make_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y_up"] = (df["log_return_target"] > 0).astype(int)
    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Baseline features: use the engineered features (returns lags, momentum, volatility)
    and optionally volume-related raw features.
    We avoid using raw 'open/high/low/close' for now to keep things stationary-ish.
    """
    engineered_prefixes = ("log_return", "momentum_", "volatility_")

    cols = []
    for c in df.columns:
        if c == "log_return_target":
            continue
        if c == "y_up":
            continue
        if c == "close_time":
            continue

        if c.startswith(engineered_prefixes):
            cols.append(c)

    # Add volume (optional but often useful)
    if "volume" in df.columns:
        cols.append("volume")

    # Deduplicate while preserving order
    seen = set()
    cols_unique = []
    for c in cols:
        if c not in seen:
            cols_unique.append(c)
            seen.add(c)

    return cols_unique


if __name__ == "__main__":
    print("Using data file:", DATA_PATH)

    # Build dataset with features + log_return_target
    df = build_dataset(DATA_PATH, horizon=1, n_lags=12)

    # Create classification target
    df = make_classification_target(df)

    feature_cols = select_feature_columns(df)
    print(f"Num rows: {len(df)} | Num features: {len(feature_cols)}")
    print("First feature columns:", feature_cols[:10])

    # Walk-forward config (in number of 4h candles)
    # 1 day = 6 candles
    candles_per_day = 6
    cfg = WalkForwardConfig(
        train_size=365 * candles_per_day * 2,  # 2 years
        val_size=90 * candles_per_day,         # 3 months
        test_size=90 * candles_per_day,        # 3 months
        step_size=90 * candles_per_day,        # slide by test window
        embargo=1,                             # 1 candle gap to be conservative
    )

    splits = list(walk_forward_splits(df, cfg))
    print(f"Num folds: {len(splits)}")
    if len(splits) == 0:
        raise RuntimeError("Not enough data for the chosen train/val/test sizes.")

    model = make_logistic_baseline(random_state=42)

    results = run_walk_forward_classification(
        df=df,
        feature_cols=feature_cols,
        target_col="y_up",
        splits=splits,
        model=model,
    )

    # Print fold reports
    for r in results:
        print(f"\n=== Fold {r.fold_id} ===")
        print(f"Train: {r.train_range[0]} -> {r.train_range[1]}")
        print(f"Val:   {r.val_range[0]} -> {r.val_range[1]}")
        print(f"Test:  {r.test_range[0]} -> {r.test_range[1]}")
        print(f"Val  : logloss={r.val_report.logloss:.4f}, auc={r.val_report.roc_auc:.4f}, brier={r.val_report.brier:.4f}, acc@0.5={r.val_report.accuracy_at_0_5:.4f}, pos_rate={r.val_report.positive_rate:.4f}")
        print(f"Test : logloss={r.test_report.logloss:.4f}, auc={r.test_report.roc_auc:.4f}, brier={r.test_report.brier:.4f}, acc@0.5={r.test_report.accuracy_at_0_5:.4f}, pos_rate={r.test_report.positive_rate:.4f}")
