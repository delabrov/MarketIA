from pathlib import Path
import pandas as pd

from src.features.target import add_log_return_target
from src.features.returns import (
    add_log_returns,
    add_return_lags,
    add_rolling_momentum,
    add_rolling_volatility,
)


def load_raw_data(path: Path) -> pd.DataFrame:
    """
    Load raw data from parquet or csv.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix == ".parquet":
        try:
            df = pd.read_parquet(path)
        except ImportError:
            raise ImportError(
                "Parquet file detected but no engine installed.\n"
                "Install with: pip install pyarrow\n"
                "Or use the CSV file instead."
            )
    elif path.suffix == ".csv":
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError("Unsupported file format. Use .parquet or .csv")

    df = df.sort_index()
    return df


def build_dataset(
    path: Path,
    horizon: int = 1,
    n_lags: int = 12,
) -> pd.DataFrame:
    """
    Full dataset builder:
        - load raw data
        - compute returns
        - compute features
        - compute target
        - drop NaNs
    """
    df = load_raw_data(path)

    # --- Features ---
    df = add_log_returns(df)
    df = add_return_lags(df, n_lags=n_lags)
    df = add_rolling_momentum(df)
    df = add_rolling_volatility(df)

    # --- Target ---
    df = add_log_return_target(df, horizon=horizon)

    # Drop NaNs created by rolling + shifting
    df = df.dropna()

    return df
