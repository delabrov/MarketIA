import numpy as np
import pandas as pd


def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 1-step log return:
        r_t = log(C_t / C_{t-1})
    """
    df = df.copy()

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    return df


def add_return_lags(df: pd.DataFrame, n_lags: int = 12) -> pd.DataFrame:
    """
    Add lagged returns:
        r_{t-1}, r_{t-2}, ..., r_{t-n}
    """
    df = df.copy()

    for i in range(1, n_lags + 1):
        df[f"log_return_lag_{i}"] = df["log_return"].shift(i)

    return df


def add_rolling_momentum(df: pd.DataFrame, windows=(3, 6, 12, 24)) -> pd.DataFrame:
    """
    Add rolling cumulative returns (momentum).
    """
    df = df.copy()

    for w in windows:
        df[f"momentum_{w}"] = (
            df["log_return"].rolling(window=w).sum()
        )

    return df


def add_rolling_volatility(df: pd.DataFrame, windows=(6, 12, 24)) -> pd.DataFrame:
    """
    Add rolling realized volatility.
    """
    df = df.copy()

    for w in windows:
        df[f"volatility_{w}"] = (
            df["log_return"].rolling(window=w).std()
        )

    return df
