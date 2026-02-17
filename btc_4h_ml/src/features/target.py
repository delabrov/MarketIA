import numpy as np
import pandas as pd


def add_log_return_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Adds future log return target:
        r_{t,h} = log(C_{t+h} / C_t)

    horizon = number of 4h candles ahead.
    """
    df = df.copy()

    df["log_return_target"] = np.log(
        df["close"].shift(-horizon) / df["close"]
    )

    return df
