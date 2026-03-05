from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def build_features(
    df: pd.DataFrame,
    use_vix: bool = True,
    use_spy: bool = True,
    use_exog: bool = True,
    use_regime_features: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build features for LSTM. Returns (feature_df, feature_cols).
    Assumes df contains AAPL OHLCV.
    """
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    open_ = df["open"].astype(float) if "open" in df.columns else close
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(np.nan, index=df.index)

    # Base returns
    logret_1d = np.log(close / close.shift(1))
    out["log_return_1d"] = logret_1d
    out["log_return_5d"] = np.log(close / close.shift(5))
    out["log_return_20d"] = np.log(close / close.shift(20))

    # EMA of returns
    out["ema_return_5"] = logret_1d.ewm(span=5, adjust=False, min_periods=5).mean()
    out["ema_return_20"] = logret_1d.ewm(span=20, adjust=False, min_periods=20).mean()

    # Realized volatility (sqrt mean of squared returns)
    out["realized_vol_5d"] = np.sqrt((logret_1d ** 2).rolling(5, min_periods=5).mean())
    out["realized_vol_20d"] = np.sqrt((logret_1d ** 2).rolling(20, min_periods=20).mean())

    # Vol ratio
    vol20 = out["realized_vol_20d"].replace(0.0, np.nan)
    out["vol_ratio_5_20"] = out["realized_vol_5d"] / vol20

    prev_close = close.shift(1)

    # Gaps and intraday moves (log)
    out["gap_log"] = np.log(open_ / prev_close.replace(0.0, np.nan))
    out["intraday_log"] = np.log(close / open_.replace(0.0, np.nan))

    # Volume z-score (log volume, 20d)
    v = np.log(volume.replace(0.0, np.nan))
    v_mean = v.rolling(20, min_periods=20).mean()
    v_std = v.rolling(20, min_periods=20).std()
    out["volume_z_20"] = (v - v_mean) / v_std.replace(0.0, np.nan)

    # Day of week (0=Mon .. 6=Sun)
    out["day_of_week"] = pd.Series(df.index.dayofweek, index=df.index).astype(float)

    # Z-score price vs MA20
    ma20 = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    out["zscore_price_vs_ma20"] = (close - ma20) / (std20 + 1e-12)

    # Candle features
    rng = (high - low).replace(0.0, np.nan)
    out["clv"] = (((close - low) - (high - close)) / rng).fillna(0.0)
    out["body_to_range"] = ((close - open_).abs() / rng).fillna(0.0)

    feature_cols = [
        "log_return_1d",
        "log_return_5d",
        "log_return_20d",
        "ema_return_5",
        "ema_return_20",
        "realized_vol_5d",
        "realized_vol_20d",
        "vol_ratio_5_20",
        "gap_log",
        "intraday_log",
        "volume_z_20",
        "day_of_week",
        "zscore_price_vs_ma20",
        "clv",
        "body_to_range",
    ]

    if use_regime_features:
        rv60 = np.sqrt((logret_1d ** 2).rolling(60, min_periods=60).mean())
        vol60 = rv60.replace(0.0, np.nan)
        out["vol_ratio_5_60"] = out["realized_vol_5d"] / vol60
        out["rolling_skew_60"] = logret_1d.rolling(60, min_periods=60).skew()
        out["rolling_kurtosis_60"] = logret_1d.rolling(60, min_periods=60).kurt()
        feature_cols += [
            "vol_ratio_5_60",
            "rolling_skew_60",
            "rolling_kurtosis_60",
        ]

    # Optional exogenous
    if use_exog:
        if use_vix:
            if "vix_level" not in df.columns:
                raise RuntimeError("[lstm] VIX data missing but use_vix=true. Enable VIX download/load.")
            vix = df["vix_level"].astype(float)
            out["vix_level"] = vix
            out["vix_change_1d"] = np.log(vix / vix.shift(1))
            vix_mean_60 = vix.rolling(60, min_periods=60).mean()
            vix_std_60 = vix.rolling(60, min_periods=60).std()
            out["vix_zscore_60"] = (vix - vix_mean_60) / (vix_std_60 + 1e-12)
            feature_cols += ["vix_level", "vix_change_1d", "vix_zscore_60"]
        if use_spy and "spy_close" in df.columns:
            spy = df["spy_close"].astype(float)
            out["spy_return_1d"] = np.log(spy / spy.shift(1))
            feature_cols += ["spy_return_1d"]

    # Clean infs
    out = out.replace([np.inf, -np.inf], np.nan)
    return out, feature_cols
