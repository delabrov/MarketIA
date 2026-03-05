# src/build_dataset.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import json


def _project_root() -> Path:
    # .../market-ml/src/build_dataset.py -> .../market-ml
    return Path(__file__).resolve().parents[2]


def _paths(root: Path) -> Dict[str, Path]:
    return {
        "data_raw": root / "data" / "raw",
        "data_processed": root / "data" / "processed",
    }


def _read_ohlcv(paths: Dict[str, Path], ticker: str) -> pd.DataFrame:
    t = ticker.lower()
    p = paths["data_raw"] / f"{t}_ohlcv.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing raw file: {p}. Run download_data.py first.")
    df = pd.read_parquet(p)

    # Ensure datetime index in UTC, sorted, unique
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Raw OHLCV must be indexed by DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Lowercase columns
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Slope of a rolling window using simple linear regression on x=0..window-1.
    Returns slope in "price units per day".
    """
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        y = y.astype(float)
        y_mean = y.mean()
        return float(((x - x_mean) * (y - y_mean)).sum() / denom)

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=slow).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


@dataclass(frozen=True)
class FeatureGroups:
    trivial: List[str]
    indicators: List[str]
    exogenous: List[str]


def _build_features_aapl(aapl: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureGroups]:
    """
    Build AAPL-only features + add close_t, close_t1 + targets.
    Output dataframe includes:
      - feature columns
      - close_t, close_t1 (for reconstruction plots)
      - target_next_log_return, target_up
    """
    df = pd.DataFrame(index=aapl.index).copy()

    close = aapl["close"].astype(float)
    open_ = aapl["open"].astype(float) if "open" in aapl.columns else close
    high = aapl["high"].astype(float) if "high" in aapl.columns else close
    low = aapl["low"].astype(float) if "low" in aapl.columns else close
    volume = aapl["volume"].astype(float) if "volume" in aapl.columns else pd.Series(np.nan, index=aapl.index)

    # --- Price anchors for plotting ---
    df["close_t"] = close
    df["close_t1"] = close.shift(-1)

    # --- Features (exact 12, non-exogenous, non-indicators) ---
    # Returns (log returns from close)
    logret = np.log(close / close.shift(1))
    df["return_1d"] = logret
    df["return_2d"] = np.log(close / close.shift(2))
    df["return_5d"] = np.log(close / close.shift(5))
    df["return_10d"] = np.log(close / close.shift(10))

    # Volatility (rolling std of return_1d)
    r1 = df["return_1d"]
    df["volatility_5d"] = r1.rolling(5, min_periods=5).std()
    df["volatility_10d"] = r1.rolling(10, min_periods=10).std()
    df["volatility_20d"] = r1.rolling(20, min_periods=20).std()

    # Volume
    vol = volume.replace(0.0, np.nan)
    df["volume_change_1d"] = np.log(vol / vol.shift(1))
    df["volume_ratio_20d"] = vol / vol.rolling(20, min_periods=20).mean()

    # Price action OHLC
    prev_close = close.shift(1).replace(0.0, np.nan)
    df["gap_1d"] = (open_ - prev_close) / prev_close
    df["range_hl_1d"] = (high - low) / close.replace(0.0, np.nan)
    df["open_to_close_1d"] = (close - open_) / open_.replace(0.0, np.nan)

    # Intraday structure features (OHLC-based)
    daily_range = (high - low)
    rng = daily_range.replace(0.0, np.nan)
    clv = ((close - low) - (high - close)) / rng
    body_to_range = (close - open_).abs() / rng
    upper_wick_ratio = (high - np.maximum(open_, close)) / rng
    lower_wick_ratio = (np.minimum(open_, close) - low) / rng
    range_ratio_20 = daily_range / daily_range.rolling(20, min_periods=20).mean()

    df["clv"] = clv.fillna(0.0)
    df["body_to_range"] = body_to_range.fillna(0.0)
    df["upper_wick_ratio"] = upper_wick_ratio.fillna(0.0)
    df["lower_wick_ratio"] = lower_wick_ratio.fillna(0.0)
    df["range_ratio_20"] = range_ratio_20.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- Targets ---
    df["target_next_log_return"] = np.log(df["close_t1"] / df["close_t"])
    df["target_up"] = (df["target_next_log_return"] > 0.0).astype(int)

    trivial = [
        "return_1d",
        "return_2d",
        "return_5d",
        "return_10d",
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "volume_change_1d",
        "volume_ratio_20d",
        "gap_1d",
        "range_hl_1d",
        "open_to_close_1d",
        "clv",
        "body_to_range",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "range_ratio_20",
    ]
    indicators: List[str] = []
    exogenous: List[str] = []  # filled later
    return df, FeatureGroups(trivial=trivial, indicators=indicators, exogenous=exogenous)


def _add_exogenous(df: pd.DataFrame, paths: Dict[str, Path], use_spy: bool, use_vix: bool) -> Tuple[pd.DataFrame, List[str]]:
    exog_cols: List[str] = []
    out = df.copy()

    if use_spy:
        spy = _read_ohlcv(paths, ticker="SPY")
        spy_close = spy["close"].astype(float)
        out["spy_return_1d"] = np.log(spy_close / spy_close.shift(1))
        out["spy_return_2d"] = np.log(spy_close / spy_close.shift(2))
        out["spy_return_10d"] = np.log(spy_close / spy_close.shift(10))
        exog_cols += ["spy_return_1d", "spy_return_2d", "spy_return_10d"]

    if use_vix:
        vix = _read_ohlcv(paths, ticker="VIX")
        # VIX level = close (not return)
        out["vix_level"] = vix["close"].astype(float)
        exog_cols += ["vix_level"]

    # Align on intersection of dates after adding exog
    out = out.sort_index()
    return out, exog_cols


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--use_spy", action="store_true")
    ap.add_argument("--use_vix", action="store_true")
    ap.add_argument("--start_date", default=None, help="YYYY-MM-DD. If set, keep data >= start_date.")
    args = ap.parse_args()

    root = _project_root()
    paths = _paths(root)
    paths["data_processed"].mkdir(parents=True, exist_ok=True)

    ticker = args.ticker.upper()
    print(f"[build] Building dataset from raw {ticker} data ...")

    aapl = _read_ohlcv(paths, ticker=ticker)
    if args.start_date:
        start = pd.to_datetime(args.start_date)
        if aapl.index.tz is not None:
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")
        aapl = aapl[aapl.index >= start]
        print(f"[build] start_date={args.start_date} | rows_after_filter={len(aapl):,}")
        if len(aapl) == 0:
            raise RuntimeError("[build] ERROR: dataset is empty after start_date filter.")
    base, groups = _build_features_aapl(aapl)

    # Add required exogenous VIX feature (always)
    df, exog_cols = _add_exogenous(base, paths, use_spy=False, use_vix=True)
    # Ensure aligned price columns exist in final dataset
    df["close_t"] = aapl["close"].astype(float)
    df["close_t1"] = aapl["close"].astype(float).shift(-1)
    groups = FeatureGroups(trivial=groups.trivial, indicators=groups.indicators, exogenous=exog_cols)

    # Final feature list
    feature_cols = groups.trivial + groups.indicators + groups.exogenous

    # Keep required plot columns too (close_t, close_t1) + targets
    keep_cols = feature_cols + ["close_t", "close_t1", "target_next_log_return", "target_up"]

    df = df[keep_cols]

    # Hard clean: remove inf, then dropna
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) == 0:
        raise RuntimeError("[build] ERROR: dataset is empty after dropna().")

    out_parquet = paths["data_processed"] / "aapl_dataset.parquet"
    out_csv = paths["data_processed"] / "aapl_dataset.csv"
    out_features = paths["data_processed"] / "aapl_features.json"
    df.to_parquet(out_parquet)
    df.to_csv(out_csv)

    features_meta = {
        "ticker": ticker,
        "feature_cols": feature_cols,
        "feature_groups": {
            "triviales": groups.trivial,
            "indicateurs": groups.indicators,
            "exogenes": groups.exogenous,
        },
        "dataset_path": str(out_parquet),
        "date_min": str(df.index.min()),
        "date_max": str(df.index.max()),
        "rows": int(len(df)),
    }
    with open(out_features, "w", encoding="utf-8") as f:
        json.dump(features_meta, f, indent=2)

    print(f"[build] Saved: {out_parquet}")
    print(f"[build] Saved: {out_csv}")
    print(f"[build] Rows: {len(df):,} | Features: {len(feature_cols)} | Targets: 2")
    print(f"[build] Date range: {df.index.min()} -> {df.index.max()}")
    print("[build] Feature NA% (post-clean) = 0 by construction.")
    print(f"[build] Features ({len(feature_cols)}):")
    print(f"  Triviales ({len(groups.trivial)}): {groups.trivial}")
    print(f"  Indicateurs ({len(groups.indicators)}): {groups.indicators}")
    print(f"  Exogènes ({len(groups.exogenous)}): {groups.exogenous}")
    print(f"[build] Saved features meta: {out_features}")
    print("[build] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
