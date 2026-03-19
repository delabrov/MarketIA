#!/usr/bin/env python3
# download_data.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yfinance as yf


def project_paths() -> Dict[str, Path]:
    """
    Root = market-ml/ (parent of src/)
    """
    ROOT = Path(__file__).resolve().parents[2]
    return {
        "ROOT": ROOT,
        "DATA_RAW": ROOT / "data" / "raw",
        "DATA_PROC": ROOT / "data" / "processed",
        "MODELS": ROOT / "models",
        "RESULTS_RF": ROOT / "results_rf",
        "FIGS_RF": ROOT / "results_rf" / "figures",
        "REPORTS_RF": ROOT / "results_rf" / "reports",
    }


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance can return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    rename_map = {
        "adj close": "adj_close",
        "adjclose": "adj_close",
    }
    df = df.rename(columns=rename_map)

    keep = ["open", "high", "low", "close", "adj_close", "volume"]
    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if ("close" not in df.columns) and ("adj_close" not in df.columns):
        raise RuntimeError(f"Downloaded data missing 'close' and 'adj_close'. Columns: {list(df.columns)}")

    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]

    df = df.dropna(subset=["close"])
    return df


def _download_via_download(
    ticker: str, start: Optional[str], end: Optional[str], use_max: bool
) -> pd.DataFrame:
    kwargs = dict(
        auto_adjust=False,
        actions=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if use_max:
        # IMPORTANT: force maximum history; otherwise yfinance can default to ~1mo
        df = yf.download(ticker, period="max", **kwargs)
    else:
        df = yf.download(ticker, start=start, end=end, **kwargs)

    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned empty dataframe (download).")

    return df


def _download_via_history(ticker: str, start: Optional[str], end: Optional[str], use_max: bool) -> pd.DataFrame:
    t = yf.Ticker(ticker)
    if use_max:
        df = t.history(period="max", auto_adjust=False, actions=False)
    else:
        df = t.history(start=start, end=end, auto_adjust=False, actions=False)

    if df is None or len(df) == 0:
        raise RuntimeError("yfinance returned empty dataframe (history).")

    # history() often returns columns with capitalized names
    df = df.reset_index().set_index("Date") if "Date" in df.columns else df
    return df


def download_clean_ohlcv(ticker: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    use_max = (start is None and end is None)

    # 1) Try yf.download
    try:
        df = _download_via_download(ticker, start, end, use_max=use_max)
    except Exception:
        df = None

    # 2) Fallback to Ticker().history
    if df is None or len(df) == 0:
        df = _download_via_history(ticker, start, end, use_max=use_max)

    df = _normalize_ohlcv(df)

    # sanity: only enforce "too short" when user expects a long history (max)
    if use_max:
        # with max history we expect thousands of rows for AAPL/SPY/VIX
        if len(df) < 500:
            raise RuntimeError(
                f"Downloaded data too short ({len(df)} rows). Something went wrong upstream. Columns: {list(df.columns)}"
            )

    return df


def save_raw(df: pd.DataFrame, out_parquet: Path, out_csv: Path) -> None:
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    df.to_csv(out_csv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="AAPL")
    ap.add_argument("--with_spy", action="store_true")
    ap.add_argument("--with_vix", action="store_true")
    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    args = ap.parse_args()

    P = project_paths()
    P["DATA_RAW"].mkdir(parents=True, exist_ok=True)

    ticker = args.ticker.upper()

    print(f"[download] Downloading {ticker} ...")
    df = download_clean_ohlcv(ticker=ticker, start=args.start, end=args.end)

    out_parquet = P["DATA_RAW"] / f"{ticker.lower()}_ohlcv.parquet"
    out_csv = P["DATA_RAW"] / f"{ticker.lower()}_ohlcv.csv"
    save_raw(df, out_parquet, out_csv)
    print(f"[download] Saved: {out_parquet}")
    print(f"[download] Saved: {out_csv}")
    print(f"[download] Rows: {len(df):,} | Columns: {list(df.columns)}")
    print(f"[download] Date range: {df.index.min()} -> {df.index.max()}")

    if args.with_spy:
        print("[download] Downloading SPY ...")
        spy = download_clean_ohlcv(ticker="SPY", start=args.start, end=args.end)
        save_raw(spy, P["DATA_RAW"] / "spy_ohlcv.parquet", P["DATA_RAW"] / "spy_ohlcv.csv")
        print(f"[download] Saved: {P['DATA_RAW'] / 'spy_ohlcv.parquet'}")
        print(f"[download] Saved: {P['DATA_RAW'] / 'spy_ohlcv.csv'}")
        print(f"[download] Rows: {len(spy):,} | Columns: {list(spy.columns)}")
        print(f"[download] Date range: {spy.index.min()} -> {spy.index.max()}")

    # Always download VIX (required exogenous feature)
    print("[download] Downloading ^VIX ...")
    vix = download_clean_ohlcv(ticker="^VIX", start=args.start, end=args.end)
    save_raw(vix, P["DATA_RAW"] / "vix_ohlcv.parquet", P["DATA_RAW"] / "vix_ohlcv.csv")
    print(f"[download] Saved: {P['DATA_RAW'] / 'vix_ohlcv.parquet'}")
    print(f"[download] Saved: {P['DATA_RAW'] / 'vix_ohlcv.csv'}")
    print(f"[download] Rows: {len(vix):,} | Columns: {list(vix.columns)}")
    print(f"[download] Date range: {vix.index.min()} -> {vix.index.max()}")

    print("[download] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
