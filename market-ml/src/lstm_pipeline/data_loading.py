from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def project_paths() -> Dict[str, Path]:
    root = Path(__file__).resolve().parents[2]  # market-ml/
    return {
        "ROOT": root,
        "DATA_RAW": root / "data" / "raw",
    }


def _read_ohlcv(paths: Dict[str, Path], ticker: str) -> pd.DataFrame:
    t = ticker.lower()
    p = paths["DATA_RAW"] / f"{t}_ohlcv.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Missing raw file: {p}. Run download_data.py first.")
    df = pd.read_parquet(p)

    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("Raw OHLCV must be indexed by DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    df.columns = [str(c).lower() for c in df.columns]
    return df


def load_base_and_exog(
    ticker: str = "AAPL",
    start_date: Optional[str] = None,
    use_vix: bool = True,
    use_spy: bool = True,
) -> pd.DataFrame:
    paths = project_paths()
    base = _read_ohlcv(paths, ticker)

    if start_date:
        start = pd.to_datetime(start_date)
        if base.index.tz is not None:
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")
        base = base[base.index >= start]
        if len(base) == 0:
            raise RuntimeError("[lstm] ERROR: dataset is empty after start_date filter.")

    out = base.copy()

    if use_vix:
        vix = _read_ohlcv(paths, "VIX")
        vix = vix[["close"]].rename(columns={"close": "vix_level"})
        out = out.join(vix, how="inner")

    if use_spy:
        spy = _read_ohlcv(paths, "SPY")
        spy = spy[["close"]].rename(columns={"close": "spy_close"})
        out = out.join(spy, how="inner")

    out = out.sort_index()
    return out
