from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"


@dataclass
class BinanceKlinesConfig:
    symbol: str = "BTCUSDT"
    interval: str = "4h"
    limit: int = 1000          # max Binance = 1000
    pause_s: float = 0.25      # pause anti rate-limit
    timeout_s: float = 20.0
    max_retries: int = 5


def _to_milliseconds(ts: str) -> int:
    """Parse timestamp string as UTC and convert to milliseconds."""
    return int(pd.Timestamp(ts, tz="UTC").value // 10**6)


def _request_with_retries(
    session: requests.Session,
    url: str,
    params: dict,
    timeout_s: float,
    max_retries: int,
) -> list:
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, params=params, timeout=timeout_s)
            if r.status_code == 429:
                # Rate limit => exponential backoff
                wait = min(2**attempt, 30)
                time.sleep(wait)
                continue

            r.raise_for_status()
            return r.json()

        except Exception as e:
            last_err = e
            wait = min(2**attempt, 30)
            time.sleep(wait)

    raise RuntimeError(f"Binance request failed after retries. Last error: {last_err}")


def fetch_ohlcv_binance(
    start: str,
    end: Optional[str] = None,
    cfg: BinanceKlinesConfig = BinanceKlinesConfig(),
) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance API (/api/v3/klines), paginating until `end`.

    Returns a DataFrame indexed by open_time (UTC), with columns including close_time.
    """
    start_ms = _to_milliseconds(start)
    end_ms = _to_milliseconds(end) if end else None

    url = BINANCE_BASE_URL + KLINES_ENDPOINT
    all_rows: list[list] = []

    with requests.Session() as session:
        next_start = start_ms

        while True:
            params = {
                "symbol": cfg.symbol,
                "interval": cfg.interval,
                "limit": cfg.limit,
                "startTime": next_start,
            }
            if end_ms is not None:
                params["endTime"] = end_ms

            data = _request_with_retries(
                session=session,
                url=url,
                params=params,
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
            )

            if not data:
                break

            all_rows.extend(data)

            last_open_time = int(data[-1][0])
            last_close_time = int(data[-1][6])

            # Stop if we reached end
            if end_ms is not None and last_close_time >= end_ms:
                break

            # Continue after last close_time (add 1ms to avoid duplicates)
            # Note: this is OK because we'll index on open_time (aligned on 4h boundaries)
            next_start = last_close_time + 1

            # Safety against infinite loops if API returns same candle
            if last_open_time == next_start:
                next_start = last_close_time + 1

            time.sleep(cfg.pause_s)

    if not all_rows:
        return pd.DataFrame()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(all_rows, columns=cols)

    # Convert numeric columns
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_asset_volume",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert times
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # Clean
    df = df.drop(columns=["ignore"])
    df = df.sort_values("open_time").reset_index(drop=True)

    # IMPORTANT: index on open_time (exact 4h grid)
    df = df.set_index("open_time", drop=True)
    df = df[~df.index.duplicated(keep="last")]

    return df
