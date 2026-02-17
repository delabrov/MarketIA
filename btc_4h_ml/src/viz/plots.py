from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_price_and_volume(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "BTC 4h - Price & Volume",
    max_points: Optional[int] = None,
) -> None:
    """
    Plot close price (line) and volume (bars) over time.
    """
    _ensure_dir(out_path.parent)

    d = df.copy()
    if max_points is not None and len(d) > max_points:
        d = d.iloc[-max_points:]

    fig = plt.figure(figsize=(12, 6))
    ax_price = fig.add_subplot(111)

    ax_price.plot(d.index, d["close"].astype(float).values)
    ax_price.set_title(title)
    ax_price.set_xlabel("Time")
    ax_price.set_ylabel("Close price")

    # Volume on secondary axis
    ax_vol = ax_price.twinx()
    ax_vol.bar(d.index, d["volume"].astype(float).values, alpha=0.25, width=0.12)
    ax_vol.set_ylabel("Volume")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_log_return_distribution(
    df: pd.DataFrame,
    out_path: Path,
    title: str = "BTC 4h - Log return distribution",
    bins: int = 120,
) -> None:
    """
    Histogram of 1-step log returns + basic stats annotation.
    Expects a 'log_return' column.
    """
    _ensure_dir(out_path.parent)

    r = df["log_return"].dropna().astype(float).values
    mu = float(np.mean(r))
    sigma = float(np.std(r))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    ax.hist(r, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("log_return")
    ax.set_ylabel("count")
    ax.axvline(mu, linestyle="--")

    txt = f"mean={mu:.6f}\nstd={sigma:.6f}\nN={len(r)}"
    ax.text(0.98, 0.95, txt, transform=ax.transAxes, ha="right", va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rolling_volatility(
    df: pd.DataFrame,
    out_path: Path,
    window: int = 24,
    title: Optional[str] = None,
    max_points: Optional[int] = None,
) -> None:
    """
    Rolling volatility of 1-step log returns.
    window=24 means 24 * 4h = 4 days.
    Expects a 'log_return' column.
    """
    _ensure_dir(out_path.parent)

    d = df.copy()
    if max_points is not None and len(d) > max_points:
        d = d.iloc[-max_points:]

    vol = d["log_return"].rolling(window=window).std()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(d.index, vol.values)
    if title is None:
        title = f"BTC 4h - Rolling volatility (window={window} candles)"
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Rolling std(log_return)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_monthly_up_rate(
    df: pd.DataFrame,
    out_path: Path,
    target_col: str = "log_return_target",
    title: str = "BTC 4h - Monthly up-rate (target > 0)",
) -> None:
    """
    Monthly proportion of 'up' outcomes based on target_col > 0.
    """
    _ensure_dir(out_path.parent)

    d = df.copy()
    y = (d[target_col] > 0).astype(int)
    monthly = y.resample("MS").mean()  # month start
    monthly = monthly.dropna()

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.plot(monthly.index, monthly.values)
    ax.axhline(0.5, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Up rate")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
