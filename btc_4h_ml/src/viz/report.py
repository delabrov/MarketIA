from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.viz.plots import (
    plot_price_and_volume,
    plot_log_return_distribution,
    plot_rolling_volatility,
    plot_monthly_up_rate,
)


@dataclass(frozen=True)
class DataReportConfig:
    max_points_price_volume: int = 4000   # limit for readability
    max_points_volatility: int = 8000     # longer view ok
    rolling_vol_window: int = 24          # 24 candles = 4 days


def generate_data_overview_report(
    raw_df: pd.DataFrame,
    featured_df: pd.DataFrame,
    out_dir: Path,
    cfg: DataReportConfig = DataReportConfig(),
) -> None:
    """
    Creates a set of standard plots to validate data/target/features.
    raw_df: should include close, volume
    featured_df: should include log_return and log_return_target
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_price_and_volume(
        raw_df,
        out_path=out_dir / "01_price_volume.png",
        max_points=cfg.max_points_price_volume,
    )

    plot_log_return_distribution(
        featured_df,
        out_path=out_dir / "02_log_return_hist.png",
    )

    plot_rolling_volatility(
        featured_df,
        out_path=out_dir / "03_rolling_volatility.png",
        window=cfg.rolling_vol_window,
        max_points=cfg.max_points_volatility,
    )

    plot_monthly_up_rate(
        featured_df,
        out_path=out_dir / "04_monthly_up_rate.png",
        target_col="log_return_target",
    )
