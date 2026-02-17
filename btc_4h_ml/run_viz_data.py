from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from src.data.dataset import load_raw_data, build_dataset
from src.viz.report import generate_data_overview_report, DataReportConfig


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "reports"

parquet_path = DATA_DIR / "btc_4h.parquet"
csv_path = DATA_DIR / "btc_4h.csv"

if parquet_path.exists():
    DATA_PATH = parquet_path
elif csv_path.exists():
    DATA_PATH = csv_path
else:
    raise FileNotFoundError(
        f"No data file found in {DATA_DIR}.\n"
        f"Expected {parquet_path.name} or {csv_path.name}.\n"
        f"Run: python run_download.py"
    )


def _utc_timestamp_dirname() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


if __name__ == "__main__":
    print("Using data file:", DATA_PATH)

    # Raw data (for price/volume)
    raw_df = load_raw_data(DATA_PATH)

    # Featured dataset (returns + momentum + volatility + target)
    featured_df = build_dataset(DATA_PATH, horizon=1, n_lags=12)

    report_dir = OUTPUT_DIR / "data_overview" / _utc_timestamp_dirname()
    cfg = DataReportConfig(
        max_points_price_volume=4000,
        max_points_volatility=10000,
        rolling_vol_window=24,
    )

    generate_data_overview_report(
        raw_df=raw_df,
        featured_df=featured_df,
        out_dir=report_dir,
        cfg=cfg,
    )

    print(f"Saved report to: {report_dir}")
    print("Files:", [p.name for p in sorted(report_dir.glob("*.png"))])
