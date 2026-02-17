from pathlib import Path

from src.data.download import fetch_ohlcv_binance, BinanceKlinesConfig
from src.data.validation import (
    check_frequency,
    check_missing_timestamps,
    basic_sanity_checks,
)

# Root du projet = dossier où se trouve CE script
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    cfg = BinanceKlinesConfig(symbol="BTCUSDT", interval="4h")

    print("Downloading BTC 4h data...")
    df = fetch_ohlcv_binance(start="2020-01-01", end=None, cfg=cfg)

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    print("Running validation...")
    check_frequency(df, expected_freq="4h")
    check_missing_timestamps(df, freq="4h")
    basic_sanity_checks(df)

    output_parquet = DATA_DIR / "btc_4h.parquet"
    output_csv = DATA_DIR / "btc_4h.csv"

    # Save parquet if possible, else fallback to csv
    try:
        df.to_parquet(output_parquet)
        print(f"Saved to {output_parquet}")
    except ImportError:
        print("Parquet engine missing (pyarrow/fastparquet). Falling back to CSV.")
        print("Install with: pip install pyarrow")
        df.to_csv(output_csv)
        print(f"Saved to {output_csv}")

    print("Files in data/raw:", [p.name for p in DATA_DIR.glob("*")])
