from pathlib import Path
from src.data.dataset import build_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

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

if __name__ == "__main__":
    print("Using data file:", DATA_PATH)
    df = build_dataset(DATA_PATH, horizon=1, n_lags=12)

    print(df.head())
    print(df.tail())
    print("Shape:", df.shape)
