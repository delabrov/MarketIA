import pandas as pd


def check_frequency(df: pd.DataFrame, expected_freq: str = "4h") -> None:
    df = df.sort_index()
    inferred = pd.infer_freq(df.index)
    print(f"Inferred frequency: {inferred}")
    if inferred != expected_freq:
        print(f"WARNING: Frequency mismatch. Expected {expected_freq}, got {inferred}.")


def check_missing_timestamps(df: pd.DataFrame, freq: str = "4h") -> None:
    df = df.sort_index()
    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=freq,
        tz="UTC",
    )
    missing = full_index.difference(df.index)

    if len(missing) > 0:
        print(f"Missing timestamps: {len(missing)}")
        print("First missing timestamps:", missing[:5].to_list())
    else:
        print("No missing timestamps detected.")


def basic_sanity_checks(df: pd.DataFrame) -> None:
    df = df.sort_index()

    print("Checking duplicates...")
    print(int(df.index.duplicated().sum()), "duplicates")

    print("Checking non-positive closes...")
    if "close" in df.columns:
        print(int((df["close"] <= 0).sum()), "non-positive closes")
    else:
        print("No 'close' column found.")

    print("Checking NaNs per column...")
    print(df.isna().sum())
