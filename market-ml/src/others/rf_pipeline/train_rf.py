#!/usr/bin/env python3
# train_rf.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def project_paths() -> Dict[str, Path]:
    ROOT = Path(__file__).resolve().parents[2]  # market-ml/
    return {
        "ROOT": ROOT,
        "DATA_PROC": ROOT / "data" / "processed",
        "MODELS": ROOT / "models",
        "REPORTS": ROOT / "results_rf" / "reports",
        "FIGS": ROOT / "results_rf" / "figures",
    }


def _load_dataset(ds_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(ds_path)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    return df


def _split_time(df: pd.DataFrame, frac_train: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split = int(len(df) * frac_train)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _compute_mu_up_down(df_train: pd.DataFrame) -> Tuple[float, float]:
    r = df_train["target_next_log_return"].astype(float)
    y = df_train["target_up"].astype(int)
    mu_up = float(r[y == 1].mean())
    mu_down = float(r[y == 0].mean())
    return mu_up, mu_down


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", type=str, default="AAPL")
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--min_samples_leaf", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--start_date", default=None, help="YYYY-MM-DD. If set, keep data >= start_date.")
    args = ap.parse_args()

    P = project_paths()
    P["MODELS"].mkdir(parents=True, exist_ok=True)
    P["REPORTS"].mkdir(parents=True, exist_ok=True)
    P["FIGS"].mkdir(parents=True, exist_ok=True)

    ticker = args.ticker.upper()
    ds_path = P["DATA_PROC"] / f"{ticker.lower()}_dataset.parquet"
    feat_path = P["DATA_PROC"] / f"{ticker.lower()}_features.json"
    if not ds_path.exists() or not feat_path.exists():
        raise FileNotFoundError(f"[train] Missing dataset/features. Run build_dataset.py first.\n{ds_path}\n{feat_path}")

    print("[train] Training RandomForestClassifier on P(next-day log return > 0) ...")

    df = _load_dataset(ds_path)
    if args.start_date:
        start = pd.to_datetime(args.start_date)
        if df.index.tz is not None:
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")
        df = df[df.index >= start]
        print(f"[train] start_date={args.start_date} | rows_after_filter={len(df):,}")
        if len(df) == 0:
            raise RuntimeError("[train] ERROR: dataset is empty after start_date filter.")

    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols: List[str] = meta["feature_cols"]
    feature_groups: Dict[str, List[str]] = meta["feature_groups"]

    train_df, test_df = _split_time(df, frac_train=0.8)

    print(f"[train] Ticker={ticker} | Rows={len(df):,} | Train={len(train_df):,} | Test={len(test_df):,}")
    print(f"[train] Features ({len(feature_cols)}):")
    print(f"  Triviales ({len(feature_groups['triviales'])}): {feature_groups['triviales']}")
    print(f"  Indicateurs ({len(feature_groups['indicateurs'])}): {feature_groups['indicateurs']}")
    print(f"  Exogènes ({len(feature_groups['exogenes'])}): {feature_groups['exogenes']}")

    X_train = train_df[feature_cols]
    y_train = train_df["target_up"].astype(int).values
    X_test = test_df[feature_cols]
    y_test = test_df["target_up"].astype(int).values

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.random_state,
    )
    params = clf.get_params(deep=True)
    print(f"[train] Params: {params}")

    clf.fit(X_train, y_train)

    p_test = clf.predict_proba(X_test)[:, 1]
    yhat = (p_test >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, yhat)),
        "roc_auc": float(roc_auc_score(y_test, p_test)) if len(np.unique(y_test)) > 1 else float("nan"),
        "log_loss": float(log_loss(y_test, np.column_stack([1 - p_test, p_test]), labels=[0, 1])),
    }

    mu_up, mu_down = _compute_mu_up_down(train_df)

    model_path = P["MODELS"] / f"{ticker.lower()}_rf_pup.joblib"
    report_path = P["REPORTS"] / f"{ticker.lower()}_rf_pup_train_report.json"

    bundle: Dict[str, Any] = {
        "model": clf,
        "feature_cols": feature_cols,
        "feature_groups": feature_groups,
        "mu_up": mu_up,
        "mu_down": mu_down,
        "params": params,
        "train_end": train_df.index.max().isoformat(),
        "test_start": test_df.index.min().isoformat(),
        "ticker": ticker,
        "dataset_path": str(ds_path),
    }
    joblib.dump(bundle, model_path)

    train_report = {
        "ticker": ticker,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "feature_cols": feature_cols,
        "feature_groups": feature_groups,
        "params": params,
        "metrics_test_classification": metrics,
        "mu_up": mu_up,
        "mu_down": mu_down,
        "train_end": bundle["train_end"],
        "test_start": bundle["test_start"],
        "model_path": str(model_path),
        "dataset_path": str(ds_path),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(train_report, f, indent=2)

    print(f"[train] Saved model bundle: {model_path}")
    print(f"[train] Saved report: {report_path}")
    print(f"[train] Test metrics (classification): {metrics}")
    print(f"[train] mu_up={mu_up:.6f}, mu_down={mu_down:.6f}")
    print("[train] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
