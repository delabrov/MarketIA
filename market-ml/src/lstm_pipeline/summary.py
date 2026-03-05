from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    if len(a) < 2:
        return float("nan")
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    c = s1.corr(s2, method=method)
    return float(c) if c is not None and np.isfinite(c) else float("nan")


def _fit_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "fit_corr": float("nan"),
            "fit_r2": float("nan"),
        }
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    y_fit = slope * y_true + intercept
    ss_res = np.sum((y_pred - y_fit) ** 2)
    ss_tot = np.sum((y_pred - np.mean(y_pred)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else float("nan")
    corr = _safe_corr(y_true, y_pred, method="pearson")
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_corr": float(corr),
        "fit_r2": float(r2),
    }


def _decile_spread(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(df) == 0:
        return float("nan")
    try:
        df["decile"] = pd.qcut(df["y_pred"], q=n_deciles, labels=False, duplicates="drop")
    except Exception:
        return float("nan")
    if df["decile"].nunique() < 2:
        return float("nan")
    top = df[df["decile"] == df["decile"].max()]
    bot = df[df["decile"] == df["decile"].min()]
    return float(top["y_true"].mean() - bot["y_true"].mean())


def metrics_by_horizon(
    preds_df: pd.DataFrame,
    metrics_pred: Dict[str, Dict[str, Dict[str, float]]],
    horizons: List[int],
    split: str = "test",
    deciles_n: int = 10,
) -> pd.DataFrame:
    rows = []
    for h in horizons:
        y_true_col = f"y_true_h{h}"
        y_pred_col = f"y_pred_h{h}"
        if y_true_col not in preds_df.columns or y_pred_col not in preds_df.columns:
            continue

        df_h = preds_df[preds_df["split"] == split].copy()
        y_true = df_h[y_true_col].astype(float).to_numpy()
        y_pred = df_h[y_pred_col].astype(float).to_numpy()
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        n = len(y_true)
        if n == 0:
            continue

        residual = y_true - y_pred

        # Pull metrics from report if available
        m = metrics_pred.get(f"horizon_{h}", {}).get(split, {})

        row = {
            "h": h,
            "n_samples": int(n),
            "ic_pearson": float(m.get("ic", np.nan)),
            "ic_spearman": float(m.get("rank_ic", np.nan)),
            "mse": float(m.get("mse", np.nan)),
            "rmse": float(m.get("rmse", np.nan)),
            "mae": float(m.get("mae", np.nan)),
            "median_ae": float(m.get("medae", np.nan)),
            "hit_ratio": float(m.get("hit_ratio", np.nan)),
        }

        fit = _fit_stats(y_true, y_pred)
        row.update(
            {
                "linear_fit_slope": fit["slope"],
                "linear_fit_intercept": fit["intercept"],
                "linear_fit_corr": fit["fit_corr"],
                "linear_fit_r2": fit["fit_r2"],
            }
        )

        row["decile_spread"] = _decile_spread(y_true, y_pred, n_deciles=deciles_n)
        row["residual_std"] = float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan")
        row["residual_skew"] = float(pd.Series(residual).skew()) if len(residual) > 2 else float("nan")
        row["residual_kurtosis"] = float(pd.Series(residual).kurt()) if len(residual) > 3 else float("nan")

        rows.append(row)

    return pd.DataFrame(rows).sort_values("h")
