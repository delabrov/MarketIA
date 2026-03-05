from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    explained_variance_score,
    balanced_accuracy_score,
)


@dataclass
class StrategyConfig:
    mode: str = "sign"  # sign or threshold
    threshold: float = 0.0
    fee_bps: float = 0.0
    slippage_bps: float = 0.0


def predict(model, loader, device: str) -> np.ndarray:
    model.eval()
    preds = []
    dev = torch.device(device)
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev)
            y = model(xb).detach().cpu().numpy()
            preds.append(y)
    if not preds:
        return np.array([])
    return np.concatenate(preds, axis=0)


def predict_array(model, X: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    dev = torch.device(device)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32, device=dev)
            y = model(xb).detach().cpu().numpy()
            preds.append(y)
    if not preds:
        return np.array([])
    return np.concatenate(preds, axis=0)


def compute_positions(y_hat: np.ndarray, cfg: StrategyConfig) -> np.ndarray:
    if cfg.mode == "threshold":
        thr = float(cfg.threshold)
        pos = np.where(np.abs(y_hat) >= thr, np.sign(y_hat), 0.0)
    else:
        pos = np.sign(y_hat)
    return pos.astype(float)


def apply_costs(positions: np.ndarray, fee_bps: float, slippage_bps: float) -> np.ndarray:
    if len(positions) == 0:
        return np.array([])
    cost_bps = float(fee_bps) + float(slippage_bps)
    if cost_bps <= 0:
        return np.zeros_like(positions, dtype=float)
    delta = np.abs(np.diff(np.concatenate([[0.0], positions])))
    return (cost_bps / 10000.0) * delta


def equity_curve(logret: np.ndarray) -> np.ndarray:
    if len(logret) == 0:
        return np.array([])
    return np.exp(np.cumsum(logret))


def max_drawdown(eq: np.ndarray) -> float:
    if len(eq) == 0:
        return float("nan")
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if len(y_true) == 0:
        return {"mse": float("nan"), "mae": float("nan"), "ic": float("nan"), "hit_ratio": float("nan")}
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ic = float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_true) > 0 and np.std(y_pred) > 0 else float("nan")
    hit = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    return {"mse": mse, "mae": mae, "ic": ic, "hit_ratio": hit}


def _safe_corr(a: np.ndarray, b: np.ndarray, method: str = "pearson") -> float:
    if len(a) == 0:
        return float("nan")
    s1 = pd.Series(a)
    s2 = pd.Series(b)
    corr = s1.corr(s2, method=method)
    return float(corr) if corr is not None and np.isfinite(corr) else float("nan")


def _autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag:
        return float("nan")
    a = x[lag:]
    b = x[:-lag]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compute_pred_metrics_extended(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    mape_epsilon: float = 1e-6,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0:
        return {}

    residual = y_true - y_pred
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    evs = float(explained_variance_score(y_true, y_pred))

    # MAPE and SMAPE
    mask = np.abs(y_true) > mape_epsilon
    if mask.any():
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))
    else:
        mape = float("nan")
    denom = np.abs(y_true) + np.abs(y_pred) + mape_epsilon
    smape = float(np.mean(2.0 * np.abs(y_true - y_pred) / denom))

    ic = _safe_corr(y_true, y_pred, method="pearson")
    rank_ic = _safe_corr(y_true, y_pred, method="spearman")

    # Directional stats
    y_true_sign = (y_true > 0).astype(int)
    y_pred_sign = (y_pred > 0).astype(int)
    hit = float(np.mean(y_true_sign == y_pred_sign))
    bal_acc = float(balanced_accuracy_score(y_true_sign, y_pred_sign)) if len(np.unique(y_true_sign)) > 1 else float("nan")

    # Residual stats
    resid_mean = float(np.mean(residual))
    resid_std = float(np.std(residual, ddof=1)) if len(residual) > 1 else float("nan")
    resid_skew = float(pd.Series(residual).skew()) if len(residual) > 2 else float("nan")
    resid_kurt = float(pd.Series(residual).kurt()) if len(residual) > 3 else float("nan")

    # Residual autocorr
    acf1 = _autocorr(residual, 1)
    acf5 = _autocorr(residual, 5)

    # IC t-stats (approx)
    n = len(y_true)
    def _ic_tstat(ic_val: float) -> float:
        if not np.isfinite(ic_val) or n < 3 or abs(ic_val) >= 1:
            return float("nan")
        return float(ic_val * np.sqrt((n - 2) / (1 - ic_val**2)))

    ic_t = _ic_tstat(ic)
    rank_ic_t = _ic_tstat(rank_ic)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "medae": medae,
        "r2": r2,
        "explained_variance": evs,
        "mape": mape,
        "smape": smape,
        "ic": ic,
        "rank_ic": rank_ic,
        "ic_tstat": ic_t,
        "rank_ic_tstat": rank_ic_t,
        "hit_ratio": hit,
        "balanced_accuracy": bal_acc,
        "resid_mean": resid_mean,
        "resid_std": resid_std,
        "resid_skew": resid_skew,
        "resid_kurtosis": resid_kurt,
        "resid_acf_lag1": acf1,
        "resid_acf_lag5": acf5,
        "n": int(n),
    }


def decile_stats(y_true: np.ndarray, y_pred: np.ndarray, n_deciles: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) == 0:
        return pd.DataFrame(columns=["decile", "count", "mean_pred", "mean_true", "std_true"])

    # Use quantile bins, drop duplicates if not enough unique values
    df["decile"] = pd.qcut(df["y_pred"], q=n_deciles, labels=False, duplicates="drop")
    df["decile"] = df["decile"].astype(int) + 1
    out = (
        df.groupby("decile")
        .agg(count=("y_true", "size"), mean_pred=("y_pred", "mean"), mean_true=("y_true", "mean"), std_true=("y_true", "std"))
        .reset_index()
    )
    return out


def compute_trading_metrics(logret: np.ndarray) -> Dict[str, float]:
    if len(logret) == 0:
        return {
            "cagr": float("nan"),
            "sharpe": float("nan"),
            "vol_ann": float("nan"),
            "max_drawdown": float("nan"),
        }
    eq = equity_curve(logret)
    n = len(logret)
    years = n / 252.0
    cagr = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 else float("nan")
    vol = float(np.std(logret) * np.sqrt(252.0))
    sharpe = float(np.mean(logret) / np.std(logret) * np.sqrt(252.0)) if np.std(logret) > 0 else float("nan")
    mdd = max_drawdown(eq)
    return {"cagr": cagr, "sharpe": sharpe, "vol_ann": vol, "max_drawdown": mdd}


def backtest_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cfg: StrategyConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos = compute_positions(y_pred, cfg)
    costs = apply_costs(pos, cfg.fee_bps, cfg.slippage_bps)
    strat_logret = pos * y_true
    strat_logret_net = strat_logret - costs
    eq = equity_curve(strat_logret)
    eq_net = equity_curve(strat_logret_net)
    return pos, strat_logret, strat_logret_net, eq_net
