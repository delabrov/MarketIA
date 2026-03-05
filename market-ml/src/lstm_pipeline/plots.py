from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_equity(dates, strat_eq, bh_eq, save_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(12, 6), layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(dates, strat_eq, label="Strategy")
    ax.plot(dates, bh_eq, label="Buy&Hold")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start=1)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, save_path)


def _drawdown(eq: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(eq)
    return (eq / peak) - 1.0


def plot_drawdown(
    dates,
    strat_eq,
    save_path: Path,
    title: str,
    bh_eq: Optional[Sequence[float]] = None,
    strat_eq_net: Optional[Sequence[float]] = None,
) -> None:
    strat_eq = np.asarray(strat_eq).reshape(-1)
    if len(strat_eq) == 0:
        return
    dates = np.asarray(dates)

    dd_strat = _drawdown(strat_eq)
    fig = plt.figure(figsize=(12, 4.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(dates, dd_strat, label="Strategy")

    dd_text = []
    min_idx = int(np.nanargmin(dd_strat))
    dd_text.append(f"Strat MDD={dd_strat[min_idx]*100:.1f}% on {pd.to_datetime(dates[min_idx]).date()}")

    if bh_eq is not None:
        bh_eq = np.asarray(bh_eq).reshape(-1)
        if len(bh_eq) == len(strat_eq):
            dd_bh = _drawdown(bh_eq)
            ax.plot(dates, dd_bh, label="Buy&Hold")
            min_idx = int(np.nanargmin(dd_bh))
            dd_text.append(f"B&H MDD={dd_bh[min_idx]*100:.1f}% on {pd.to_datetime(dates[min_idx]).date()}")

    if strat_eq_net is not None:
        strat_eq_net = np.asarray(strat_eq_net).reshape(-1)
        if len(strat_eq_net) == len(strat_eq):
            dd_net = _drawdown(strat_eq_net)
            ax.plot(dates, dd_net, label="Strategy net")
            min_idx = int(np.nanargmin(dd_net))
            dd_text.append(f"Net MDD={dd_net[min_idx]*100:.1f}% on {pd.to_datetime(dates[min_idx]).date()}")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(
        0.02,
        0.98,
        "\n".join(dd_text),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="0.5"),
    )

    _save(fig, save_path)


def plot_return_hist(returns, save_path: Path, title: str) -> None:
    r = np.asarray(returns).reshape(-1)
    if len(r) == 0:
        return
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.hist(r, bins=60, alpha=0.8, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_scatter(y_true, y_pred, save_path: Path, title: str) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0:
        return
    fig = plt.figure(figsize=(6, 6), layout="constrained")
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=10, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_pred_vs_true_enhanced(y_true, y_pred, save_path: Path, title: str) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0:
        return
    fig = plt.figure(figsize=(6.5, 6.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=10, alpha=0.5)

    # y=x line
    min_v = float(np.nanmin([y_true.min(), y_pred.min()]))
    max_v = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([min_v, max_v], [min_v, max_v], color="black", linewidth=1.0, alpha=0.7, label="y=x")

    # regression line
    if len(y_true) >= 2:
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        xs = np.array([min_v, max_v])
        ax.plot(xs, slope * xs + intercept, color="tab:red", linewidth=1.2, label="Fit")
        corr = np.corrcoef(y_true, y_pred)[0, 1] if np.std(y_true) > 0 and np.std(y_pred) > 0 else np.nan
        ax.text(
            0.02,
            0.98,
            f"slope={slope:.3f}\nintercept={intercept:.3f}\ncorr={corr:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="0.5"),
        )

    ax.set_title(title)
    ax.set_xlabel("y_true")
    ax.set_ylabel("y_pred")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, save_path)


def plot_loss(history: Dict[str, list], save_path: Path, title: str) -> None:
    tr = history.get("train_loss", [])
    va = history.get("val_loss", [])
    if not tr:
        return
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(tr, label="train")
    if va:
        ax.plot(va, label="val")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, save_path)


def plot_returns_true_vs_pred(
    dates,
    y_true,
    y_pred,
    save_path: Path,
    title: str,
    smoothing_window: int = 5,
) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    dates = np.asarray(dates)
    if len(y_true) == 0:
        return

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(dates, y_true, color="tab:blue", alpha=0.35, label="True")
    ax.plot(dates, y_pred, color="tab:orange", alpha=0.35, label="Pred")

    if smoothing_window and smoothing_window > 1 and len(y_true) >= smoothing_window:
        w = smoothing_window
        true_s = np.convolve(y_true, np.ones(w) / w, mode="valid")
        pred_s = np.convolve(y_pred, np.ones(w) / w, mode="valid")
        dates_s = dates[w - 1 :]
        ax.plot(dates_s, true_s, color="tab:blue", label=f"True (MA{w})")
        ax.plot(dates_s, pred_s, color="tab:orange", label=f"Pred (MA{w})")

    resid = y_true - y_pred
    axr.plot(dates, resid, color="black", linewidth=0.8, label="Residuals")
    axr.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

    ax.set_title(title)
    ax.set_ylabel("Log return")
    ax.grid(True, alpha=0.3)
    ax.legend()

    axr.set_xlabel("Date")
    axr.set_ylabel("Residual")
    axr.grid(True, alpha=0.3)
    axr.legend()

    _save(fig, save_path)


def plot_residuals_time(dates, residuals, save_path: Path, title: str) -> None:
    residuals = np.asarray(residuals).reshape(-1)
    dates = np.asarray(dates)
    if len(residuals) == 0:
        return
    fig = plt.figure(figsize=(12, 4.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(dates, residuals, color="black", linewidth=0.8)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_residuals_acf(residuals, max_lag: int, save_path: Path, title: str) -> None:
    r = np.asarray(residuals).reshape(-1)
    if len(r) == 0:
        return
    lags = np.arange(1, max_lag + 1)
    acf = []
    for k in lags:
        if len(r) <= k:
            acf.append(np.nan)
        else:
            a = r[k:]
            b = r[:-k]
            if np.std(a) == 0 or np.std(b) == 0:
                acf.append(np.nan)
            else:
                acf.append(np.corrcoef(a, b)[0, 1])
    fig = plt.figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.bar(lags, acf, color="tab:blue", alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, save_path)


def plot_residuals_hist(residuals, save_path: Path, title: str) -> None:
    r = np.asarray(residuals).reshape(-1)
    if len(r) == 0:
        return
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.hist(r, bins=60, alpha=0.8, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    _save(fig, save_path)


def plot_decile_means(decile_df: pd.DataFrame, save_path: Path, title: str) -> None:
    if decile_df is None or len(decile_df) == 0:
        return
    fig = plt.figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.bar(decile_df["decile"], decile_df["mean_true"], alpha=0.8, label="Mean true")
    ax.plot(decile_df["decile"], decile_df["mean_pred"], color="black", marker="o", label="Mean pred")
    ax.set_title(title)
    ax.set_xlabel("Decile (1=lowest pred)")
    ax.set_ylabel("Mean true return")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, save_path)


def plot_compare_horizons_metrics(
    metrics_df: pd.DataFrame,
    h1: int,
    h2: int,
    save_path: Path,
    title: str,
) -> None:
    if metrics_df is None or len(metrics_df) == 0:
        return
    m1 = metrics_df[metrics_df["h"] == h1]
    m2 = metrics_df[metrics_df["h"] == h2]
    if m1.empty or m2.empty:
        return

    m1 = m1.iloc[0]
    m2 = m2.iloc[0]

    labels = ["IC", "RankIC", "RMSE", "MAE"]
    v1 = [m1["ic_pearson"], m1["ic_spearman"], m1["rmse"], m1["mae"]]
    v2 = [m2["ic_pearson"], m2["ic_spearman"], m2["rmse"], m2["mae"]]

    x = np.arange(len(labels))
    width = 0.35

    fig = plt.figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.bar(x - width / 2, v1, width, label=f"h={h1}")
    ax.bar(x + width / 2, v2, width, label=f"h={h2}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, save_path)


def plot_compare_returns_and_residuals(
    dates,
    y_true_h1,
    y_pred_h1,
    y_true_h5,
    y_pred_h5,
    save_path: Path,
    title: str,
    smoothing_window: int = 5,
) -> None:
    dates = np.asarray(dates)
    y_true_h1 = np.asarray(y_true_h1).reshape(-1)
    y_pred_h1 = np.asarray(y_pred_h1).reshape(-1)
    y_true_h5 = np.asarray(y_true_h5).reshape(-1)
    y_pred_h5 = np.asarray(y_pred_h5).reshape(-1)
    if len(dates) == 0:
        return

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax1r = fig.add_subplot(gs[1], sharex=ax1)
    ax2 = fig.add_subplot(gs[2], sharex=ax1)
    ax2r = fig.add_subplot(gs[3], sharex=ax1)

    # h1
    ax1.plot(dates, y_true_h1, alpha=0.35, label="True h1")
    ax1.plot(dates, y_pred_h1, alpha=0.35, label="Pred h1")
    if smoothing_window and smoothing_window > 1 and len(y_true_h1) >= smoothing_window:
        w = smoothing_window
        true_s = np.convolve(y_true_h1, np.ones(w) / w, mode="valid")
        pred_s = np.convolve(y_pred_h1, np.ones(w) / w, mode="valid")
        dates_s = dates[w - 1 :]
        ax1.plot(dates_s, true_s, label=f"True h1 (MA{w})")
        ax1.plot(dates_s, pred_s, label=f"Pred h1 (MA{w})")
    ax1.set_title(f"{title} — h=1")
    ax1.set_ylabel("Log return")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    resid1 = y_true_h1 - y_pred_h1
    ax1r.plot(dates, resid1, color="black", linewidth=0.8, label="Residuals h1")
    ax1r.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax1r.set_ylabel("Residual")
    ax1r.grid(True, alpha=0.3)
    ax1r.legend()

    # h5
    ax2.plot(dates, y_true_h5, alpha=0.35, label="True h5")
    ax2.plot(dates, y_pred_h5, alpha=0.35, label="Pred h5")
    if smoothing_window and smoothing_window > 1 and len(y_true_h5) >= smoothing_window:
        w = smoothing_window
        true_s = np.convolve(y_true_h5, np.ones(w) / w, mode="valid")
        pred_s = np.convolve(y_pred_h5, np.ones(w) / w, mode="valid")
        dates_s = dates[w - 1 :]
        ax2.plot(dates_s, true_s, label=f"True h5 (MA{w})")
        ax2.plot(dates_s, pred_s, label=f"Pred h5 (MA{w})")
    ax2.set_title(f"{title} — h=5")
    ax2.set_ylabel("Log return")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    resid5 = y_true_h5 - y_pred_h5
    ax2r.plot(dates, resid5, color="black", linewidth=0.8, label="Residuals h5")
    ax2r.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax2r.set_xlabel("Date")
    ax2r.set_ylabel("Residual")
    ax2r.grid(True, alpha=0.3)
    ax2r.legend()

    _save(fig, save_path)


def plot_compare_deciles(
    decile_df: pd.DataFrame,
    h1: int,
    h2: int,
    save_path: Path,
    title: str,
) -> None:
    if decile_df is None or len(decile_df) == 0:
        return
    d1 = decile_df[decile_df["horizon"] == h1]
    d2 = decile_df[decile_df["horizon"] == h2]
    if d1.empty or d2.empty:
        return
    fig = plt.figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(d1["decile"], d1["mean_true"], marker="o", label=f"h={h1}")
    ax.plot(d2["decile"], d2["mean_true"], marker="o", label=f"h={h2}")
    ax.set_title(title)
    ax.set_xlabel("Decile (1=lowest pred)")
    ax.set_ylabel("Mean true return")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, save_path)


def _rolling_sharpe_series(r: np.ndarray, window: int) -> np.ndarray:
    if len(r) < window:
        return np.array([])
    roll_mean = np.convolve(r, np.ones(window) / window, mode="valid")
    roll_std = np.array([np.std(r[i : i + window]) for i in range(len(r) - window + 1)])
    return np.where(roll_std > 0, roll_mean / roll_std * np.sqrt(252.0), np.nan)


def plot_rolling_sharpe(
    dates,
    strat_logret,
    window: int,
    save_path: Path,
    title: str,
    bh_logret: Optional[Sequence[float]] = None,
    strat_logret_net: Optional[Sequence[float]] = None,
) -> None:
    r = np.asarray(strat_logret).reshape(-1)
    if len(r) < window:
        return
    dates = np.asarray(dates)
    sharpe = _rolling_sharpe_series(r, window)
    dates_ = dates[window - 1 :]

    fig = plt.figure(figsize=(10, 4.5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(dates_, sharpe, label="Strategy")

    if strat_logret_net is not None:
        r_net = np.asarray(strat_logret_net).reshape(-1)
        if len(r_net) >= window:
            sh_net = _rolling_sharpe_series(r_net, window)
            ax.plot(dates_, sh_net, label="Strategy net")

    if bh_logret is not None:
        r_bh = np.asarray(bh_logret).reshape(-1)
        if len(r_bh) >= window:
            sh_bh = _rolling_sharpe_series(r_bh, window)
            ax.plot(dates_, sh_bh, label="Buy&Hold")

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling Sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, save_path)


def plot_price_train_val_test_predictions(
    dates,
    true_price: np.ndarray,
    pred_train: np.ndarray,
    pred_val: np.ndarray,
    pred_test: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    save_path: Path,
    title: str,
) -> None:
    dates = np.asarray(dates)
    true_price = np.asarray(true_price).reshape(-1)
    pred_train = np.asarray(pred_train).reshape(-1)
    pred_val = np.asarray(pred_val).reshape(-1)
    pred_test = np.asarray(pred_test).reshape(-1)

    n = min(len(dates), len(true_price), len(pred_train), len(pred_val), len(pred_test))
    if n == 0:
        return

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(dates[:n], true_price[:n], color="black", label="True price")
    ax.plot(dates[:n], pred_train[:n], color="tab:green", label="Pred train")
    ax.plot(dates[:n], pred_val[:n], color="tab:orange", label="Pred val")
    ax.plot(dates[:n], pred_test[:n], color="tab:red", label="Pred test")

    text = (
        f"Train RMSE={metrics['train']['rmse']:.4f}, R2={metrics['train']['r2']:.3f}\n"
        f"Val   RMSE={metrics['val']['rmse']:.4f}, R2={metrics['val']['r2']:.3f}\n"
        f"Test  RMSE={metrics['test']['rmse']:.4f}, R2={metrics['test']['r2']:.3f}"
    )
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="0.5"),
    )

    # Residuals per split
    resid_train = true_price[:n] - pred_train[:n]
    resid_val = true_price[:n] - pred_val[:n]
    resid_test = true_price[:n] - pred_test[:n]

    axr.plot(dates[:n], resid_train, color="tab:green", label="Residuals train", linewidth=0.8)
    axr.plot(dates[:n], resid_val, color="tab:orange", label="Residuals val", linewidth=0.8)
    axr.plot(dates[:n], resid_test, color="tab:red", label="Residuals test", linewidth=0.8)
    axr.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()

    axr.set_xlabel("Date")
    axr.set_ylabel("Residual")
    axr.grid(True, alpha=0.3)
    axr.legend()

    _save(fig, save_path)


def plot_price_test_zoom_with_residuals(
    dates,
    true_price: np.ndarray,
    pred_test: np.ndarray,
    rmse: float,
    r2: float,
    save_path: Path,
    title: str,
) -> None:
    dates = np.asarray(dates)
    true_price = np.asarray(true_price).reshape(-1)
    pred_test = np.asarray(pred_test).reshape(-1)

    n = min(len(dates), len(true_price), len(pred_test))
    if n == 0:
        return

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharex=ax)

    ax.plot(dates[:n], true_price[:n], color="black", label="True price")
    pred_line, = ax.plot(dates[:n], pred_test[:n], color="tab:red", label="Pred test")

    resid = true_price[:n] - pred_test[:n]
    sigma = float(np.nanstd(resid, ddof=1)) if np.isfinite(resid).any() else float("nan")
    if np.isfinite(sigma):
        upper = pred_test[:n] + 3.0 * sigma
        lower = pred_test[:n] - 3.0 * sigma
        ax.fill_between(dates[:n], lower, upper, color=pred_line.get_color(), alpha=0.15, label="±3σ")
    axr.plot(dates[:n], resid, color="black", linewidth=0.8, label="Residuals")
    axr.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(
        0.02,
        0.98,
        f"RMSE={rmse:.4f}\nR²={r2:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="0.5"),
    )

    axr.set_xlabel("Date")
    axr.set_ylabel("Residual")
    axr.grid(True, alpha=0.3)
    axr.legend()

    _save(fig, save_path)


def plot_feature_importance(
    feature_names: Sequence[str],
    importance_mean: np.ndarray,
    importance_std: np.ndarray,
    save_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    names = np.asarray(feature_names)
    imp = np.asarray(importance_mean).reshape(-1)
    std = np.asarray(importance_std).reshape(-1)
    if len(names) == 0 or len(imp) == 0:
        return

    exog_feats = {"vix_level", "vix_change_1d", "vix_zscore_60", "spy_return_1d"}
    indicator_feats = {
        "ema_return_5",
        "ema_return_20",
        "realized_vol_5d",
        "realized_vol_20d",
        "vol_ratio_5_20",
        "vol_ratio_5_60",
        "rolling_skew_60",
        "rolling_kurtosis_60",
        "volume_z_20",
    }
    complex_feats = {"amihud_illiquidity_20d", "downside_vol_20d"}
    colors = {
        "simple": "tab:blue",
        "indicator": "tab:orange",
        "complex": "tab:purple",
        "exogenous": "tab:green",
    }

    order = np.argsort(imp)[::-1]
    if top_n is not None:
        order = order[: min(top_n, len(order))]

    names = names[order]
    imp = imp[order]
    std = std[order]
    cats = []
    bar_colors = []
    for n in names:
        if n in exog_feats:
            cat = "exogenous"
        elif n in indicator_feats:
            cat = "indicator"
        elif n in complex_feats:
            cat = "complex"
        else:
            cat = "simple"
        cats.append(cat)
        bar_colors.append(colors[cat])

    fig = plt.figure(figsize=(10, max(4, len(names) * 0.35)), layout="constrained")
    ax = fig.add_subplot(111)
    ax.barh(names[::-1], imp[::-1], xerr=std[::-1], alpha=0.8, color=bar_colors[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    legend_handles = [
        Patch(color=colors["simple"], label="Simple"),
        Patch(color=colors["indicator"], label="Indicator"),
        Patch(color=colors["complex"], label="Complex"),
        Patch(color=colors["exogenous"], label="Exogenous"),
    ]
    ax.legend(handles=legend_handles, loc="best")
    _save(fig, save_path)
