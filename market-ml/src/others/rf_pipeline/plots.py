# src/plots.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Helpers
# ----------------------------
def _save_close(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def feature_color(name: str) -> str:
    """
    Code couleur (3 types) :
      - Triviales: bleu
      - Indicateurs: orange
      - Exogènes: vert
    (Matplotlib accepte des couleurs nommées simples.)
    """
    n = name.lower()
    # Exogènes
    if n.startswith("spy_") or n.startswith("vix_"):
        return "tab:green"
    # Indicateurs
    if n.startswith("rsi_") or n.startswith("macd") or "sma" in n or n.startswith("dist_"):
        return "tab:orange"
    # Triviales / default
    return "tab:blue"


# ----------------------------
# Core plots used by evaluate.py
# ----------------------------
def plot_equity_curve(
    dates,
    strat_eq: np.ndarray,
    bh_eq: np.ndarray,
    save_path: Path,
    title: str = "Equity curve: Strategy vs Buy&Hold",
) -> None:
    strat_eq = np.asarray(strat_eq).reshape(-1)
    bh_eq = np.asarray(bh_eq).reshape(-1)

    fig = plt.figure(figsize=(12, 6), layout="constrained")
    ax = fig.add_subplot(111)

    ax.plot(dates, strat_eq, label="Strategy")
    ax.plot(dates, bh_eq, label="Buy&Hold")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start=1)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_close(fig, save_path)


def plot_strategy_vs_buyhold_vs_mc(
    dates,
    strat_eq: np.ndarray,
    bh_eq: np.ndarray,
    mc_curves: np.ndarray,
    mc_mean: np.ndarray,
    mc_std: np.ndarray,
    save_path: Path,
    mc_block_size: int | None = None,
    title: str = "Equity curve: Strategy vs Buy&Hold vs Random Monte Carlo",
) -> None:
    strat_eq = np.asarray(strat_eq).reshape(-1)
    bh_eq = np.asarray(bh_eq).reshape(-1)
    mc_curves = np.asarray(mc_curves)
    mc_mean = np.asarray(mc_mean).reshape(-1)
    mc_std = np.asarray(mc_std).reshape(-1)

    n = len(dates)
    n = min(n, len(strat_eq), len(bh_eq))
    if mc_mean.size:
        n = min(n, len(mc_mean))
    if mc_std.size:
        n = min(n, len(mc_std))
    if n == 0:
        return

    dates_ = dates[:n]
    strat_ = strat_eq[:n]
    bh_ = bh_eq[:n]

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    ax = fig.add_subplot(111)

    # Monte Carlo curves (gray, transparent)
    if mc_curves.ndim == 2 and mc_curves.shape[1] >= n:
        for curve in mc_curves[:, :n]:
            ax.plot(dates_, curve, color="gray", alpha=0.15, linewidth=0.8)

    # Monte Carlo 1-sigma band
    if mc_mean.size and mc_std.size:
        mean_ = mc_mean[:n]
        std_ = mc_std[:n]
        upper1 = mean_ + std_
        lower1 = np.maximum(mean_ - std_, 0.0)

        # Mean (solid black), bounds as dashed greys (two tones)
        ax.plot(dates_, mean_, color="black", linestyle="-", linewidth=1.6, label="Random mean")
        ax.plot(dates_, upper1, color="0.6", linestyle="-", linewidth=1.2, label="Random 1 sigma")
        ax.plot(dates_, lower1, color="0.6", linestyle="-", linewidth=1.2, label="_nolegend_")

    # Strategy and Buy & Hold
    ax.plot(dates_, strat_, color="red", label="Strategy", linewidth=1.8)
    ax.plot(dates_, bh_, color="blue", label="Buy&Hold", linewidth=1.6)

    # p-value annotation (timing-null Monte Carlo)
    p_value = None
    percentile = None
    if mc_curves.ndim == 2 and mc_curves.shape[1] >= n and len(strat_) > 0:
        final_obs = float(strat_[-1])
        final_mc = mc_curves[:, n - 1].astype(float)
        if final_mc.size:
            p_value = (np.sum(final_mc >= final_obs) + 1) / (final_mc.size + 1)
            percentile = 100.0 * np.mean(final_mc < final_obs)

    if p_value is not None:
        bs = f", block={mc_block_size}" if mc_block_size else ""
        text = f"p(shuffle{bs})={p_value:.3f}"
        if percentile is not None:
            text += f" ({percentile:.1f}th pct)"
        ax.text(
            0.02,
            0.98,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="0.5"),
        )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (start=1)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_close(fig, save_path)


def plot_logret_prediction(
    dates,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    save_path: Path,
    title: str = "Next-day log return: true vs predicted",
) -> None:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    residuals = np.asarray(residuals).reshape(-1)

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ax1.plot(dates, y_true, label="True")
    ax1.plot(dates, y_pred, label="Predicted")
    ax1.set_title(title)
    ax1.set_ylabel("log return")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(0.0, linestyle="--")
    ax2.plot(dates, residuals, label="Residuals")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("err")
    ax2.grid(True, alpha=0.3)

    _save_close(fig, save_path)


def plot_residual_hist(
    residuals: np.ndarray,
    save_path: Path,
    title: str = "Residuals histogram",
    bins: int = 60,
) -> None:
    residuals = np.asarray(residuals).reshape(-1)

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    ax = fig.add_subplot(111)

    ax.hist(residuals, bins=bins, alpha=0.9)
    ax.axvline(0.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    _save_close(fig, save_path)


def plot_price_reconstruction_last365(
    dates,
    close_t: np.ndarray,
    close_t1: np.ndarray,
    pred_logret: np.ndarray,
    residuals: np.ndarray,
    out_path: Path,
    title: str = "Reconstructed price (last 365 days) with ±3σ band",
) -> None:
    """
    One-step prediction:
      pred_close_{t+1} = close_t * exp(pred_logret_t)
    where pred_logret_t is the next-day log return from t -> t+1.
    Band computed in log-return space (one-step, no sqrt(t)):
      upper/lower = pred_close_{t+1} * exp(±3 * sigma)
    """
    close_t = np.asarray(close_t).reshape(-1)
    close_t1 = np.asarray(close_t1).reshape(-1)
    pred_logret = np.asarray(pred_logret).reshape(-1)
    residuals = np.asarray(residuals).reshape(-1)

    n = min(len(close_t), len(close_t1), len(pred_logret), len(residuals), len(dates))
    if n < 2:
        return

    # Align to t+1 (next-day close)
    dates_ = dates[:n]
    base_close = close_t[: n - 1]
    true_next = close_t1[: n - 1]
    pred_lr = pred_logret[: n - 1]
    resid = residuals[: n - 1]
    dates_aligned = dates_[1:]

    last = min(365, len(base_close))
    dates_aligned = dates_aligned[-last:]
    base_close = base_close[-last:]
    true_next = true_next[-last:]
    pred_lr = pred_lr[-last:]
    resid = resid[-last:]

    sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0

    pred_next = base_close * np.exp(pred_lr)
    upper = pred_next * np.exp(3.0 * sigma)
    lower = pred_next * np.exp(-3.0 * sigma)

    # residuals in price space: true (t+1) - predicted (t+1)
    price_resid = true_next - pred_next

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # True / predicted
    ax1.plot(dates_aligned, true_next, label="True price")
    line_pred, = ax1.plot(dates_aligned, pred_next, label="Predicted price")
    # band, same color as predicted curve
    ax1.fill_between(dates_aligned, lower, upper, alpha=0.25, color=line_pred.get_color(), label="±3σ band")

    ax1.set_title(title)
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.axhline(0.0, linestyle="--")
    ax2.plot(dates_aligned, price_resid, color="black", label="Residuals (price)")
    # RMSD in price space
    rmsd = float(np.sqrt(np.mean(price_resid**2))) if len(price_resid) > 0 else float("nan")
    ax2.text(
        0.01,
        0.95,
        f"RMSD: {rmsd:.4f}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
    )
    ax2.set_xlabel("Date")
    ax2.set_ylabel("err")
    ax2.grid(True, alpha=0.3)

    _save_close(fig, out_path)


def plot_walkforward_logloss(
    dates,
    logloss_values: np.ndarray,
    out_path: Path,
    title: str = "Walk-forward logloss over time",
) -> None:
    logloss_values = np.asarray(logloss_values).reshape(-1)
    if len(logloss_values) == 0:
        return

    dates = np.asarray(dates)
    n = min(len(dates), len(logloss_values))
    if n == 0:
        return

    dates_ = dates[:n]
    ll_ = logloss_values[:n]

    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(111)
    ax.plot(dates_, ll_, marker="o", linewidth=1.4)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Log loss")
    ax.grid(True, alpha=0.3)

    _save_close(fig, out_path)


def plot_price_train_test_predictions(
    dates,
    close_t: np.ndarray,
    pred_logret_train: np.ndarray,
    pred_logret_test: np.ndarray,
    split_idx: int,
    metrics_cls: dict,
    metrics_ret: dict,
    out_path: Path,
    title: str = "Price with train/test intervals and predictions",
) -> None:
    dates = np.asarray(dates)
    close_t = np.asarray(close_t).reshape(-1)
    pred_logret_train = np.asarray(pred_logret_train).reshape(-1)
    pred_logret_test = np.asarray(pred_logret_test).reshape(-1)

    n = min(len(dates), len(close_t))
    if n < 2:
        return

    dates = dates[:n]
    close_t = close_t[:n]

    split_idx = int(max(1, min(split_idx, n - 1)))

    fig = plt.figure(figsize=(12, 6), layout="constrained")
    ax = fig.add_subplot(111)

    # True price curve
    ax.plot(dates, close_t, color="tab:blue", label="True price")

    # Train predictions (aligned to next-day dates)
    train_len = min(split_idx, len(pred_logret_train))
    if train_len >= 2:
        train_pred = close_t[: train_len - 1] * np.exp(pred_logret_train[: train_len - 1])
        train_dates = dates[1:train_len]
        ax.plot(train_dates, train_pred, color="tab:orange", label="Predicted (train)")

    # Test predictions (aligned to next-day dates)
    test_len = min(n - split_idx, len(pred_logret_test))
    if test_len >= 2:
        base = close_t[split_idx : split_idx + test_len - 1]
        test_pred = base * np.exp(pred_logret_test[: test_len - 1])
        test_dates = dates[split_idx + 1 : split_idx + test_len]
        ax.plot(test_dates, test_pred, color="tab:green", label="Predicted (test)")

    # Train/Test intervals
    ax.axvspan(dates[0], dates[split_idx - 1], color="tab:blue", alpha=0.06, label="Train interval")
    ax.axvspan(dates[split_idx], dates[-1], color="tab:orange", alpha=0.06, label="Test interval")
    ax.axvline(dates[split_idx], color="black", linestyle="--", linewidth=1.0, alpha=0.6)

    # Test metrics text box
    metrics_text = (
        f"Test accuracy: {metrics_cls.get('accuracy', float('nan')):.3f}\n"
        f"Test ROC AUC: {metrics_cls.get('roc_auc', float('nan')):.3f}\n"
        f"Test log loss: {metrics_cls.get('log_loss', float('nan')):.3f}\n"
        f"Test RMSE: {metrics_ret.get('rmse', float('nan')):.6f}"
    )
    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _save_close(fig, out_path)


# ----------------------------
# Feature importance (your exact signatures)
# ----------------------------
def plot_feature_importance_mdi(
    feature_names: Sequence[str],
    importances: np.ndarray,
    out_path: Path,
    title: str = "Feature importance (MDI)",
    top_k: Optional[int] = None,
) -> None:
    imp = np.asarray(importances).reshape(-1)
    names = list(feature_names)
    if len(names) != len(imp):
        raise ValueError("feature_names and importances must have same length.")

    order = np.argsort(imp)[::-1]
    if top_k is not None:
        order = order[:top_k]

    names_s = [names[i] for i in order]
    imp_s = imp[order]
    colors = [feature_color(n) for n in names_s]

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    ax = fig.add_subplot(111)

    ax.barh(names_s[::-1], imp_s[::-1], color=colors[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)

    _save_close(fig, out_path)


def plot_permutation_importance(
    feature_names: Sequence[str],
    importances_mean: np.ndarray,
    importances_std: Optional[np.ndarray],
    out_path: Path,
    title: str = "Permutation importance (mean decrease in score)",
    xlabel: str = "Decrease in score (ROC AUC)",
    top_k: Optional[int] = None,
) -> None:
    mean = np.asarray(importances_mean).reshape(-1)
    std = None if importances_std is None else np.asarray(importances_std).reshape(-1)

    names = list(feature_names)
    if len(names) != len(mean):
        raise ValueError("feature_names and importances_mean must have same length.")
    if std is not None and len(std) != len(mean):
        raise ValueError("importances_std must have same length as importances_mean.")

    order = np.argsort(mean)[::-1]
    if top_k is not None:
        order = order[:top_k]

    names_s = [names[i] for i in order]
    mean_s = mean[order]
    std_s = None if std is None else std[order]
    colors = [feature_color(n) for n in names_s]

    fig = plt.figure(figsize=(12, 7), layout="constrained")
    ax = fig.add_subplot(111)

    ax.axvline(0.0, linestyle="--")
    ax.barh(names_s[::-1], mean_s[::-1], color=colors[::-1])
    if std_s is not None:
        ax.errorbar(
            mean_s[::-1],
            np.arange(len(names_s)),
            xerr=std_s[::-1],
            fmt="none",
            ecolor="black",
            elinewidth=1.5,
            capsize=0,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.3)

    _save_close(fig, out_path)
