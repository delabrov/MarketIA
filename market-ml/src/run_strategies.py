#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _PROJECT_ROOT / ".cache"
_MPL_DIR = _PROJECT_ROOT / ".mplconfig"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from strategies.backtest import (
    compute_equity,
    compute_metrics,
    compute_positions,
    compute_returns,
    load_close_from_raw,
    load_preds,
    load_vix_zscore_60_from_raw,
)


def _find_latest_preds(project_root: Path) -> Path:
    base = project_root / "results_lstm"
    cands = sorted(
        [
            p
            for p in base.glob("**/preds/*h1*_preds.csv")
            if "feature_ablation" not in str(p)
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        cands = sorted(
            [p for p in base.glob("**/preds/*_preds.csv") if "feature_ablation" not in str(p)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not cands:
        raise FileNotFoundError("No preds CSV found in results_lstm.")
    return cands[0]


def _load_report_if_exists(preds_path: Path) -> dict:
    run_dir = preds_path.parent.parent
    reports_dir = run_dir / "reports"
    report_files = sorted(reports_dir.glob("*_report.json"))
    if not report_files:
        return {}
    try:
        with open(report_files[0], "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _plot_equity(
    df: pd.DataFrame,
    equity_df: pd.DataFrame,
    out_path: Path,
    fee_bps: float,
    title_suffix: str,
) -> None:
    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    if "close" in df.columns and np.isfinite(df["close"]).any():
        ax1.plot(df.index, df["close"], label="AAPL close")
    else:
        proxy = np.exp(np.log1p(df["y_true"].fillna(0.0)).cumsum())
        ax1.plot(df.index, proxy, label="AAPL proxy from returns")
    ax1.set_title("AAPL Price")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Final comparison set.
    strategy_order = [
        "buyhold",
        "regime_overlay_longonly_base",
        "regime_overlay_condvol_highvol",
        "regime_overlay_condvol_p80",
    ]
    for c in strategy_order:
        if c in equity_df.columns:
            ax2.plot(equity_df.index, equity_df[c], label=c)
    ax2.set_title(f"Equity Curves ({title_suffix}, fee_bps={fee_bps:g})")
    ax2.set_ylabel("Equity (start=1)")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_drawdown(equity_df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    dd = equity_df / equity_df.cummax() - 1.0
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(111)
    strategy_order = [
        "buyhold",
        "regime_overlay_longonly_base",
        "regime_overlay_condvol_highvol",
        "regime_overlay_condvol_p80",
    ]
    for c in strategy_order:
        if c in dd.columns:
            ax.plot(dd.index, dd[c], label=c)
    ax.set_title(f"Drawdown Comparison ({title_suffix})")
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_regime_overlay_exposures(positions_df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(111)
    strategy_order = [
        "regime_overlay_longonly_base",
        "regime_overlay_condvol_highvol",
        "regime_overlay_condvol_p80",
    ]
    for c in strategy_order:
        if c in positions_df.columns:
            ax.plot(positions_df.index, positions_df[c], label=c)
    ax.set_title(f"Regime Overlay Percentile Exposures ({title_suffix})")
    ax.set_ylabel("Exposure")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_regime_overlay_vol_percentile_thresholds(vol_df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 1.1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    ax1.plot(vol_df.index, vol_df["realized_vol_20"], label="realized_vol_20")
    ax1.set_title("Realized Volatility (20d)")
    ax1.set_ylabel("Vol")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(vol_df.index, vol_df["vol_thr_p80"], label="thr_p80")
    ax2.set_title(f"Rolling Percentile Thresholds (252d) ({title_suffix})")
    ax2.set_ylabel("Threshold")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_regime_overlay_risk_coefficients(risk_df: pd.DataFrame, out_path: Path, title_suffix: str) -> None:
    fig = plt.figure(figsize=(12, 5), layout="constrained")
    ax = fig.add_subplot(111)
    order = [
        "risk_regime_overlay_condvol_highvol",
    ]
    for c in order:
        if c in risk_df.columns:
            ax.plot(risk_df.index, risk_df[c], label=c)
    ax.set_title(f"Regime Overlay Conditional Vol Risk Coefficients ({title_suffix})")
    ax.set_ylabel("risk_t")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_indicators_overview(ind_df: pd.DataFrame, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(ind_df.index, ind_df["close"], label="Close")
    ax1.plot(ind_df.index, ind_df["ema20"], label="EMA20")
    ax1.plot(ind_df.index, ind_df["ema50"], label="EMA50")
    ax1.set_title("AAPL + EMA20/EMA50")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ind_df.index, ind_df["rsi14"], label="RSI(14)")
    ax2.axhline(35.0, color="gray", linestyle="--", linewidth=1.0, label="35")
    ax2.axhline(65.0, color="gray", linestyle=":", linewidth=1.0, label="65")
    ax2.set_title("RSI(14)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(ind_df.index, ind_df["macd_line"], label="MACD")
    ax3.plot(ind_df.index, ind_df["macd_signal"], label="Signal")
    ax3.set_title("MACD(12,26,9)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_indicator_coefficient(ind_df: pd.DataFrame, pos_filtered: pd.Series, out_path: Path) -> None:
    fig = plt.figure(figsize=(12, 7), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    close = pd.Series(ind_df["close"], index=ind_df.index).astype(float)
    pos = pd.Series(pos_filtered, index=ind_df.index).astype(float).fillna(0.0)
    in_position = pos.abs() > 1e-8
    close_in = close.where(in_position, np.nan)
    close_out = close.where(~in_position, np.nan)

    ax1.plot(ind_df.index, close_in, color="green", label="Position confirmed")
    ax1.plot(ind_df.index, close_out, color="gray", label="Pas de confirmation")
    ax1.set_title("AAPL Price (indicator confirmation)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ind_df.index, ind_df["coeff_total"], label="coeff_total")
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax2.set_title("Indicator Coefficient")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_zscore_vs_indicator_filtered_signal(
    index: pd.DatetimeIndex,
    pos_zscore: pd.Series,
    coeff_total: pd.Series,
    pos_filtered: pd.Series,
    out_path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)

    ax1.plot(index, pos_zscore, label="zscore_threshold")
    ax1.set_title("ML Signal: zscore_threshold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(index, coeff_total, label="coeff_total")
    ax2.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax2.set_title("Indicator Coefficient")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3.plot(index, pos_filtered, label="zscore_threshold_indicators")
    ax3.set_title("Confirmed Position (indicator filter)")
    ax3.set_xlabel("Date")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=str, default=None, help="Path to LSTM preds CSV.")
    ap.add_argument("--outdir", type=str, default=None, help="Output dir. Default: <run_dir>/strategies")
    ap.add_argument("--fee_bps", type=float, default=0.0)
    ap.add_argument("--allow_short", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--z_window", type=int, default=252)
    ap.add_argument("--alpha_scale", type=float, default=1.0)
    ap.add_argument("--smooth_lambda", type=float, default=0.1)
    ap.add_argument("--deadband", type=float, default=0.05)
    ap.add_argument("--vol_target_ann", type=float, default=0.10)
    ap.add_argument("--lev_max", type=float, default=2.0)
    ap.add_argument("--realized_vol_window", type=int, default=20)
    ap.add_argument("--regime_threshold", type=float, default=1.0)
    ap.add_argument("--regime_feature_name", type=str, default="vix_zscore_60")
    ap.add_argument("--stress_strategy_name", type=str, default="vol_target")
    ap.add_argument("--normal_strategy_name", type=str, default="hybrid_vol_alpha")
    ap.add_argument("--core_base_exposure", type=float, default=1.0)
    ap.add_argument("--core_k", type=float, default=0.5)
    ap.add_argument("--core_min_exposure", type=float, default=0.5)
    ap.add_argument("--core_max_exposure", type=float, default=1.5)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--high_exposure", type=float, default=1.5)
    ap.add_argument("--neutral_exposure", type=float, default=1.0)
    ap.add_argument("--low_exposure", type=float, default=0.5)
    ap.add_argument("--overlay_min_exposure", type=float, default=0.25)
    ap.add_argument("--overlay_max_exposure", type=float, default=1.5)
    ap.add_argument("--stress_multiplier", type=float, default=0.5)
    ap.add_argument(
        "--extended_start_date",
        type=str,
        default="2015-01-01",
        help="Additional strategy backtest start date (YYYY-MM-DD) on full preds history.",
    )
    args = ap.parse_args()

    project_root = _PROJECT_ROOT
    preds_path = Path(args.preds) if args.preds else _find_latest_preds(project_root)
    outdir = Path(args.outdir) if args.outdir else preds_path.parent.parent / "strategies"

    reports_dir = outdir / "reports"
    preds_dir = outdir / "preds"
    plots_dir = outdir / "plots"
    for d in [reports_dir, preds_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    obsolete_plots = [
        "aapl_zscore_vs_indicator_filtered_signal.png",
        "aapl_regime_overlay_smoothing_exposures.png",
        "aapl_regime_overlay_exposures.png",
        "aapl_regime_overlay_volscale_exposures.png",
        "aapl_regime_overlay_volscale_risk_coefficients.png",
        "aapl_regime_overlay_condvol_risk_coefficients.png",
        "aapl_regime_overlay_condvol_exposures.png",
        "aapl_regime_overlay_percentile_exposures.png",
        "aapl_indicators_overview.png",
        "aapl_indicator_coefficient.png",
    ]
    for name in obsolete_plots:
        p = plots_dir / name
        if p.exists():
            p.unlink()

    df_all = load_preds(preds_path)
    df = df_all.copy()
    if "split" in df.columns and (df["split"].str.lower() == "test").any():
        df = df[df["split"].str.lower() == "test"].copy()

    if "close" not in df_all.columns or not np.isfinite(df_all["close"]).any():
        df_all["close"] = load_close_from_raw(df_all.index, ticker="AAPL", project_root=project_root)
    if args.regime_feature_name not in df_all.columns and args.regime_feature_name == "vix_zscore_60":
        vix_z = load_vix_zscore_60_from_raw(df_all.index, project_root=project_root)
        if vix_z.notna().any():
            df_all["vix_zscore_60"] = vix_z
        else:
            print("[strategies] Warning: vix_zscore_60 unavailable, using realized-vol fallback for regime.")
    df = df_all.reindex(df.index).copy()

    cfg = {
        "allow_short": bool(args.allow_short),
        "common": {
            "z_window": int(args.z_window),
            "alpha_scale": float(args.alpha_scale),
            "smooth_lambda": float(args.smooth_lambda),
            "deadband": float(args.deadband),
            "vol_target_ann": float(args.vol_target_ann),
            "lev_max": float(args.lev_max),
            "realized_vol_window": int(args.realized_vol_window),
        },
        "zscore_threshold": {"window": int(args.z_window), "entry_z": 1.0, "exit_z": 0.3},
        "zscore_threshold_indicators": {"confirm_threshold": 2.0 / 3.0},
        "rolling_quantile": {"window": int(args.z_window), "q_hi": 0.8, "q_lo": 0.2},
        "vol_target": {"pred_window": 60, "vol_target_ann": float(args.vol_target_ann), "clip_signal": 2.0, "lev_max": float(args.lev_max)},
        "hybrid_vol_alpha": {
            "z_window": int(args.z_window),
            "alpha_scale": float(args.alpha_scale),
            "vol_target_ann": float(args.vol_target_ann),
            "lev_max": float(args.lev_max),
        },
        "hybrid_vol_alpha_longonly": {
            "z_window": int(args.z_window),
            "alpha_scale": float(args.alpha_scale),
            "vol_target_ann": float(args.vol_target_ann),
            "lev_max": float(args.lev_max),
        },
        "regime_switching": {
            "regime_threshold": float(args.regime_threshold),
            "regime_feature_name": str(args.regime_feature_name),
            "stress_strategy_name": str(args.stress_strategy_name),
            "normal_strategy_name": str(args.normal_strategy_name),
        },
        "core_satellite_longonly": {
            "z_window": int(args.z_window),
            "base_exposure": float(args.core_base_exposure),
            "k": float(args.core_k),
            "min_exposure": float(args.core_min_exposure),
            "max_exposure": float(args.core_max_exposure),
        },
        "longonly_threshold": {
            "z_window": int(args.z_window),
            "threshold": float(args.threshold),
            "high_exposure": float(args.high_exposure),
            "neutral_exposure": float(args.neutral_exposure),
            "low_exposure": float(args.low_exposure),
        },
        "regime_overlay_longonly": {
            "regime_threshold": float(args.regime_threshold),
            "stress_multiplier": float(args.stress_multiplier),
            "min_exposure": float(args.overlay_min_exposure),
            "max_exposure": float(args.overlay_max_exposure),
        },
    }

    positions_all = compute_positions(df, cfg)
    selected_strategies = {
        "buyhold",
        "regime_overlay_longonly_base",
        "regime_overlay_condvol_highvol",
        "regime_overlay_condvol_p80",
    }
    positions = {k: v for k, v in positions_all.items() if k in selected_strategies}
    positions_df = pd.DataFrame(positions, index=df.index)
    returns_df = compute_returns(df, positions, fee_bps=float(args.fee_bps))
    equity_df = compute_equity(returns_df)
    metrics_df = compute_metrics(returns_df, positions_df)

    report = _load_report_if_exists(preds_path)
    ticker = str(report.get("ticker", "AAPL"))
    horizon = int(report.get("prediction_horizon", 1))
    title_suffix = f"{ticker} h={horizon}"

    metrics_df.to_csv(reports_dir / "metrics.csv")
    with open(reports_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(json.loads(metrics_df.to_json(orient="index")), f, indent=2)

    equity_df.to_csv(preds_dir / "equity_curves.csv")
    returns_df.to_csv(preds_dir / "strategy_returns.csv")
    diagnostics_df = pd.DataFrame(index=df.index)
    diagnostics_df["close"] = df["close"]
    diagnostics_df["regime_overlay_longonly_base"] = positions_df["regime_overlay_longonly_base"]
    diagnostics_df["regime_overlay_condvol_highvol"] = positions_df["regime_overlay_condvol_highvol"]
    diagnostics_df["regime_overlay_condvol_p80"] = positions_df["regime_overlay_condvol_p80"]
    base_pos = positions_df["regime_overlay_longonly_base"].replace(0.0, np.nan)
    diagnostics_df["risk_regime_overlay_condvol_highvol"] = (
        positions_df["regime_overlay_condvol_highvol"] / base_pos
    ).clip(lower=0.0, upper=1.0).fillna(1.0)
    diagnostics_df["risk_regime_overlay_condvol_p80"] = (
        positions_df["regime_overlay_condvol_p80"] / base_pos
    ).clip(lower=0.0, upper=1.0).fillna(1.0)
    if "close" in df.columns and np.isfinite(pd.to_numeric(df["close"], errors="coerce")).any():
        close = pd.to_numeric(df["close"], errors="coerce")
        hist_ret = np.log(close / close.shift(1))
    else:
        hist_ret = pd.to_numeric(df["y_true"], errors="coerce").shift(1)
    diagnostics_df["realized_vol_20"] = hist_ret.rolling(window=20, min_periods=20).std()
    diagnostics_df["vol_thr_p80"] = diagnostics_df["realized_vol_20"].rolling(window=252, min_periods=252).quantile(0.80)
    diagnostics_df.to_csv(preds_dir / "indicator_diagnostics.csv")

    _plot_equity(
        df=df,
        equity_df=equity_df,
        out_path=plots_dir / "aapl_strategies_equity_vs_buyhold.png",
        fee_bps=float(args.fee_bps),
        title_suffix=title_suffix,
    )
    _plot_drawdown(
        equity_df=equity_df,
        out_path=plots_dir / "aapl_strategies_drawdown.png",
        title_suffix=title_suffix,
    )
    _plot_regime_overlay_exposures(
        positions_df=positions_df,
        out_path=plots_dir / "aapl_regime_overlay_final_exposures.png",
        title_suffix=title_suffix,
    )
    _plot_regime_overlay_vol_percentile_thresholds(
        vol_df=diagnostics_df,
        out_path=plots_dir / "aapl_regime_overlay_vol_percentile_thresholds.png",
        title_suffix=title_suffix,
    )

    print(f"[strategies] preds: {preds_path}")
    print(f"[strategies] outdir: {outdir}")
    print("[strategies] metrics:")
    display_strategies = [
        "buyhold",
        "regime_overlay_longonly_base",
        "regime_overlay_condvol_highvol",
        "regime_overlay_condvol_p80",
    ]
    display_cols = [
        "cagr",
        "sharpe",
        "max_drawdown",
        "annualized_vol",
        "calmar_ratio",
        "turnover",
        "average_holding_period",
        "hit_ratio",
        "n_trades",
    ]
    display_idx = [s for s in display_strategies if s in metrics_df.index]
    print(metrics_df.loc[display_idx, display_cols].round(4))

    # Additional wider-interval test from a configurable start date on full history.
    start_ts = pd.to_datetime(args.extended_start_date, utc=True, errors="coerce")
    if pd.notna(start_ts):
        df_ext = df_all[df_all.index >= start_ts].copy()
        if len(df_ext) > 0:
            positions_all_ext = compute_positions(df_ext, cfg)
            positions_ext = {k: v for k, v in positions_all_ext.items() if k in selected_strategies}
            positions_ext_df = pd.DataFrame(positions_ext, index=df_ext.index)
            returns_ext_df = compute_returns(df_ext, positions_ext, fee_bps=float(args.fee_bps))
            equity_ext_df = compute_equity(returns_ext_df)
            metrics_ext_df = compute_metrics(returns_ext_df, positions_ext_df)

            start_label = start_ts.strftime("%Y%m%d")
            metrics_ext_df.to_csv(reports_dir / f"metrics_since_{start_label}.csv")
            with open(reports_dir / f"metrics_since_{start_label}.json", "w", encoding="utf-8") as f:
                json.dump(json.loads(metrics_ext_df.to_json(orient="index")), f, indent=2)
            equity_ext_df.to_csv(preds_dir / f"equity_curves_since_{start_label}.csv")

            _plot_equity(
                df=df_ext,
                equity_df=equity_ext_df,
                out_path=plots_dir / f"aapl_strategies_equity_vs_buyhold_since_{start_label}.png",
                fee_bps=float(args.fee_bps),
                title_suffix=f"{title_suffix} since {start_ts.date()}",
            )

            display_ext_idx = [s for s in display_strategies if s in metrics_ext_df.index]
            print(f"[strategies] extended interval from {start_ts.date()} metrics:")
            print(metrics_ext_df.loc[display_ext_idx, display_cols].round(4))
        else:
            print(f"[strategies] extended interval from {args.extended_start_date}: no rows available.")
    else:
        print(f"[strategies] invalid --extended_start_date='{args.extended_start_date}', skipping extended test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
