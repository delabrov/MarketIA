# src/evaluate.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from .diagnostics import alignment_audit, leakage_audit
from .nested_wf_eval import nested_walk_forward_eval

from .strategy import (
    strategy_equity_from_proba,
    buy_hold_equity,
    monte_carlo_random_equity,
)

from .plots import (
    plot_equity_curve,
    plot_logret_prediction,
    plot_price_reconstruction_last365,
    plot_price_train_test_predictions,
    plot_feature_importance_mdi,
    plot_permutation_importance,
    plot_residual_hist,
    plot_strategy_vs_buyhold_vs_mc,
    plot_walkforward_logloss,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _paths(root: Path) -> Dict[str, Path]:
    return {
        "data_processed": root / "data" / "processed",
        "models": root / "models",
        "results_reports": root / "results_rf" / "reports",
        "results_figures": root / "results_rf" / "figures",
    }


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _find_close_col(df: pd.DataFrame) -> str | None:
    # For next-day log-return reconstruction, anchor on close_t (today's close),
    # not close_t1. close_t1 is used only as a fallback if close_t is missing.
    for c in ["close_t", "close", "adj_close", "close_t1"]:
        if c in df.columns:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--model_name", default="aapl_rf_pup.joblib")
    ap.add_argument("--dataset_name", default="aapl_dataset.parquet")
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--perm_repeats", type=int, default=10)
    ap.add_argument("--nested_wf", action="store_true", default=False)
    ap.add_argument("--nested_wf_fast", action="store_true", default=False)
    ap.add_argument("--outer_train_years", type=int, default=10)
    ap.add_argument("--outer_test_months", type=int, default=6)
    ap.add_argument("--inner_folds", type=int, default=5)
    ap.add_argument("--inner_min_train_days", type=int, default=252 * 3)
    ap.add_argument("--purge_gap_days", type=int, default=1)
    ap.add_argument("--start_date", default=None, help="YYYY-MM-DD. If set, keep data >= start_date.")
    ap.add_argument("--leakage_audit", action="store_true", default=False)
    ap.add_argument("--leakage_smoke_model", action="store_true", default=False)
    args = ap.parse_args()

    root = _project_root()
    p = _paths(root)
    p["results_reports"].mkdir(parents=True, exist_ok=True)
    p["results_figures"].mkdir(parents=True, exist_ok=True)

    model_path = p["models"] / args.model_name
    data_path = p["data_processed"] / args.dataset_name
    report_path = p["results_reports"] / "aapl_rf_pup_eval_report.json"

    print("[eval] Loading model + dataset ...")

    bundle = joblib.load(model_path)
    df = pd.read_parquet(data_path)
    if args.start_date:
        start = pd.to_datetime(args.start_date)
        if df.index.tz is not None:
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")
        df = df[df.index >= start]
        print(f"[eval] start_date={args.start_date} | rows_after_filter={len(df):,}")
        if len(df) == 0:
            raise RuntimeError("[eval] ERROR: dataset is empty after start_date filter.")

    clf: RandomForestClassifier = bundle["model"]
    feature_cols = list(bundle["feature_cols"])
    mu_up = float(bundle["mu_up"])
    mu_down = float(bundle["mu_down"])

    t = args.ticker.lower()

    print(f"[eval] Model: {model_path.name}")
    print(f"[eval] Features ({len(feature_cols)}): {feature_cols}")

    required = list(feature_cols) + ["target_up", "target_next_log_return", "close_t", "close_t1"]
    alignment_audit(df, required, strict=True)

    if args.leakage_audit:
        audit = leakage_audit(
            df=df,
            feature_cols=feature_cols,
            target_col="target_next_log_return",
            close_cols=["close_t", "close", "adj_close", "open", "high", "low", "volume", "target_next_log_return", "target_up"],
            max_lag=5,
            corr_threshold=0.20,
            identity_corr_threshold=0.999,
            run_smoke_model=args.leakage_smoke_model,
            random_state=42,
        )
        if not audit.get("passed", False):
            raise ValueError("[leakage] Audit failed. See report above for details.")

    n = len(df)
    split = int(round(n * (1.0 - args.test_frac)))
    split = max(1, min(split, n - 1))

    test_df = df.iloc[split:].copy()
    test_start = test_df.index.min()
    print(f"[eval] mu_up={mu_up:.6f} mu_down={mu_down:.6f} | test_start={str(test_start)}")

    X_test = test_df[feature_cols]
    y_test = test_df["target_up"].astype(int).to_numpy()
    logret_test = test_df["target_next_log_return"].astype(float).to_numpy()
    dates = test_df.index

    # Train predictions (for interval plot)
    train_df = df.iloc[:split].copy()
    X_train = train_df[feature_cols]
    proba_train = clf.predict_proba(X_train)[:, 1]
    pred_logret_train = proba_train * mu_up + (1.0 - proba_train) * mu_down

    proba = clf.predict_proba(X_test)[:, 1]
    pred_logret = proba * mu_up + (1.0 - proba) * mu_down
    residuals = logret_test - pred_logret

    y_hat = (proba >= 0.5).astype(int)
    metrics_cls = {
        "accuracy": _safe_float(accuracy_score(y_test, y_hat)),
        "roc_auc": _safe_float(roc_auc_score(y_test, proba)),
        "log_loss": _safe_float(log_loss(y_test, proba, labels=[0, 1])),
    }
    print(f"[eval] Test metrics (classification): {metrics_cls}")

    mse = mean_squared_error(logret_test, pred_logret)
    metrics_ret = {
        "mae": _safe_float(mean_absolute_error(logret_test, pred_logret)),
        "mse": _safe_float(mse),
        "rmse": _safe_float(np.sqrt(mse)),
        "r2": _safe_float(r2_score(logret_test, pred_logret)),
    }
    print(f"[eval] Test metrics (expected logret): {metrics_ret}")

    strat_logret, strat_eq, positions = strategy_equity_from_proba(logret_test, proba, threshold=0.5)
    bh_eq = buy_hold_equity(logret_test)

    print(f"[eval] Strategy (NO costs): strat={float(strat_eq[-1]):.4f} vs buyhold={float(bh_eq[-1]):.4f}")

    fig_dir = p["results_figures"]

    # Plots
    plot_equity_curve(
        dates=dates,
        strat_eq=strat_eq,
        bh_eq=bh_eq,
        save_path=fig_dir / f"{t}_equity_curve_strategy_vs_buyhold.png",
    )

    # Monte Carlo random investment baseline
    mc_curves, mc_mean, mc_std = monte_carlo_random_equity(
        logret_next=logret_test,
        n_sims=200,
        p_long=0.5,
        seed=42,
        positions_strategy=positions,
        block_size=20,
    )
    plot_strategy_vs_buyhold_vs_mc(
        dates=dates,
        strat_eq=strat_eq,
        bh_eq=bh_eq,
        mc_curves=mc_curves,
        mc_mean=mc_mean,
        mc_std=mc_std,
        mc_block_size=20,
        save_path=fig_dir / f"{t}_equity_curve_strategy_vs_buyhold_vs_random_mc.png",
    )

    plot_logret_prediction(
        dates=dates,
        y_true=logret_test,
        y_pred=pred_logret,
        residuals=residuals,
        save_path=fig_dir / f"{t}_logret_true_vs_pred_with_residuals.png",
    )

    plot_residual_hist(
        residuals=residuals,
        save_path=fig_dir / f"{t}_residuals_hist_logret.png",
        title="Residuals (log return)",
    )

    # Price curve with train/test intervals and predictions
    plot_price_train_test_predictions(
        dates=df.index,
        close_t=df["close_t"].astype(float).to_numpy(),
        pred_logret_train=pred_logret_train,
        pred_logret_test=pred_logret,
        split_idx=split,
        metrics_cls=metrics_cls,
        metrics_ret=metrics_ret,
        out_path=fig_dir / f"{t}_price_train_test_predictions.png",
    )

    if {"close_t", "close_t1"}.issubset(test_df.columns):
        close_t = test_df["close_t"].astype(float).to_numpy()
        close_t1 = test_df["close_t1"].astype(float).to_numpy()
        plot_price_reconstruction_last365(
            dates=dates,
            close_t=close_t,
            close_t1=close_t1,
            pred_logret=pred_logret,
            residuals=residuals,
            out_path=fig_dir / f"{t}_price_true_vs_pred_last365_band3sigma_with_residuals.png",
        )
    else:
        print("[eval] NOTE: ['close_t', 'close_t1'] not found -> skipping price reconstruction plot.")

    # Next-day prediction (one-step) for the last available date
    if {"close_t"}.issubset(test_df.columns) and len(test_df) > 0:
        close_t_last = float(test_df["close_t"].iloc[-1])
        pred_lr_last = float(pred_logret[-1])
        sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

        pred_next = close_t_last * np.exp(pred_lr_last)
        lower = pred_next * np.exp(-3.0 * sigma)
        upper = pred_next * np.exp(3.0 * sigma)

        pred_ret = np.exp(pred_lr_last) - 1.0
        ret_lower = np.exp(pred_lr_last - 3.0 * sigma) - 1.0
        ret_upper = np.exp(pred_lr_last + 3.0 * sigma) - 1.0

        last_date = str(test_df.index[-1])
        print(f"[eval] Next-day prediction from {last_date}:")
        print(f"  pred_close_t1={pred_next:.2f} | band_3sigma=[{lower:.2f}, {upper:.2f}]")
        print(
            f"  pred_logret={pred_lr_last:.6f} | pred_return={pred_ret*100:.3f}% | "
            f"band_3sigma_return=[{ret_lower*100:.3f}%, {ret_upper*100:.3f}%]"
        )

    # Feature importance (MDI)
    mdi = np.asarray(clf.feature_importances_, dtype=float)
    plot_feature_importance_mdi(
        feature_names=feature_cols,
        importances=mdi,
        out_path=fig_dir / f"{t}_feature_importance_mdi.png",
        title="Feature importance (MDI)",
    )

    # Permutation importance
    perm = permutation_importance(
        estimator=clf,
        X=X_test,
        y=y_test,
        n_repeats=int(args.perm_repeats),
        random_state=42,
        n_jobs=-1,
        scoring="accuracy",
    )
    plot_permutation_importance(
        feature_names=feature_cols,
        importances_mean=perm.importances_mean,
        importances_std=perm.importances_std,
        out_path=fig_dir / f"{t}_feature_importance_permutation.png",
        title=f"Feature importance (Permutation, n_repeats={args.perm_repeats})",
        xlabel="Decrease in score (accuracy)",
    )

    nested_wf = None
    if args.nested_wf:
        print("[eval] Nested walk-forward validation ...")
        base_params = clf.get_params()
        outer_train_years = args.outer_train_years
        outer_test_months = args.outer_test_months
        inner_folds = args.inner_folds
        inner_min_train_days = args.inner_min_train_days
        if args.nested_wf_fast:
            base_params["n_estimators"] = 200
            print("[eval] Nested WF fast mode enabled.")
            outer_train_years = 5
            outer_test_months = 3
            inner_folds = 3
            inner_min_train_days = 504
        nested_wf = nested_walk_forward_eval(
            df=df,
            feature_cols=feature_cols,
            target_col="target_up",
            base_model_params=base_params,
            outer_train_years=outer_train_years,
            outer_test_months=outer_test_months,
            inner_folds=inner_folds,
            inner_min_train_size=inner_min_train_days,
            purge_gap_days=args.purge_gap_days,
            random_state=42,
            fast_mode=args.nested_wf_fast,
        )

        folds = nested_wf.get("outer", {}).get("folds", [])
        overall = nested_wf.get("outer", {}).get("overall", {})
        skipped = nested_wf.get("outer", {}).get("skipped", [])

        for f in folds:
            print(f"[eval][WF] Fold {f['fold']}:")
            print(f"  train: {f['train_start']} -> {f['train_end']} (n={f['n_train']})")
            for i, s in enumerate(f.get("inner_splits", []), start=1):
                print(f"    inner {i}: train {s['train_start']} -> {s['train_end']} | "
                      f"val {s['val_start']} -> {s['val_end']}")
            print(f"  best params: {f['best_params']} | best mean logloss={f['inner_summary']['best_mean_logloss']:.4f}")
            print(f"  test: {f['test_start']} -> {f['test_end']} (n={f['n_test']})")
            print(f"  metrics: logloss={f['metrics']['log_loss']:.4f} "
                  f"auc={f['metrics']['roc_auc']:.4f} acc={f['metrics']['accuracy']:.4f}")

        if skipped:
            print(f"[eval][WF] Skipped folds: {skipped}")

        if overall:
            print(
                "[eval][WF] Overall: "
                f"logloss={overall.get('log_loss_mean', float('nan')):.4f}±{overall.get('log_loss_std', float('nan')):.4f} | "
                f"auc={overall.get('roc_auc_mean', float('nan')):.4f}±{overall.get('roc_auc_std', float('nan')):.4f} | "
                f"acc={overall.get('accuracy_mean', float('nan')):.4f}±{overall.get('accuracy_std', float('nan')):.4f}"
            )

        if folds:
            last = folds[-1]
            print(f"[eval][WF] Last fold proba length: {last.get('proba_len', 'n/a')} (n_test={last.get('n_test', 'n/a')})")

        # Plot logloss over time (outer test end dates)
        if folds:
            wf_dates = pd.to_datetime([f["test_end"] for f in folds])
            wf_ll = [f["metrics"]["log_loss"] for f in folds]
            plot_walkforward_logloss(
                dates=wf_dates,
                logloss_values=wf_ll,
                out_path=fig_dir / f"{t}_walkforward_logloss.png",
                title="Walk-forward logloss over time",
            )

    report: Dict[str, Any] = {
        "ticker": args.ticker,
        "model_file": model_path.name,
        "dataset_file": data_path.name,
        "features": feature_cols,
        "mu_up": _safe_float(mu_up),
        "mu_down": _safe_float(mu_down),
        "test_start": str(test_start),
        "metrics_classification": metrics_cls,
        "metrics_expected_logret": metrics_ret,
        "strategy": {
            "final_equity_strat": _safe_float(strat_eq[-1]),
            "final_equity_buyhold": _safe_float(bh_eq[-1]),
        },
    }
    if nested_wf is not None:
        report["nested_walk_forward"] = nested_wf

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[eval] Saved report: {report_path}")
    print(f"[eval] Saved figures in: {fig_dir}")
    print("[eval] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
