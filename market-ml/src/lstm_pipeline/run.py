from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import mean_squared_error, r2_score

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from .data_loading import load_base_and_exog, project_paths
from .features import build_features
from .dataset import SplitConfig, compute_split_masks, scale_features, make_sequences, create_loaders
from .model import LSTMRegressor
from .train import TrainConfig, train_model
from .evaluate import (
    predict_array,
    compute_pred_metrics_extended,
    decile_stats,
)
from .plots import (
    plot_loss,
    plot_feature_importance,
    plot_price_train_val_test_predictions,
    plot_price_test_zoom_with_residuals,
    plot_returns_true_vs_pred,
    plot_residuals_acf,
    plot_residuals_hist,
    plot_pred_vs_true_enhanced,
    plot_decile_means,
    plot_compare_horizons_metrics,
    plot_compare_returns_and_residuals,
    plot_compare_deciles,
)

# Reuse RF downloader to keep data fresh (same raw structure)
from others.rf_pipeline import download_data as rf_download
from .importance import permutation_importance, ablation_importance


def load_config(path: Path) -> Dict[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Please add pyyaml to requirements.")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_device(cfg: Dict[str, Any]) -> str:
    device = cfg.get("device", "auto")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _download_if_needed(cfg: Dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    start_date = data_cfg.get("start_date")
    end_date = data_cfg.get("end_date")
    do_download = bool(data_cfg.get("download", True))
    use_exog = bool(feat_cfg.get("use_exog", feat_cfg.get("use_exogenous", False)))
    use_vix = bool(data_cfg.get("use_vix", False)) if use_exog else False
    use_spy = bool(data_cfg.get("use_spy", False)) if use_exog else False

    if not do_download:
        return

    ticker = data_cfg.get("ticker", "AAPL")
    print("[lstm] Downloading data ...")
    P = rf_download.project_paths()
    P["DATA_RAW"].mkdir(parents=True, exist_ok=True)
    df_main = rf_download.download_clean_ohlcv(ticker=ticker, start=start_date, end=end_date)
    rf_download.save_raw(df_main, P["DATA_RAW"] / f"{ticker.lower()}_ohlcv.parquet", P["DATA_RAW"] / f"{ticker.lower()}_ohlcv.csv")
    if use_spy:
        df_spy = rf_download.download_clean_ohlcv(ticker="SPY", start=start_date, end=end_date)
        rf_download.save_raw(df_spy, P["DATA_RAW"] / "spy_ohlcv.parquet", P["DATA_RAW"] / "spy_ohlcv.csv")
    if use_vix:
        df_vix = rf_download.download_clean_ohlcv(ticker="^VIX", start=start_date, end=end_date)
        rf_download.save_raw(df_vix, P["DATA_RAW"] / "vix_ohlcv.parquet", P["DATA_RAW"] / "vix_ohlcv.csv")


def _run_single(cfg: Dict[str, Any], horizon: int) -> Dict[str, str]:
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    seq_cfg = cfg.get("sequence", {})
    split_cfg = cfg.get("split", {})
    model_cfg = cfg.get("model", {})
    out_cfg = cfg.get("outputs", {})
    fi_cfg = cfg.get("feature_importance", {})
    plots_cfg = cfg.get("plots", {})
    metrics_cfg = cfg.get("metrics", {})

    seed = int(model_cfg.get("seed", 42))
    set_seed(seed)

    ticker = data_cfg.get("ticker", "AAPL")
    start_date = data_cfg.get("start_date")
    use_exog = bool(feat_cfg.get("use_exog", feat_cfg.get("use_exogenous", False)))
    use_vix = bool(data_cfg.get("use_vix", False)) if use_exog else False
    use_spy = bool(data_cfg.get("use_spy", False)) if use_exog else False
    use_regime = bool(feat_cfg.get("use_regime_features", False))

    seq_len = int(seq_cfg.get("length", 60))

    outputs_root = Path(project_paths()["ROOT"]) / out_cfg.get("root", "results_lstm")
    base_run_name = out_cfg.get("run_name", f"{ticker.lower()}_lstm")
    run_name = f"{base_run_name}_h{horizon}"
    run_dir = outputs_root / run_name
    model_dir = run_dir / "models"
    plots_dir = run_dir / "plots"
    reports_dir = run_dir / "reports"
    preds_dir = run_dir / "preds"
    logs_dir = run_dir / "logs"
    for d in [model_dir, plots_dir, reports_dir, preds_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"[lstm] Loading data for {ticker} (h={horizon}) ...")
    df_raw = load_base_and_exog(ticker=ticker, start_date=start_date, use_vix=use_vix, use_spy=use_spy)

    features, feature_cols = build_features(
        df_raw,
        use_vix=use_vix,
        use_spy=use_spy,
        use_exog=use_exog,
        use_regime_features=use_regime,
    )
    added = [f for f in ["vix_zscore_60"] if f in feature_cols]
    if added:
        print(f"[lstm] Added features: {', '.join(added)}")

    # Target: log(C_{t+h}/C_t)
    close = df_raw.loc[features.index, "close"].astype(float)
    df_feat = features.copy()
    df_feat["target"] = np.log(close.shift(-horizon) / close)
    df_feat = df_feat.dropna(subset=feature_cols + ["target"])
    if len(df_feat) == 0:
        raise RuntimeError("[lstm] ERROR: empty dataset after feature/target construction.")

    index = df_feat.index
    split = SplitConfig(
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        train_end_date=split_cfg.get("train_end_date"),
        val_end_date=split_cfg.get("val_end_date"),
    )
    train_mask, val_mask, test_mask = compute_split_masks(index, split)
    train_dates = index[train_mask]
    val_dates = index[val_mask]
    test_dates = index[test_mask]
    if len(train_dates) and len(val_dates) and len(test_dates):
        print(
            f"[lstm] Split: train={len(train_dates)} ({train_dates.min().date()}→{train_dates.max().date()}), "
            f"val={len(val_dates)} ({val_dates.min().date()}→{val_dates.max().date()}), "
            f"test={len(test_dates)} ({test_dates.min().date()}→{test_dates.max().date()})"
        )

    X_scaled, scaler = scale_features(df_feat[feature_cols], train_mask)
    y = df_feat["target"].values.astype(float).reshape(-1, 1)
    dates = df_feat.index.values

    X_seq, y_seq, dates_seq, idxs = make_sequences(X_scaled, y, dates, seq_len)

    train_loader, val_loader, test_loader = create_loaders(
        X_seq, y_seq, idxs, train_mask, val_mask, test_mask, batch_size=int(model_cfg.get("batch_size", 64))
    )

    input_dim = X_seq.shape[2] if X_seq.size else len(feature_cols)
    model = LSTMRegressor(
        input_dim=input_dim,
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.0)),
        output_dim=1,
    )

    device = _get_device(model_cfg)
    train_cfg = TrainConfig(
        epochs=int(model_cfg.get("epochs", 50)),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0)),
        patience=int(model_cfg.get("patience", 8)),
        grad_clip=float(model_cfg.get("grad_clip", 1.0)),
        loss=str(model_cfg.get("loss", "mse")),
        device=device,
        verbose=bool(model_cfg.get("verbose", True)),
    )

    print(f"[lstm] Samples: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"[lstm] Features={len(feature_cols)} | Seq={seq_len} | Horizon={horizon} | Device={device}")

    model, history = train_model(model, train_loader, val_loader, train_cfg)

    # Predictions (ordered)
    train_mask_seq = train_mask[idxs]
    val_mask_seq = val_mask[idxs]
    test_mask_seq = test_mask[idxs]

    X_train = X_seq[train_mask_seq]
    X_val = X_seq[val_mask_seq]
    X_test = X_seq[test_mask_seq]

    y_true_train = y_seq[train_mask_seq].reshape(-1)
    y_true_val = y_seq[val_mask_seq].reshape(-1)
    y_true_test = y_seq[test_mask_seq].reshape(-1)

    dates_train = dates_seq[train_mask_seq]
    dates_val = dates_seq[val_mask_seq]
    dates_test = dates_seq[test_mask_seq]

    y_pred_train = predict_array(model, X_train, device, batch_size=int(model_cfg.get("batch_size", 64))).reshape(-1)
    y_pred_val = predict_array(model, X_val, device, batch_size=int(model_cfg.get("batch_size", 64))).reshape(-1)
    y_pred_test = predict_array(model, X_test, device, batch_size=int(model_cfg.get("batch_size", 64))).reshape(-1)

    mape_eps = float(metrics_cfg.get("mape_epsilon", 1e-6))
    metrics_pred = {
        "train": compute_pred_metrics_extended(y_true_train, y_pred_train, mape_epsilon=mape_eps),
        "val": compute_pred_metrics_extended(y_true_val, y_pred_val, mape_epsilon=mape_eps),
        "test": compute_pred_metrics_extended(y_true_test, y_pred_test, mape_epsilon=mape_eps),
    }

    # Deciles (test)
    deciles_enabled = bool(plots_cfg.get("deciles_enabled", True))
    deciles_n = int(plots_cfg.get("deciles_n", 10))
    decile_df = None
    decile_csv = None
    if deciles_enabled:
        decile_df = decile_stats(y_true_test, y_pred_test, n_deciles=deciles_n)
        if len(decile_df) > 0:
            decile_csv = reports_dir / f"{ticker.lower()}_pred_deciles_h{horizon}.csv"
            decile_df.to_csv(decile_csv, index=False)

    # Save model + scaler
    model_path = model_dir / f"{ticker.lower()}_lstm_h{horizon}.pt"
    torch.save(model.state_dict(), model_path)
    scaler_path = model_dir / f"{ticker.lower()}_scaler_h{horizon}.joblib"
    joblib.dump(scaler, scaler_path)

    # Save preds
    n_seq = len(dates_seq)
    y_pred_all = np.full(n_seq, np.nan, dtype=float)
    split_labels = np.empty(n_seq, dtype=object)
    split_labels[train_mask_seq] = "train"
    split_labels[val_mask_seq] = "val"
    split_labels[test_mask_seq] = "test"
    y_pred_all[train_mask_seq] = y_pred_train
    y_pred_all[val_mask_seq] = y_pred_val
    y_pred_all[test_mask_seq] = y_pred_test

    y_true_all = y_seq.reshape(-1)
    residual_all = y_true_all - y_pred_all

    preds_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_seq),
            "split": split_labels,
            "y_true": y_true_all,
            "y_pred": y_pred_all,
            "residual": residual_all,
        }
    )
    if deciles_enabled and len(y_pred_test) > 0:
        decile_labels = np.full(n_seq, np.nan)
        s = pd.Series(y_pred_test)
        try:
            d = pd.qcut(s, q=deciles_n, labels=False, duplicates="drop") + 1
            decile_labels[test_mask_seq] = d.values
        except Exception:
            pass
        preds_df["decile"] = decile_labels

    preds_path = preds_dir / f"{ticker.lower()}_lstm_h{horizon}_preds.csv"
    preds_df.to_csv(preds_path, index=False)

    # Price plots (optional)
    price_metrics = None
    if bool(plots_cfg.get("price_plots_enabled", False)):
        close_series = df_raw.loc[df_feat.index, "close"].astype(float)
        close_t = close_series.values
        close_t1 = close_series.shift(-1).values
        close_t_seq = close_t[idxs]
        close_t1_seq = close_t1[idxs]

        pred_price_train = np.full_like(close_t1_seq, np.nan, dtype=float)
        pred_price_val = np.full_like(close_t1_seq, np.nan, dtype=float)
        pred_price_test = np.full_like(close_t1_seq, np.nan, dtype=float)

        pred_price_train[train_mask_seq] = close_t_seq[train_mask_seq] * np.exp(y_pred_train)
        pred_price_val[val_mask_seq] = close_t_seq[val_mask_seq] * np.exp(y_pred_val)
        pred_price_test[test_mask_seq] = close_t_seq[test_mask_seq] * np.exp(y_pred_test)

        def _valid_xy(a, b):
            a = np.asarray(a, dtype=float)
            b = np.asarray(b, dtype=float)
            m = np.isfinite(a) & np.isfinite(b)
            return a[m], b[m]

        def _rmse(a, b) -> float:
            a2, b2 = _valid_xy(a, b)
            if len(a2) == 0:
                return float("nan")
            return float(np.sqrt(mean_squared_error(a2, b2)))

        def _r2(a, b) -> float:
            a2, b2 = _valid_xy(a, b)
            if len(a2) == 0:
                return float("nan")
            return float(r2_score(a2, b2))

        price_metrics = {
            "train": {
                "rmse": _rmse(close_t1_seq[train_mask_seq], pred_price_train[train_mask_seq]),
                "r2": _r2(close_t1_seq[train_mask_seq], pred_price_train[train_mask_seq]),
            },
            "val": {
                "rmse": _rmse(close_t1_seq[val_mask_seq], pred_price_val[val_mask_seq]),
                "r2": _r2(close_t1_seq[val_mask_seq], pred_price_val[val_mask_seq]),
            },
            "test": {
                "rmse": _rmse(close_t1_seq[test_mask_seq], pred_price_test[test_mask_seq]),
                "r2": _r2(close_t1_seq[test_mask_seq], pred_price_test[test_mask_seq]),
            },
        }

        plot_price_train_val_test_predictions(
            dates=pd.to_datetime(dates_seq),
            true_price=close_t1_seq,
            pred_train=pred_price_train,
            pred_val=pred_price_val,
            pred_test=pred_price_test,
            metrics=price_metrics,
            save_path=plots_dir / f"{ticker.lower()}_price_train_val_test_predictions_h{horizon}.png",
            title=f"AAPL Price: True vs Predicted (h={horizon})",
        )

        plot_price_test_zoom_with_residuals(
            dates=pd.to_datetime(dates_seq[test_mask_seq]),
            true_price=close_t1_seq[test_mask_seq],
            pred_test=pred_price_test[test_mask_seq],
            rmse=price_metrics["test"]["rmse"],
            r2=price_metrics["test"]["r2"],
            save_path=plots_dir / f"{ticker.lower()}_price_test_zoom_with_residuals_h{horizon}.png",
            title=f"AAPL Price: Test Zoom (h={horizon})",
        )

    # Feature importance (optional)
    fi_results = {}
    if bool(fi_cfg.get("enabled", False)):
        fi_method = str(fi_cfg.get("method", "permutation")).lower()
        fi_metric = str(fi_cfg.get("metric", "ic")).lower()
        fi_repeats = int(fi_cfg.get("n_repeats", 5))
        fi_block = fi_cfg.get("block_size", None)
        fi_topn = int(fi_cfg.get("top_n", 20))
        fi_batch = int(fi_cfg.get("batch_size", model_cfg.get("batch_size", 64)))

        if fi_method in ("permutation", "both"):
            perm = permutation_importance(
                model=model,
                X=X_test,
                y=y_true_test,
                feature_names=feature_cols,
                metric=fi_metric,
                n_repeats=fi_repeats,
                block_size=fi_block,
                batch_size=fi_batch,
                device=device,
                seed=seed,
                strategy_mode="sign",
                strategy_threshold=0.0,
                horizon_index=0,
            )
            perm_df = pd.DataFrame(
                {
                    "feature": perm["feature_names"],
                    "importance_mean": perm["importance_mean"],
                    "importance_std": perm["importance_std"],
                    "baseline_score": perm["baseline_score"],
                    "metric_name": perm["metric"],
                }
            )
            perm_csv = reports_dir / f"{ticker.lower()}_feature_importance_permutation_h{horizon}.csv"
            perm_df.to_csv(perm_csv, index=False)
            plot_feature_importance(
                feature_names=perm["feature_names"],
                importance_mean=perm["importance_mean"],
                importance_std=perm["importance_std"],
                save_path=plots_dir / f"{ticker.lower()}_feature_importance_permutation_h{horizon}.png",
                title=f"Permutation Importance (h={horizon})",
                top_n=fi_topn,
            )
            fi_results["permutation"] = {
                "baseline_score": float(perm["baseline_score"]),
                "metric": perm["metric"],
                "horizon": int(horizon),
                "csv": str(perm_csv),
            }

        if fi_method in ("ablation", "both"):
            abl = ablation_importance(
                model=model,
                X=X_test,
                y=y_true_test,
                feature_names=feature_cols,
                metric=fi_metric,
                batch_size=fi_batch,
                device=device,
                strategy_mode="sign",
                strategy_threshold=0.0,
                horizon_index=0,
            )
            abl_df = pd.DataFrame(
                {
                    "feature": abl["feature_names"],
                    "importance_mean": abl["importance_mean"],
                    "importance_std": abl["importance_std"],
                    "baseline_score": abl["baseline_score"],
                    "metric_name": abl["metric"],
                }
            )
            abl_csv = reports_dir / f"{ticker.lower()}_feature_importance_ablation_h{horizon}.csv"
            abl_df.to_csv(abl_csv, index=False)
            plot_feature_importance(
                feature_names=abl["feature_names"],
                importance_mean=abl["importance_mean"],
                importance_std=abl["importance_std"],
                save_path=plots_dir / f"{ticker.lower()}_feature_importance_ablation_h{horizon}.png",
                title=f"Ablation Importance (h={horizon})",
                top_n=fi_topn,
            )
            fi_results["ablation"] = {
                "baseline_score": float(abl["baseline_score"]),
                "metric": abl["metric"],
                "horizon": int(horizon),
                "csv": str(abl_csv),
            }

    # Report
    report = {
        "ticker": ticker,
        "features": feature_cols,
        "use_regime_features": use_regime,
        "prediction_horizon": horizon,
        "seq_len": seq_len,
        "rows_total": int(len(df_feat)),
        "train_rows": int(train_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "metrics_prediction": metrics_pred,
        "metrics_price_prediction": price_metrics,
        "metrics_trading_test": None,
        "metrics_trading_test_net": None,
        "feature_importance": fi_results,
        "pred_deciles_csv": str(decile_csv) if decile_csv else None,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "preds_path": str(preds_path),
        "config": cfg,
    }
    report_path = reports_dir / f"{ticker.lower()}_lstm_h{horizon}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Plots (per-horizon)
    smoothing_window = int(plots_cfg.get("smoothing_window", 5))
    returns_plots_enabled = bool(plots_cfg.get("returns_plots_enabled", True))
    residuals_plots_enabled = bool(plots_cfg.get("residuals_plots_enabled", True))
    acf_max_lag = int(plots_cfg.get("acf_max_lag", 20))

    if returns_plots_enabled:
        plot_returns_true_vs_pred(
            dates=pd.to_datetime(dates_test),
            y_true=y_true_test,
            y_pred=y_pred_test,
            save_path=plots_dir / f"{ticker.lower()}_returns_true_vs_pred_test_h{horizon}.png",
            title=f"Returns: True vs Predicted (Test, h={horizon})",
            smoothing_window=smoothing_window,
        )
        plot_pred_vs_true_enhanced(
            y_true=y_true_test,
            y_pred=y_pred_test,
            save_path=plots_dir / f"{ticker.lower()}_pred_vs_true_test_h{horizon}.png",
            title=f"Predicted vs True Returns (Test, h={horizon})",
        )

    if residuals_plots_enabled:
        residuals_test = y_true_test - y_pred_test
        plot_residuals_hist(
            residuals=residuals_test,
            save_path=plots_dir / f"{ticker.lower()}_residuals_hist_test_h{horizon}.png",
            title=f"Residuals Histogram (Test, h={horizon})",
        )
        plot_residuals_acf(
            residuals=residuals_test,
            max_lag=acf_max_lag,
            save_path=plots_dir / f"{ticker.lower()}_residuals_acf_test_h{horizon}.png",
            title=f"Residuals ACF (Test, h={horizon}, lags 1..{acf_max_lag})",
        )

    if deciles_enabled and decile_df is not None and len(decile_df) > 0:
        plot_decile_means(
            decile_df=decile_df,
            save_path=plots_dir / f"{ticker.lower()}_decile_mean_true_h{horizon}.png",
            title=f"Decile Mean True Return (Test, h={horizon})",
        )

    plot_loss(
        history=history,
        save_path=plots_dir / f"{ticker.lower()}_loss_h{horizon}.png",
        title=f"Train/Val Loss (h={horizon})",
    )

    # Minimal terminal summary
    test_metrics = metrics_pred.get("test", {})
    if test_metrics:
        print(
            "[lstm] Test metrics: "
            f"h={horizon} | RMSE={test_metrics.get('rmse', float('nan')):.6f}, "
            f"MAE={test_metrics.get('mae', float('nan')):.6f}, "
            f"IC={test_metrics.get('ic', float('nan')):.3f}, "
            f"RankIC={test_metrics.get('rank_ic', float('nan')):.3f}, "
            f"Hit={test_metrics.get('hit_ratio', float('nan')):.3f}"
        )

    print(f"[lstm] Saved report: {report_path}")
    print(f"[lstm] Saved preds: {preds_path}")

    return {
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "preds_path": str(preds_path),
    }


def main(config_path: Path, horizon: Optional[int] = None, multi_run: bool = False) -> int:
    cfg = load_config(config_path)

    # Download once for multi-run
    _download_if_needed(cfg)

    if multi_run:
        run_horizons = cfg.get("run_horizons", [1, 3])
        run_horizons = [int(h) for h in run_horizons if int(h) > 0]
        if not run_horizons:
            run_horizons = [1, 3]
        for h in run_horizons:
            _run_single(cfg, h)
        return 0

    h = int(horizon) if horizon is not None else int(cfg.get("prediction_horizon", 1))
    _run_single(cfg, h)
    return 0


def cli() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=str(Path(__file__).resolve().parents[2] / "configs" / "lstm.yaml"))
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--multi-run", action="store_true", default=False)
    args = ap.parse_args()
    return main(Path(args.config), horizon=args.horizon, multi_run=args.multi_run)


if __name__ == "__main__":
    raise SystemExit(cli())
