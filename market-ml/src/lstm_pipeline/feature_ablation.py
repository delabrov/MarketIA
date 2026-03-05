from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .data_loading import load_base_and_exog
from .features import build_features
from .dataset import SplitConfig, compute_split_masks, scale_features, make_sequences, create_loaders
from .model import LSTMRegressor
from .train import TrainConfig, train_model
from .evaluate import compute_pred_metrics_extended, predict_array


@dataclass
class AblationRunResult:
    feature: str
    kept_features_count: int
    metrics: Dict[str, float]
    delta_ic: float
    delta_rmse: float
    metrics_std: Optional[Dict[str, float]] = None


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def _config_hash(cfg: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        payload = repr(cfg).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def prepare_base_dataset(cfg: Dict[str, Any], horizon: int = 1) -> Tuple[pd.DataFrame, List[str], np.ndarray, np.ndarray, np.ndarray, SplitConfig]:
    data_cfg = cfg.get("data", {})
    feat_cfg = cfg.get("features", {})
    split_cfg = cfg.get("split", {})

    ticker = data_cfg.get("ticker", "AAPL")
    start_date = data_cfg.get("start_date")
    use_vix = bool(data_cfg.get("use_vix", False))
    use_spy = bool(data_cfg.get("use_spy", False))
    use_exog = bool(feat_cfg.get("use_exog", feat_cfg.get("use_exogenous", False)))
    use_regime = bool(feat_cfg.get("use_regime_features", False))

    df_raw = load_base_and_exog(ticker=ticker, start_date=start_date, use_vix=use_vix, use_spy=use_spy)
    features, feature_cols = build_features(
        df_raw,
        use_vix=use_vix,
        use_spy=use_spy,
        use_exog=use_exog,
        use_regime_features=use_regime,
    )

    close = df_raw.loc[features.index, "close"].astype(float)
    df_feat = features.copy()
    df_feat["target"] = np.log(close.shift(-horizon) / close)
    df_feat = df_feat.dropna(subset=feature_cols + ["target"])
    if len(df_feat) == 0:
        raise RuntimeError("[ablation] Empty dataset after feature/target construction.")

    index = df_feat.index
    split = SplitConfig(
        train_ratio=float(split_cfg.get("train_ratio", 0.7)),
        val_ratio=float(split_cfg.get("val_ratio", 0.15)),
        test_ratio=float(split_cfg.get("test_ratio", 0.15)),
        train_end_date=split_cfg.get("train_end_date"),
        val_end_date=split_cfg.get("val_end_date"),
    )
    train_mask, val_mask, test_mask = compute_split_masks(index, split)

    X_all = df_feat[feature_cols].to_numpy(dtype=float)
    y_all = df_feat["target"].to_numpy(dtype=float)
    dates = index.to_numpy()

    return df_feat, feature_cols, X_all, y_all, dates, split


def train_eval_for_features(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    dates: np.ndarray,
    feature_cols: List[str],
    split: SplitConfig,
    cfg: Dict[str, Any],
    *,
    seed: int = 42,
    save_model: bool = False,
    model_path: Optional[Path] = None,
    scaler_path: Optional[Path] = None,
) -> Dict[str, float]:
    set_seed(seed)

    model_cfg = cfg.get("model", {})
    seq_cfg = cfg.get("sequence", {})
    metrics_cfg = cfg.get("metrics", {})

    seq_len = int(seq_cfg.get("length", 60))
    train_mask, val_mask, test_mask = compute_split_masks(pd.to_datetime(dates), split)

    if isinstance(X_all, np.ndarray):
        X_all = pd.DataFrame(X_all, columns=feature_cols)
    X_scaled, scaler = scale_features(X_all, train_mask)
    y = y_all.reshape(-1, 1)
    X_seq, y_seq, dates_seq, idxs = make_sequences(X_scaled, y, dates, seq_len)

    train_loader, val_loader, test_loader = create_loaders(
        X_seq,
        y_seq,
        idxs,
        train_mask,
        val_mask,
        test_mask,
        batch_size=int(model_cfg.get("batch_size", 64)),
    )
    test_mask_seq = test_mask[idxs]

    device = str(model_cfg.get("device", "auto"))
    if device == "auto":
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    model = LSTMRegressor(
        input_dim=len(feature_cols),
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        num_layers=int(model_cfg.get("num_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        output_dim=1,
    )

    train_cfg = TrainConfig(
        epochs=int(model_cfg.get("epochs", 50)),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 0.0)),
        patience=int(model_cfg.get("patience", 8)),
        grad_clip=float(model_cfg.get("grad_clip", 1.0)),
        loss=str(model_cfg.get("loss", "mse")),
        device=device,
        verbose=False,
    )

    model, _ = train_model(model, train_loader, val_loader, train_cfg)

    if save_model and model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        __import__("torch").save(model.state_dict(), model_path)
    if save_model and scaler_path is not None:
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        __import__("joblib").dump(scaler, scaler_path)

    y_pred_test = predict_array(
        model,
        X_seq[test_mask_seq],
        device=device,
        batch_size=int(model_cfg.get("batch_size", 256)),
    ).reshape(-1)
    y_true_test = y_seq[test_mask_seq].reshape(-1)

    metrics = compute_pred_metrics_extended(y_true_test, y_pred_test, mape_epsilon=float(metrics_cfg.get("mape_epsilon", 1e-6)))
    return metrics


def _aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
    if not metrics_list:
        return {}, None
    keys = metrics_list[0].keys()
    stacked = {k: np.array([m.get(k, np.nan) for m in metrics_list], dtype=float) for k in keys}
    mean = {k: float(np.nanmean(v)) for k, v in stacked.items()}
    std = None
    if len(metrics_list) > 1:
        std = {k: float(np.nanstd(v, ddof=1)) if len(v) > 1 else float("nan") for k, v in stacked.items()}
    return mean, std


def run_ablation(
    cfg: Dict[str, Any],
    *,
    horizon: int = 1,
    mode: str = "drop",
    features_subset: Optional[List[str]] = None,
    seed: int = 42,
    n_repeats: int = 1,
    max_features_tested: Optional[int] = None,
    save_models: bool = False,
    output_root: Optional[Path] = None,
    core_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df_feat, feature_cols, X_all, y_all, dates, split = prepare_base_dataset(cfg, horizon=horizon)

    if features_subset:
        feature_cols = [f for f in feature_cols if f in set(features_subset)]
    if max_features_tested is not None:
        feature_cols = feature_cols[: int(max_features_tested)]

    if output_root is None:
        output_root = Path(cfg.get("outputs", {}).get("root", "results_lstm"))

    base_metrics_runs = []
    for r in range(int(n_repeats)):
        base_metrics_runs.append(
            train_eval_for_features(
                X_all=df_feat[feature_cols],
                y_all=y_all,
                dates=dates,
                feature_cols=feature_cols,
                split=split,
                cfg=cfg,
                seed=seed + r,
                save_model=False,
            )
        )
    base_metrics, base_std = _aggregate_metrics(base_metrics_runs)

    results: List[AblationRunResult] = []
    mode = str(mode).lower()
    if mode not in {"drop", "add"}:
        raise ValueError("mode must be 'drop' or 'add'")

    if mode == "add":
        if not core_features:
            raise RuntimeError("[ablation] add mode requires core_features list.")
        base_set = [f for f in core_features if f in df_feat.columns]
        candidates = [f for f in feature_cols if f not in base_set]
        for f in candidates:
            feats = base_set + [f]
            metrics_runs = []
            for r in range(int(n_repeats)):
                metrics_runs.append(
                    train_eval_for_features(
                        X_all=df_feat[feats],
                        y_all=y_all,
                        dates=dates,
                        feature_cols=feats,
                        split=split,
                        cfg=cfg,
                        seed=seed + r,
                        save_model=save_models,
                    )
                )
            metrics, metrics_std = _aggregate_metrics(metrics_runs)
            delta_ic = float(base_metrics.get("ic", np.nan) - metrics.get("ic", np.nan))
            delta_rmse = float(metrics.get("rmse", np.nan) - base_metrics.get("rmse", np.nan))
            results.append(
                AblationRunResult(
                    feature=f,
                    kept_features_count=len(feats),
                    metrics=metrics,
                    delta_ic=delta_ic,
                    delta_rmse=delta_rmse,
                    metrics_std=metrics_std,
                )
            )
            print(
                f"[ablation] add {f}: IC={metrics.get('ic', np.nan):.4f} "
                f"ΔIC={delta_ic:+.4f} | RMSE={metrics.get('rmse', np.nan):.6f} "
                f"ΔRMSE={delta_rmse:+.6f}"
            )
    else:
        for f in feature_cols:
            feats = [c for c in feature_cols if c != f]
            metrics_runs = []
            for r in range(int(n_repeats)):
                metrics_runs.append(
                train_eval_for_features(
                    X_all=df_feat[feats],
                    y_all=y_all,
                    dates=dates,
                    feature_cols=feats,
                        split=split,
                        cfg=cfg,
                        seed=seed + r,
                        save_model=save_models,
                    )
                )
            metrics, metrics_std = _aggregate_metrics(metrics_runs)
            delta_ic = float(base_metrics.get("ic", np.nan) - metrics.get("ic", np.nan))
            delta_rmse = float(metrics.get("rmse", np.nan) - base_metrics.get("rmse", np.nan))
            results.append(
                AblationRunResult(
                    feature=f,
                    kept_features_count=len(feats),
                    metrics=metrics,
                    delta_ic=delta_ic,
                    delta_rmse=delta_rmse,
                    metrics_std=metrics_std,
                )
            )
            print(
                f"[ablation] drop {f}: IC={metrics.get('ic', np.nan):.4f} "
                f"ΔIC={delta_ic:+.4f} | RMSE={metrics.get('rmse', np.nan):.6f} "
                f"ΔRMSE={delta_rmse:+.6f}"
            )

    rows = []
    for r in results:
        row = {
            "feature": r.feature,
            "kept_features_count": r.kept_features_count,
            "ic": r.metrics.get("ic", np.nan),
            "rmse": r.metrics.get("rmse", np.nan),
            "mae": r.metrics.get("mae", np.nan),
            "rank_ic": r.metrics.get("rank_ic", np.nan),
            "hit_ratio": r.metrics.get("hit_ratio", np.nan),
            "delta_ic": r.delta_ic,
            "delta_rmse": r.delta_rmse,
        }
        if r.metrics_std:
            row["ic_std"] = r.metrics_std.get("ic", np.nan)
            row["rmse_std"] = r.metrics_std.get("rmse", np.nan)
            row["mae_std"] = r.metrics_std.get("mae", np.nan)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    summary = {
        "baseline": base_metrics,
        "baseline_std": base_std,
        "mode": mode,
        "horizon": int(horizon),
        "seed": int(seed),
        "n_repeats": int(n_repeats),
        "n_features": int(len(feature_cols)),
        "config_hash": _config_hash(cfg),
    }
    return df_out, summary


def save_summary(df_out: pd.DataFrame, summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_dir / "feature_ablation_results.csv", index=False)
    with open(out_dir / "feature_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_delta_barh(
    df_out: pd.DataFrame,
    *,
    value_col: str,
    save_path: Path,
    title: str,
    top_n: int = 30,
) -> None:
    if df_out is None or len(df_out) == 0 or value_col not in df_out.columns:
        return
    df_plot = df_out.sort_values(value_col, ascending=False)
    df_plot = df_plot.head(top_n)
    fig = plt.figure(figsize=(10, max(4, len(df_plot) * 0.35)), layout="constrained")
    ax = fig.add_subplot(111)
    ax.barh(df_plot["feature"][::-1], df_plot[value_col][::-1], alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel(value_col)
    ax.grid(True, axis="x", alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_delta_scatter(
    df_out: pd.DataFrame,
    *,
    save_path: Path,
    title: str,
) -> None:
    if df_out is None or len(df_out) == 0:
        return
    fig = plt.figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(111)
    ax.scatter(df_out["delta_ic"], df_out["delta_rmse"], alpha=0.7)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Delta IC (base - ablated)")
    ax.set_ylabel("Delta RMSE (ablated - base)")
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
