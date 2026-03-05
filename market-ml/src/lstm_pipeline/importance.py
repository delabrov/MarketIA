from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def block_permutation_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    b = max(int(block_size), 1)
    blocks = [np.arange(i, min(i + b, n)) for i in range(0, n, b)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order]).astype(int)


def _predict_array(model, X: np.ndarray, device: str, batch_size: int = 256) -> np.ndarray:
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


def _strategy_sharpe(y_true: np.ndarray, y_pred: np.ndarray, mode: str, threshold: float) -> float:
    if len(y_true) == 0:
        return float("nan")
    if mode == "threshold":
        pos = np.where(np.abs(y_pred) >= threshold, np.sign(y_pred), 0.0)
    else:
        pos = np.sign(y_pred)
    logret = pos * y_true
    std = np.std(logret)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(np.mean(logret) / std * np.sqrt(252.0))


def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _score(metric: str, y_true: np.ndarray, y_pred: np.ndarray, mode: str, threshold: float) -> float:
    m = metric.lower()
    if m == "mse":
        return float(mean_squared_error(y_true, y_pred))
    if m == "sharpe":
        return _strategy_sharpe(y_true, y_pred, mode=mode, threshold=threshold)
    return _ic(y_true, y_pred)


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    metric: str = "ic",
    n_repeats: int = 5,
    block_size: Optional[int] = None,
    batch_size: int = 256,
    device: str = "cpu",
    seed: int = 42,
    strategy_mode: str = "sign",
    strategy_threshold: float = 0.0,
    horizon_index: int = 0,
) -> Dict[str, object]:
    if X.ndim != 3:
        raise ValueError("X must be [N, L, d]")
    n, _, d = X.shape
    if d != len(feature_names):
        raise ValueError("feature_names length does not match X.shape[2]")

    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    if y.ndim == 2:
        y = y[:, horizon_index]
    y = y.reshape(-1)

    base_pred = _predict_array(model, X, device=device, batch_size=batch_size)
    if base_pred.ndim == 2:
        base_pred = base_pred[:, horizon_index]
    base_score = _score(metric, y, base_pred, strategy_mode, strategy_threshold)

    importances = np.zeros((d, n_repeats), dtype=float)

    for i in range(d):
        for r in range(n_repeats):
            Xp = X.copy()
            if block_size is not None and block_size > 1:
                perm_idx = block_permutation_indices(n, block_size, rng)
            else:
                perm_idx = rng.permutation(n)
            Xp[:, :, i] = X[perm_idx, :, i]
            y_pred = _predict_array(model, Xp, device=device, batch_size=batch_size)
            if y_pred.ndim == 2:
                y_pred = y_pred[:, horizon_index]
            score = _score(metric, y, y_pred, strategy_mode, strategy_threshold)

            if metric.lower() == "mse":
                importances[i, r] = score - base_score
            else:
                importances[i, r] = base_score - score

    imp_mean = importances.mean(axis=1)
    imp_std = importances.std(axis=1, ddof=1) if n_repeats > 1 else np.zeros(d)

    return {
        "feature_names": feature_names,
        "importance_mean": imp_mean,
        "importance_std": imp_std,
        "baseline_score": float(base_score),
        "metric": metric,
    }


def ablation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    metric: str = "ic",
    batch_size: int = 256,
    device: str = "cpu",
    strategy_mode: str = "sign",
    strategy_threshold: float = 0.0,
    horizon_index: int = 0,
) -> Dict[str, object]:
    if X.ndim != 3:
        raise ValueError("X must be [N, L, d]")
    n, _, d = X.shape
    if d != len(feature_names):
        raise ValueError("feature_names length does not match X.shape[2]")

    y = np.asarray(y)
    if y.ndim == 2:
        y = y[:, horizon_index]
    y = y.reshape(-1)
    base_pred = _predict_array(model, X, device=device, batch_size=batch_size)
    if base_pred.ndim == 2:
        base_pred = base_pred[:, horizon_index]
    base_score = _score(metric, y, base_pred, strategy_mode, strategy_threshold)

    importances = np.zeros(d, dtype=float)
    for i in range(d):
        Xp = X.copy()
        Xp[:, :, i] = 0.0
        y_pred = _predict_array(model, Xp, device=device, batch_size=batch_size)
        if y_pred.ndim == 2:
            y_pred = y_pred[:, horizon_index]
        score = _score(metric, y, y_pred, strategy_mode, strategy_threshold)
        if metric.lower() == "mse":
            importances[i] = score - base_score
        else:
            importances[i] = base_score - score

    return {
        "feature_names": feature_names,
        "importance_mean": importances,
        "importance_std": np.zeros(d),
        "baseline_score": float(base_score),
        "metric": metric,
    }
