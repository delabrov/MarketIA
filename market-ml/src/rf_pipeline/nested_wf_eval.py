# src/nested_wf_eval.py

from __future__ import annotations

from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from tqdm.auto import tqdm


def _to_py(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    return x


def _ts(x) -> str:
    try:
        return pd.Timestamp(x).date().isoformat()
    except Exception:
        return str(x)


def _make_inner_splits(
    n: int,
    inner_folds: int,
    min_train_size: int,
    purge_gap: int,
) -> List[Tuple[int, int, int]]:
    """
    Returns list of (train_end, val_start, val_end) indices.
    """
    splits: List[Tuple[int, int, int]] = []
    if n <= min_train_size + 5:
        return splits

    remaining = n - min_train_size
    val_size = max(1, remaining // inner_folds)

    for k in range(inner_folds):
        train_end = min_train_size + k * val_size
        val_start = train_end + purge_gap
        val_end = min(n, val_start + val_size)
        if val_start >= n or (val_end - val_start) < 5:
            break
        splits.append((train_end, val_start, val_end))

    return splits


def nested_walk_forward_eval(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    base_model_params: dict,
    outer_train_years: int = 10,
    outer_test_months: int = 6,
    inner_folds: int = 5,
    inner_min_train_size: int = 756,
    purge_gap_days: int = 1,
    random_state: int = 42,
    fast_mode: bool = False,
) -> dict:
    d = df.copy()
    d = d.sort_index()

    # Ensure required columns exist
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns for nested WFV: {missing}")

    # Basic sizes (approx trading days)
    outer_train_size = int(round(outer_train_years * 252))
    outer_test_size = int(round(outer_test_months * 21))
    purge_gap = max(0, int(purge_gap_days))
    warmup_days = 252  # ignore very early regime and feature warmup

    n = len(d)
    start_test_idx = warmup_days + outer_train_size + purge_gap

    if n < start_test_idx + max(10, outer_test_size):
        return {
            "outer": {
                "folds": [],
                "overall": {
                    "log_loss_mean": float("nan"),
                    "log_loss_std": float("nan"),
                    "roc_auc_mean": float("nan"),
                    "roc_auc_std": float("nan"),
                    "accuracy_mean": float("nan"),
                    "accuracy_std": float("nan"),
                    "folds_used": 0,
                },
                "skipped": [{"reason": "dataset too small for nested WFV"}],
            }
        }

    if fast_mode:
        n_estimators_grid = [200]
        min_samples_leaf_grid = [5, 10]
        max_features_grid = ["sqrt"]
        max_depth_grid = [None, 6]
    else:
        n_estimators_grid = [300, 600]
        min_samples_leaf_grid = [3, 5, 10]
        max_features_grid = ["sqrt", 0.5]
        max_depth_grid = [None, 6, 10]

    param_grid = []
    for n_estimators in n_estimators_grid:
        for min_samples_leaf in min_samples_leaf_grid:
            for max_features in max_features_grid:
                for max_depth in max_depth_grid:
                    param_grid.append(
                        {
                            "n_estimators": n_estimators,
                            "min_samples_leaf": min_samples_leaf,
                            "max_features": max_features,
                            "max_depth": max_depth,
                        }
                    )

    folds: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    outer_starts = list(range(start_test_idx, n, outer_test_size))
    for k, test_start in enumerate(tqdm(outer_starts, desc="Nested WF outer folds"), start=1):
        test_end = min(n, test_start + outer_test_size)
        if (test_end - test_start) < 10:
            break

        train_end = test_start - purge_gap
        train_start = max(warmup_days, train_end - outer_train_size)

        outer_train = d.iloc[train_start:train_end]
        outer_test = d.iloc[test_start:test_end]

        if len(outer_train) < inner_min_train_size:
            skipped.append({"fold": k, "reason": "outer train window too small"})
            continue

        y_outer_train = outer_train[target_col].astype(int).values
        if len(np.unique(y_outer_train)) < 2:
            skipped.append({"fold": k, "reason": "outer train has single class"})
            continue

        inner_splits = _make_inner_splits(len(outer_train), inner_folds, inner_min_train_size, purge_gap)
        if not inner_splits:
            skipped.append({"fold": k, "reason": "no valid inner splits"})
            continue

        # Inner tuning
        best_params = None
        best_logloss = float("inf")
        best_auc = float("nan")
        best_folds_used = 0

        total_evals = len(param_grid) * len(inner_splits)
        inner_pbar = tqdm(total=total_evals, desc="Inner CV", leave=False)

        for params in param_grid:
            loglosses = []
            aucs = []
            valid_folds = 0

            for train_end_i, val_start_i, val_end_i in inner_splits:
                inner_train = outer_train.iloc[:train_end_i]
                inner_val = outer_train.iloc[val_start_i:val_end_i]

                y_train = inner_train[target_col].astype(int).values
                if len(np.unique(y_train)) < 2:
                    inner_pbar.update(1)
                    continue

                X_train = inner_train[feature_cols]
                y_val = inner_val[target_col].astype(int).values
                X_val = inner_val[feature_cols]

                if len(X_val) == 0:
                    inner_pbar.update(1)
                    continue
                if len(np.unique(y_val)) < 2:
                    inner_pbar.update(1)
                    continue

                model_params = dict(base_model_params)
                model_params.update(params)
                model_params["class_weight"] = "balanced_subsample"
                model_params["random_state"] = random_state

                model = RandomForestClassifier(**model_params)
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_val)[:, 1]

                ll = log_loss(y_val, proba, labels=[0, 1])
                loglosses.append(float(ll))
                valid_folds += 1

                if len(np.unique(y_val)) > 1:
                    try:
                        aucs.append(float(roc_auc_score(y_val, proba)))
                    except Exception:
                        aucs.append(float("nan"))
                inner_pbar.update(1)

            if valid_folds < 2:
                continue

            mean_ll = float(np.mean(loglosses))
            mean_auc = float(np.nanmean(aucs)) if aucs else float("nan")

            if mean_ll < best_logloss:
                best_logloss = mean_ll
                best_auc = mean_auc
                best_params = params
                best_folds_used = valid_folds

        inner_pbar.close()

        if best_params is None:
            skipped.append({"fold": k, "reason": "no valid inner params"})
            continue

        # Fit final model on outer train with best params
        X_outer_train = outer_train[feature_cols]
        y_outer_train = outer_train[target_col].astype(int).values
        X_outer_test = outer_test[feature_cols]
        y_outer_test = outer_test[target_col].astype(int).values

        final_params = dict(base_model_params)
        final_params.update(best_params)
        final_params["class_weight"] = "balanced_subsample"
        final_params["random_state"] = random_state

        final_model = RandomForestClassifier(**final_params)
        final_model.fit(X_outer_train, y_outer_train)
        proba_test = final_model.predict_proba(X_outer_test)[:, 1]
        y_hat = (proba_test >= 0.5).astype(int)

        ll = float(log_loss(y_outer_test, proba_test, labels=[0, 1]))
        acc = float(accuracy_score(y_outer_test, y_hat))
        if len(np.unique(y_outer_test)) > 1:
            try:
                auc = float(roc_auc_score(y_outer_test, proba_test))
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")

        idx = d.index
        fold_info = {
            "fold": k,
            "train_start": _ts(idx[train_start]),
            "train_end": _ts(idx[train_end - 1]),
            "test_start": _ts(idx[test_start]),
            "test_end": _ts(idx[test_end - 1]),
            "n_train": int(len(outer_train)),
            "n_test": int(len(outer_test)),
            "best_params": {k: _to_py(v) for k, v in best_params.items()},
            "inner_summary": {
                "best_mean_logloss": _to_py(best_logloss),
                "best_mean_roc_auc": _to_py(best_auc),
                "folds_used": int(best_folds_used),
                "total_folds": int(len(inner_splits)),
            },
            "inner_splits": [
                {
                    "train_start": _ts(outer_train.index[0]),
                    "train_end": _ts(outer_train.index[train_end_i - 1]),
                    "val_start": _ts(outer_train.index[val_start_i]),
                    "val_end": _ts(outer_train.index[val_end_i - 1]),
                }
                for (train_end_i, val_start_i, val_end_i) in inner_splits
                if val_start_i < len(outer_train)
            ],
            "metrics": {
                "log_loss": _to_py(ll),
                "roc_auc": _to_py(auc),
                "accuracy": _to_py(acc),
            },
            "proba_len": int(len(proba_test)),
        }
        folds.append(fold_info)

        val_start = fold_info["inner_splits"][0]["val_start"] if fold_info["inner_splits"] else "n/a"
        val_end = fold_info["inner_splits"][-1]["val_end"] if fold_info["inner_splits"] else "n/a"
        tqdm.write(
            f"[WF] Fold {k}: train {_ts(idx[train_start])}->{_ts(idx[train_end - 1])} "
            f"(n={len(outer_train)}), val {val_start}->{val_end}, "
            f"test {_ts(idx[test_start])}->{_ts(idx[test_end - 1])} (n={len(outer_test)}) | "
            f"best={fold_info['best_params']} | logloss={ll:.4f} auc={auc:.4f} acc={acc:.4f}"
        )

    # Overall summary
    logloss_vals = [f["metrics"]["log_loss"] for f in folds if np.isfinite(f["metrics"]["log_loss"])]
    auc_vals = [f["metrics"]["roc_auc"] for f in folds if np.isfinite(f["metrics"]["roc_auc"])]
    acc_vals = [f["metrics"]["accuracy"] for f in folds if np.isfinite(f["metrics"]["accuracy"])]

    overall = {
        "log_loss_mean": _to_py(np.mean(logloss_vals)) if logloss_vals else float("nan"),
        "log_loss_std": _to_py(np.std(logloss_vals, ddof=1)) if len(logloss_vals) > 1 else float("nan"),
        "roc_auc_mean": _to_py(np.mean(auc_vals)) if auc_vals else float("nan"),
        "roc_auc_std": _to_py(np.std(auc_vals, ddof=1)) if len(auc_vals) > 1 else float("nan"),
        "accuracy_mean": _to_py(np.mean(acc_vals)) if acc_vals else float("nan"),
        "accuracy_std": _to_py(np.std(acc_vals, ddof=1)) if len(acc_vals) > 1 else float("nan"),
        "folds_used": int(len(folds)),
        "outer_train_years": int(outer_train_years),
        "outer_test_months": int(outer_test_months),
        "inner_folds": int(inner_folds),
        "inner_min_train_size": int(inner_min_train_size),
        "purge_gap_days": int(purge_gap),
    }

    return {
        "outer": {
            "folds": folds,
            "overall": overall,
            "skipped": skipped,
        }
    }
