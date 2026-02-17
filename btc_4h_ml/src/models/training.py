from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import pandas as pd

from src.evaluation.metrics import classification_report, ClassificationReport


@dataclass
class FoldResult:
    fold_id: int
    train_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_range: Tuple[pd.Timestamp, pd.Timestamp]
    test_range: Tuple[pd.Timestamp, pd.Timestamp]
    val_report: ClassificationReport
    test_report: ClassificationReport


def fit_predict_proba(model, X_train, y_train, X_eval) -> pd.Series:
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_eval)[:, 1]
    return pd.Series(proba, index=X_eval.index)


def run_walk_forward_classification(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    splits,
    model,
) -> List[FoldResult]:
    """
    Train model per fold, evaluate on val and test.
    """
    results: List[FoldResult] = []

    for fold_id, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        train_df = df.loc[train_idx]
        val_df = df.loc[val_idx]
        test_df = df.loc[test_idx]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col].astype(int)

        X_val = val_df[feature_cols]
        y_val = val_df[target_col].astype(int)

        X_test = test_df[feature_cols]
        y_test = test_df[target_col].astype(int)

        val_proba = fit_predict_proba(model, X_train, y_train, X_val)
        test_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        val_rep = classification_report(y_val, val_proba)
        test_rep = classification_report(y_test, test_proba)

        res = FoldResult(
            fold_id=fold_id,
            train_range=(train_idx.min().to_pydatetime(), train_idx.max().to_pydatetime()),
            val_range=(val_idx.min().to_pydatetime(), val_idx.max().to_pydatetime()),
            test_range=(test_idx.min().to_pydatetime(), test_idx.max().to_pydatetime()),
            val_report=val_rep,
            test_report=test_rep,
        )
        results.append(res)

    return results
