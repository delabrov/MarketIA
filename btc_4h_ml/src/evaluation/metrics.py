from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss


@dataclass(frozen=True)
class ClassificationReport:
    logloss: float
    roc_auc: float
    brier: float
    accuracy_at_0_5: float
    positive_rate: float


def classification_report(y_true, y_proba) -> ClassificationReport:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    # safety clip for logloss
    eps = 1e-12
    y_proba = np.clip(y_proba, eps, 1 - eps)

    ll = float(log_loss(y_true, y_proba))
    # roc_auc requires both classes present; handle edge case
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")
    brier = float(brier_score_loss(y_true, y_proba))

    y_pred = (y_proba >= 0.5).astype(int)
    acc = float((y_pred == y_true).mean())

    pos_rate = float(y_true.mean())

    return ClassificationReport(
        logloss=ll,
        roc_auc=auc,
        brier=brier,
        accuracy_at_0_5=acc,
        positive_rate=pos_rate,
    )
