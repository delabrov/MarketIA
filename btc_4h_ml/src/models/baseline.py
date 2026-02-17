from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def make_logistic_baseline(random_state: int = 42) -> Pipeline:
    """
    A strong and clean baseline for tabular ML:
    StandardScaler + LogisticRegression (L2).
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    n_jobs=None,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return model
