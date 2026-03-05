# diagnostics.py
from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd


def alignment_audit(df: pd.DataFrame, required_cols: List[str], strict: bool = True) -> List[str]:
    """
    Checks basic dataset integrity + required columns.

    Returns list of missing required columns.
    If strict=True, raises RuntimeError when missing.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing and strict:
        raise RuntimeError(f"[eval] Missing required columns: {missing}. Rebuild dataset.")
    # index checks
    if not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("[eval] Dataset index must be DatetimeIndex.")
    if df.index.tz is None:
        raise RuntimeError("[eval] Dataset index must be timezone-aware (utc=True).")
    if not df.index.is_monotonic_increasing:
        raise RuntimeError("[eval] Dataset index must be sorted ascending.")

    # Optional leakage sanity warning (doesn't fail)
    if "return_1d" in df.columns and "target_next_log_return" in df.columns:
        corr = df["return_1d"].corr(df["target_next_log_return"])
        if corr is not None and np.isfinite(corr) and abs(corr) > 0.9:
            print(f"[eval] WARNING: corr(return_1d, target_next_log_return)={corr:.3f} (possible leakage?)")

    return missing


def expected_log_return_from_prob(p_up: np.ndarray, mu_up: float, mu_down: float) -> np.ndarray:
    return p_up * mu_up + (1.0 - p_up) * mu_down


def sharpe(logrets: np.ndarray, ann_factor: float = 252.0) -> float:
    if len(logrets) < 2:
        return float("nan")
    m = float(np.mean(logrets))
    s = float(np.std(logrets, ddof=1))
    if s == 0.0:
        return float("nan")
    return float((m / s) * np.sqrt(ann_factor))


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (equity / peak) - 1.0
    return float(np.min(dd))


def turnover(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(positions))))


def strategy_returns(realized_logret_next: np.ndarray, p_up: np.ndarray, threshold: float = 0.5):
    pos = (p_up >= threshold).astype(float)
    strat = pos * realized_logret_next
    return strat, pos


def random_baseline_equity(realized_logret_next: np.ndarray, n_sims: int = 200, p_long: float = 0.5, seed: int = 42):
    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    T = len(realized_logret_next)

    finals = []
    curves = []

    for _ in tqdm(range(n_sims), desc="[eval] Random baseline sims"):
        pos = (rng.random(T) < p_long).astype(float)
        r = pos * realized_logret_next
        eq = np.exp(np.cumsum(r))
        finals.append(eq[-1] if len(eq) else 1.0)

    for _ in tqdm(range(n_sims), desc="[eval] Random equity curves"):
        pos = (rng.random(T) < p_long).astype(float)
        r = pos * realized_logret_next
        eq = np.exp(np.cumsum(r))
        curves.append(eq)

    return {"finals": np.array(finals, dtype=float), "curves": curves}


def leakage_audit(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    *,
    close_cols: Optional[List[str]] = None,
    max_lag: int = 5,
    corr_threshold: float = 0.20,
    identity_corr_threshold: float = 0.999,
    run_smoke_model: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Lightweight anti-leakage checks for time-series features.
    Returns a JSON-serializable dict and prints a concise report.
    """
    d = df.copy()
    if target_col not in d.columns:
        raise ValueError(f"[leakage] target_col '{target_col}' not found in df")

    y = d[target_col].astype(float)
    if close_cols is None:
        close_cols = ["close_t", "close", "adj_close", "open", "high", "low", "volume", "target_next_log_return", "target_up"]

    # Name-based flags
    name_flags = []
    for f in feature_cols:
        name = f.lower()
        if any(tok in name for tok in ["t1", "future", "lead", "target"]):
            name_flags.append(f)

    # Correlation lag scan
    suspicious_corrs = []
    for f in feature_cols:
        if f not in d.columns:
            continue
        s = d[f].astype(float)
        corr_by_lag: Dict[int, float] = {}
        best_lag = 0
        best_corr = 0.0
        for k in range(-max_lag, max_lag + 1):
            ys = y.shift(-k)
            m = s.notna() & ys.notna() & np.isfinite(s) & np.isfinite(ys)
            if m.sum() < 50:
                corr = np.nan
            else:
                corr = s[m].corr(ys[m])
            corr_by_lag[k] = float(corr) if corr is not None and np.isfinite(corr) else float("nan")
            if np.isfinite(corr_by_lag[k]) and abs(corr_by_lag[k]) > abs(best_corr):
                best_corr = corr_by_lag[k]
                best_lag = k
        if best_lag > 0 and np.isfinite(best_corr) and abs(best_corr) >= corr_threshold:
            suspicious_corrs.append(
                {
                    "feature": f,
                    "best_lag": int(best_lag),
                    "best_corr": float(best_corr),
                    "corr_by_lag": {str(k): float(v) for k, v in corr_by_lag.items()},
                }
            )

    # Identity / shift checks vs basic columns
    identity_matches = []
    for base in close_cols:
        if base not in d.columns:
            continue
        base_series = d[base].astype(float)
        for shift in [-1, 0]:
            shifted = base_series.shift(shift)
            for f in feature_cols:
                if f not in d.columns:
                    continue
                s = d[f].astype(float)
                m = s.notna() & shifted.notna() & np.isfinite(s) & np.isfinite(shifted)
                if m.sum() < 50:
                    continue
                corr = s[m].corr(shifted[m])
                if corr is not None and np.isfinite(corr) and abs(corr) >= identity_corr_threshold:
                    identity_matches.append(
                        {
                            "feature": f,
                            "matched_series": f"{base}.shift({shift})",
                            "corr": float(corr),
                        }
                    )

    smoke_test = None
    if run_smoke_model:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score

        sample = d.tail(2000).copy() if len(d) > 2000 else d.copy()
        sample = sample[feature_cols + [target_col]].dropna()
        if len(sample) >= 200:
            X = sample[feature_cols]
            y_bin = sample[target_col].astype(int).values
            rng = np.random.default_rng(random_state)
            y_shuffled = rng.permutation(y_bin)
            split = int(len(sample) * 0.8)
            X_train = X.iloc[:split]
            y_train = y_shuffled[:split]
            X_test = X.iloc[split:]
            y_test = y_shuffled[split:]
            if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=3,
                    random_state=random_state,
                    n_jobs=-1,
                )
                model.fit(X_train, y_train)
                proba = model.predict_proba(X_test)[:, 1]
                y_hat = (proba >= 0.5).astype(int)
                acc = float(accuracy_score(y_test, y_hat))
                try:
                    auc = float(roc_auc_score(y_test, proba))
                except Exception:
                    auc = float("nan")
                smoke_test = {"accuracy": acc, "roc_auc": auc, "n": int(len(sample))}
            else:
                smoke_test = {"accuracy": float("nan"), "roc_auc": float("nan"), "n": int(len(sample))}
        else:
            smoke_test = {"accuracy": float("nan"), "roc_auc": float("nan"), "n": int(len(sample))}

    passed = True
    if suspicious_corrs or identity_matches or name_flags:
        passed = False
    if smoke_test and (smoke_test.get("accuracy", 0) > 0.55 or smoke_test.get("roc_auc", 0) > 0.55):
        passed = False

    # Print report
    print("[leakage] Audit summary:")
    print(f"  suspicious_corrs={len(suspicious_corrs)} | identity_matches={len(identity_matches)} | name_flags={len(name_flags)}")
    if suspicious_corrs:
        print("  Top suspicious (feature, lag, corr):")
        for s in suspicious_corrs[:10]:
            print(f"    {s['feature']} | lag={s['best_lag']} | corr={s['best_corr']:.3f}")
    if identity_matches:
        print("  Identity-like matches:")
        for m in identity_matches[:10]:
            print(f"    {m['feature']} ~ {m['matched_series']} | corr={m['corr']:.6f}")
    if name_flags:
        print(f"  Name flags: {name_flags}")
    if smoke_test:
        print(
            f"  Smoke test (shuffled target): acc={smoke_test['accuracy']:.3f} "
            f"auc={smoke_test['roc_auc']:.3f} n={smoke_test['n']}"
        )
    if passed:
        print("[leakage] PASSED: no strong positive-lag correlations, no identity matches, smoke test near random.")
    else:
        print("[leakage] FAILED: potential leakage signals detected.")

    return {
        "suspicious_corrs": suspicious_corrs,
        "identity_matches": identity_matches,
        "name_flags": name_flags,
        "smoke_test": smoke_test,
        "passed": bool(passed),
    }
