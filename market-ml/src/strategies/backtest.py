from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .indicators import build_indicator_signals
from .strategies import (
    apply_volatility_risk_overlay,
    core_satellite_longonly_strategy,
    hybrid_vol_alpha_strategy,
    longonly_threshold_strategy,
    regime_overlay_longonly_strategy,
    regime_switching_strategy,
    rolling_quantile_strategy,
    vol_target_strategy,
    zscore_threshold_strategy,
    zscore_threshold_indicators_strategy,
)


# In LSTM preds CSV, y_true is already the forward return for the modeled horizon.
TARGET_IS_FORWARD_RETURN = True


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    raise KeyError(f"Missing expected column. Tried: {candidates}")


def load_preds(path: str | Path) -> pd.DataFrame:
    """
    Load predictions CSV into a dataframe indexed by datetime.
    Required normalized columns in output: y_true, y_pred.
    Optional: close.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preds file not found: {p}")
    df = pd.read_csv(p)
    date_col = _find_column(df, ["date", "Date", "datetime", "timestamp"])
    y_true_col = _find_column(df, ["y_true", "target_h1", "target", "ret", "return"])
    y_pred_col = _find_column(df, ["y_pred", "pred", "prediction", "y_hat"])

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col)

    out = pd.DataFrame(index=df.index)
    out["y_true"] = pd.to_numeric(df[y_true_col], errors="coerce")
    out["y_pred"] = pd.to_numeric(df[y_pred_col], errors="coerce")
    if "split" in df.columns:
        out["split"] = df["split"].astype(str)

    close_candidates = ["close", "adj_close", "close_t", "price", "aapl_close"]
    for c in close_candidates:
        if c in df.columns:
            out["close"] = pd.to_numeric(df[c], errors="coerce")
            break
    return out


def load_close_from_raw(
    index: pd.DatetimeIndex,
    ticker: str = "AAPL",
    project_root: Path | None = None,
) -> pd.Series:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    p_parquet = project_root / "data" / "raw" / f"{ticker.lower()}_ohlcv.parquet"
    p_csv = project_root / "data" / "raw" / f"{ticker.lower()}_ohlcv.csv"
    raw = None
    if p_csv.exists():
        try:
            raw = pd.read_csv(p_csv, index_col=0, parse_dates=True)
        except Exception:
            raw = None
    if raw is None and p_parquet.exists():
        try:
            raw = pd.read_parquet(p_parquet)
        except Exception:
            raw = None
    if raw is None:
        return pd.Series(index=index, dtype=float, name="close")
    if not isinstance(raw.index, pd.DatetimeIndex):
        return pd.Series(index=index, dtype=float, name="close")
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")
    raw = raw.sort_index()
    col = "close" if "close" in raw.columns else ("adj_close" if "adj_close" in raw.columns else None)
    if col is None:
        return pd.Series(index=index, dtype=float, name="close")
    s = raw[col].astype(float).rename("close")
    return s.reindex(index)


def load_vix_zscore_60_from_raw(index: pd.DatetimeIndex, project_root: Path | None = None) -> pd.Series:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    p_parquet = project_root / "data" / "raw" / "vix_ohlcv.parquet"
    p_csv = project_root / "data" / "raw" / "vix_ohlcv.csv"
    raw = None
    if p_csv.exists():
        try:
            raw = pd.read_csv(p_csv, index_col=0, parse_dates=True)
        except Exception:
            raw = None
    if raw is None and p_parquet.exists():
        try:
            raw = pd.read_parquet(p_parquet)
        except Exception:
            raw = None
    if raw is None or "close" not in raw.columns:
        return pd.Series(index=index, dtype=float, name="vix_zscore_60")
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    else:
        raw.index = raw.index.tz_convert("UTC")
    vix = raw["close"].astype(float).sort_index()
    mu = vix.rolling(window=60, min_periods=60).mean()
    sd = vix.rolling(window=60, min_periods=60).std()
    z = (vix - mu) / (sd + 1e-12)
    return z.rename("vix_zscore_60").reindex(index)


def compute_positions(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, pd.Series]:
    pred = df["y_pred"].astype(float)
    r = df["y_true"].astype(float)
    allow_short = bool(config.get("allow_short", True))
    common = config.get("common", {})
    z_window = int(common.get("z_window", 252))
    alpha_scale = float(common.get("alpha_scale", 1.0))
    smooth_lambda = float(common.get("smooth_lambda", 0.1))
    deadband = float(common.get("deadband", 0.05))
    realized_vol_window = int(common.get("realized_vol_window", 20))
    vol_target_ann = float(common.get("vol_target_ann", 0.10))
    lev_max = float(common.get("lev_max", 2.0))

    realized_vol = r.rolling(window=realized_vol_window, min_periods=realized_vol_window).std()
    realized_vol_long = r.rolling(window=126, min_periods=126).std()

    a_cfg = config.get("zscore_threshold", {})
    b_cfg = config.get("rolling_quantile", {})
    c_cfg = config.get("vol_target", {})
    zi_cfg = config.get("zscore_threshold_indicators", {})
    h_cfg = config.get("hybrid_vol_alpha", {})
    hl_cfg = config.get("hybrid_vol_alpha_longonly", {})
    rg_cfg = config.get("regime_switching", {})
    csl_cfg = config.get("core_satellite_longonly", {})
    lot_cfg = config.get("longonly_threshold", {})
    rol_cfg = config.get("regime_overlay_longonly", {})

    pos_a = zscore_threshold_strategy(
        pred=pred,
        window=int(a_cfg.get("window", z_window)),
        entry_z=float(a_cfg.get("entry_z", 1.0)),
        exit_z=float(a_cfg.get("exit_z", 0.3)),
        allow_short=allow_short,
        smooth_lambda=float(a_cfg.get("smooth_lambda", smooth_lambda)),
        deadband=float(a_cfg.get("deadband", deadband)),
    )
    pos_b = rolling_quantile_strategy(
        pred=pred,
        window=int(b_cfg.get("window", z_window)),
        q_hi=float(b_cfg.get("q_hi", 0.8)),
        q_lo=float(b_cfg.get("q_lo", 0.2)),
        allow_short=allow_short,
        smooth_lambda=float(b_cfg.get("smooth_lambda", smooth_lambda)),
        deadband=float(b_cfg.get("deadband", deadband)),
    )
    pos_c = vol_target_strategy(
        pred=pred,
        realized_vol=realized_vol,
        pred_window=int(c_cfg.get("pred_window", 60)),
        vol_target_ann=float(c_cfg.get("vol_target_ann", vol_target_ann)),
        clip_signal=float(c_cfg.get("clip_signal", 2.0)),
        lev_max=float(c_cfg.get("lev_max", lev_max)),
        smooth_lambda=float(c_cfg.get("smooth_lambda", smooth_lambda)),
        deadband=float(c_cfg.get("deadband", deadband)),
    )

    if "coeff_total" in df.columns:
        coeff_total = pd.to_numeric(df["coeff_total"], errors="coerce").fillna(0.0)
    elif "close" in df.columns:
        coeff_total = build_indicator_signals(pd.to_numeric(df["close"], errors="coerce")).reindex(df.index)["coeff_total"].fillna(0.0)
    else:
        coeff_total = pd.Series(0.0, index=df.index)

    pos_zi = zscore_threshold_indicators_strategy(
        pos_ml=pos_a,
        coeff_total=coeff_total,
        confirm_threshold=float(zi_cfg.get("confirm_threshold", 2.0 / 3.0)),
    )

    pos_h = hybrid_vol_alpha_strategy(
        pred=pred,
        realized_vol=realized_vol,
        z_window=int(h_cfg.get("z_window", z_window)),
        alpha_scale=float(h_cfg.get("alpha_scale", alpha_scale)),
        vol_target_ann=float(h_cfg.get("vol_target_ann", vol_target_ann)),
        lev_max=float(h_cfg.get("lev_max", lev_max)),
        smooth_lambda=float(h_cfg.get("smooth_lambda", smooth_lambda)),
        deadband=float(h_cfg.get("deadband", deadband)),
        long_only=False,
    )
    pos_hl = hybrid_vol_alpha_strategy(
        pred=pred,
        realized_vol=realized_vol,
        z_window=int(hl_cfg.get("z_window", z_window)),
        alpha_scale=float(hl_cfg.get("alpha_scale", alpha_scale)),
        vol_target_ann=float(hl_cfg.get("vol_target_ann", vol_target_ann)),
        lev_max=float(hl_cfg.get("lev_max", lev_max)),
        smooth_lambda=float(hl_cfg.get("smooth_lambda", smooth_lambda)),
        deadband=float(hl_cfg.get("deadband", deadband)),
        long_only=True,
    )

    pos_core = core_satellite_longonly_strategy(
        pred=pred,
        z_window=int(csl_cfg.get("z_window", z_window)),
        base_exposure=float(csl_cfg.get("base_exposure", 1.0)),
        k=float(csl_cfg.get("k", 0.5)),
        min_exposure=float(csl_cfg.get("min_exposure", 0.5)),
        max_exposure=float(csl_cfg.get("max_exposure", 1.5)),
    )
    pos_lot = longonly_threshold_strategy(
        pred=pred,
        z_window=int(lot_cfg.get("z_window", z_window)),
        threshold=float(lot_cfg.get("threshold", 1.0)),
        high_exposure=float(lot_cfg.get("high_exposure", 1.5)),
        neutral_exposure=float(lot_cfg.get("neutral_exposure", 1.0)),
        low_exposure=float(lot_cfg.get("low_exposure", 0.5)),
    )

    regime_feature_name = str(rg_cfg.get("regime_feature_name", "vix_zscore_60"))
    if regime_feature_name in df.columns:
        regime_signal = pd.to_numeric(df[regime_feature_name], errors="coerce")
    elif regime_feature_name == "vix_zscore_60":
        # Fallback: z-score of long realized volatility from returns.
        rv_mu = realized_vol_long.rolling(window=252, min_periods=252).mean()
        rv_sd = realized_vol_long.rolling(window=252, min_periods=252).std()
        regime_signal = (realized_vol_long - rv_mu) / (rv_sd + 1e-12)
    else:
        rv_mu = realized_vol_long.rolling(window=252, min_periods=252).mean()
        rv_sd = realized_vol_long.rolling(window=252, min_periods=252).std()
        regime_signal = (realized_vol_long - rv_mu) / (rv_sd + 1e-12)

    if regime_signal.isna().all() and "close" in df.columns:
        # Secondary fallback: simple trend regime from EMA20/EMA50.
        ind = build_indicator_signals(pd.to_numeric(df["close"], errors="coerce")).reindex(df.index)
        regime_signal = np.where(ind["ema20"] > ind["ema50"], 0.0, 2.0)
        regime_signal = pd.Series(regime_signal, index=df.index, dtype=float)

    stress_name = str(rg_cfg.get("stress_strategy_name", "vol_target"))
    normal_name = str(rg_cfg.get("normal_strategy_name", "hybrid_vol_alpha"))
    pos_map = {
        "vol_target": pos_c,
        "zscore_threshold": pos_a,
        "zscore_threshold_indicators": pos_zi,
        "rolling_quantile": pos_b,
        "hybrid_vol_alpha": pos_h,
        "hybrid_vol_alpha_longonly": pos_hl,
    }
    stress_pos = pos_map.get(stress_name, pos_c)
    normal_pos = pos_map.get(normal_name, pos_h)
    pos_regime = regime_switching_strategy(
        regime_signal=regime_signal,
        stress_position=stress_pos,
        normal_position=normal_pos,
        regime_threshold=float(rg_cfg.get("regime_threshold", 1.0)),
        smooth_lambda=float(rg_cfg.get("smooth_lambda", 0.0)),
        deadband=float(rg_cfg.get("deadband", 0.0)),
    )

    pos_rol = regime_overlay_longonly_strategy(
        candidate_pos=pos_lot,
        regime_signal=regime_signal,
        regime_threshold=float(rol_cfg.get("regime_threshold", 1.0)),
        stress_multiplier=float(rol_cfg.get("stress_multiplier", 0.5)),
        min_exposure=float(rol_cfg.get("min_exposure", 0.25)),
        max_exposure=float(rol_cfg.get("max_exposure", 1.5)),
        smooth_lambda=float(rol_cfg.get("smooth_lambda", 0.0)),
        deadband=float(rol_cfg.get("deadband", 0.0)),
    )

    # Step-3 experiment: fixed base [0.5, 1.5], no smoothing, volatility risk overlay.
    pos_rol_base = regime_overlay_longonly_strategy(
        candidate_pos=pos_lot,
        regime_signal=regime_signal,
        regime_threshold=float(rol_cfg.get("regime_threshold", 1.0)),
        stress_multiplier=float(rol_cfg.get("stress_multiplier", 0.5)),
        min_exposure=0.50,
        max_exposure=1.50,
        smooth_lambda=0.0,
        deadband=0.0,
    ).fillna(0.0)

    if "close" in df.columns and np.isfinite(pd.to_numeric(df["close"], errors="coerce")).any():
        close = pd.to_numeric(df["close"], errors="coerce")
        hist_ret = np.log(close / close.shift(1))
    else:
        # Fallback remains causal with respect to target-aligned series.
        hist_ret = r.shift(1)
    hist_ret = pd.Series(hist_ret, index=df.index).astype(float)

    # Best fixed reference from step-3: w20 / vt15 / rf03.
    pos_rol_vol_ref, risk_ref = apply_volatility_risk_overlay(
        base_position=pos_rol_base,
        realized_returns=hist_ret,
        vol_window=20,
        vol_target_ann=0.15,
        risk_floor=0.30,
        risk_cap=1.00,
        max_exposure_final=1.5,
    )
    pos_rol_vol_ref = pos_rol_vol_ref.fillna(0.0)
    risk_ref = risk_ref.fillna(1.0)

    realized_vol = hist_ret.rolling(window=20, min_periods=20).std()
    vol_threshold = float(realized_vol.dropna().median()) if realized_vol.notna().any() else float("inf")
    high_vol = realized_vol > vol_threshold

    vol_thr_p60 = realized_vol.rolling(window=252, min_periods=252).quantile(0.60)
    vol_thr_p70 = realized_vol.rolling(window=252, min_periods=252).quantile(0.70)
    vol_thr_p80 = realized_vol.rolling(window=252, min_periods=252).quantile(0.80)
    high_vol_p60 = realized_vol > vol_thr_p60
    high_vol_p70 = realized_vol > vol_thr_p70
    high_vol_p80 = realized_vol > vol_thr_p80

    def _apply_conditional(mask: pd.Series) -> pd.Series:
        mask = pd.Series(mask, index=pos_rol_base.index).fillna(False).astype(bool)
        out = pd.Series(np.where(mask, pos_rol_base * risk_ref, pos_rol_base), index=pos_rol_base.index)
        out = out.clip(lower=0.0, upper=1.5)
        return out.where(np.isfinite(out), 0.0).fillna(0.0)

    pos_rol_cond_highvol = _apply_conditional(high_vol)
    pos_rol_cond_p60 = _apply_conditional(high_vol_p60)
    pos_rol_cond_p70 = _apply_conditional(high_vol_p70)
    pos_rol_cond_p80 = _apply_conditional(high_vol_p80)

    return {
        "zscore_threshold": pos_a.fillna(0.0),
        "zscore_threshold_indicators": pos_zi.fillna(0.0),
        "rolling_quantile": pos_b.fillna(0.0),
        "vol_target": pos_c.fillna(0.0),
        "hybrid_vol_alpha": pos_h.fillna(0.0),
        "hybrid_vol_alpha_longonly": pos_hl.fillna(0.0),
        "regime_switching": pos_regime.fillna(0.0),
        "core_satellite_longonly": pos_core.fillna(0.0),
        "longonly_threshold": pos_lot.fillna(0.0),
        "regime_overlay_longonly": pos_rol.fillna(0.0),
        "regime_overlay_longonly_base": pos_rol_base,
        "regime_overlay_volscale_w20_vt15_rf03": pos_rol_vol_ref,
        "regime_overlay_condvol_highvol": pos_rol_cond_highvol,
        "regime_overlay_condvol_p60": pos_rol_cond_p60,
        "regime_overlay_condvol_p70": pos_rol_cond_p70,
        "regime_overlay_condvol_p80": pos_rol_cond_p80,
        "buyhold": pd.Series(1.0, index=df.index, name="buyhold"),
    }


def compute_returns(df: pd.DataFrame, positions: Dict[str, pd.Series], fee_bps: float = 0.0) -> pd.DataFrame:
    r = df["y_true"].astype(float).fillna(0.0)
    out = pd.DataFrame(index=df.index)
    fee = float(fee_bps) / 1e4

    for name, p in positions.items():
        p = p.reindex(df.index).fillna(0.0).astype(float)
        if TARGET_IS_FORWARD_RETURN:
            gross = p * r
        else:
            gross = p.shift(1).fillna(0.0) * r
        turnover_abs = p.diff().abs().fillna(p.abs())
        cost = fee * turnover_abs
        if name == "buyhold":
            out[name] = gross
        else:
            out[name] = gross - cost
    return out


def compute_equity(returns_df: pd.DataFrame) -> pd.DataFrame:
    equity = np.exp(np.log1p(returns_df).cumsum())
    equity = equity.replace([np.inf, -np.inf], np.nan)
    return equity


def _max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def _average_holding_period(position: pd.Series, eps: float = 1e-8) -> float:
    active = position.abs() > eps
    if not active.any():
        return float("nan")
    grp = (active != active.shift(1)).cumsum()
    lengths = active.groupby(grp).sum()
    lengths = lengths[lengths > 0]
    if len(lengths) == 0:
        return float("nan")
    return float(lengths.mean())


def compute_metrics(returns_df: pd.DataFrame, positions_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ann_factor = 252.0
    for col in returns_df.columns:
        r = returns_df[col].astype(float).dropna()
        if len(r) == 0:
            continue
        eq = np.exp(np.log1p(r).cumsum())
        years = max(len(r) / ann_factor, 1e-12)
        cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)
        vol = float(r.std(ddof=1) * np.sqrt(ann_factor)) if len(r) > 1 else float("nan")
        sharpe = float((r.mean() / r.std(ddof=1)) * np.sqrt(ann_factor)) if len(r) > 1 and r.std(ddof=1) > 0 else float("nan")
        mdd = _max_drawdown(eq)

        p = positions_df[col].reindex(r.index).fillna(0.0).astype(float)
        delta = p.diff().abs().fillna(p.abs())
        turnover = float(delta.mean())
        n_trades = int((delta > 1e-8).sum())
        in_mkt = p.abs() > 0
        if in_mkt.any():
            hit = float((r[in_mkt] > 0).mean())
        else:
            hit = float("nan")
        avg_hold = _average_holding_period(p)
        calmar = float(cagr / abs(mdd)) if np.isfinite(mdd) and mdd < 0 else float("nan")

        rows.append(
            {
                "strategy": col,
                "n_days": int(len(r)),
                "cagr": cagr,
                "annualized_vol": vol,
                "ann_vol": vol,
                "sharpe": sharpe,
                "max_drawdown": mdd,
                "calmar_ratio": calmar,
                "turnover": turnover,
                "n_trades": n_trades,
                "hit_ratio": hit,
                "average_holding_period": avg_hold,
            }
        )
    out = pd.DataFrame(rows).set_index("strategy")
    return out.sort_index()
