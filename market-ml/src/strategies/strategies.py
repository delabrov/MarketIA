from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_zscore_signal(pred: pd.Series, window: int = 252) -> pd.Series:
    pred = pd.Series(pred).astype(float)
    mu = pred.rolling(window=window, min_periods=window).mean()
    sigma = pred.rolling(window=window, min_periods=window).std()
    z = (pred - mu) / (sigma + 1e-12)
    return pd.Series(z, index=pred.index, name="z_signal")


def apply_position_postprocess(
    pos_raw: pd.Series,
    smooth_lambda: float = 0.0,
    deadband: float = 0.0,
) -> pd.Series:
    """
    Apply optional exponential smoothing and deadband on positions.
    """
    s = pd.Series(pos_raw).astype(float).fillna(0.0)
    lam = float(np.clip(smooth_lambda, 0.0, 1.0))
    db = float(max(deadband, 0.0))

    out = np.zeros(len(s), dtype=float)
    prev = 0.0
    for i, raw in enumerate(s.to_numpy()):
        if not np.isfinite(raw):
            raw = 0.0
        if i == 0 or lam <= 0.0:
            curr = raw
        else:
            curr = lam * raw + (1.0 - lam) * prev
        if abs(curr - prev) < db:
            curr = prev
        out[i] = curr
        prev = curr
    return pd.Series(out, index=s.index, name=s.name)


def alpha_engine(
    pred: pd.Series,
    z_window: int = 252,
    alpha_scale: float = 1.0,
    smooth_lambda: float = 0.1,
    deadband: float = 0.05,
    long_only: bool = False,
) -> pd.Series:
    """
    Robust directional alpha:
    1) rolling z-score of prediction
    2) bounded exposure via tanh(alpha_scale * z)
    3) optional smoothing/deadband
    """
    pred = pd.Series(pred).astype(float)
    mu = pred.rolling(window=z_window, min_periods=z_window).mean()
    sigma = pred.rolling(window=z_window, min_periods=z_window).std()
    z = (pred - mu) / (sigma + 1e-12)
    alpha = np.tanh(float(alpha_scale) * z)
    alpha = pd.Series(alpha, index=pred.index, name="alpha")
    if long_only:
        alpha = alpha.clip(lower=0.0)
    alpha = alpha.where(np.isfinite(alpha), 0.0).fillna(0.0)
    return apply_position_postprocess(alpha, smooth_lambda=smooth_lambda, deadband=deadband)


def zscore_threshold_strategy(
    pred: pd.Series,
    window: int = 252,
    entry_z: float = 1.0,
    exit_z: float = 0.3,
    allow_short: bool = True,
    smooth_lambda: float = 0.0,
    deadband: float = 0.0,
) -> pd.Series:
    """
    Stateful sparse strategy based on rolling z-score of predictions.

    Rules:
    - Flat -> Long if z > entry_z
    - Flat -> Short if z < -entry_z (if allow_short)
    - Long -> Flat if z < exit_z
    - Short -> Flat if z > -exit_z
    """
    pred = pd.Series(pred).astype(float)
    z = rolling_zscore_signal(pred, window=window)

    pos = np.zeros(len(pred), dtype=float)
    current = 0.0
    for i, zi in enumerate(z.to_numpy()):
        if not np.isfinite(zi):
            pos[i] = 0.0
            current = 0.0
            continue
        if current == 0.0:
            if zi > entry_z:
                current = 1.0
            elif allow_short and zi < -entry_z:
                current = -1.0
        elif current > 0.0:
            if zi < exit_z:
                current = 0.0
        else:
            if zi > -exit_z:
                current = 0.0
        pos[i] = current
    out = pd.Series(pos, index=pred.index, name="zscore_threshold")
    return apply_position_postprocess(out, smooth_lambda=smooth_lambda, deadband=deadband)


def core_satellite_longonly_strategy(
    pred: pd.Series,
    z_window: int = 252,
    base_exposure: float = 1.0,
    k: float = 0.5,
    min_exposure: float = 0.5,
    max_exposure: float = 1.5,
) -> pd.Series:
    """
    Long-only dynamic exposure around a core allocation:
    pos_t = clip(base_exposure + k * tanh(z_t), min_exposure, max_exposure)
    """
    z = rolling_zscore_signal(pred, window=z_window).fillna(0.0)
    pos = float(base_exposure) + float(k) * np.tanh(z)
    pos = pd.Series(pos, index=z.index).clip(lower=max(0.0, float(min_exposure)), upper=float(max_exposure))
    pos = pos.where(np.isfinite(pos), 0.0).fillna(0.0)
    return pd.Series(pos, index=z.index, name="core_satellite_longonly")


def longonly_threshold_strategy(
    pred: pd.Series,
    z_window: int = 252,
    threshold: float = 1.0,
    high_exposure: float = 1.5,
    neutral_exposure: float = 1.0,
    low_exposure: float = 0.5,
) -> pd.Series:
    """
    Long-only discrete 3-level exposure from z-score thresholding.
    """
    z = rolling_zscore_signal(pred, window=z_window)
    hi = float(high_exposure)
    ne = float(neutral_exposure)
    lo = float(low_exposure)
    pos = np.where(z > float(threshold), hi, np.where(z < -float(threshold), lo, ne))
    pos = pd.Series(pos, index=z.index).clip(lower=0.0)
    pos = pos.where(np.isfinite(z), 0.0).fillna(0.0)
    return pd.Series(pos, index=z.index, name="longonly_threshold")


def regime_overlay_longonly_strategy(
    candidate_pos: pd.Series,
    regime_signal: pd.Series,
    regime_threshold: float = 1.0,
    stress_multiplier: float = 0.5,
    min_exposure: float = 0.25,
    max_exposure: float = 1.5,
    smooth_lambda: float = 0.0,
    deadband: float = 0.0,
) -> pd.Series:
    """
    Apply a stress overlay to a long-only candidate exposure.
    If regime_signal > threshold, reduce exposure with stress_multiplier.
    """
    candidate = pd.Series(candidate_pos).astype(float).fillna(0.0)
    regime = pd.Series(regime_signal, index=candidate.index).astype(float)
    mult = np.where(regime > float(regime_threshold), float(stress_multiplier), 1.0)
    pos = candidate * pd.Series(mult, index=candidate.index)
    lower = max(0.0, float(min_exposure))
    upper = float(max_exposure)
    pos = pd.Series(pos, index=candidate.index).clip(lower=lower, upper=upper)
    pos = apply_position_postprocess(pos, smooth_lambda=smooth_lambda, deadband=deadband)
    # Keep hard exposure bounds after smoothing.
    pos = pd.Series(pos, index=candidate.index).clip(lower=lower, upper=upper)
    pos = pos.where(np.isfinite(pos), 0.0).fillna(0.0)
    return pd.Series(pos, index=candidate.index, name="regime_overlay_longonly")


def apply_volatility_risk_overlay(
    base_position: pd.Series,
    realized_returns: pd.Series,
    vol_window: int = 20,
    vol_target_ann: float = 0.10,
    risk_floor: float = 0.30,
    risk_cap: float = 1.00,
    max_exposure_final: float = 1.5,
) -> tuple[pd.Series, pd.Series]:
    """
    Apply a causal volatility risk overlay on top of a base long-only exposure.

    pos_final_t = base_pos_t * risk_t
    risk_t      = clip(vol_target_ann / (realized_vol_t * sqrt(252)), risk_floor, risk_cap)
    """
    base = pd.Series(base_position).astype(float).fillna(0.0)
    ret = pd.Series(realized_returns, index=base.index).astype(float)
    rv = ret.rolling(window=int(vol_window), min_periods=int(vol_window)).std()
    risk = float(vol_target_ann) / (rv * np.sqrt(252.0) + 1e-12)
    risk = pd.Series(risk, index=base.index).clip(lower=float(risk_floor), upper=float(risk_cap))
    risk = risk.where(np.isfinite(risk), 1.0).fillna(1.0)

    pos = base * risk
    pos = pd.Series(pos, index=base.index).clip(lower=0.0, upper=float(max_exposure_final))
    pos = pos.where(np.isfinite(pos), 0.0).fillna(0.0)
    return pd.Series(pos, index=base.index), pd.Series(risk, index=base.index)


def rolling_quantile_strategy(
    pred: pd.Series,
    window: int = 252,
    q_hi: float = 0.8,
    q_lo: float = 0.2,
    allow_short: bool = True,
    smooth_lambda: float = 0.0,
    deadband: float = 0.0,
) -> pd.Series:
    """
    Long/short strategy using rolling quantile thresholds of predictions.
    """
    pred = pd.Series(pred).astype(float)
    hi = pred.rolling(window=window, min_periods=window).quantile(q_hi)
    lo = pred.rolling(window=window, min_periods=window).quantile(q_lo)

    long_sig = (pred > hi).astype(float)
    if allow_short:
        short_sig = (pred < lo).astype(float) * -1.0
    else:
        short_sig = pd.Series(0.0, index=pred.index)

    pos = long_sig + short_sig
    pos = pos.where(np.isfinite(pos), 0.0).fillna(0.0)
    out = pd.Series(pos, index=pred.index, name="rolling_quantile")
    return apply_position_postprocess(out, smooth_lambda=smooth_lambda, deadband=deadband)


def vol_target_strategy(
    pred: pd.Series,
    realized_vol: pd.Series,
    pred_window: int = 60,
    vol_target_ann: float = 0.10,
    clip_signal: float = 2.0,
    lev_max: float = 2.0,
    smooth_lambda: float = 0.0,
    deadband: float = 0.0,
) -> pd.Series:
    """
    Continuous position sizing with volatility targeting.

    position = tanh(signal) * leverage
    signal   = clip(pred / rolling_std(pred), -clip_signal, clip_signal)
    leverage = clip(vol_target_ann / (realized_vol * sqrt(252)), 0, lev_max)
    """
    pred = pd.Series(pred).astype(float)
    realized_vol = pd.Series(realized_vol, index=pred.index).astype(float)

    pred_std = pred.rolling(window=pred_window, min_periods=pred_window).std()
    signal = pred / (pred_std + 1e-12)
    signal = signal.clip(lower=-clip_signal, upper=clip_signal)

    lev = vol_target_ann / (realized_vol * np.sqrt(252.0) + 1e-12)
    lev = lev.clip(lower=0.0, upper=lev_max)

    pos = np.tanh(signal) * lev
    pos = pos.where(np.isfinite(pos), 0.0).fillna(0.0)
    out = pd.Series(pos, index=pred.index, name="vol_target")
    return apply_position_postprocess(out, smooth_lambda=smooth_lambda, deadband=deadband)


def hybrid_vol_alpha_strategy(
    pred: pd.Series,
    realized_vol: pd.Series,
    z_window: int = 252,
    alpha_scale: float = 1.0,
    vol_target_ann: float = 0.10,
    lev_max: float = 2.0,
    smooth_lambda: float = 0.1,
    deadband: float = 0.05,
    long_only: bool = False,
) -> pd.Series:
    """
    Hybrid strategy:
    position_t = lev_t * alpha_t
    with lev_t from vol targeting and alpha_t from alpha_engine.
    """
    pred = pd.Series(pred).astype(float)
    realized_vol = pd.Series(realized_vol, index=pred.index).astype(float)

    alpha = alpha_engine(
        pred=pred,
        z_window=z_window,
        alpha_scale=alpha_scale,
        smooth_lambda=smooth_lambda,
        deadband=deadband,
        long_only=long_only,
    )
    lev = vol_target_ann / (realized_vol * np.sqrt(252.0) + 1e-12)
    lev = lev.clip(lower=0.0, upper=lev_max)
    pos = (lev * alpha).where(np.isfinite(lev * alpha), 0.0).fillna(0.0)
    name = "hybrid_vol_alpha_longonly" if long_only else "hybrid_vol_alpha"
    return pd.Series(pos, index=pred.index, name=name)


def regime_switching_strategy(
    regime_signal: pd.Series,
    stress_position: pd.Series,
    normal_position: pd.Series,
    regime_threshold: float = 1.0,
    smooth_lambda: float = 0.0,
    deadband: float = 0.0,
) -> pd.Series:
    """
    Generic regime-switching strategy:
    if regime_signal > threshold -> stress_position else normal_position.
    """
    regime_signal = pd.Series(regime_signal).astype(float)
    stress_position = pd.Series(stress_position, index=regime_signal.index).astype(float).fillna(0.0)
    normal_position = pd.Series(normal_position, index=regime_signal.index).astype(float).fillna(0.0)
    use_stress = regime_signal > float(regime_threshold)
    pos = pd.Series(np.where(use_stress, stress_position, normal_position), index=regime_signal.index, name="regime_switching")
    pos = pos.where(np.isfinite(pos), 0.0).fillna(0.0)
    return apply_position_postprocess(pos, smooth_lambda=smooth_lambda, deadband=deadband)


def zscore_threshold_indicators_strategy(
    pos_ml: pd.Series,
    coeff_total: pd.Series,
    confirm_threshold: float = 2.0 / 3.0,
) -> pd.Series:
    """
    Confirmation-filtered version of zscore_threshold.

    Keep ML position only if:
    - abs(coeff_total) >= confirm_threshold
    - sign(pos_ml) == sign(coeff_total)
    Else set position to 0.
    """
    pos_ml = pd.Series(pos_ml).astype(float)
    coeff_total = pd.Series(coeff_total, index=pos_ml.index).astype(float)
    sign_ml = np.sign(pos_ml)
    sign_coeff = np.sign(coeff_total)
    confirmed = (coeff_total.abs() >= float(confirm_threshold)) & (sign_ml == sign_coeff) & (sign_ml != 0.0)
    pos = pos_ml.where(confirmed, 0.0).where(np.isfinite(pos_ml), 0.0).fillna(0.0)
    return pd.Series(pos, index=pos_ml.index, name="zscore_threshold_indicators")
