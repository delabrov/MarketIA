from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    close = pd.Series(close).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def build_indicator_signals(close: pd.Series) -> pd.DataFrame:
    """
    Build EMA-cross, RSI(14), MACD(12,26,9) indicator values and discrete signals.
    Signals are in {-1, 0, +1}. coeff_total is the simple average of the 3 signals.
    """
    close = pd.Series(close).astype(float)
    out = pd.DataFrame(index=close.index)
    out["close"] = close

    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    sig_ema = np.where(ema20 > ema50, 1.0, np.where(ema20 < ema50, -1.0, 0.0))
    sig_ema = pd.Series(sig_ema, index=close.index).where(np.isfinite(ema20) & np.isfinite(ema50), 0.0)

    rsi14 = _rsi_wilder(close, period=14)
    sig_rsi = np.where(rsi14 < 35.0, 1.0, np.where(rsi14 > 65.0, -1.0, 0.0))
    sig_rsi = pd.Series(sig_rsi, index=close.index).where(np.isfinite(rsi14), 0.0)

    ema12 = close.ewm(span=12, adjust=False, min_periods=26).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    sig_macd = np.where(macd_line > macd_signal, 1.0, np.where(macd_line < macd_signal, -1.0, 0.0))
    sig_macd = pd.Series(sig_macd, index=close.index).where(np.isfinite(macd_line) & np.isfinite(macd_signal), 0.0)

    coeff_total = (sig_ema + sig_rsi + sig_macd) / 3.0

    out["ema20"] = ema20
    out["ema50"] = ema50
    out["signal_ema"] = sig_ema.astype(float)
    out["rsi14"] = rsi14
    out["signal_rsi"] = sig_rsi.astype(float)
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["signal_macd"] = sig_macd.astype(float)
    out["coeff_total"] = coeff_total.astype(float)
    return out
