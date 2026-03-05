# src/strategy.py

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def strategy_equity_from_proba(
    logret_next: np.ndarray,
    proba_up: np.ndarray,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Current strategy: long if P(up) >= threshold, else stay in cash.
    Returns (strategy_logret, strategy_equity, positions).
    """
    r = np.asarray(logret_next).reshape(-1)
    p = np.asarray(proba_up).reshape(-1)
    n = min(len(r), len(p))
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    r = r[:n]
    p = p[:n]
    pos = (p >= threshold).astype(float)
    strat_logret = pos * r
    strat_eq = np.exp(np.cumsum(strat_logret))
    return strat_logret, strat_eq, pos


def buy_hold_equity(logret_next: np.ndarray) -> np.ndarray:
    r = np.asarray(logret_next).reshape(-1)
    if len(r) == 0:
        return np.array([])
    return np.exp(np.cumsum(r))


def monte_carlo_random_equity(
    logret_next: np.ndarray,
    n_sims: int = 200,
    p_long: float = 0.5,
    seed: int = 42,
    positions_strategy: Optional[np.ndarray] = None,
    block_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Monte Carlo random investment (timing null):
    - If positions_strategy is provided, generate random curves by block-permuting
      the strategy positions over time (preserve exposure, break timing).
    - Otherwise fallback to i.i.d. random positions (legacy).
    - Returns equity curves and (mean, std) across sims at each time step.
    """
    r = np.asarray(logret_next).reshape(-1)
    T = len(r)
    if T == 0 or n_sims <= 0:
        return np.empty((0, T)), np.array([]), np.array([])

    rng = np.random.default_rng(seed)
    if positions_strategy is not None:
        pos = np.asarray(positions_strategy).reshape(-1).astype(float)
        if len(pos) != T:
            n = min(len(pos), T)
            pos = pos[:n]
            r = r[:n]
            T = n
        if T == 0:
            return np.empty((0, 0)), np.array([]), np.array([])

        # Block permutation of positions
        def _block_perm_idx(n: int, b: int, rng_: np.random.Generator) -> np.ndarray:
            b = max(int(b), 1)
            blocks = [np.arange(i, min(i + b, n)) for i in range(0, n, b)]
            order = rng_.permutation(len(blocks))
            return np.concatenate([blocks[i] for i in order]).astype(int)

        logrets = np.empty((n_sims, T), dtype=float)
        for i in range(n_sims):
            perm_idx = _block_perm_idx(T, block_size, rng)
            pos_perm = pos[perm_idx]
            logrets[i, :] = pos_perm * r
        eq = np.exp(np.cumsum(logrets, axis=1))
    else:
        # Legacy i.i.d. random positions (fallback)
        positions = (rng.random((n_sims, T)) < p_long).astype(float)
        logrets = positions * r[None, :]
        eq = np.exp(np.cumsum(logrets, axis=1))

    mean = np.mean(eq, axis=0)
    std = np.std(eq, axis=0, ddof=1) if n_sims > 1 else np.zeros(T)
    return eq, mean, std
