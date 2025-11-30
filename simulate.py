from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple

__all__ = [
    "simulate_gaussian_returns",
    "simulate_portfolio_paths",
    "mc_var_cvar",
    "block_bootstrap_returns",
]

def simulate_gaussian_returns(
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    n_steps: int,
    n_paths: int = 10_000,
    periods_per_year: int = 252,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assets = list(mu_annual.index)
    mu_step = mu_annual.values / periods_per_year
    cov_step = cov_annual.values / periods_per_year

    try:
        L = np.linalg.cholesky(cov_step)
    except np.linalg.LinAlgError:
        jitter = 1e-10 * np.eye(cov_step.shape[0])
        L = np.linalg.cholesky(cov_step + jitter)

    Z = rng.standard_normal((n_steps, n_paths, len(assets)))  # iid N(0,1)
    shocks = Z @ L.T
    step_rets = mu_step + shocks
    return step_rets


def simulate_portfolio_paths(
    weights: pd.Series,
    mu_annual: pd.Series,
    cov_annual: pd.DataFrame,
    n_steps: int,
    n_paths: int = 10_000,
    periods_per_year: int = 252,
    initial_wealth: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    assets = list(mu_annual.index)
    w = weights.reindex(assets).fillna(0.0).values
    w = w / w.sum() if w.sum() != 0 else w

    step_rets = simulate_gaussian_returns(
        mu_annual, cov_annual, n_steps, n_paths, periods_per_year, seed
    )

    port_step_rets = (step_rets @ w)  # (T, P)

    wealth = np.empty((n_steps + 1, n_paths), dtype=float)
    wealth[0, :] = initial_wealth
    np.multiply.accumulate(1.0 + port_step_rets, axis=0, out=wealth[1:, :])
    wealth[1:, :] *= initial_wealth
    return wealth

def mc_var_cvar(
    wealth_paths: np.ndarray,
    level: float = 0.05,
) -> Tuple[float, float]:
    W0 = wealth_paths[0, 0]
    WT = wealth_paths[-1, :]
    horizon_returns = WT / W0 - 1.0
    q = np.quantile(horizon_returns, level)
    cvar = horizon_returns[horizon_returns <= q].mean()
    return float(q), float(cvar)

def block_bootstrap_returns(
    returns: pd.DataFrame,
    n_steps: int,
    n_paths: int = 10_000,
    block_size: int = 20,
    seed: Optional[int] = None,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = returns.to_numpy()  # (T, N)
    T, N = R.shape
    n_blocks = int(np.ceil(n_steps / block_size))
    out = np.empty((n_steps, n_paths, N), dtype=float)

    for p in range(n_paths):
        chunks = []
        for _ in range(n_blocks):
            start = rng.integers(0, max(1, T - block_size + 1))
            chunk = R[start:start + block_size, :]
            chunks.append(chunk)
        path = np.vstack(chunks)[:n_steps, :]
        out[:, p, :] = path
    return out
