# portfolio.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

__all__ = ["estimate_mu_cov", "portfolio_perf"]

def estimate_mu_cov(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> Tuple[pd.Series, pd.DataFrame]:
    mu = returns.mean() * periods_per_year
    cov = returns.cov(ddof=0) * periods_per_year
    return mu, cov

def portfolio_perf(
    w: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
) -> tuple[float, float, float]:
    w = np.asarray(w, dtype=float)
    if not np.isclose(w.sum(), 1.0):
        w = w / w.sum()
    mu_vec = mu.values
    C = cov.values
    pret = float(w @ mu_vec)
    pvol = float(np.sqrt(max(w @ C @ w, 0.0)))
    sharpe = (pret - rf) / (pvol if pvol > 0 else np.nan)
    return pret, pvol, sharpe
