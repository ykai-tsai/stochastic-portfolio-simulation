# metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["sharpe_ratio", "var", "cvar"]

def _to_simple(returns: pd.DataFrame, log: bool) -> pd.DataFrame:
    return np.expm1(returns) if log else returns

def sharpe_ratio(
    returns: pd.DataFrame,
    rf: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    if isinstance(rf, pd.Series):
        rf = rf.reindex(returns.index).fillna(0.0)
    ex = returns.sub(rf, axis=0) if isinstance(rf, pd.Series) else (returns - rf)

    mu = ex.mean() * periods_per_year
    sigma = ex.std(ddof=0) * np.sqrt(periods_per_year)
    return mu / sigma.replace(0.0, np.nan)

def var(
    returns: pd.DataFrame,
    level: float = 0.05,
    returns_are_log: bool = True,
) -> pd.Series:
    simple = _to_simple(returns, returns_are_log)
    return simple.quantile(level)

def cvar(
    returns: pd.DataFrame,
    level: float = 0.05,
    returns_are_log: bool = True,
) -> pd.Series:
    simple = _to_simple(returns, returns_are_log)
    q = simple.quantile(level)
    es = pd.Series(index=simple.columns, dtype=float)
    for c in simple.columns:
        tail = simple[c][simple[c] <= q[c]]
        es[c] = tail.mean()
    return es
