# optimize.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Iterable
from scipy.optimize import minimize

__all__ = ["global_min_variance", "min_vol_for_target_return", "max_sharpe", "efficient_frontier"]

def _align(mu: pd.Series, cov: pd.DataFrame):
    assets = list(mu.index)
    cov = cov.loc[assets, assets]
    return mu.values.astype(float), cov.values.astype(float), assets

def _bounds(n: int, lb: float | Iterable[float] = 0.0, ub: float | Iterable[float] = 1.0):
    if isinstance(lb, (int, float)): lb = [float(lb)] * n
    if isinstance(ub, (int, float)): ub = [float(ub)] * n
    return tuple((l, u) for l, u in zip(lb, ub))

def _solve(
    obj, x0, bounds, cons, jac=None, method="SLSQP", options=None
) -> np.ndarray:
    res = minimize(obj, x0=x0, jac=jac, bounds=bounds, constraints=cons, method=method, options=options)
    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")
    w = res.x
    w[w < 0] = 0.0
    s = w.sum()
    return w / s if s != 0 else w

def global_min_variance(
    cov: pd.DataFrame,
    lb: float | Iterable[float] = 0.0,
    ub: float | Iterable[float] = 1.0,
) -> pd.Series:
    assets = list(cov.columns)
    C = cov.loc[assets, assets].values
    n = len(assets)
    bounds = _bounds(n, lb, ub)
    x0 = np.ones(n) / n

    def f(w, C=C): return float(w @ C @ w)
    def g(w, C=C): return (2 * C @ w).astype(float)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w = _solve(f, x0, bounds, cons, jac=g)
    return pd.Series(w, index=assets, name="GMV")

def min_vol_for_target_return(
    mu: pd.Series,
    cov: pd.DataFrame,
    target_mu: float,
    lb: float | Iterable[float] = 0.0,
    ub: float | Iterable[float] = 1.0,
) -> pd.Series:
    m, C, assets = _align(mu, cov)
    n = len(assets)
    bounds = _bounds(n, lb, ub)
    x0 = np.ones(n) / n

    def f(w, C=C): return float(w @ C @ w)
    def g(w, C=C): return (2 * C @ w).astype(float)

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w, m=m, t=target_mu: w @ m - t},
    ]
    w = _solve(f, x0, bounds, cons, jac=g)
    return pd.Series(w, index=assets, name=f"MinVol@{target_mu:.4f}")

def max_sharpe(
    mu: pd.Series,
    cov: pd.DataFrame,
    rf: float = 0.0,
    lb: float | Iterable[float] = 0.0,
    ub: float | Iterable[float] = 1.0,
) -> pd.Series:
    m, C, assets = _align(mu, cov)
    n = len(assets)
    bounds = _bounds(n, lb, ub)
    x0 = np.ones(n) / n
    ones = np.ones(n)
    m_ex = m - rf * ones

    def f(w, m=m_ex, C=C):
        ret = w @ m
        vol = np.sqrt(max(w @ C @ w, 1e-18))
        return -ret / vol

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    w = _solve(f, x0, bounds, cons)
    return pd.Series(w, index=assets, name="MaxSharpe")

def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    lb: float | Iterable[float] = 0.0,
    ub: float | Iterable[float] = 1.0,
) -> pd.DataFrame:
    m, C, assets = _align(mu, cov)

    w_gmv = global_min_variance(cov, lb=lb, ub=ub).values
    mu_gmv = float(w_gmv @ m)
    t_min, t_max = min(mu_gmv, m.max()), max(mu_gmv, m.max())
    targets = np.linspace(t_min, t_max, n_points)

    W, R, V = [], [], []
    for t in targets:
        w = min_vol_for_target_return(mu, cov, target_mu=t, lb=lb, ub=ub).values
        W.append(w)
        R.append(float(w @ m))
        V.append(float(np.sqrt(max(w @ C @ w, 0.0))))

    df = pd.DataFrame({"ret": R, "vol": V})
    df.attrs["weights"] = W
    return df
