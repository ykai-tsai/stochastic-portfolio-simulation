# plots.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional, Tuple, Union

__all__ = ["plot_frontier", "plot_points"]

def _ax(ax=None):
    return ax if ax is not None else plt.gca()

def plot_frontier(frontier_df: pd.DataFrame, ax=None, label: str = "Efficient Frontier"):
    """
    Expects columns ['vol','ret'] (from optimize.efficient_frontier).
    """
    ax = _ax(ax)
    v = frontier_df["vol"].to_numpy()
    r = frontier_df["ret"].to_numpy()
    order = np.argsort(v)
    ax.plot(v[order], r[order], label=label)
    ax.set_xlabel("Volatility (annualized)")
    ax.set_ylabel("Return (annualized)")
    ax.legend()
    return ax

def plot_points(points: Iterable[Tuple[float, float]], labels: Optional[Iterable[str]] = None, ax=None, s: int = 40):
    """
    Overlay (vol, ret) points. labels is optional.
    """
    ax = _ax(ax)
    pts = list(points)
    vs = [p[0] for p in pts]
    rs = [p[1] for p in pts]
    ax.scatter(vs, rs, s=s)
    if labels:
        for (v, r), name in zip(pts, labels):
            ax.annotate(str(name), (v, r), xytext=(5, 5), textcoords="offset points")
    return ax
