from __future__ import annotations
import pandas as pd
import numpy as np
import yfinance as yf
from collections import defaultdict
from typing import Dict, Tuple

def _ensure_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

def load_prices(
    tickers: list[str] | str,
    start: str,
    end: str,
    fields: tuple[str, ...] = ("Adj Close", "Volume"),
    tz_policy: str = "naive",  # "naive" | "per_tz"
):
    tickers = _ensure_list(tickers)

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    ).sort_index()

    by_field = {}
    for f in fields:
        if f in raw.columns.get_level_values(0):
            df = raw[f].copy()
            df = df.dropna(how="all")
            by_field[f] = df

    if tz_policy == "naive":
        return by_field

    elif tz_policy == "per_tz":
        tz_by_ticker = {}
        for t in by_field[fields[0]].columns:
            try:
                tzname = yf.Ticker(t).fast_info.timezone
            except Exception:
                tzname = None
            tz_by_ticker[t] = tzname or "Unknown"

        per_tz = defaultdict(dict)
        for tzname in set(tz_by_ticker.values()):
            tz_tickers = [t for t, tz in tz_by_ticker.items() if tz == tzname]
            if not tz_tickers:
                continue
            for f, df in by_field.items():
                sub = df[tz_tickers].copy()
                if tzname != "Unknown":
                    if sub.index.tz is None:
                        sub.index = sub.index.tz_localize(tzname)
                    else:
                        sub.index = sub.index.tz_convert(tzname)
                per_tz[tzname][f] = sub
        return dict(per_tz)

    else:
        raise ValueError("tz_policy must be 'naive' or 'per_tz'")
    
def to_returns(prices: pd.DataFrame, method: str = "log") -> pd.DataFrame:
    prices = prices.sort_index()
    if method == "log":
        rets = np.log(prices / prices.shift(1))
    elif method == "simple":
        rets = prices.pct_change()
    else:
        raise ValueError("method must be 'log' or 'simple'")
    return rets.dropna(how="all")

def annualize_ret_vol(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> Tuple[pd.Series, pd.Series]:
    mu = returns.mean() * periods_per_year
    sigma = returns.std(ddof=0) * np.sqrt(periods_per_year)
    return mu, sigma

def compute_dollar_volume(prices: pd.DataFrame, volume: pd.DataFrame) -> pd.DataFrame:
    idx = prices.index.intersection(volume.index)
    cols = prices.columns.intersection(volume.columns)
    return prices.loc[idx, cols] * volume.loc[idx, cols]

def resample_to_period_end(df: pd.DataFrame, freq: str = "M", how: str = "last") -> pd.DataFrame:
    if how == "last":
        out = df.resample(freq).last()
    elif how == "mean":
        out = df.resample(freq).mean()
    else:
        raise ValueError("how must be 'last' or 'mean'")
    return out.dropna(how="all")

def bundle_ticker(prices: pd.DataFrame, volume: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Build a per-ticker view with multiple fields as columns.
    """
    cols = [c for c in [ticker] if c in prices.columns]
    volcols = [c for c in [ticker] if c in volume.columns]
    df = pd.DataFrame()
    if cols:
        df["Adj Close"] = prices[cols[0]]
    if volcols:
        df["Volume"] = volume[volcols[0]]
    return df.dropna(how="all")

def to_panel(by_field: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    if not by_field:
        return pd.DataFrame()
    return pd.concat(by_field, axis=1)

def from_panel(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in panel.columns.get_level_values(0):
        raise KeyError(f"field '{field}' not found in panel")
    return panel[field]
