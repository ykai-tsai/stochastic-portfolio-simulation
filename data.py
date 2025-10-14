from __future__ import annotations
import pandas as pd
import yfinance as yf
from collections import defaultdict

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
    

a = load_prices(['GOOG', 'AMD'], start='2025-01-01', end='2025-10-12')

print(a)
