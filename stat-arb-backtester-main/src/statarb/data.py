"""Daily price data from Yahoo Finance (via yfinance) with local CSV caching.

Closes are auto-adjusted (splits and dividends), so both the spread legs
and the buy-and-hold benchmark are measured on total-return-consistent
prices. Each symbol is cached to data/<symbol>.csv so repeated research
runs are fast and reproducible offline.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[2] / "data"


def fetch_daily(symbol: str, cache_dir: Path = CACHE_DIR, max_age_days: float = 1.0) -> pd.DataFrame:
    """Return adjusted daily OHLCV for a symbol, using a local CSV cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol.lower()}.csv"

    if cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days <= max_age_days:
            return pd.read_csv(cache_file, parse_dates=["Date"], index_col="Date")

    df = yf.Ticker(symbol).history(period="max", auto_adjust=True)
    if df.empty:
        raise ValueError(f"Yahoo Finance returned no data for {symbol!r}")

    df = df.sort_index()
    df.index = df.index.tz_localize(None)  # naive dates align cleanly across symbols
    df.index.name = "Date"
    df.to_csv(cache_file)
    return df


def load_closes(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Aligned adjusted closes for several symbols over [start, end].

    Only dates where every symbol traded are kept, so spreads are always
    computed on synchronous prices.
    """
    closes = {}
    for sym in symbols:
        closes[sym] = fetch_daily(sym)["Close"]
    df = pd.DataFrame(closes).loc[start:end].dropna()
    if df.empty:
        raise ValueError(f"No overlapping data for {symbols} in [{start}, {end}]")
    return df
