"""Cointegration testing and hedge-ratio estimation for candidate pairs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint


@dataclass
class PairResult:
    """Engle-Granger scan result for one candidate pair (fit on train data only)."""

    leg_a: str
    leg_b: str
    pvalue: float          # Engle-Granger cointegration p-value
    hedge_ratio: float     # beta from OLS of log(A) on log(B)
    half_life_days: float  # mean-reversion half-life of the spread
    spread_std: float      # std of the train spread


def hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    """OLS hedge ratio: log(A) = alpha + beta * log(B) + eps."""
    X = sm.add_constant(log_b.to_numpy())
    model = sm.OLS(log_a.to_numpy(), X).fit()
    return float(model.params[1])


def spread(log_a: pd.Series, log_b: pd.Series, beta: float) -> pd.Series:
    """Log-price spread per unit of leg A: s = log(A) - beta * log(B)."""
    return log_a - beta * log_b


def half_life(s: pd.Series) -> float:
    """Half-life of mean reversion from an AR(1) fit on the spread.

    ds_t = lambda * s_{t-1} + eps  =>  half-life = -ln(2) / lambda.
    A non-negative lambda means no mean reversion (returns inf).
    """
    lag = s.shift(1).dropna()
    ds = s.diff().dropna()
    lag, ds = lag.align(ds, join="inner")
    X = sm.add_constant(lag.to_numpy() - lag.mean())
    lam = sm.OLS(ds.to_numpy(), X).fit().params[1]
    if lam >= 0:
        return float("inf")
    return float(-np.log(2) / lam)


def scan_pairs(closes: pd.DataFrame, candidates: list[tuple[str, str]]) -> list[PairResult]:
    """Engle-Granger test on every candidate pair, sorted by p-value.

    `closes` must be TRAIN data only — using test data here would leak the
    future into pair selection.
    """
    results = []
    log_prices = np.log(closes)
    for a, b in candidates:
        if a not in closes.columns or b not in closes.columns:
            continue
        la, lb = log_prices[a], log_prices[b]
        _, pvalue, _ = coint(la, lb)
        beta = hedge_ratio(la, lb)
        s = spread(la, lb, beta)
        results.append(
            PairResult(
                leg_a=a,
                leg_b=b,
                pvalue=float(pvalue),
                hedge_ratio=beta,
                half_life_days=half_life(s),
                spread_std=float(s.std()),
            )
        )
    return sorted(results, key=lambda r: r.pvalue)
