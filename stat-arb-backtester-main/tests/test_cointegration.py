import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from statarb.cointegration import half_life, hedge_ratio, scan_pairs, spread


def _synthetic_pair(n=2000, beta=1.5, seed=0):
    """B is a random walk; log(A) = beta*log(B) + stationary AR(1) noise."""
    rng = np.random.default_rng(seed)
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + np.log(100)
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.9 * eps[i - 1] + rng.normal(0, 0.01)
    log_a = beta * log_b + eps
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    return pd.DataFrame({"A": np.exp(log_a), "B": np.exp(log_b)}, index=idx)


def test_hedge_ratio_recovers_beta():
    closes = _synthetic_pair(beta=1.5)
    est = hedge_ratio(np.log(closes["A"]), np.log(closes["B"]))
    assert est == pytest.approx(1.5, abs=0.05)


def test_cointegrated_pair_detected():
    closes = _synthetic_pair()
    results = scan_pairs(closes, [("A", "B")])
    assert results[0].pvalue < 0.05


def test_independent_walks_not_cointegrated():
    rng = np.random.default_rng(7)
    n = 2000
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    closes = pd.DataFrame({
        "A": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
        "B": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
    }, index=idx)
    results = scan_pairs(closes, [("A", "B")])
    assert results[0].pvalue > 0.05


def test_half_life_of_ar1_matches_theory():
    # AR(1) with phi = 0.9 has half-life -ln(2)/ln(0.9) ~ 6.58 steps
    rng = np.random.default_rng(3)
    n = 20000
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.9 * eps[i - 1] + rng.normal(0, 1)
    hl = half_life(pd.Series(eps))
    assert hl == pytest.approx(-np.log(2) / np.log(0.9), rel=0.15)


def test_random_walk_has_no_half_life():
    rng = np.random.default_rng(4)
    walk = pd.Series(np.cumsum(rng.normal(0, 1, 5000)))
    assert half_life(walk) > 500 or half_life(walk) == float("inf")
