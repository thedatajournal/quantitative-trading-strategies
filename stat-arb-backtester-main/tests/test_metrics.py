import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from statarb.metrics import (
    annualized_return,
    annualized_vol,
    hit_rate,
    max_drawdown,
    sharpe,
)


def test_annualized_return_exact_year():
    # 252 days of the exact daily rate that compounds to +10%
    daily = (1.10) ** (1 / 252) - 1
    r = pd.Series([daily] * 252)
    assert annualized_return(r) == pytest.approx(0.10, abs=1e-9)


def test_sharpe_known_value():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 10_000))
    expected = r.mean() / r.std() * np.sqrt(252)
    assert sharpe(r) == pytest.approx(expected, rel=1e-12)


def test_sharpe_zero_vol_is_zero_not_nan():
    assert sharpe(pd.Series([0.0] * 100)) == 0.0


def test_max_drawdown_simple():
    # equity 1 -> 2 -> 1 -> 3: worst decline is -50%
    eq = pd.Series([1.0, 2.0, 1.0, 3.0])
    assert max_drawdown(eq) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_is_zero():
    eq = pd.Series(np.linspace(1, 2, 50))
    assert max_drawdown(eq) == 0.0


def test_hit_rate_ignores_flat_days():
    r = pd.Series([0.0, 0.0, 0.01, -0.01, 0.02, 0.0])
    assert hit_rate(r) == pytest.approx(2 / 3)


def test_vol_scaling():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    assert annualized_vol(r) == pytest.approx(r.std() * np.sqrt(252), rel=1e-12)
