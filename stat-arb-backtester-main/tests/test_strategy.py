import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from statarb.backtest import backtest_pair
from statarb.strategy import positions_from_zscore, zscore


def test_positions_enter_short_above_entry_and_exit_below_exit():
    z = pd.Series([0.0, 1.0, 2.5, 1.5, 0.4, 0.0])
    pos = positions_from_zscore(z, entry=2.0, exit_=0.5)
    assert list(pos) == [0, 0, -1, -1, 0, 0]


def test_positions_enter_long_below_negative_entry():
    z = pd.Series([0.0, -2.5, -1.0, -0.2])
    pos = positions_from_zscore(z, entry=2.0, exit_=0.5)
    assert list(pos) == [0, 1, 1, 0]


def test_blowout_stop_flattens():
    z = pd.Series([0.0, 2.5, 4.5, 3.0])
    pos = positions_from_zscore(z, entry=2.0, exit_=0.5, stop=4.0)
    # stopped out at 4.5 and does NOT re-enter at 3.0 (already in no-position
    # state and 3.0 > entry means a fresh short... it re-enters):
    assert list(pos) == [0, -1, 0, -1]


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        positions_from_zscore(pd.Series([0.0]), entry=1.0, exit_=2.0)


def test_zscore_uses_only_past_window():
    s = pd.Series(np.arange(100, dtype=float))
    z = zscore(s, window=10)
    assert z.iloc[:9].isna().all()  # not enough history yet
    assert np.isfinite(z.iloc[10:]).all()


def test_backtest_no_lookahead():
    """The return earned on day t must use the position decided at t-1.

    Construct prices where a huge spread move happens on the LAST day and
    verify the strategy cannot profit from a position entered that same day.
    """
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    a = pd.Series(100.0, index=idx)
    b = pd.Series(100.0, index=idx)
    # flat spread until the final day, when A jumps 10%
    a.iloc[-1] = 110.0
    closes = pd.DataFrame({"A": a, "B": b})

    res = backtest_pair(closes, "A", "B", beta=1.0, z_window=20,
                        entry=2.0, exit_=0.5, stop=100.0, tc=0.0)
    # Any signal from the jump appears at the last close — too late to trade.
    # With a flat spread before it, no position was ever held, so PnL == 0.
    assert res.returns.abs().sum() == pytest.approx(0.0)


def test_transaction_costs_reduce_returns():
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    # mean-reverting spread: log(A) oscillates around log(B)
    noise = np.sin(np.linspace(0, 40, n)) * 0.05 + rng.normal(0, 0.005, n)
    b = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    a = b * np.exp(noise)
    closes = pd.DataFrame({"A": a, "B": b})

    free = backtest_pair(closes, "A", "B", beta=1.0, tc=0.0)
    costly = backtest_pair(closes, "A", "B", beta=1.0, tc=0.002)
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]
    assert free.n_trades == costly.n_trades  # costs change PnL, not signals
