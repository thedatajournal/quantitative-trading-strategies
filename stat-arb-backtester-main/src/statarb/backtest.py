"""Dollar-neutral spread backtest with transaction costs.

PnL model (per unit of gross exposure):
    spread_ret_t = (dlog(A)_t - beta * dlog(B)_t) / (1 + |beta|)
    strat_ret_t  = pos_{t-1} * spread_ret_t - tc * |pos_t - pos_{t-1}|

Positions are shifted one day: a signal computed at close t earns the
spread's return from t to t+1, never the same day's — no lookahead.
The (1 + |beta|) divisor normalizes PnL to the gross capital deployed
across both legs. `tc` charges a round-trip-style cost on every unit of
position change, covering commissions and slippage on both legs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import positions_from_zscore, zscore


@dataclass
class BacktestResult:
    returns: pd.Series = field(repr=False)      # daily strategy returns (net)
    equity: pd.Series = field(repr=False)       # cumulative growth of $1
    positions: pd.Series = field(repr=False)    # daily position in {-1, 0, +1}
    z: pd.Series = field(repr=False)            # z-score series used
    n_trades: int = 0                           # number of position changes


def backtest_pair(
    closes: pd.DataFrame,
    leg_a: str,
    leg_b: str,
    beta: float,
    z_window: int = 60,
    entry: float = 2.0,
    exit_: float = 0.5,
    stop: float = 4.0,
    tc: float = 0.0005,
) -> BacktestResult:
    """Backtest a fixed-beta pairs strategy on the given close prices.

    `beta` must come from a training window that precedes `closes` (or from
    its earliest portion) — estimating it on the same data being evaluated
    would leak information.
    """
    log_a = np.log(closes[leg_a])
    log_b = np.log(closes[leg_b])
    s = log_a - beta * log_b

    z = zscore(s, z_window)
    pos = positions_from_zscore(z, entry=entry, exit_=exit_, stop=stop)

    gross = 1.0 + abs(beta)
    spread_ret = (log_a.diff() - beta * log_b.diff()) / gross

    held = pos.shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(0.0)
    strat_ret = held * spread_ret - tc * turnover
    strat_ret = strat_ret.fillna(0.0)

    equity = (1.0 + strat_ret).cumprod()
    n_trades = int((turnover > 0).sum())

    return BacktestResult(
        returns=strat_ret, equity=equity, positions=pos, z=z, n_trades=n_trades
    )
