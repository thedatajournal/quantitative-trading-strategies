"""statarb — a pairs-trading statistical arbitrage research library.

Modules:
    data          — daily price download (Stooq) with local CSV caching
    cointegration — Engle-Granger pair scanning, hedge ratios, half-life
    strategy      — z-score mean-reversion signal generation (no lookahead)
    backtest      — dollar-neutral spread backtest with transaction costs
    metrics       — annualized return/vol, Sharpe, Sortino, drawdown, hit rate
    plots         — matplotlib tearsheet (equity, drawdown, z-score)
"""

__version__ = "1.0.0"
