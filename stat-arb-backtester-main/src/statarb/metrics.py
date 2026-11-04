"""Performance and risk metrics for daily return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(returns: pd.Series) -> float:
    """Geometric annualized return."""
    if len(returns) == 0:
        return 0.0
    total = float((1 + returns).prod())
    years = len(returns) / TRADING_DAYS
    if years == 0 or total <= 0:
        return 0.0
    return total ** (1 / years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * np.sqrt(TRADING_DAYS))


def sharpe(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Annualized Sharpe ratio vs a constant risk-free rate."""
    excess = returns - rf_annual / TRADING_DAYS
    vol = excess.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(excess.mean() / vol * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf_annual: float = 0.0) -> float:
    """Like Sharpe but penalizes only downside volatility."""
    excess = returns - rf_annual / TRADING_DAYS
    downside = excess[excess < 0]
    dvol = downside.std()
    if dvol == 0 or np.isnan(dvol):
        return 0.0
    return float(excess.mean() / dvol * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    """Worst peak-to-trough decline of an equity curve (negative number)."""
    peak = equity.cummax()
    dd = equity / peak - 1
    return float(dd.min())


def hit_rate(returns: pd.Series) -> float:
    """Fraction of active (non-zero) days that were profitable."""
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def summarize(returns: pd.Series, equity: pd.Series, rf_annual: float = 0.0) -> dict:
    return {
        "annualized_return": annualized_return(returns),
        "annualized_vol": annualized_vol(returns),
        "sharpe": sharpe(returns, rf_annual),
        "sortino": sortino(returns, rf_annual),
        "max_drawdown": max_drawdown(equity),
        "hit_rate": hit_rate(returns),
        "n_days": int(len(returns)),
    }
