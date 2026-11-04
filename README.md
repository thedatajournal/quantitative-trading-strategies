# stat-arb-backtester

A pairs-trading statistical-arbitrage research library and backtester in Python.
Scans candidate equity pairs for cointegration, trades the spread with a z-score
mean-reversion rule, and evaluates strictly out-of-sample with transaction costs.

![Tearsheet](results/tearsheet.png)

## Methodology

1. **Pair selection (train window only, 2019–2023).** Engle–Granger cointegration
   test on log prices of 10 pre-specified candidate pairs of related large-caps
   (statsmodels). Hedge ratio β from OLS; mean-reversion half-life from an AR(1)
   fit on the spread. Pairs with p < 0.05 qualify — a rule committed *before*
   looking at test data.
2. **Signal.** Rolling 60-day z-score of the spread `log(A) − β·log(B)`.
   Enter at |z| > 2 (short the rich leg, long the cheap one), exit at |z| < 0.5,
   blow-out stop at |z| > 4.
3. **Backtest (test window, 2024 – Jul 2026).** β frozen from train. Signals are
   applied to the *next* day's returns (no lookahead — enforced by a unit test).
   PnL is normalized per unit of gross exposure and charged 5 bps per unit of
   position change (both legs, incl. slippage). Qualifying pairs are combined
   equal-weight.

## Results (out-of-sample, net of costs)

Train scan: **V/MA** (p=0.001, half-life 17d), **KO/PEP** (p=0.030, 39d),
**HD/LOW** (p=0.046, 38d) qualified.

| Metric | Portfolio | Best pair (V/MA) | SPY |
|---|---|---|---|
| Annualized return | −2.8% | +0.0% | +22.0% |
| Annualized vol | 3.0% | 3.5% | 15.9% |
| Sharpe | −0.94 | 0.03 | 1.33 |
| Max drawdown | −7.5% | −5.6% | −18.8% |
| Correlation to SPY | +0.11 | −0.04 | 1.00 |

**Finding: naive pairs trading on liquid US large-caps is no longer profitable
net of costs.** The strategy achieves what it promises structurally — genuine
market neutrality (|corr to SPY| ≈ 0.1, one-fifth of SPY's volatility) — but the
edge itself has decayed. Notably, KO/PEP was strongly cointegrated in-sample yet
lost −8.4%/yr out-of-sample: in-sample cointegration is *evidence*, not a
*guarantee*, and regime shifts break historical relationships. This is
consistent with the well-documented alpha decay of published anomalies
(McLean & Pontiff, 2016).

**What I'd try next:** intraday rather than daily frequency (mean reversion is
faster than a 60-day window can capture), a Kalman filter for a time-varying
hedge ratio, a cointegration-stability filter (rolling p-value monitoring to
drop breaking pairs like KO/PEP), and a wider universe of less-crowded pairs.

## Engineering

- Vectorized NumPy/pandas throughout; explicit state machine only for the
  entry/exit logic where readability wins.
- **19 unit tests** (pytest), including a lookahead-bias test that plants a
  price jump on the final day and asserts the strategy cannot profit from it,
  and statistical tests that recover known AR(1) half-lives and planted
  hedge ratios from synthetic data.
- Data layer: adjusted daily closes from Yahoo Finance with local CSV caching
  for reproducible offline runs.

## Run it

```bash
pip install -r requirements.txt
python scripts/run_backtest.py   # downloads data, scans, backtests, saves tearsheet
python -m pytest tests/ -q       # 19 tests
```

## Disclaimer

Research/educational project. Not investment advice; past (or simulated)
performance does not indicate future results.
