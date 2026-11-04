"""End-to-end research run.

1. Download daily closes for all candidate tickers (Stooq, cached locally).
2. Scan candidate pairs for cointegration on the TRAIN window only.
3. Fix the hedge ratio from train, then backtest the best pair on the
   held-out TEST window with transaction costs.
4. Print a metrics table vs SPY buy-and-hold and save a tearsheet + JSON.

Usage: python scripts/run_backtest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from statarb.backtest import backtest_pair
from statarb.cointegration import scan_pairs
from statarb.data import load_closes
from statarb.metrics import summarize
from statarb.plots import tearsheet

# ---------------- configuration ----------------
TRAIN_START, TRAIN_END = "2019-01-01", "2023-12-31"
TEST_START, TEST_END = "2024-01-01", "2026-07-10"

CANDIDATES = [
    ("KO", "PEP"),    # beverages
    ("XOM", "CVX"),   # integrated oil
    ("V", "MA"),      # card networks
    ("HD", "LOW"),    # home improvement
    ("WMT", "TGT"),   # big-box retail
    ("JPM", "BAC"),   # money-center banks
    ("UPS", "FDX"),   # parcel logistics
    ("MCD", "YUM"),   # fast food
    ("DUK", "SO"),    # regulated utilities
    ("CAT", "DE"),    # heavy machinery
]

Z_WINDOW = 60
ENTRY, EXIT, STOP = 2.0, 0.5, 4.0
TC = 0.0005  # 5 bps per unit of position change (both legs incl. slippage)
BENCHMARK = "SPY"
RESULTS_DIR = ROOT / "results"
# ------------------------------------------------


def main() -> None:
    tickers = sorted({t for pair in CANDIDATES for t in pair} | {BENCHMARK})
    print(f"Loading {len(tickers)} tickers from Yahoo Finance (cached in data/)...")
    closes = load_closes(tickers, TRAIN_START, TEST_END)
    train = closes.loc[TRAIN_START:TRAIN_END]
    test = closes.loc[TEST_START:TEST_END]
    print(f"Train: {train.index[0].date()} -> {train.index[-1].date()} ({len(train)} days)")
    print(f"Test:  {test.index[0].date()} -> {test.index[-1].date()} ({len(test)} days)\n")

    print("Engle-Granger cointegration scan (train window only):")
    scan = scan_pairs(train, CANDIDATES)
    header = f"{'pair':12} {'p-value':>8} {'beta':>7} {'half-life':>10}"
    print(header)
    print("-" * len(header))
    for r in scan:
        hl = f"{r.half_life_days:9.1f}d" if np.isfinite(r.half_life_days) else "      none"
        print(f"{r.leg_a}/{r.leg_b:<8} {r.pvalue:8.4f} {r.hedge_ratio:7.3f} {hl}")

    tradeable = [r for r in scan if r.pvalue < 0.05 and np.isfinite(r.half_life_days)]
    if not tradeable:
        print("\nNo pair passes p < 0.05 on train — refusing to trade noise. Exiting.")
        return
    labels = [f"{r.leg_a}/{r.leg_b}" for r in tradeable]
    print(f"\nTrading all pairs with train p < 0.05, equal weight: {', '.join(labels)}")

    # Equal-weight portfolio across every pair that passed the pre-committed
    # significance rule — diversification, not cherry-picking the winner.
    pair_results = {}
    for r in tradeable:
        pair_results[f"{r.leg_a}/{r.leg_b}"] = backtest_pair(
            test, r.leg_a, r.leg_b, beta=r.hedge_ratio,
            z_window=Z_WINDOW, entry=ENTRY, exit_=EXIT, stop=STOP, tc=TC,
        )
    port_ret = pd.concat([res.returns for res in pair_results.values()], axis=1).mean(axis=1)
    port_equity = (1.0 + port_ret).cumprod()
    label = " + ".join(pair_results)
    n_trades_total = sum(res.n_trades for res in pair_results.values())
    result = pair_results[labels[0]]  # best pair, used for the z-score panel

    bench_ret = test[BENCHMARK].pct_change().fillna(0.0)
    bench_equity = (1 + bench_ret).cumprod()

    print("\nPer-pair out-of-sample Sharpe:")
    for name, res in pair_results.items():
        s = summarize(res.returns, res.equity)
        print(f"  {name:8} sharpe {s['sharpe']:+5.2f}  ann.ret {s['annualized_return']:+7.2%}  trades {res.n_trades}")

    strat = summarize(port_ret, port_equity)
    bench = summarize(bench_ret, bench_equity)

    print(f"\nOut-of-sample results ({TEST_START} -> {TEST_END}), net of {TC*1e4:.0f} bps costs:")
    rows = [
        ("Annualized return", "annualized_return", "{:+.2%}"),
        ("Annualized vol", "annualized_vol", "{:.2%}"),
        ("Sharpe", "sharpe", "{:.2f}"),
        ("Sortino", "sortino", "{:.2f}"),
        ("Max drawdown", "max_drawdown", "{:+.2%}"),
        ("Hit rate (active days)", "hit_rate", "{:.1%}"),
    ]
    print(f"{'metric':24} {'strategy':>10} {'SPY':>10}")
    for name, key, fmt in rows:
        print(f"{name:24} {fmt.format(strat[key]):>10} {fmt.format(bench[key]):>10}")
    print(f"Position changes across all pairs: {n_trades_total}")

    corr = float(port_ret.corr(bench_ret))
    print(f"Correlation to SPY: {corr:+.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary = {
        "pairs": [
            {"pair": f"{r.leg_a}/{r.leg_b}", "pvalue_train": r.pvalue,
             "hedge_ratio_train": r.hedge_ratio, "half_life_days_train": r.half_life_days,
             **summarize(pair_results[f"{r.leg_a}/{r.leg_b}"].returns,
                         pair_results[f"{r.leg_a}/{r.leg_b}"].equity)}
            for r in tradeable
        ],
        "train": [TRAIN_START, TRAIN_END],
        "test": [TEST_START, TEST_END],
        "params": {"z_window": Z_WINDOW, "entry": ENTRY, "exit": EXIT, "stop": STOP, "tc": TC},
        "portfolio": strat,
        "spy_benchmark": bench,
        "n_trades": n_trades_total,
        "correlation_to_spy": corr,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    from statarb.backtest import BacktestResult
    port_view = BacktestResult(returns=port_ret, equity=port_equity,
                               positions=result.positions, z=result.z,
                               n_trades=n_trades_total)
    png = tearsheet(port_view, bench_equity, label, ENTRY, EXIT, RESULTS_DIR / "tearsheet.png")
    print(f"\nSaved {RESULTS_DIR / 'summary.json'}")
    print(f"Saved {png}")


if __name__ == "__main__":
    main()
