"""Tearsheet: equity curve vs benchmark, drawdown, and z-score panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — always render to file
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult


def tearsheet(
    result: BacktestResult,
    benchmark_equity: pd.Series | None,
    pair_label: str,
    entry: float,
    exit_: float,
    out_path: str | Path,
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1.4]})

    ax = axes[0]
    ax.plot(result.equity.index, result.equity, label=f"Stat-arb {pair_label} (net of costs)", lw=1.6)
    if benchmark_equity is not None:
        ax.plot(benchmark_equity.index, benchmark_equity, label="SPY buy & hold", lw=1.2, alpha=0.8)
    ax.set_ylabel("Growth of $1")
    ax.set_title(f"Pairs-trading backtest — {pair_label} (out-of-sample)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    dd = result.equity / result.equity.cummax() - 1
    ax.fill_between(dd.index, dd, 0, color="tab:red", alpha=0.4)
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(result.z.index, result.z, lw=0.8, color="tab:purple", label="Spread z-score")
    for level, style in [(entry, "--"), (-entry, "--"), (exit_, ":"), (-exit_, ":")]:
        ax.axhline(level, color="gray", ls=style, lw=0.8)
    in_pos = result.positions != 0
    ax.fill_between(result.z.index, ax.get_ylim()[0], ax.get_ylim()[1],
                    where=in_pos, alpha=0.08, color="tab:green", label="In position")
    ax.set_ylabel("Z-score")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
