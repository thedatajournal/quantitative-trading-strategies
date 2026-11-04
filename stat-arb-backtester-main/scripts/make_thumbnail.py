"""Generate a 1280x640 social-preview thumbnail for the repo.

Uses the real V/MA spread z-score from the cached backtest data as the hero
visual, with a title overlay. Output: results/social-preview.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from statarb.data import load_closes
from statarb.cointegration import hedge_ratio, spread
from statarb.strategy import zscore, positions_from_zscore

# ---- palette ----
BG = "#0A1428"        # deep navy
ACCENT = "#38BDF8"    # sky/cyan (the signal)
BAND = "#5B6B85"      # muted band lines
GREEN = "#34D399"     # in-position shading
INK = "#F1F5F9"       # near-white text
SUB = "#94A3B8"       # secondary text
GOLD = "#FBBF24"      # small highlight

# Prefer a clean system sans; fall back to DejaVu.
for fam in ("Segoe UI", "Arial", "Helvetica"):
    try:
        font_manager.findfont(fam, fallback_to_default=False)
        plt.rcParams["font.family"] = fam
        break
    except Exception:
        continue

# ---- compute the real z-score (V/MA, test window) ----
closes = load_closes(["V", "MA"], "2019-01-01", "2026-07-10")
train = closes.loc["2019-01-01":"2023-12-31"]
beta = hedge_ratio(np.log(train["V"]), np.log(train["MA"]))
test = closes.loc["2024-01-01":"2026-07-10"]
s = spread(np.log(test["V"]), np.log(test["MA"]), beta)
z = zscore(s, 60).dropna()
pos = positions_from_zscore(z, entry=2.0, exit_=0.5, stop=4.0)
x = np.arange(len(z))

# ---- figure ----
fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG)
ax.set_xlim(0, len(z) - 1)
ax.set_ylim(-4.2, 4.2)
ax.axis("off")

# entry / exit reference bands
for lvl in (2, -2):
    ax.axhline(lvl, color=BAND, lw=1.2, ls="--", alpha=0.7)
for lvl in (0.5, -0.5):
    ax.axhline(lvl, color=BAND, lw=1.0, ls=":", alpha=0.5)
ax.axhline(0, color=BAND, lw=1.0, alpha=0.35)

# shade in-position stretches
in_pos = (pos.reindex(z.index).fillna(0) != 0).to_numpy()
ax.fill_between(x, -4.2, 4.2, where=in_pos, color=GREEN, alpha=0.06, lw=0)

# the signal line (glow + core)
ax.plot(x, z.to_numpy(), color=ACCENT, lw=4.5, alpha=0.18, solid_capstyle="round")
ax.plot(x, z.to_numpy(), color=ACCENT, lw=1.9, alpha=0.95, solid_capstyle="round")

# left-side dark gradient for text legibility
grad = np.linspace(1, 0, 256).reshape(1, -1)
ax.imshow(
    grad, extent=[0, len(z) - 1, -4.2, 4.2], aspect="auto",
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("g", [BG, BG]),
    alpha=0, zorder=2,
)
# proper alpha gradient via RGBA image
rgba = np.zeros((1, 256, 4))
bg_rgb = matplotlib.colors.to_rgb(BG)
rgba[..., 0], rgba[..., 1], rgba[..., 2] = bg_rgb
rgba[..., 3] = np.linspace(0.97, 0.0, 256) ** 1.1
ax.imshow(rgba, extent=[0, len(z) - 1, -4.2, 4.2], aspect="auto", zorder=3)

# ---- text overlay (figure coords) ----
def txt(xf, yf, s, size, color, weight="normal", spacing=None, alpha=1.0):
    t = fig.text(xf, yf, s, fontsize=size, color=color, weight=weight,
                 ha="left", va="center", alpha=alpha, zorder=10)
    if spacing:
        t.set_fontproperties(t.get_fontproperties())
    return t

txt(0.055, 0.86, "Q U A N T I T A T I V E   R E S E A R C H", 13, ACCENT, "bold")
txt(0.052, 0.70, "Statistical Arbitrage", 46, INK, "bold")
txt(0.052, 0.575, "Backtester", 46, INK, "bold")
txt(0.055, 0.44, "Pairs trading · Engle–Granger cointegration ·", 18, SUB)
txt(0.055, 0.385, "strict out-of-sample evaluation with costs", 18, SUB)

# tech row
txt(0.055, 0.25, "Python", 15, INK, "bold")
txt(0.135, 0.25, "· NumPy · pandas · statsmodels · pytest", 15, SUB)

# market-neutral highlight
fig.text(0.055, 0.135, "Market-neutral  ", fontsize=15, color=GOLD, weight="bold",
         ha="left", va="center", zorder=10)
fig.text(0.185, 0.135, "ρ ≈ 0.1 to S&P 500  ·  1/5 the volatility",
         fontsize=15, color=SUB, ha="left", va="center", zorder=10)

# github handle bottom-right
fig.text(0.965, 0.06, "github.com/IanWaters0/stat-arb-backtester",
         fontsize=13, color=SUB, ha="right", va="center", zorder=10)

out = ROOT / "results" / "social-preview.png"
out.parent.mkdir(exist_ok=True)
fig.savefig(out, facecolor=BG, dpi=100)
plt.close(fig)
print("Saved:", out, "(1280x640)")
