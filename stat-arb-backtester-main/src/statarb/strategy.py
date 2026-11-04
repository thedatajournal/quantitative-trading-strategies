"""Z-score mean-reversion signals for a spread series.

Signal convention:
    +1  long the spread  (long A, short beta*B)  — entered when z < -entry
    -1  short the spread (short A, long beta*B)  — entered when z >  entry
     0  flat                                     — after |z| falls below exit,
                                                   or a blowout stop at |z| > stop

Positions are generated with data available at each close and are meant to
be APPLIED to the next day's return by the backtester (which shifts them),
so there is no lookahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score of the spread using only past-and-current data."""
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / std


def positions_from_zscore(
    z: pd.Series,
    entry: float = 2.0,
    exit_: float = 0.5,
    stop: float = 4.0,
) -> pd.Series:
    """Stateful entry/exit logic over a z-score series.

    A small explicit loop keeps the state machine readable; daily series
    are a few thousand points so speed is a non-issue.
    """
    if not (0 <= exit_ < entry < stop):
        raise ValueError("Thresholds must satisfy 0 <= exit < entry < stop")

    pos = np.zeros(len(z))
    current = 0
    for i, zi in enumerate(z.to_numpy()):
        if np.isnan(zi):
            current = 0
        elif current == 0:
            if zi > entry:
                current = -1
            elif zi < -entry:
                current = 1
        else:
            # exit on reversion to the mean, or stop out on a blowout
            if abs(zi) < exit_ or abs(zi) > stop:
                current = 0
        pos[i] = current
    return pd.Series(pos, index=z.index, name="position")
