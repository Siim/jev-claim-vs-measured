"""Turning numbers into five named levels.

The vendor's documentation says the model is weak at numeric precision and recommends
passing *named buckets* while keeping all arithmetic in code. So every feature in these
experiments is reduced, in code, to one of five ordered levels:

    0 = far below usual   1 = below usual   2 = usual   3 = above usual   4 = far above usual

and only the matching plain-English phrase is sent to the model (see the ``wording_*``
modules). Nothing here looks at the future: a level at time *t* uses data up to *t* only.
"""

import numpy as np
import pandas as pd

LEVELS = [0, 1, 2, 3, 4]


def _pick_level(values, conditions):
    """Apply five mutually exclusive conditions; missing inputs stay missing."""
    levels = np.select(conditions, LEVELS, default=np.nan)
    return pd.Series(levels, index=values.index).where(values.notna())


def level_from_percentile(percentile):
    """Level from a percentile between 0 and 1.

    bottom 10% -> 0, next 20% -> 1, middle 40% -> 2, next 20% -> 3, top 10% -> 4
    """
    p = percentile
    return _pick_level(p, [p <= 0.10, p <= 0.30, p < 0.70, p < 0.90, p >= 0.90])


def level_from_zscore(zscore):
    """Level from a z-score, cut at -1.5, -0.5, +0.5 and +1.5."""
    z = zscore
    return _pick_level(z, [z <= -1.5, z <= -0.5, z < 0.5, z < 1.5, z >= 1.5])


def level_from_range_position(position):
    """Level from a position inside a high-low range (0 = at the low, 1 = at the high)."""
    x = position
    return _pick_level(x, [x <= 0.10, x <= 0.35, x < 0.65, x < 0.90, x >= 0.90])


def trailing_percentile(series, window, minimum_history):
    """Percentile of each value versus its own trailing window (the current value included)."""
    return series.rolling(window, min_periods=minimum_history).rank(pct=True)
