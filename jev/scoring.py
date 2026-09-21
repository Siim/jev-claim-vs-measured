"""Turning the model's answers into trades, and judging them.

The trade rule is the same in every experiment and was fixed before any outcome was seen:

    score = P(buy) - P(sell)
    buy when score >= +0.25, sell when score <= -0.25, otherwise do nothing.
"""

from collections import namedtuple

import numpy as np

ACT_THRESHOLD = 0.25

TradeSummary = namedtuple("TradeSummary", "trades gross_bp hit_rate")


def signal_from_score(score, threshold=ACT_THRESHOLD):
    """+1 = buy, -1 = sell, 0 = do nothing."""
    score = np.asarray(score, dtype=float)
    return np.where(score >= threshold, 1, np.where(score <= -threshold, -1, 0))


def summarise_trades(signal, forward_return_bp):
    """Number of trades, average gross profit per trade (bp), and hit rate.

    The hit rate is measured over trades whose outcome was *not flat*. On one-second bars
    the price often does not move at all; a coin flip gets 50% of the rest, so that is the
    fair benchmark.
    """
    signal = np.asarray(signal)
    forward_return_bp = np.asarray(forward_return_bp, dtype=float)

    traded = signal != 0
    if not traded.any():
        return TradeSummary(0, np.nan, np.nan)

    profit_bp = signal[traded] * forward_return_bp[traded]
    moved = profit_bp != 0
    hit_rate = float((profit_bp[moved] > 0).mean()) if moved.any() else np.nan
    return TradeSummary(int(traded.sum()), float(profit_bp.mean()), hit_rate)


def time_shift_null(
    signal, forward_return_bp, groups, rng, draws=300, smallest_shift=500, smallest_group=1200
):
    """How unusual is the real profit compared with the same signal at the wrong time?

    For each draw, the signal is rotated in time *within each group* (one group = one
    symbol-day) by a random number of bars. That keeps everything about the signal - how
    often it trades, how long it stays on one side - and destroys only its alignment with
    the market. Returns ``(z, p)``: the z-score of the real gross profit against those
    draws, and the share of draws that did at least as well.
    """
    signal = np.asarray(signal)
    forward_return_bp = np.asarray(forward_return_bp, dtype=float)
    groups = np.asarray(groups)

    real = summarise_trades(signal, forward_return_bp).gross_bp
    rows_of_group = [np.where(groups == name)[0] for name in np.unique(groups)]

    shuffled_profits = []
    for _ in range(draws):
        shifted = signal.copy()
        for rows in rows_of_group:
            if len(rows) > smallest_group:
                shift = int(rng.integers(smallest_shift, len(rows) - smallest_shift))
                shifted[rows] = np.roll(signal[rows], shift)
        shuffled_profits.append(summarise_trades(shifted, forward_return_bp).gross_bp)

    z = (real - np.nanmean(shuffled_profits)) / np.nanstd(shuffled_profits)
    p = (1 + sum(value >= real for value in shuffled_profits)) / (draws + 1)
    return z, p


def print_headline(numbers):
    """Print the block that ``check_headline_numbers.py`` reads."""
    print("\nHEADLINE NUMBERS")
    for name, value in numbers.items():
        print(f"  {name} = {value}")
