#!/usr/bin/env python3
"""Tape experiment, step 1 (optional): rebuild the tape inputs from Binance yourself.

    python 20_tape_rebuild_inputs.py        (no API key; streams ~180 MB; about 3 minutes)
    python 22_tape_evaluate.py --rebuilt    (then re-run the test on what you built)

Every aggregated trade on BTCUSDT and ETHUSDT perpetuals for five days is streamed from
Binance's public archive, grouped into bars of 50 trades, and turned into feature levels and
forward returns (see jev/tape_bars.py). Nothing raw is written to disk.

The five days are 2026-09-14 .. 2026-09-18: the five most recent complete weekdays when the
test was written, chosen before looking at them.
"""

import os

import numpy as np
import pandas as pd

from jev.tape_bars import ENTRY_DELAYS, FEATURE_NAMES, HOLDING_PERIODS, build_symbol_day, return_column

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DAYS = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]
PRICE_BASIS = "last_trade"  # the spread on BTC and ETH is one tick (~0.01 bp)

SHIPPED = f"{DATA}/tape_levels.parquet"
REBUILT = f"{DATA}/rebuilt_tape_levels.parquet"


def identical(shipped, rebuilt):
    if len(shipped) != len(rebuilt):
        return False
    same_levels = (shipped[FEATURE_NAMES].values == rebuilt[FEATURE_NAMES].values).all()
    same_returns = all(
        np.allclose(shipped[return_column(d, h)].values, rebuilt[return_column(d, h)].values, equal_nan=True)
        for d in ENTRY_DELAYS
        for h in HOLDING_PERIODS
    )
    return bool(same_levels and same_returns)


def main():
    rebuilt = pd.concat(
        [build_symbol_day(s, day, PRICE_BASIS) for s in SYMBOLS for day in DAYS], ignore_index=True
    )
    rebuilt.to_parquet(REBUILT, index=False, compression="zstd")

    print(f"\ndecision bars {len(rebuilt):,}; median bar {rebuilt['bar_seconds'].median():.2f} s")
    print(f"distinct states that occur: {len(rebuilt[FEATURE_NAMES].drop_duplicates()):,} of 15,625")
    print(
        "rebuilt file identical to the shipped one (levels and forward returns):",
        identical(pd.read_parquet(SHIPPED), rebuilt),
    )


if __name__ == "__main__":
    main()
