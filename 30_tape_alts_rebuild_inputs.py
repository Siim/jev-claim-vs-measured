#!/usr/bin/env python3
"""Tape experiment on alts, step 1 (optional): rebuild the alt tape inputs from Binance yourself.

    python 30_tape_alts_rebuild_inputs.py        (no API key; streams ~460 MB; about 10 minutes)
    python 31_tape_alts_evaluate.py --rebuilt    (then re-run the test on what you built)

Same procedure as the BTC/ETH tape test with one change, forced by the spreads: one tick is
2-3 bp on small alts, so last-trade prices bounce between bid and ask. Prices here are a
mid-price estimate and the bid-ask gap is kept as a spread estimate (see jev/tape_bars.py).

The coins come from data/tape_alts_symbols.json. They were selected by a volume-rank rule,
described in 31_tape_alts_evaluate.py and in method/METHOD.md.
"""

import json
import os

import numpy as np
import pandas as pd

from jev.tape_bars import ENTRY_DELAYS, FEATURE_NAMES, HOLDING_PERIODS, build_symbol_day, return_column

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

REFERENCE_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DAYS = ["2026-09-14", "2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18"]
PRICE_BASIS = "mid_proxy"

SHIPPED = f"{DATA}/tape_alts_levels.parquet"
REBUILT = f"{DATA}/rebuilt_tape_alts_levels.parquet"


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
    symbols = REFERENCE_SYMBOLS + json.load(open(f"{DATA}/tape_alts_symbols.json"))
    rebuilt = pd.concat(
        [build_symbol_day(s, day, PRICE_BASIS) for s in symbols for day in DAYS], ignore_index=True
    )
    rebuilt.to_parquet(REBUILT, index=False, compression="zstd")

    print(f"\ndecision bars {len(rebuilt):,} across {rebuilt['symbol'].nunique()} coins")
    print(
        "rebuilt file identical to the shipped one (levels and forward returns):",
        identical(pd.read_parquet(SHIPPED), rebuilt),
    )


if __name__ == "__main__":
    main()
