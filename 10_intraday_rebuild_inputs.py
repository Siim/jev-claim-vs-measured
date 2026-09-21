#!/usr/bin/env python3
"""15-60 minute experiment, step 1 (optional): rebuild the market inputs from Binance yourself.

    python 10_intraday_rebuild_inputs.py        (no API key; streams ~170 MB; 10-20 minutes)
    python 12_intraday_evaluate.py --rebuilt    (then re-run the test on what you built)

Source: Binance's public archive, one small file per symbol per day:
    https://data.binance.vision/data/futures/um/daily/metrics/<SYMBOL>/<SYMBOL>-metrics-<DATE>.zip
Each file has one row every five minutes: open interest, open-interest value, the top-trader
long/short account ratio and the taker buy/sell volume ratio. The price used here is
open-interest value / open interest, a mark-price proxy. Nothing raw is written to disk.

Everything is causal: a level at time t uses rows at or before t, and the forward return
starts ONE BAR LATER (entry at t + 5 min; exits at t + 20 min and t + 65 min).

A KNOWN WRINKLE, REPORTED RATHER THAN HIDDEN
The shipped inputs were built from the archive as downloaded on 2026-07-29. Binance has since
RELABELLED the timestamps of rows from about April 2025 onward by -5 minutes: identical values,
moved labels, every column moving together (so features and outcomes stay aligned). A fresh
rebuild therefore samples a 15-minute grid that is one bar off in that period. This script
shows it from public data alone: each shipped row is compared with the rebuild at the same
timestamp AND at the timestamp five minutes earlier.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from jev import binance_archive
from jev.buckets import (
    level_from_percentile,
    level_from_range_position,
    level_from_zscore,
    trailing_percentile,
)
from jev.wording_intraday import INTRADAY_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FEATURES = list(INTRADAY_FEATURES)

FIRST_DAY = "2023-11-15"  # six weeks of warm-up before the first decision
LAST_DAY = "2026-06-30"
FIRST_DECISION = "2024-01-01"
COLUMNS_NEEDED = [
    "create_time",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]

BARS_PER_DAY = 288  # five-minute bars
SEVEN_DAYS = 7 * BARS_PER_DAY
THIRTY_DAYS = 30 * BARS_PER_DAY


def download_one_day(job):
    symbol, day = job
    content = binance_archive.download(binance_archive.metrics_url(f"{symbol}USDT", day.date()), timeout=60)
    if content is None:
        return None
    rows = binance_archive.read_zipped_csv(content, usecols=COLUMNS_NEEDED)
    rows["symbol"] = symbol
    return rows


def five_minute_grid(rows):
    """One symbol's rows on a complete five-minute grid (missing bars become NaN)."""
    rows = rows.copy()
    rows["create_time"] = pd.to_datetime(rows["create_time"]).dt.floor("5min")
    rows = rows.drop_duplicates("create_time").set_index("create_time").sort_index()
    return rows.reindex(pd.date_range(rows.index.min(), rows.index.max(), freq="5min"))


def levels_and_outcomes(grid, symbol):
    """Six feature levels and two forward returns for every five-minute bar of one symbol."""
    price = grid["sum_open_interest_value"] / grid["sum_open_interest"]
    log_price = np.log(price)

    def versus_last_7_days(series):
        return level_from_percentile(trailing_percentile(series, SEVEN_DAYS, minimum_history=1000))

    taker_flow_30m = (
        np.log(grid["sum_taker_long_short_vol_ratio"].clip(lower=1e-6)).rolling(6, min_periods=4).mean()
    )
    open_interest_change_1h = np.log(grid["sum_open_interest"]) - np.log(grid["sum_open_interest"].shift(12))

    top_traders = grid["count_toptrader_long_short_ratio"]
    trailing = top_traders.rolling(THIRTY_DAYS, min_periods=4000)
    top_trader_zscore = (top_traders - trailing.mean()) / trailing.std()

    low_24h = price.rolling(BARS_PER_DAY, min_periods=200).min()
    high_24h = price.rolling(BARS_PER_DAY, min_periods=200).max()
    position_in_range = (price - low_24h) / (high_24h - low_24h).replace(0, np.nan)

    table = pd.DataFrame(
        {
            "price_15m": versus_last_7_days(log_price - log_price.shift(3)),
            "price_4h": versus_last_7_days(log_price - log_price.shift(48)),
            "taker_flow_30m": versus_last_7_days(taker_flow_30m),
            "open_interest_1h": versus_last_7_days(open_interest_change_1h),
            "top_trader_accounts": level_from_zscore(top_trader_zscore),
            "range_24h": level_from_range_position(position_in_range),
            # enter one bar after the decision (t + 5 min), exit 15 / 60 minutes after entry
            "ret_15m_bp": (log_price.shift(-4) - log_price.shift(-1)) * 1e4,
            "ret_60m_bp": (log_price.shift(-13) - log_price.shift(-1)) * 1e4,
        }
    )
    table.index.name = "ts"
    table = table.reset_index()
    table["symbol"] = symbol
    return table


def share_of_shipped_rows_reproduced(shipped, rebuilt, minutes_earlier):
    """For each shipped row stamped t: does the rebuild agree at t minus `minutes_earlier`?"""
    candidate = rebuilt.copy()
    candidate["ts"] = candidate["ts"] + pd.Timedelta(minutes=minutes_earlier)
    both = shipped.merge(candidate, on=["ts", "symbol"], how="left", suffixes=("", "_rebuilt"))

    same_levels = (both[FEATURES].values == both[[f"{f}_rebuilt" for f in FEATURES]].values).all(axis=1)
    same_outcome = ((both["ret_15m_bp"] - both["ret_15m_bp_rebuilt"]).abs() < 1e-6) | (
        both["ret_15m_bp"].isna() & both["ret_15m_bp_rebuilt"].isna()
    )
    return pd.Series(same_levels, index=both["ts"]), pd.Series(same_outcome.values, index=both["ts"])


def main():
    symbols = json.load(open(f"{DATA}/intraday_symbols.json"))
    jobs = [(symbol, day) for symbol in symbols for day in pd.date_range(FIRST_DAY, LAST_DAY, freq="D")]
    print(f"{len(jobs):,} daily files to stream", flush=True)

    parts = []
    with ThreadPoolExecutor(12) as pool:
        for done, rows in enumerate(pool.map(download_one_day, jobs), start=1):
            if rows is not None:
                parts.append(rows)
            if done % 1000 == 0:
                print(f"   {done:,}/{len(jobs):,}", flush=True)
    raw = pd.concat(parts, ignore_index=True)
    print(f"rows {len(raw):,}; files missing from the archive: {len(jobs) - len(parts)}", flush=True)

    every_bar = pd.concat(
        [
            levels_and_outcomes(five_minute_grid(raw[raw["symbol"] == s].drop(columns="symbol")), s)
            for s in symbols
        ],
        ignore_index=True,
    )
    every_bar = every_bar[every_bar["ts"] >= "2023-12-31"].dropna(subset=FEATURES)
    every_bar[FEATURES] = every_bar[FEATURES].astype("int8")

    # The decision clock: every 15 minutes from 2024-01-01.
    on_the_clock = (every_bar["ts"] >= FIRST_DECISION) & (every_bar["ts"].dt.minute % 15 == 0)
    decisions = every_bar[on_the_clock].reset_index(drop=True)
    decisions[["ts"] + FEATURES + ["symbol"]].to_parquet(
        f"{DATA}/rebuilt_intraday_levels.parquet", index=False
    )
    decisions[["ts", "symbol", "ret_15m_bp", "ret_60m_bp"]].to_parquet(
        f"{DATA}/rebuilt_intraday_outcomes.parquet", index=False
    )
    print(
        f"wrote {len(decisions):,} rebuilt decisions -> now run: python 12_intraday_evaluate.py --rebuilt",
        flush=True,
    )

    # Compare with the shipped files, at the same timestamp and five minutes earlier.
    shipped = pd.read_parquet(f"{DATA}/intraday_levels.parquet").merge(
        pd.read_parquet(f"{DATA}/intraday_outcomes.parquet"), on=["ts", "symbol"]
    )
    levels_same, outcome_same = share_of_shipped_rows_reproduced(shipped, every_bar, minutes_earlier=0)
    levels_5min, outcome_5min = share_of_shipped_rows_reproduced(shipped, every_bar, minutes_earlier=5)

    by_quarter = (
        pd.DataFrame(
            {
                "outcome: same stamp %": outcome_same,
                "outcome: 5 min earlier %": outcome_5min,
                "outcome: either %": outcome_same | outcome_5min,
                "levels: same stamp %": levels_same,
                "levels: 5 min earlier %": levels_5min,
                "levels: either %": levels_same | levels_5min,
            }
        )
        .groupby(pd.Grouper(freq="QS"))
        .mean()
        * 100
    )
    print("\nshare of SHIPPED rows reproduced by the rebuild, by quarter:")
    print(by_quarter.round(2).to_string())
    print(
        f"\nOVERALL: outcomes reproduced {100 * (outcome_same | outcome_5min).mean():.3f}%,"
        f" all six levels reproduced {100 * (levels_same | levels_5min).mean():.3f}%"
        "\n(levels can differ for a few weeks after the relabelling date, where trailing windows straddle it)"
    )


if __name__ == "__main__":
    main()
