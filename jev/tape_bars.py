"""Raw trades -> bars of 50 trades -> feature levels and forward returns.

Used by the two tape experiments. Everything is causal: a level at the close of bar *t*
uses trades up to that close only, and forward returns start at that close or later.

Two price conventions
---------------------
``last_trade``  the bar's price is its last trade. Fine for BTC and ETH, where the spread is
                one tick (about 0.01 bp) and trade prices are as good as mid prices.
``mid_proxy``   the bar's price is the mean of the last *buyer-initiated* trade (which
                happened at the ask) and the last *seller-initiated* trade (at the bid).
                Needed for small alts, where one tick is 2-3 bp and last-trade prices
                bounce between bid and ask. Their gap is reported as a spread estimate.
                Caveat: when one side has not traded for a while this proxy lags the true
                mid, which makes "follow the flow" rules look better than they are.
"""

import numpy as np
import pandas as pd

from jev import binance_archive
from jev.buckets import level_from_percentile, trailing_percentile
from jev.wording_tape import TAPE_FEATURES

TRADES_PER_BAR = 50
TRAILING_BARS = 2000  # each feature is ranked against its own last 2,000 bars ...
MINIMUM_HISTORY = 1000  # ... once at least 1,000 of them exist (so the start of each day is unused)

ENTRY_DELAYS = (0, 1)  # bars between the decision and the entry (0 = no latency at all)
HOLDING_PERIODS = (1, 10, 50)  # bars held

FEATURE_NAMES = list(TAPE_FEATURES)


def return_column(entry_delay, holding_period):
    return f"ret_bp_delay{entry_delay}_hold{holding_period}"


def load_trades(symbol, day):
    """All aggregated trades of one symbol-day, oldest first."""
    content = binance_archive.download(binance_archive.agg_trades_url(symbol, day), timeout=300)
    if content is None:
        raise SystemExit(f"{symbol} {day} is not in the Binance archive")

    raw = binance_archive.read_zipped_csv(content, header=None, low_memory=False)
    first_cell_is_a_number = str(raw.iloc[0, 0]).replace(".", "").isdigit()
    if not first_cell_is_a_number:  # some files carry a header row, some do not
        raw = raw.iloc[1:]

    # columns of the archive: 1 = price, 2 = quantity, 5 = transaction time (ms), 6 = buyer is maker
    trades = raw[[1, 2, 5, 6]]
    trades.columns = ["price", "quantity", "time_ms", "buyer_is_maker"]
    trades = trades.astype({"price": float, "quantity": float, "time_ms": "int64"})
    trades["buyer_is_maker"] = trades["buyer_is_maker"].astype(str).str.lower().isin(["true", "1"])
    return trades.sort_values("time_ms", kind="stable").reset_index(drop=True)


def build_bars(trades, price_basis):
    """Group consecutive trades into bars of 50. Returns one row per complete bar."""
    trades = trades.copy()

    if price_basis == "mid_proxy":
        # A buyer-initiated trade (the buyer is NOT the maker) happens at the ask, and vice versa.
        last_ask = trades["price"].where(~trades["buyer_is_maker"]).ffill()
        last_bid = trades["price"].where(trades["buyer_is_maker"]).ffill()
        trades["bar_price"] = (last_ask + last_bid) / 2
        trades["spread_bp"] = (last_ask - last_bid).clip(lower=0) / trades["bar_price"] * 1e4
    elif price_basis == "last_trade":
        trades["bar_price"] = trades["price"]
    else:
        raise ValueError(price_basis)

    # If the buyer was the maker, the aggressor SOLD: count that volume as negative.
    trades["signed_quantity"] = np.where(trades["buyer_is_maker"], -trades["quantity"], trades["quantity"])

    complete_bars = len(trades) // TRADES_PER_BAR
    trades["bar"] = np.arange(len(trades)) // TRADES_PER_BAR
    trades = trades[trades["bar"] < complete_bars]  # drop the unfinished last bar

    grouped = trades.groupby("bar")
    bars = pd.DataFrame(
        {
            "close": grouped["bar_price"].last(),
            "start_ms": grouped["time_ms"].first(),
            "end_ms": grouped["time_ms"].last(),
            "volume": grouped["quantity"].sum(),
            "signed_volume": grouped["signed_quantity"].sum(),
        }
    )
    if price_basis == "mid_proxy":
        bars["spread_proxy_bp"] = grouped["spread_bp"].last()
    return bars


def levels_and_outcomes(bars, symbol, day):
    """The six feature levels at each bar close, plus what the price did afterwards."""

    def level(series):
        return level_from_percentile(trailing_percentile(series, TRAILING_BARS, MINIMUM_HISTORY))

    log_price = np.log(bars["close"])
    bar_return = log_price.diff()
    seconds_for_last_5_bars = ((bars["end_ms"] - bars["start_ms"].shift(4)) / 1000.0).clip(lower=1e-3)

    table = pd.DataFrame(
        {
            "flow_last_50_trades": level(bars["signed_volume"] / bars["volume"]),
            "flow_last_1000_trades": level(
                bars["signed_volume"].rolling(20).sum() / bars["volume"].rolling(20).sum()
            ),
            "price_last_50_trades": level(bar_return),
            "price_last_few_minutes": level(log_price - log_price.shift(100)),
            "tape_speed": level(5.0 / seconds_for_last_5_bars),
            "volatility": level(bar_return.rolling(20).std()),
        }
    )

    # Forward returns in basis points: enter `delay` bars after the decision, hold `hold` bars.
    for delay in ENTRY_DELAYS:
        for hold in HOLDING_PERIODS:
            entry = log_price.shift(-delay)
            exit_ = log_price.shift(-(delay + hold))
            table[return_column(delay, hold)] = ((exit_ - entry) * 1e4).astype("float32")

    table["bar_seconds"] = ((bars["end_ms"] - bars["start_ms"]) / 1000.0).astype("float32")
    if "spread_proxy_bp" in bars:
        table["spread_proxy_bp"] = bars["spread_proxy_bp"].astype("float32")
    table["ts"] = pd.to_datetime(bars["end_ms"], unit="ms")
    table["symbol"] = symbol.replace("USDT", "")
    table["day"] = day

    # Keep bars where every level exists and the longest forward return is known.
    table = table.dropna(subset=FEATURE_NAMES + [return_column(1, 50)])
    table[FEATURE_NAMES] = table[FEATURE_NAMES].astype("int8")
    return table


def build_symbol_day(symbol, day, price_basis):
    """Download one symbol-day and return its decision bars. Prints a one-line summary."""
    trades = load_trades(symbol, day)
    bars = build_bars(trades, price_basis)
    table = levels_and_outcomes(bars, symbol, day)

    median_bar_seconds = (bars["end_ms"] - bars["start_ms"]).median() / 1000
    summary = (
        f"  {symbol:10s} {day}: {len(trades):>9,} trades -> {len(table):>7,} decision bars,"
        f" median bar {median_bar_seconds:6.2f} s"
    )
    if "spread_proxy_bp" in bars:
        summary += f", median spread estimate {bars['spread_proxy_bp'].median():5.2f} bp"
    print(summary, flush=True)
    return table
