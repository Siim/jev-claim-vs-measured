#!/usr/bin/env python3
"""15-60 minute experiment, step 2: ask the model about EVERY possible state, once.

Six features x five levels = 15,625 states. The answers are written to
data/intraday_table.parquet: that table is the model's complete trading behaviour on these
inputs, and the evaluation (step 3) simply looks each market moment up in it.

    python 11_intraday_price_all_states.py

With the shipped cache this makes zero API calls and needs no key.
"""

import itertools
import os

import pandas as pd

from jev.client import ask_many
from jev.scoring import ACT_THRESHOLD, print_headline
from jev.wording_intraday import INTRADAY_FEATURES, INTRADAY_QUESTIONS, build_state

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "data", "intraday_table.parquet")


def main():
    every_state = list(itertools.product(range(5), repeat=len(INTRADAY_FEATURES)))
    replies = ask_many(
        [(build_state(levels), INTRADAY_QUESTIONS) for levels in every_state],
        max_parallel=14,
        progress_every=2000,
    )

    rows = []
    for levels, reply in zip(every_state, replies):
        if "error" in reply:
            continue
        answers = reply["answers"]
        action = answers["action"]["probabilities"]
        row = dict(zip(INTRADAY_FEATURES, levels))
        row["p_up_15m"] = answers["up15"]["noul"]
        row["p_up_60m"] = answers["up60"]["noul"]
        row["p_buy"] = action["buy"]
        row["p_sell"] = action["sell"]
        row["p_wait"] = action["wait"]
        row["action_confidence"] = answers["action"]["confidence"]
        rows.append(row)
    table = pd.DataFrame(rows)

    if os.path.exists(TABLE_PATH):
        shipped = pd.read_parquet(TABLE_PATH)[list(table.columns)].reset_index(drop=True)
        print("table identical to the shipped one:", shipped.equals(table))
    table.to_parquet(TABLE_PATH, index=False)

    score = table["p_buy"] - table["p_sell"]
    buy_states = int((score >= ACT_THRESHOLD).sum())
    sell_states = int((score <= -ACT_THRESHOLD).sum())
    fresh_calls = sum(1 for reply in replies if not reply.get("from_cache"))

    print(
        f"states priced               {len(table):,}"
        f"  (failed: {len(every_state) - len(table)}, fresh API calls: {fresh_calls})"
    )
    print(f"states where it would BUY   {buy_states:,}")
    print(f"states where it would SELL  {sell_states:,}   <- the inputs are symmetric, the answers are not")
    print(f"mean P(up in 15 min)        {table['p_up_15m'].mean():.3f}   <- the true base rate is 0.50")

    print_headline(
        {
            "states": len(table),
            "buy_states": buy_states,
            "sell_states": sell_states,
            "mean_p_up_15m": round(float(table["p_up_15m"].mean()), 3),
        }
    )


if __name__ == "__main__":
    main()
