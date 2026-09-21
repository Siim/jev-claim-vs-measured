#!/usr/bin/env python3
"""Tape experiment, step 3: the model on raw trades, one decision about every second.

    python 22_tape_evaluate.py              (no API key, no downloads)
    python 22_tape_evaluate.py --rebuilt    (use the inputs you rebuilt with step 1)

This is the scale the "HFT" claim is made at. 15.5 million raw trades on BTCUSDT and ETHUSDT
(2026-09-14 .. 09-18) were grouped into bars of 50 trades - about one second each. At the
close of every bar the model's answer is looked up and the frozen trade rule applied.

Two entry assumptions are reported:
    no delay      enter at the decision bar's close. Zero latency: the model's best case,
                  and only possible with a precomputed lookup table.
    1 bar late    enter one bar (~1 s) later: a live API call (0.3-0.5 s) plus an order.

Three yardsticks are computed on the identical bars:
    ONE-LINE RULE   "follow the last 50 trades when their flow is extreme". No model at all.
    FITTED MODEL    XGBoost fitted on days 1-3 and tested on days 4-5.
    PERFECT ORACLE  the average absolute move: what a predictor that is NEVER wrong would earn.

For scale: a round trip costs 4 bp with maker orders (before adverse selection) and 10 bp with
taker orders. The hit rate is measured over outcomes that were not flat.
"""

import os
import sys

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score

from jev.scoring import print_headline, signal_from_score, summarise_trades, time_shift_null
from jev.tape_bars import ENTRY_DELAYS, FEATURE_NAMES, HOLDING_PERIODS, return_column

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

MAKER_ROUND_TRIP_BP = 4.0
TAKER_ROUND_TRIP_BP = 10.0
DELAY_NAME = {0: "no delay", 1: "1 bar late"}


def bars_label(count):
    return f"{count} bar" if count == 1 else f"{count} bars"


def load_decisions(use_rebuilt_inputs):
    prefix = "rebuilt_" if use_rebuilt_inputs else ""
    print(f"inputs: data/{prefix}tape_levels.parquet")
    bars = pd.read_parquet(f"{DATA}/{prefix}tape_levels.parquet")
    answers = pd.read_parquet(f"{DATA}/tape_table.parquet")

    decisions = bars.merge(answers, on=FEATURE_NAMES, how="left")
    decisions["score"] = decisions["p_buy"] - decisions["p_sell"]
    assert decisions["score"].notna().all(), "a tape state is missing from the answer table"
    return decisions.sort_values(["symbol", "ts"]).reset_index(drop=True)


def one_line_rule(decisions):
    """Buy after a bar of extreme buying, sell after a bar of extreme selling. That is all."""
    flow = decisions["flow_last_50_trades"]
    return np.where(flow == 4, 1, np.where(flow == 0, -1, 0))


def fitted_model_signal(decisions, forward_return_bp, train_rows, test_rows):
    """XGBoost fitted on the training days; trade its top and bottom 10% of test predictions."""
    model = xgboost.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=300,
        n_jobs=2,
        verbosity=0,
    )
    features = decisions[FEATURE_NAMES].values
    model.fit(features[train_rows], np.clip(forward_return_bp[train_rows], -50, 50))
    prediction = model.predict(features[test_rows])
    low, high = np.quantile(prediction, [0.1, 0.9])
    return np.where(prediction >= high, 1, np.where(prediction <= low, -1, 0))


def main():
    rng = np.random.default_rng(20260921)
    decisions = load_decisions(use_rebuilt_inputs="--rebuilt" in sys.argv)

    model_signal = signal_from_score(decisions["score"])
    rule_signal = one_line_rule(decisions)
    symbol_day = (decisions["symbol"] + decisions["day"]).values

    days = sorted(decisions["day"].unique())
    train_rows = decisions["day"].isin(days[:3]).values
    test_rows = ~train_rows

    bar_seconds = decisions["bar_seconds"].median()
    print(
        f"decision bars {len(decisions):,}  ("
        + ", ".join(f"{s} {n:,}" for s, n in decisions["symbol"].value_counts().sort_index().items())
        + ")"
    )
    print(
        f"median bar {bar_seconds:.2f} s, so holding 1 / 10 / 50 bars is about"
        f" {bar_seconds:.0f} s / {10 * bar_seconds:.0f} s / {50 * bar_seconds:.0f} s"
    )
    print(
        f"the model wants to act on {100 * (model_signal != 0).mean():.1f}% of bars"
        f" (BUY {100 * (model_signal == 1).mean():.1f}%, SELL {100 * (model_signal == -1).mean():.1f}%);"
        f" its average P(up) is {decisions['p_up_50_trades'].mean():.3f}"
    )

    print(
        "\nright % = hit rate over non-flat outcomes; gross = average profit per trade in bp, before any cost"
    )
    print(
        f"{'entry':<11}{'hold':<9}| {'ORACLE':>7} |"
        f" {'MODEL right%':>12} {'gross':>8} {'z':>6} |"
        f" {'ONE-LINE RULE right%':>20} {'gross':>8} |"
        f" {'FITTED right%':>13} {'gross':>8}"
    )
    headline = {"decisions": len(decisions)}
    for delay in ENTRY_DELAYS:
        for hold in HOLDING_PERIODS:
            forward_return = decisions[return_column(delay, hold)].values.astype(float)

            model = summarise_trades(model_signal, forward_return)
            z, _ = time_shift_null(model_signal, forward_return, symbol_day, rng)
            rule = summarise_trades(rule_signal, forward_return)
            fitted = summarise_trades(
                fitted_model_signal(decisions, forward_return, train_rows, test_rows),
                forward_return[test_rows],
            )
            oracle_bp = np.abs(forward_return).mean()

            print(
                f"{DELAY_NAME[delay]:<11}{bars_label(hold):<9}| {oracle_bp:>7.2f} |"
                f" {100 * model.hit_rate:>12.2f} {model.gross_bp:>+8.3f} {z:>+6.1f} |"
                f" {100 * rule.hit_rate:>20.2f} {rule.gross_bp:>+8.3f} |"
                f" {100 * fitted.hit_rate:>13.2f} {fitted.gross_bp:>+8.3f}"
            )
            key = f"delay{delay}_hold{hold}"
            headline[f"oracle_bp_{key}"] = round(float(oracle_bp), 2)
            headline[f"model_hit_pct_{key}"] = round(100 * model.hit_rate, 2)
            headline[f"model_gross_bp_{key}"] = round(model.gross_bp, 3)
            headline[f"rule_hit_pct_{key}"] = round(100 * rule.hit_rate, 2)
            headline[f"rule_gross_bp_{key}"] = round(rule.gross_bp, 3)
    headline["model_trades"] = int((model_signal != 0).sum())

    print("\nthe model, 1 bar late, by coin:")
    for symbol in sorted(decisions["symbol"].unique()):
        rows = (decisions["symbol"] == symbol).values
        parts = []
        for hold in HOLDING_PERIODS:
            result = summarise_trades(model_signal[rows], decisions.loc[rows, return_column(1, hold)].values)
            parts.append(
                f"{bars_label(hold)}: right {100 * result.hit_rate:.2f}%, gross {result.gross_bp:+.3f} bp"
            )
        print(f"   {symbol}: " + "   ".join(parts))

    late = decisions[decisions[return_column(1, 1)] != 0]
    auc = roc_auc_score((late[return_column(1, 1)] > 0).astype(int), late["p_up_50_trades"])
    print(
        f"\nAUC of P(up after 50 trades) against what happened, 1 bar late: {auc:.4f}   (0.500 = coin flip)"
    )

    best = max(headline[f"model_gross_bp_delay1_hold{hold}"] for hold in HOLDING_PERIODS)
    print(
        f"\nRULE FIXED IN ADVANCE: the model must gross more than {MAKER_ROUND_TRIP_BP} bp per trade,"
        " 1 bar late."
    )
    print(f"Its best is {best:+.3f} bp -> FAIL, short by a factor of {MAKER_ROUND_TRIP_BP / best:.0f}.")
    print(
        f"Even the perfect oracle grosses only {headline['oracle_bp_delay1_hold1']}"
        f" / {headline['oracle_bp_delay1_hold10']} / {headline['oracle_bp_delay1_hold50']} bp"
        f" against {TAKER_ROUND_TRIP_BP:.0f} bp of taker costs."
    )

    print_headline(headline)


if __name__ == "__main__":
    main()
