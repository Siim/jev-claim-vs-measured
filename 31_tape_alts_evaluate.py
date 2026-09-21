#!/usr/bin/env python3
"""Tape experiment on ALTS, step 2: the same test on six alt perpetuals, coin by coin.

    python 31_tape_alts_evaluate.py              (no API key, no downloads)
    python 31_tape_alts_evaluate.py --rebuilt    (use the inputs you rebuilt with step 1)

Why alts could be different: their moves per bar are several times larger relative to fees,
their spreads are wider, and their tapes are thinner.

The coins were picked by a rule, not by hand: the alt USDT perpetuals trading on 2026-09-21
were ranked by 24-hour volume and ranks 1-4 (ZEC, SOL, AKE, XRP) and 25 / 50 / 75 / 100
(ONDO, ZIL, FLOCK, GUN) were taken. BTC and ETH are included for reference. The model's
answers come from the SAME table as the BTC/ETH tape test, so nothing was tuned to these coins.

Prices here are a mid-price estimate (see jev/tape_bars.py). It lags the true mid slightly,
which flatters "follow the flow" rules - so every number below is generous to the model.

All results are for an entry ONE BAR LATE. The hit rate is over outcomes that were not flat.
"""

import os
import sys

import numpy as np
import pandas as pd
import xgboost

from jev.scoring import print_headline, signal_from_score, summarise_trades, time_shift_null
from jev.tape_bars import FEATURE_NAMES, HOLDING_PERIODS, return_column

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

COINS = ["BTC", "ETH", "ZEC", "SOL", "AKE", "XRP", "ONDO", "ZIL", "FLOCK", "GUN"]
REFERENCE_COINS = ["BTC", "ETH"]
FEWEST_BARS_TO_EVALUATE = 3000
MAKER_ROUND_TRIP_BP = 4.0
TAKER_FEES_ROUND_TRIP_BP = 10.0  # plus one spread, added per coin below
ENTRY_DELAY = 1


def load_decisions(use_rebuilt_inputs):
    prefix = "rebuilt_" if use_rebuilt_inputs else ""
    print(f"inputs: data/{prefix}tape_alts_levels.parquet")
    bars = pd.read_parquet(f"{DATA}/{prefix}tape_alts_levels.parquet")
    answers = pd.read_parquet(f"{DATA}/tape_table.parquet")

    decisions = bars.merge(answers, on=FEATURE_NAMES, how="left")
    decisions["score"] = decisions["p_buy"] - decisions["p_sell"]
    decisions = decisions.sort_values(["symbol", "ts"]).reset_index(drop=True)

    decisions["model_signal"] = signal_from_score(decisions["score"])
    flow = decisions["flow_last_50_trades"]
    decisions["rule_signal"] = np.where(flow == 4, 1, np.where(flow == 0, -1, 0))  # the one-line rule
    return decisions


def add_fitted_model_predictions(decisions, train_rows):
    """One XGBoost per holding period, fitted on days 1-3 of ALL coins pooled.

    Pooling is natural here because every feature is a percentile of the coin's own history.
    """
    for hold in HOLDING_PERIODS:
        target = decisions[return_column(ENTRY_DELAY, hold)].values.astype(float)
        clip_at = np.nanpercentile(np.abs(target[train_rows]), 99)
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
        model.fit(
            decisions.loc[train_rows, FEATURE_NAMES].values, np.clip(target[train_rows], -clip_at, clip_at)
        )
        decisions[f"fitted_prediction_hold{hold}"] = model.predict(decisions[FEATURE_NAMES].values)


def evaluate_coin(coin_decisions, test_days, rng):
    """Everything we report for one coin, as a dict."""
    result = {
        "bars": len(coin_decisions),
        "bar_seconds": float(coin_decisions["bar_seconds"].median()),
        "spread_bp": float(coin_decisions["spread_proxy_bp"].median()),
    }
    result["taker_cost_bp"] = TAKER_FEES_ROUND_TRIP_BP + result["spread_bp"]
    test = coin_decisions[coin_decisions["day"].isin(test_days)]

    for hold in HOLDING_PERIODS:
        forward_return = coin_decisions[return_column(ENTRY_DELAY, hold)].values.astype(float)
        result[f"oracle_{hold}"] = float(np.abs(forward_return).mean())

        model = summarise_trades(coin_decisions["model_signal"].values, forward_return)
        rule = summarise_trades(coin_decisions["rule_signal"].values, forward_return)
        result[f"model_hit_{hold}"], result[f"model_gross_{hold}"] = model.hit_rate, model.gross_bp
        result[f"rule_hit_{hold}"], result[f"rule_gross_{hold}"] = rule.hit_rate, rule.gross_bp

        prediction = test[f"fitted_prediction_hold{hold}"]
        low, high = np.quantile(prediction, [0.1, 0.9])
        fitted_signal = np.where(prediction >= high, 1, np.where(prediction <= low, -1, 0))
        result[f"fitted_gross_{hold}"] = summarise_trades(
            fitted_signal, test[return_column(ENTRY_DELAY, hold)].values
        ).gross_bp

    # How unusual is the model's profit? (same decisions, wrong time; 1-bar and 10-bar holds)
    for hold in (1, 10):
        forward_return = coin_decisions[return_column(ENTRY_DELAY, hold)].values
        z, _ = time_shift_null(
            coin_decisions["model_signal"].values, forward_return, coin_decisions["day"].values, rng
        )
        result[f"model_z_{hold}"] = float(z)

    result["oracle_beats_taker_cost_at"] = [
        hold for hold in HOLDING_PERIODS if result[f"oracle_{hold}"] > result["taker_cost_bp"]
    ]
    return result


def first_winning_hold(result):
    """From which holding period would a perfect predictor beat this coin's taker costs?"""
    winning = result["oracle_beats_taker_cost_at"]
    return f"from {winning[0]} bars" if winning else "never"


def main():
    rng = np.random.default_rng(20260921)
    decisions = load_decisions(use_rebuilt_inputs="--rebuilt" in sys.argv)
    days = sorted(decisions["day"].unique())
    add_fitted_model_predictions(decisions, train_rows=decisions["day"].isin(days[:3]).values)

    results = {}
    print(
        "\ngross = average profit per trade in bp before any cost, holding 1 / 10 / 50 bars;"
        " right% is for a 1-bar hold"
    )
    print(
        f"{'coin':<6}{'bars':>9}{'bar s':>7}{'spread':>8} | {'ORACLE 1/10/50':>20} |"
        f" {'MODEL right%':>12} {'gross 1/10/50':>22} {'z(1)':>6} {'z(10)':>6} |"
        f" {'RULE right%':>11} {'gross 1/10/50':>22} | {'FITTED gross 1/10/50':>22} | {'taker cost':>10}"
    )
    for coin in COINS:
        coin_decisions = decisions[decisions["symbol"] == coin]
        if len(coin_decisions) < FEWEST_BARS_TO_EVALUATE:
            print(
                f"{coin:<6}{len(coin_decisions):>9,}   too thin for the procedure fixed in advance"
                " (1,000 bars a day are needed to warm up) -> not evaluated"
            )
            continue

        r = evaluate_coin(coin_decisions, test_days=days[3:], rng=rng)
        results[coin] = r

        def three(prefix):
            return " ".join(f"{r[f'{prefix}_{hold}']:>+7.3f}" for hold in HOLDING_PERIODS)

        print(
            f"{coin:<6}{r['bars']:>9,}{r['bar_seconds']:>7.1f}{r['spread_bp']:>8.2f} |"
            f" {r['oracle_1']:>6.2f} {r['oracle_10']:>6.2f} {r['oracle_50']:>6.2f} |"
            f" {100 * r['model_hit_1']:>12.2f} {three('model_gross')}"
            f" {r['model_z_1']:>+6.1f} {r['model_z_10']:>+6.1f} |"
            f" {100 * r['rule_hit_1']:>11.2f} {three('rule_gross')} |"
            f" {three('fitted_gross')} | {r['taker_cost_bp']:>7.1f} bp"
        )

    alts = [coin for coin in results if coin not in REFERENCE_COINS]
    rule_wins_1 = sum(results[c]["rule_gross_1"] > results[c]["model_gross_1"] for c in results)
    rule_wins_10 = sum(results[c]["rule_gross_10"] > results[c]["model_gross_10"] for c in results)
    alts_clearing = sum(
        max(results[c][f"model_gross_{hold}"] for hold in HOLDING_PERIODS) > MAKER_ROUND_TRIP_BP for c in alts
    )

    print(
        f"\nThe one-line rule out-earns the model on {rule_wins_1} of {len(results)} coins at a 1-bar hold,"
        f" and on {rule_wins_10} of {len(results)} at 10 bars."
    )
    print(
        "Where would a PERFECT predictor beat taker costs? "
        + "; ".join(f"{c}: {first_winning_hold(results[c])}" for c in results)
    )
    print(
        f"\nRULE FIXED IN ADVANCE: the model must gross more than {MAKER_ROUND_TRIP_BP} bp per trade"
        " on at least 4 of the 8 alts."
    )
    print(f"Alts where it does: {alts_clearing} -> {'PASS' if alts_clearing >= 4 else 'FAIL'}")

    headline = {
        "decisions": len(decisions),
        "coins_evaluated": len(results),
        "rule_beats_model_at_1_bar": rule_wins_1,
        "alts_clearing_4bp": alts_clearing,
    }
    for coin in ("BTC", "ZEC", "AKE", "SOL"):
        headline[f"{coin}_model_hit_pct"] = round(100 * results[coin]["model_hit_1"], 2)
        headline[f"{coin}_model_gross_10_bp"] = round(results[coin]["model_gross_10"], 3)
        headline[f"{coin}_rule_gross_10_bp"] = round(results[coin]["rule_gross_10"], 3)
    print_headline(headline)


if __name__ == "__main__":
    main()
