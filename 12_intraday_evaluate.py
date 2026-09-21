#!/usr/bin/env python3
"""15-60 minute experiment, step 3: did the model's decisions make money?

    python 12_intraday_evaluate.py              (no API key, no downloads)
    python 12_intraday_evaluate.py --rebuilt    (use the inputs you rebuilt with step 1)

What happens here
-----------------
1. Every 15 minutes, for six liquid Binance perpetuals (2024-01 .. 2026-06), we know the
   six feature levels (data/intraday_levels.parquet).
2. We look that state up in the model's answer table (data/intraday_table.parquet).
3. Frozen trade rule: buy when P(buy) - P(sell) >= 0.25, sell when it is <= -0.25.
4. We enter ONE 5-minute bar after the decision and hold for 15 or 60 minutes
   (data/intraday_outcomes.parquet holds those forward returns, in basis points).
5. We report the hit rate and the average GROSS profit per trade, and compare the profit
   with a null in which the same decisions are made at the wrong time.
6. Ceiling check: a gradient-boosted model is FITTED on the first half of the data and
   tested on the second half, on the same six features. If even a fitted model finds
   nothing, the inputs carry nothing - and no zero-shot model could find it either.

For scale: a round trip on Binance costs about 8.4 bp with maker orders (including measured
adverse selection) and about 10.5 bp with taker orders.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import roc_auc_score

from jev.scoring import ACT_THRESHOLD, print_headline
from jev.wording_intraday import INTRADAY_FEATURES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FEATURES = list(INTRADAY_FEATURES)

MAKER_ROUND_TRIP_BP = 8.42
TAKER_ROUND_TRIP_BP = 10.5
SECOND_HALF_STARTS = pd.Timestamp("2025-04-01")
NULL_DRAWS = 500
SMALLEST_SHIFT = 96  # bars; one day of 15-minute decisions
HORIZONS = {"15 min": "ret_15m_bp", "60 min": "ret_60m_bp"}


def load_decisions(use_rebuilt_inputs):
    """One row per decision: the six levels, the model's answers, and what happened next."""
    prefix = "rebuilt_" if use_rebuilt_inputs else ""
    print(f"inputs: data/{prefix}intraday_levels.parquet + data/{prefix}intraday_outcomes.parquet")

    levels = pd.read_parquet(f"{DATA}/{prefix}intraday_levels.parquet")
    answers = pd.read_parquet(f"{DATA}/intraday_table.parquet")
    outcomes = pd.read_parquet(f"{DATA}/{prefix}intraday_outcomes.parquet")

    decisions = levels.merge(answers, on=FEATURES, how="left")
    decisions["score"] = decisions["p_buy"] - decisions["p_sell"]
    decisions = decisions.merge(outcomes, on=["ts", "symbol"], how="left")
    decisions = decisions.dropna(subset=list(HORIZONS.values()))
    decisions["day"] = decisions["ts"].dt.normalize()
    return decisions


def trade_statistics(decisions, return_column):
    """Trades taken, average gross profit (bp), day-clustered t-statistic, and hit rate.

    Hit rate here = share of trades that made money. (At 15 minutes a flat outcome is rare,
    so this is the same as the hit rate over non-flat outcomes to within 0.1 point.)
    """
    trades = decisions[decisions["score"].abs() >= ACT_THRESHOLD]
    profit_bp = np.sign(trades["score"]) * trades[return_column]

    daily_mean = profit_bp.groupby(trades["day"]).mean()
    t_statistic = daily_mean.mean() / daily_mean.std(ddof=1) * np.sqrt(len(daily_mean))
    return len(trades), float(profit_bp.mean()), float(t_statistic), float((profit_bp > 0).mean())


def time_shift_null(decisions, return_column, rng):
    """Gross profit of the same decisions made at the WRONG time.

    Each contract's series of scores is rotated in time by a random amount (at least a day).
    That keeps how often the model trades and how long it stays on one side, and destroys
    only the alignment between its decisions and the market.
    """
    per_symbol = [
        (group["score"].to_numpy(), group[return_column].to_numpy())
        for _, group in decisions.groupby("symbol")
    ]

    null_profits = []
    for _ in range(NULL_DRAWS):
        total_profit = 0.0
        total_trades = 0
        for score, forward_return in per_symbol:
            shift = int(rng.integers(SMALLEST_SHIFT, len(score) - SMALLEST_SHIFT))
            shifted_score = np.roll(score, shift)
            traded = np.abs(shifted_score) >= ACT_THRESHOLD
            total_profit += float((np.sign(shifted_score[traded]) * forward_return[traded]).sum())
            total_trades += int(traded.sum())
        null_profits.append(total_profit / total_trades)
    return np.array(null_profits)


def fitted_model_ceiling(decisions, return_column):
    """Fit XGBoost on the first half, trade its top and bottom 10% of predictions on the second."""
    train = decisions[decisions["ts"] < SECOND_HALF_STARTS]
    test = decisions[decisions["ts"] >= SECOND_HALF_STARTS]

    model = xgboost.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.8,
        min_child_weight=200,
        n_jobs=2,
        verbosity=0,
    )
    model.fit(train[FEATURES].values, train[return_column].clip(-200, 200).values)
    prediction = model.predict(test[FEATURES].values)

    realised = test[return_column].values
    go_long = prediction >= np.quantile(prediction, 0.9)
    go_short = prediction <= np.quantile(prediction, 0.1)
    gross_bp = (realised[go_long].sum() - realised[go_short].sum()) / (go_long.sum() + go_short.sum())
    correlation = np.corrcoef(prediction, realised)[0, 1]
    return float(gross_bp), float(correlation)


def main():
    rng = np.random.default_rng(20260920)
    decisions = load_decisions(use_rebuilt_inputs="--rebuilt" in sys.argv)
    symbols = json.load(open(f"{DATA}/intraday_symbols.json"))
    headline = {"decisions": len(decisions)}

    print(f"contracts {symbols}")
    print(f"decisions {len(decisions):,}   from {decisions['ts'].min()} to {decisions['ts'].max()}")
    for label, column in HORIZONS.items():
        print(f"typical move over {label}: {decisions[column].abs().mean():.1f} bp")
    print(
        f"the model's average P(up in 15 min) is {decisions['p_up_15m'].mean():.3f};"
        f" the price actually rose {100 * (decisions['ret_15m_bp'] > 0).mean():.1f}% of the time"
    )

    print("\n1. CAN IT TELL UP FROM DOWN?   (AUC: 0.500 = coin flip, 1.000 = perfect)")
    for probability, (label, column) in zip(["p_up_15m", "p_up_60m"], HORIZONS.items()):
        moved = decisions[decisions[column] != 0]
        auc = roc_auc_score((moved[column] > 0).astype(int), moved[probability])
        print(f"   P(up) vs what happened over {label}: AUC {auc:.4f}")
        headline[f"auc_{column}"] = round(float(auc), 4)

    print("\n2. THE TRADE RULE   (act when |P(buy) - P(sell)| >= 0.25; enter one 5-minute bar late)")
    halves = {
        "whole period": decisions,
        "first half": decisions[decisions["ts"] < SECOND_HALF_STARTS],
        "second half": decisions[decisions["ts"] >= SECOND_HALF_STARTS],
    }
    for label, column in HORIZONS.items():
        print(f"   holding {label}:")
        for period, subset in halves.items():
            trades, gross, t_statistic, hit_rate = trade_statistics(subset, column)
            print(
                f"      {period:<13s} {trades:>8,} trades   right {100 * hit_rate:5.2f}% of the time"
                f"   gross {gross:+.3f} bp/trade (t {t_statistic:+.2f})"
                f"   after maker costs {gross - MAKER_ROUND_TRIP_BP:+.2f}"
                f"   after taker costs {gross - TAKER_ROUND_TRIP_BP:+.2f}"
            )

        trades, gross, _, hit_rate = trade_statistics(decisions, column)
        null_profits = time_shift_null(decisions, column, rng)
        z = (gross - null_profits.mean()) / null_profits.std()
        print(
            f"      same decisions at the wrong time:"
            f" {null_profits.mean():+.3f} +/- {null_profits.std():.3f} bp"
            f"   ->   the real result is {z:+.2f} standard deviations from that"
        )
        headline[f"trades_{column}"] = trades
        headline[f"hit_rate_pct_{column}"] = round(100 * hit_rate, 2)
        headline[f"gross_bp_{column}"] = round(gross, 3)
        headline[f"null_z_{column}"] = round(float(z), 2)

    print("\n3. CEILING CHECK   (XGBoost FITTED on the first half, tested on the second, same six features)")
    for label, column in HORIZONS.items():
        gross, correlation = fitted_model_ceiling(decisions, column)
        print(
            f"   holding {label}: trading its top/bottom 10% of predictions grosses {gross:+.3f} bp/trade"
            f"   (correlation of prediction with outcome: {correlation:+.4f})"
        )
        headline[f"fitted_model_gross_bp_{column}"] = round(gross, 3)
    print(
        f"   Costs are {MAKER_ROUND_TRIP_BP}-{TAKER_ROUND_TRIP_BP} bp."
        " These inputs do not predict the next 15-60 minutes."
    )

    print_headline(headline)


if __name__ == "__main__":
    main()
