"""FROZEN WORDING - the 15-60 minute test.

Everything the model is told, and everything it is asked, is in this file. The wording was
written before any model answer was compared with any market outcome, and it has not been
edited since: one wording, one run. (Re-wording a question after seeing results would be
fitting the prompt to the data. The response cache is keyed on these exact strings, so any
edit would show up as cache misses.)

No dates, no symbols and no price levels are sent, so the model cannot recognise the period
and recall what happened next. All arithmetic stays in code; the model only sees which of
five levels each feature is at.

Six features x five levels = 15,625 possible states. Each was priced once: that table IS the
model's trading behaviour on these inputs.
"""

INSTRUMENT = "a cryptocurrency perpetual futures contract on a large exchange"

INTRADAY_FEATURES = {
    # Return over the last 15 minutes, as a percentile of the contract's own last 7 days.
    "price_15m": [
        "the price dropped sharply in the last fifteen minutes",  # 0: far below usual
        "the price slipped in the last fifteen minutes",  # 1: below usual
        "the price was flat in the last fifteen minutes",  # 2: usual
        "the price edged up in the last fifteen minutes",  # 3: above usual
        "the price jumped sharply in the last fifteen minutes",  # 4: far above usual
    ],
    # Return over the last 4 hours, as a percentile of the contract's own last 7 days.
    "price_4h": [
        "the price has fallen strongly over the last four hours",  # 0: far below usual
        "the price has drifted down over the last four hours",  # 1: below usual
        "the price has gone sideways over the last four hours",  # 2: usual
        "the price has drifted up over the last four hours",  # 3: above usual
        "the price has risen strongly over the last four hours",  # 4: far above usual
    ],
    # Taker buy/sell volume ratio over the last 30 minutes, percentile of the last 7 days.
    "taker_flow_30m": [
        "aggressive selling has dominated the last thirty minutes",  # 0: far below usual
        "sellers have been somewhat more aggressive than buyers",  # 1: below usual
        "aggressive buying and selling have been balanced",  # 2: usual
        "buyers have been somewhat more aggressive than sellers",  # 3: above usual
        "aggressive buying has dominated the last thirty minutes",  # 4: far above usual
    ],
    # Change in open interest over the last hour, percentile of the last 7 days.
    "open_interest_1h": [
        "open interest dropped sharply in the last hour (positions are being closed)",  # 0: far below usual
        "open interest declined in the last hour",  # 1: below usual
        "open interest was steady in the last hour",  # 2: usual
        "open interest increased in the last hour",  # 3: above usual
        "open interest jumped sharply in the last hour (new positions are being opened)",  # 4: far above usual
    ],
    # Top-trader long/short account ratio as a z-score vs the last 30 days (cuts at +-0.5 and +-1.5).
    "top_trader_accounts": [
        "far more top-trader accounts are short than is usual for this contract",  # 0: far below usual
        "somewhat more top-trader accounts are short than usual",  # 1: below usual
        "top-trader accounts are positioned about as usual",  # 2: usual
        "somewhat more top-trader accounts are long than usual",  # 3: above usual
        "far more top-trader accounts are long than is usual for this contract",  # 4: far above usual
    ],
    # Where the price sits inside its 24-hour high-low range (cuts at 10 / 35 / 65 / 90%).
    "range_24h": [
        "the price is at the bottom of its range of the last twenty-four hours",  # 0: far below usual
        "the price is in the lower part of its range of the last twenty-four hours",  # 1: below usual
        "the price is in the middle of its range of the last twenty-four hours",  # 2: usual
        "the price is in the upper part of its range of the last twenty-four hours",  # 3: above usual
        "the price is at the top of its range of the last twenty-four hours",  # 4: far above usual
    ],
}

INTRADAY_QUESTIONS = {
    "up15": {
        "type": "noul",
        "instructions": "Will the price of this contract be higher fifteen minutes from now than it is now?",
        "criteria": {
            "true": "The price is higher fifteen minutes from now.",
            "false": "The price is lower fifteen minutes from now.",
        },
    },
    "up60": {
        "type": "noul",
        "instructions": "Will the price of this contract be higher sixty minutes from now than it is now?",
        "criteria": {
            "true": "The price is higher sixty minutes from now.",
            "false": "The price is lower sixty minutes from now.",
        },
    },
    "action": {
        "type": "choice",
        "instructions": "What is the best action for a short-term trader in this contract right now?",
        "criteria": {
            "buy": "Buy now for a short-term gain.",
            "sell": "Sell short now for a short-term gain.",
            "wait": "Do nothing; there is no short-term edge right now.",
        },
    },
}


def build_state(levels):
    """The JSON state sent to the model for one tuple of levels (one level per feature)."""
    assert len(levels) == len(INTRADAY_FEATURES) and all(0 <= int(level) <= 4 for level in levels)
    state = {"instrument": INSTRUMENT}
    for feature, level in zip(INTRADAY_FEATURES, levels):
        state[feature] = INTRADAY_FEATURES[feature][int(level)]
    return state
