"""FROZEN WORDING - the tape test (bars of 50 trades, about one second each).

Everything the model is told, and everything it is asked, is in this file. The wording was
written before any model answer was compared with any market outcome, and it has not been
edited since: one wording, one run. (Re-wording a question after seeing results would be
fitting the prompt to the data. The response cache is keyed on these exact strings, so any
edit would show up as cache misses.)

No dates, no symbols and no price levels are sent, so the model cannot recognise the period
and recall what happened next. All arithmetic stays in code; the model only sees which of
five levels each feature is at.

Every level below is the percentile of the feature versus the trailing 2,000 bars:
bottom 10% -> 0, next 20% -> 1, middle 40% -> 2, next 20% -> 3, top 10% -> 4.
Six features x five levels = 15,625 possible states, each priced once.
"""

INSTRUMENT = "a cryptocurrency perpetual futures contract on a large exchange"

TAPE_FEATURES = {
    # Signed taker volume of the last bar (50 trades): buys minus sells, as a share of volume.
    "flow_last_50_trades": [
        "aggressive selling dominated the last fifty trades",  # 0: far below usual
        "sellers were somewhat more aggressive than buyers in the last fifty trades",  # 1: below usual
        "aggressive buying and selling were balanced in the last fifty trades",  # 2: usual
        "buyers were somewhat more aggressive than sellers in the last fifty trades",  # 3: above usual
        "aggressive buying dominated the last fifty trades",  # 4: far above usual
    ],
    # The same over the last 20 bars (1,000 trades).
    "flow_last_1000_trades": [
        "aggressive selling has dominated the last thousand trades",  # 0: far below usual
        "sellers have been somewhat more aggressive over the last thousand trades",  # 1: below usual
        "aggressive buying and selling have been balanced over the last thousand trades",  # 2: usual
        "buyers have been somewhat more aggressive over the last thousand trades",  # 3: above usual
        "aggressive buying has dominated the last thousand trades",  # 4: far above usual
    ],
    # Return of the last bar.
    "price_last_50_trades": [
        "the price dropped sharply during the last fifty trades",  # 0: far below usual
        "the price slipped during the last fifty trades",  # 1: below usual
        "the price was unchanged during the last fifty trades",  # 2: usual
        "the price edged up during the last fifty trades",  # 3: above usual
        "the price jumped sharply during the last fifty trades",  # 4: far above usual
    ],
    # Return over the last 100 bars (roughly two minutes on BTC).
    "price_last_few_minutes": [
        "the price has fallen strongly over the last few minutes",  # 0: far below usual
        "the price has drifted down over the last few minutes",  # 1: below usual
        "the price has gone sideways over the last few minutes",  # 2: usual
        "the price has drifted up over the last few minutes",  # 3: above usual
        "the price has risen strongly over the last few minutes",  # 4: far above usual
    ],
    # Bars per second over the last 5 bars: how fast trades are arriving.
    "tape_speed": [
        "trades are arriving much more slowly than usual",  # 0: far below usual
        "trades are arriving somewhat more slowly than usual",  # 1: below usual
        "trades are arriving at the usual pace",  # 2: usual
        "trades are arriving somewhat faster than usual",  # 3: above usual
        "trades are arriving much faster than usual",  # 4: far above usual
    ],
    # Standard deviation of bar returns over the last 20 bars.
    "volatility": [
        "price swings are much smaller than usual",  # 0: far below usual
        "price swings are somewhat smaller than usual",  # 1: below usual
        "price swings are typical",  # 2: usual
        "price swings are somewhat larger than usual",  # 3: above usual
        "price swings are much larger than usual",  # 4: far above usual
    ],
}

TAPE_QUESTIONS = {
    "up50": {
        "type": "noul",
        "instructions": "Will the price of this contract be higher after the next fifty trades than it is now?",
        "criteria": {
            "true": "The price is higher after the next fifty trades.",
            "false": "The price is lower after the next fifty trades.",
        },
    },
    "up500": {
        "type": "noul",
        "instructions": "Will the price of this contract be higher after the next five hundred trades than it is now?",
        "criteria": {
            "true": "The price is higher after the next five hundred trades.",
            "false": "The price is lower after the next five hundred trades.",
        },
    },
    "action": {
        "type": "choice",
        "instructions": "What is the best action for a high-frequency trader in this contract right now?",
        "criteria": {
            "buy": "Buy now for a gain over the next few seconds.",
            "sell": "Sell short now for a gain over the next few seconds.",
            "wait": "Do nothing; there is no edge right now.",
        },
    },
}


def build_state(levels):
    """The JSON state sent to the model for one tuple of levels (one level per feature)."""
    assert len(levels) == len(TAPE_FEATURES) and all(0 <= int(level) <= 4 for level in levels)
    state = {"instrument": INSTRUMENT}
    for feature, level in zip(TAPE_FEATURES, levels):
        state[feature] = TAPE_FEATURES[feature][int(level)]
    return state
