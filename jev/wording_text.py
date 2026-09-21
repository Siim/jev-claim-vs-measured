"""FROZEN WORDING - the text test (exchange announcement titles).

The state is simply the announcement title. Three questions are asked about it in one request.
Written before any result was seen and not edited since.
"""

TEXT_QUESTIONS = {
    "event_type": {
        "type": "choice",
        "instructions": "What kind of exchange announcement is this title?",
        "criteria": {
            "spot_delisting": "The exchange will delist (remove) one or more tokens from spot trading.",
            "perp_delisting": "The futures exchange will delist or settle one or more perpetual or futures contracts.",
            "spot_listing": "The exchange will list a new token for spot trading.",
            "perp_launch": "The futures exchange will launch a new perpetual or futures contract.",
            "monitoring_tag": "Tokens are added to, or removed from, the Monitoring Tag or Seed Tag lists.",
            "risk_parameter_change": "Changes to leverage, margin tiers, collateral ratios, funding rate caps or intervals, tick size or price limits.",
            "pair_change": "Trading pairs, margin pairs or collateral are added or removed, without delisting the token itself.",
            "airdrop_or_launchpool": "HODLer airdrops, Launchpool, Megadrop, Launchpad, Alpha or pre-market programs for a new token.",
            "network_or_wallet": "Network upgrades, hard forks, token swaps, rebrands, redenominations, deposit or withdrawal suspensions.",
            "promotion_or_product": "Campaigns, competitions, rewards, fee promotions, new product features, Earn, Loans or Pay products.",
            "other": "Anything else.",
        },
    },
    "direction": {
        "type": "choice",
        "instructions": "In which direction is the price of the named token most likely to move right after this announcement?",
        "criteria": {
            "up": "The price is likely to rise.",
            "down": "The price is likely to fall.",
            "unclear": "No clear direction, or no specific token is named.",
        },
    },
    "impact": {
        "type": "score",
        "instructions": "How large an immediate price move in a specific token is this announcement likely to cause?",
        "criteria": [
            "No effect on the price of any token.",
            "A minor price effect.",
            "A large immediate price move in a specific token.",
        ],
    },
}
