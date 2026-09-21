"""FROZEN vocabulary for the intraday test (T3) and the text-triage test (T4).

Why buckets: the vendor's own docs say jev-1.13 is weak at numeric precision and recommend
passing named buckets, keeping all arithmetic in code. Why anonymous: no dates, no symbols,
no price levels are sent, so the model cannot recognise the period and look the answer up.

Every T3 feature has five ordered levels 0..4. A state is a tuple of six levels, so the
model's whole decision function is a FINITE TABLE of 5**6 = 15,625 rows. It was priced once.
The wording below was frozen BEFORE any model output was joined to any market outcome and
was never edited afterwards. (The response cache is keyed by a hash of these exact strings.)
"""
INSTRUMENT = 'a cryptocurrency perpetual futures contract on a large exchange'

T3 = {
 "price_15m": [
  "the price dropped sharply in the last fifteen minutes",
  "the price slipped in the last fifteen minutes",
  "the price was flat in the last fifteen minutes",
  "the price edged up in the last fifteen minutes",
  "the price jumped sharply in the last fifteen minutes"
 ],
 "price_4h": [
  "the price has fallen strongly over the last four hours",
  "the price has drifted down over the last four hours",
  "the price has gone sideways over the last four hours",
  "the price has drifted up over the last four hours",
  "the price has risen strongly over the last four hours"
 ],
 "taker_flow_30m": [
  "aggressive selling has dominated the last thirty minutes",
  "sellers have been somewhat more aggressive than buyers",
  "aggressive buying and selling have been balanced",
  "buyers have been somewhat more aggressive than sellers",
  "aggressive buying has dominated the last thirty minutes"
 ],
 "open_interest_1h": [
  "open interest dropped sharply in the last hour (positions are being closed)",
  "open interest declined in the last hour",
  "open interest was steady in the last hour",
  "open interest increased in the last hour",
  "open interest jumped sharply in the last hour (new positions are being opened)"
 ],
 "top_trader_accounts": [
  "far more top-trader accounts are short than is usual for this contract",
  "somewhat more top-trader accounts are short than usual",
  "top-trader accounts are positioned about as usual",
  "somewhat more top-trader accounts are long than usual",
  "far more top-trader accounts are long than is usual for this contract"
 ],
 "range_24h": [
  "the price is at the bottom of its range of the last twenty-four hours",
  "the price is in the lower part of its range of the last twenty-four hours",
  "the price is in the middle of its range of the last twenty-four hours",
  "the price is in the upper part of its range of the last twenty-four hours",
  "the price is at the top of its range of the last twenty-four hours"
 ]
}

T3_Q = {
 "up15": {
  "type": "noul",
  "instructions": "Will the price of this contract be higher fifteen minutes from now than it is now?",
  "criteria": {
   "true": "The price is higher fifteen minutes from now.",
   "false": "The price is lower fifteen minutes from now."
  }
 },
 "up60": {
  "type": "noul",
  "instructions": "Will the price of this contract be higher sixty minutes from now than it is now?",
  "criteria": {
   "true": "The price is higher sixty minutes from now.",
   "false": "The price is lower sixty minutes from now."
  }
 },
 "action": {
  "type": "choice",
  "instructions": "What is the best action for a short-term trader in this contract right now?",
  "criteria": {
   "buy": "Buy now for a short-term gain.",
   "sell": "Sell short now for a short-term gain.",
   "wait": "Do nothing; there is no short-term edge right now."
  }
 }
}

T4_Q = {
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
   "other": "Anything else."
  }
 },
 "direction": {
  "type": "choice",
  "instructions": "In which direction is the price of the named token most likely to move right after this announcement?",
  "criteria": {
   "up": "The price is likely to rise.",
   "down": "The price is likely to fall.",
   "unclear": "No clear direction, or no specific token is named."
  }
 },
 "impact": {
  "type": "score",
  "instructions": "How large an immediate price move in a specific token is this announcement likely to cause?",
  "criteria": [
   "No effect on the price of any token.",
   "A minor price effect.",
   "A large immediate price move in a specific token."
  ]
 }
}


def build_state(spec, levels):
    keys = list(spec)
    assert len(levels) == len(keys) and all(0 <= int(l) <= 4 for l in levels)
    st = {"instrument": INSTRUMENT}
    for k, l in zip(keys, levels):
        st[k] = spec[k][int(l)]
    return st
