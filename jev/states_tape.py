"""FROZEN vocabulary for the TAPE test (50-trade bars, about one second each).
Six features x five levels = 15,625 states. Each level is the percentile of the feature versus the
trailing 2,000 bars, cut at 10/30/70/90. Written before any outcome was joined; never edited."""
INSTRUMENT = 'a cryptocurrency perpetual futures contract on a large exchange'

T5 = {
 "flow_last_50_trades": [
  "aggressive selling dominated the last fifty trades",
  "sellers were somewhat more aggressive than buyers in the last fifty trades",
  "aggressive buying and selling were balanced in the last fifty trades",
  "buyers were somewhat more aggressive than sellers in the last fifty trades",
  "aggressive buying dominated the last fifty trades"
 ],
 "flow_last_1000_trades": [
  "aggressive selling has dominated the last thousand trades",
  "sellers have been somewhat more aggressive over the last thousand trades",
  "aggressive buying and selling have been balanced over the last thousand trades",
  "buyers have been somewhat more aggressive over the last thousand trades",
  "aggressive buying has dominated the last thousand trades"
 ],
 "price_last_50_trades": [
  "the price dropped sharply during the last fifty trades",
  "the price slipped during the last fifty trades",
  "the price was unchanged during the last fifty trades",
  "the price edged up during the last fifty trades",
  "the price jumped sharply during the last fifty trades"
 ],
 "price_last_few_minutes": [
  "the price has fallen strongly over the last few minutes",
  "the price has drifted down over the last few minutes",
  "the price has gone sideways over the last few minutes",
  "the price has drifted up over the last few minutes",
  "the price has risen strongly over the last few minutes"
 ],
 "tape_speed": [
  "trades are arriving much more slowly than usual",
  "trades are arriving somewhat more slowly than usual",
  "trades are arriving at the usual pace",
  "trades are arriving somewhat faster than usual",
  "trades are arriving much faster than usual"
 ],
 "volatility": [
  "price swings are much smaller than usual",
  "price swings are somewhat smaller than usual",
  "price swings are typical",
  "price swings are somewhat larger than usual",
  "price swings are much larger than usual"
 ]
}

T5_Q = {
 "up50": {
  "type": "noul",
  "instructions": "Will the price of this contract be higher after the next fifty trades than it is now?",
  "criteria": {
   "true": "The price is higher after the next fifty trades.",
   "false": "The price is lower after the next fifty trades."
  }
 },
 "up500": {
  "type": "noul",
  "instructions": "Will the price of this contract be higher after the next five hundred trades than it is now?",
  "criteria": {
   "true": "The price is higher after the next five hundred trades.",
   "false": "The price is lower after the next five hundred trades."
  }
 },
 "action": {
  "type": "choice",
  "instructions": "What is the best action for a high-frequency trader in this contract right now?",
  "criteria": {
   "buy": "Buy now for a gain over the next few seconds.",
   "sell": "Sell short now for a gain over the next few seconds.",
   "wait": "Do nothing; there is no edge right now."
  }
 }
}


def build_state(levels):
    st = {"instrument": INSTRUMENT}
    for k, l in zip(T5, levels):
        st[k] = T5[k][int(l)]
    return st
