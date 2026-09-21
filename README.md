# Jev for trading: claim vs. measured

A post with ~400k views says TypeSafe's **Jev** is the fastest AI model ever built for trading,
makes calibrated buy/sell decisions in under 100 ms, and shows how to build an HFT system on it.
The article behind it contains no backtest, no P&L and no hit rate. So I ran the tests: on the raw
tape at one decision per second, and at 15–60 minute horizons. It cost **$0.97** of API credit.
Everything needed to check me is in this repository.

![claim vs measured](card/jev_claim_vs_measured.png)

| the claim | what I measured | reproduce with |
|---|---|---|
| Calibrated buy/sell decisions | Mean P(up) **0.44–0.45** while price rose **50%** of the time. On perfectly symmetric inputs it says SELL far more often than BUY: **7,013 vs 3,312** states (15-min test), **4,665 vs 827** (tape test) | `11_…`, `21_…` |
| Decisions in under 100 ms | Sequential, warm connection from the EU: median **294 ms**, p99 447 ms. Under load, 35,685 calls: median 295 ms, **0.00% under 100 ms** | `01_…`, `00_…` |
| HFT — on the raw tape, one decision per second | **298,787 decisions** on 50-trade bars, BTC + ETH, 5 days. It is right **53.4%** of the time one second later and grosses **+0.04 bp/trade** (58.7% and +0.16 bp with *zero* latency). A one-line rule — follow the last 50 trades when their flow is extreme — is right **58.1%**. Fees are **4–10 bp** | `22_tape_evaluate.py` |
| …and at 15–60 minutes | **298,549 trades**, 6 Binance majors, 2.5 years: right **48.4%** of the time, gross **−0.30 bp/trade**. Replication on rebuilt data: 48.6%, −0.16 bp | `12_intraday_evaluate.py` |
| …and on alts | Six alt perps picked by a volume-rank rule, **626k more decisions**: right **48–52%** of the time one bar late. The one-line rule beats it on **7 of 8** symbols (on the hot coin AKE: +3.3 bp vs +0.24 bp). Nothing clears even the 4 bp maker fee; taker costs there are 11–16 bp | `31_tape_alts_evaluate.py` |
| (any model at all, on BTC and ETH) | A predictor that is **never wrong** would gross **1.0 bp** at 1 s, **3.8 bp** at 12 s, **8.7 bp** at 1 min. Taker fees are 10 bp round trip: on the two busiest coins, directional "HFT" at retail fees loses money with a crystal ball. (On alts moves are 3–10× larger and a perfect predictor *would* clear fees from about 10 bars; the model is nowhere near perfect there — see above) | `22_…`, `31_…` |
| (the inputs) | A fitted XGBoost on the same features ties the one-line rule on the tape and finds nothing at 15–60 minutes | `12_…`, `22_…` |

**Read the two HFT rows together.** At one-second scale order flow genuinely continues for a moment, and
the model's one strong rule is "follow aggressive flow", so it is right more often than not there — and
still earns a hundredth of the fees, and still loses to one line of code. At 15–60 minutes that effect is
gone and it is slightly worse than a coin flip. Neither is a trading edge.

**Fair is fair.** On text the model is good: **99.3%** accurate on exchange announcement titles when
its confidence is ≥ 0.8 (1,145 of 1,313 labelled titles; 88.2% with no gating), and several of its
"errors" are my regex being wrong. It is a solid, cheap, fast text classifier. It is not a price oracle.

## Verify it in a few minutes — no API key needed

Every model response used here is in `data/jev_cache.sqlite` (35,685 responses), keyed by a hash of
the exact request. The scripts read the cache and make **zero API calls**.

```bash
git clone https://github.com/Siim/jev-claim-vs-measured && cd jev-claim-vs-measured
pip install -r requirements.txt

python check_headline_numbers.py         # re-runs everything below and checks each published number (~8 min)
```

or one experiment at a time:

```bash
python 12_intraday_evaluate.py           # 15-60 minutes: hit rate, bp/trade, AUC, null, fitted-model ceiling
python 22_tape_evaluate.py               # the raw tape, one decision per second, vs a one-line rule and a perfect oracle
python 31_tape_alts_evaluate.py          # the same on six alts, coin by coin
python 40_text_triage.py                 # fair is fair: the text test, 99.3% when confident
python 01_latency_from_cache.py          # latency of all 35,685 recorded calls
python 11_intraday_price_all_states.py   # the model's complete 15,625-row decision table (15-60 minute test)
python 21_tape_price_all_states.py       # ... and for the tape tests
```

Each script starts with a plain-English description of what it does and why. The expected output of
every script is in `results/`.

## Reading the code

The experiments are the numbered scripts in the repository root; they are meant to be read top to
bottom. The numbering groups them: `0x` API probes, `1x` the 15–60 minute experiment, `2x` the tape
experiment, `3x` the tape experiment on alts, `4x` the text test. Within an experiment: rebuild the
inputs (optional) → price every state → evaluate.

| read this | to check |
|---|---|
| `jev/wording_intraday.py`, `jev/wording_tape.py`, `jev/wording_text.py` | every word the model was told and asked — frozen before any outcome was seen |
| `jev/buckets.py` | how a number becomes one of five levels |
| `jev/tape_bars.py` | how raw trades become bars, features and forward returns (and that nothing looks ahead) |
| `jev/scoring.py` | the trade rule, the hit rate, and the "same decisions at the wrong time" null |
| `jev/client.py` | the API client and the response cache |

## Don't trust my cache? Re-price it yourself (~$1)

Delete `data/jev_cache.sqlite`, set `TYPESAFE_API_KEY`, and run `11_…`, `21_…` and `40_…` again. The model is not
deterministic (identical requests differ by about ±0.02), so your table will differ in the second
decimal and your trade count by a little; the conclusions will not. `00_probe_latency_and_determinism.py`
measures latency and determinism from your own location for about a cent.

## Don't trust my market data? Rebuild it from Binance and re-run the test

```bash
python 20_tape_rebuild_inputs.py            # raw trades -> 50-trade bars, ~180 MB streamed, ~3 min, no key
python 22_tape_evaluate.py --rebuilt        # the tape test on the inputs you just built (they come out identical)
python 30_tape_alts_rebuild_inputs.py       # the alt tapes, ~460 MB streamed, ~10 min, no key
python 31_tape_alts_evaluate.py --rebuilt
python 10_intraday_rebuild_inputs.py        # the 15-60 minute inputs, ~170 MB streamed, 10-20 min, no key
python 12_intraday_evaluate.py --rebuilt
```

For the 15–60 minute test:

| | primary run (shipped inputs) | replication (rebuilt from the archive, 2026-09-21) |
|---|---|---|
| trades | 298,549 | 298,437 |
| hit rate, 15 min | **48.37%** | **48.58%** |
| gross bp/trade, 15 min / 60 min | **−0.30 / −0.47** | **−0.16 / −0.32** |
| AUC of P(up 15m) | 0.4805 | 0.4825 |
| z vs time-shift null, 15 min | −3.88 | −2.15 |
| fitted XGBoost ceiling, 15 / 60 min | +0.18 / +0.53 bp | −0.31 / −0.27 bp |
| cost floor | ~8.4–10.5 bp | ~8.4–10.5 bp |

**A data wrinkle, reported rather than hidden.** The shipped inputs were built from the archive as
downloaded on 2026-07-29, before the test was designed. Binance has since *relabelled* the timestamps
of rows from about April 2025 onward by −5 minutes (identical values, moved labels; every column moves
together, so features and outcomes stay aligned with each other). `10_intraday_rebuild_inputs.py` demonstrates this from
public data alone: **99.99% of the shipped outcomes and 99.96% of the shipped feature rows are
reproduced** by the rebuild, at the same stamp through 2024 and at the stamp five minutes earlier
afterwards (`results/10_intraday_rebuild_inputs.txt`). A fresh rebuild therefore samples a 15-minute
grid that is one bar off in the later period, which makes the replication an independent draw rather
than a copy. Both runs say the same thing.

## The alts, per symbol

Entry one bar late; hit rate over non-flat outcomes; gross bp per trade at 1 / 10 / 50 bars. Prices are a
mid-price proxy, which *flatters* momentum rules (it lags the true mid), so these are upper bounds.

| coin | decisions | bar | spread | the model: right at 1 bar | the model: gross 1 / 10 / 50 | one-line rule: right | one-line rule: gross 1 / 10 / 50 | taker cost |
|---|---|---|---|---|---|---|---|---|
| BTC | 144,547 | 1.0 s | 0.3 bp | 56.2% | +0.07 / +0.15 / +0.13 | 68.5% | +0.30 / +0.62 / +0.66 | 10.3 bp |
| ETH | 154,240 | 1.3 s | 0.4 bp | 52.2% | +0.05 / +0.11 / −0.09 | 57.2% | +0.25 / +0.40 / +0.38 | 10.4 bp |
| ZEC | 294,046 | 0.8 s | 1.2 bp | 52.4% | +0.11 / +0.29 / −0.04 | 57.9% | +0.54 / +1.23 / +1.14 | 11.2 bp |
| SOL | 27,036 | 11 s | 1.0 bp | 50.6% | −0.04 / −0.34 / −1.38 | 51.5% | +0.02 / +0.06 / −0.68 | 11.0 bp |
| AKE | 255,792 | 0.4 s | 6.3 bp | 50.3% | +0.24 / +0.24 / −1.19 | 53.2% | +1.42 / +3.11 / +3.34 | 16.3 bp |
| XRP | 32,063 | 8.7 s | 0.8 bp | 50.9% | +0.04 / +0.41 / +1.98 | 50.4% | −0.01 / −0.18 / +0.47 | 10.8 bp |
| ONDO | 3,373 | 31 s | 2.7 bp | 48.1% | −0.72 / −8.3 / −36.0 | 47.0% | −0.57 / +0.48 / +5.54 | 12.7 bp |
| FLOCK | 13,545 | 8.5 s | 5.5 bp | 50.0% | −0.19 / +2.33 / +13.7 | 50.4% | +0.02 / −0.10 / +3.87 | 15.5 bp |

ZIL and GUN were also selected by the rule but are too thin for the procedure (fewer than the 1,000 bars a
day needed to warm up the trailing windows). ONDO and FLOCK are thin too: their 50-bar numbers (−36 and
+13.7 bp) are overlapping windows on a few thousand bars, of opposite signs, with a 1-bar hit rate of a coin.
The rule fixed in advance needed the model to clear 4 bp on at least four of eight alts. One did.

## How the test works (short version; details in `method/METHOD.md`)

1. All arithmetic stays in code. Each of six features is reduced to one of five levels and written
   as a plain-English phrase. No dates, symbols or prices are sent, so the model cannot look
   anything up.
2. Six features × five levels = 15,625 possible states. Each was priced once. That table **is** the
   model's trading behaviour on these inputs, and you can read it: with everything neutral it says
   P(up) = 0.42 and "wait"; its one strong rule is "follow aggressive taker flow".
3. Tape test: 15.5 million raw trades → bars of 50 trades (~1 s). 15–60 minute test: a decision every
   15 minutes on six liquid perpetuals for 2.5 years. Entry **one bar late** in both (the tape test also
   reports zero latency, the model's best case); frozen rule: act when |P(buy) − P(sell)| ≥ 0.25.
4. Wording, rule, horizons, null and pass bar were fixed before any output met any outcome.
   One wording, one run.

## Scope — what this does and does not show

- It shows that **jev-1.13.0, asked for direction on bucketed market state, has no tradeable edge on
  liquid Binance perpetuals from one second to one hour**, that where it is right more often than not
  (tick scale) a one-line rule is better, and that its probabilities are not calibrated to market
  outcomes (its calibration is for the semantic judgments it was trained on).
- It does **not** test on-chain order books, which is the venue the article points to; the tape test
  is the closest public, reproducible equivalent (one decision per second on the busiest perpetuals). The article offers no evidence for those either; the burden is on the claim.
- Prices are a 5-minute mark-price proxy, not fills, and Binance's own relabelling shows the stamps
  are only good to about one bar. With a gross edge that is negative, neither distinction can help the model.
- If the vendor or the author publishes real results on a real venue, I will link them here.

## Files

```
check_headline_numbers.py            re-runs every experiment and checks each published number
00_probe_latency_and_determinism.py  needs a key (one cent): latency, determinism, what moves the model's answer
01_latency_from_cache.py             latency of every recorded call
10_ / 11_ / 12_intraday_*.py         the 15-60 minute experiment: rebuild inputs / price all states / evaluate
20_ / 21_ / 22_tape_*.py             the tape experiment (BTC, ETH): rebuild inputs / price all states / evaluate
30_ / 31_tape_alts_*.py              the tape experiment on alts: rebuild inputs / evaluate
40_text_triage.py                    the text test

jev/                                 frozen wording, bucketing, bar building, scoring, API client

data/intraday_levels.parquet         the six feature levels at 523,736 decision times (every 15 min, 6 coins)
data/intraday_outcomes.parquet       what the price did next (bp), entering one bar late
data/intraday_table.parquet          the model's answers for all 15,625 possible states
data/tape_levels.parquet             298,787 one-second decision bars on BTC and ETH: levels + forward returns
data/tape_alts_levels.parquet        925,874 decision bars on ten coins (mid-price estimate)
data/tape_table.parquet              the model's answers for all 15,625 possible tape states
data/announcement_titles.parquet     4,435 exchange announcement titles (the text test)
data/jev_cache.sqlite                every model response: request hash -> answers, tokens, latency

results/                             captured output of every script, including the replication on rebuilt data
method/METHOD.md                     how each test was built, and the rules fixed in advance
card/                                the summary image and its HTML source
```

Not affiliated with TypeSafe or Binance. Model `jev-1.13.0`, September 2026. Not financial advice;
it is the opposite of a recommendation. MIT licence.
