# Method

The test rules below were written down before any model output was joined to any market outcome
(in a private research ledger; this is the part that concerns the published tests). The wording
in `jev/states.py` was frozen at the same time and never edited: one wording, one run. Re-wording
a question after seeing results would be fitting the prompt to the data.

## Why the state is words, not numbers

The vendor's documentation says the model is weak at numeric precision, reads dates as text, loses
accuracy as the state grows, and recommends passing named buckets while keeping arithmetic in code.
So every feature is reduced, in code, to one of five ordered levels and rendered as a plain-English
phrase. No dates, no symbols and no price levels are sent, so the model cannot recognise the
period and recall what happened next.

A state is therefore a tuple of six levels and the model's decision function is a finite table of
5^6 = 15,625 rows. It was priced once. A live system built this way would do a table lookup with
zero latency, so the latency numbers here are about the API, not about what such a design needs.

## Intraday test (T3)

- Contracts: BTC, ETH, XRP, BNB, DOGE, LTC USD-M perpetuals — the six with the largest median
  open-interest value in 2023 among 22 long-listed majors (a liquidity rule; no outcome involved).
- Clock: a decision every 15 minutes, 2024-01-01 .. 2026-06-30 (523,361 usable stamps).
- Features (each vs the contract's OWN trailing window, cut at the 10/30/70/90th percentiles):
  15-minute return, 4-hour return, 30-minute taker buy/sell flow, 1-hour open-interest change
  (all vs trailing 7 days); top-trader account long/short ratio as a z-score vs trailing 30 days
  (cut at -1.5/-0.5/+0.5/+1.5); position in the 24-hour range (0.10/0.35/0.65/0.90).
- Questions: P(price higher in 15 min), P(price higher in 60 min), and a buy / sell / wait choice.
- Trade rule (frozen): act when |P(buy) - P(sell)| >= 0.25, in that direction.
- Execution: entry ONE 5-minute bar after the decision stamp; exit 15 or 60 minutes after entry.
- Costs for context: Binance VIP0 round trip is about 8.4 bp maker/maker including a measured
  adverse-selection penalty, about 10.5 bp taker/taker. Results are reported GROSS against that floor.
- Null: each contract's score series is circularly shifted in time 500 times (keeps the score's
  persistence and duty cycle, breaks its alignment with the market).
- Pass bar (set in advance): gross > 8.4 bp/trade with null p < 0.005 in BOTH halves. It was missed
  by a factor of about 28 and with the wrong sign.
- Ceiling check (a diagnostic, not a candidate): an XGBoost regressor fitted on the first half and
  read on the second half, on the same six features. If a fitted model cannot find anything in the
  inputs, no zero-shot prior over those inputs can either.

The primary run used inputs built from Binance's public archive as downloaded on 2026-07-29. The
archive's timestamps for rows after about April 2025 have since been relabelled by -5 minutes (see the
README and script 02); the replication on a fresh rebuild is reported next to the primary run.

Time bars, not dollar/volume bars. Prices are open-interest value / open interest sampled every
five minutes, which is a mark-price proxy, not a tradeable fill.

## Text test (T4)

4,435 unique Binance announcement titles since 2022. One request per title: a closed-set event
type, a direction, and an impact score. Ground truth is a handful of regexes on the classes they
define exactly (1,313 titles). Confidence gating is the vendor's recommended pattern; it works.

## What is NOT in this repository

Daily-horizon tests of the same model (as a per-contract selector, as a tilt on an existing
long/short book, and as a market-regime gate) were run with the same discipline and also failed,
but they depend on a private portfolio codebase and are not reproducible from here, so no number
from them is claimed in this repository.
