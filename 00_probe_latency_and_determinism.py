#!/usr/bin/env python3
"""Probe the API directly: how fast is it, is it deterministic, and what moves its answer?

NEEDS AN API KEY (costs about one cent). No market data is involved.

    python 00_probe_latency_and_determinism.py

Three measurements:
  (a) latency of 40 sequential requests on one warm connection;
  (b) five identical requests - do we get five identical answers?
  (c) the "response surface": start from a fully neutral state, move ONE feature through
      its five levels, and watch the model's answer. This shows which inputs the model
      actually reacts to.

The output captured when the experiments were run is in results/00_probe_latency_and_determinism.txt
"""

import asyncio
import statistics
import time

import aiohttp

from jev.client import API_URL, MODEL, ask, ask_many, read_api_key
from jev.wording_intraday import INTRADAY_FEATURES, INTRADAY_QUESTIONS, build_state

NEUTRAL_LEVEL = 2
FEATURES = list(INTRADAY_FEATURES)


async def measure_latency(requests_to_send=40):
    headers = {"Authorization": f"Bearer {read_api_key()}", "Content-Type": "application/json"}
    latencies_ms = []
    input_tokens = []

    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(requests_to_send):
            # a different state every time, so nothing can be served from a vendor-side cache
            levels = [(i * (position + 1)) % 5 for position in range(len(FEATURES))]
            body = {"state": build_state(levels), "model": MODEL, "questions": INTRADAY_QUESTIONS}

            started = time.perf_counter()
            async with session.post(API_URL, json=body) as response:
                payload = await response.json()
            latencies_ms.append(1000 * (time.perf_counter() - started))
            input_tokens.append(payload["usage"]["input_tokens"])

    warm = sorted(latencies_ms[1:])  # the first request also pays for the TLS handshake

    def percentile(share):
        return warm[min(len(warm) - 1, int(share * len(warm)))]

    print(
        f"(a) LATENCY: {requests_to_send} sequential requests, 3 questions each,"
        f" ~{int(statistics.mean(input_tokens))} input tokens"
    )
    print(f"    first request (cold connection) {latencies_ms[0]:.0f} ms")
    print(
        f"    then: median {percentile(0.50):.0f} ms | p90 {percentile(0.90):.0f} ms"
        f" | p99 {percentile(0.99):.0f} ms"
    )


def measure_determinism(repeats=5):
    state = build_state((4, 3, 4, 4, 4, 4))
    answers = set()
    for _ in range(repeats):
        reply = ask(state, INTRADAY_QUESTIONS, use_cache=False)["answers"]
        answers.add((reply["up15"]["noul"], reply["up60"]["noul"], reply["action"]["probabilities"]["buy"]))
    print(f"(b) DETERMINISM: {repeats} identical requests -> {len(answers)} distinct answers")
    for answer in sorted(answers):
        print(f"    P(up 15m) {answer[0]:.2f}   P(up 60m) {answer[1]:.2f}   P(buy) {answer[2]:.2f}")


def measure_response_surface():
    jobs = []
    moved_feature = []
    for position, feature in enumerate(FEATURES):
        for level in range(5):
            levels = [NEUTRAL_LEVEL] * len(FEATURES)
            levels[position] = level
            jobs.append((build_state(levels), INTRADAY_QUESTIONS))
            moved_feature.append(feature)
    replies = ask_many(jobs)

    readings = {
        "P(up in 15 min)": lambda a: a["up15"]["noul"],
        "P(buy) - P(sell)": lambda a: a["action"]["probabilities"]["buy"]
        - a["action"]["probabilities"]["sell"],
        "P(wait)": lambda a: a["action"]["probabilities"]["wait"],
    }
    print("(c) RESPONSE SURFACE: one feature moved through levels 0..4, all others neutral")
    for name, read in readings.items():
        print(f"    [{name}]            level 0      1      2      3      4")
        for feature in FEATURES:
            values = [
                read(reply["answers"]) for which, reply in zip(moved_feature, replies) if which == feature
            ]
            print(f"      {feature:22s} " + " ".join(f"{value:6.3f}" for value in values))


if __name__ == "__main__":
    asyncio.run(measure_latency())
    measure_determinism()
    measure_response_surface()
