#!/usr/bin/env python3
"""How fast was the API, measured over every call made for these experiments?

Every cached response carries the wall-clock latency of the call that produced it. The calls
were made from the EU with 12-14 parallel connections at about 17 requests per second.

    python 01_latency_from_cache.py        (no API key needed)
"""

import sqlite3

import numpy as np

from jev.client import CACHE_PATH
from jev.scoring import print_headline

PRICE_PER_MILLION_INPUT_TOKENS = 0.042  # USD; output tokens are free


def main():
    connection = sqlite3.connect(CACHE_PATH)
    rows = connection.execute("select latency_ms, input_tokens from responses").fetchall()
    latency_ms = np.array([row[0] for row in rows])
    input_tokens = np.array([row[1] for row in rows])

    share_under_100ms = 100 * (latency_ms < 100).mean()
    cost = input_tokens.sum() * PRICE_PER_MILLION_INPUT_TOKENS / 1e6

    print(f"API calls recorded          {len(latency_ms):,}")
    print(f"latency, median             {np.percentile(latency_ms, 50):.0f} ms")
    print(f"latency, 90th percentile    {np.percentile(latency_ms, 90):.0f} ms")
    print(f"latency, 99th percentile    {np.percentile(latency_ms, 99):.0f} ms")
    print(f"share of calls under 100 ms {share_under_100ms:.2f}%")
    print(f"input tokens                {input_tokens.sum():,}  ->  ${cost:.2f}")

    print_headline(
        {
            "api_calls": len(latency_ms),
            "median_latency_ms": round(float(np.percentile(latency_ms, 50))),
            "share_under_100ms_pct": round(float(share_under_100ms), 2),
        }
    )


if __name__ == "__main__":
    main()
