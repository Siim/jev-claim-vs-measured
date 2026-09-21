"""A minimal client for TypeSafe's Jev API, with a response cache.

How the API works
-----------------
You POST a ``state`` (any JSON) plus a set of typed ``questions`` to one endpoint and get
back one typed answer per question:

* ``noul``   - a yes/no question; the answer is the probability of "yes"
* ``choice`` - pick one option; the answer is a probability for every option

Why there is a cache
--------------------
Every response is stored in ``data/jev_cache.sqlite`` under a hash of the exact request.
The repository ships with every response used by the experiments, so **all scripts run
without an API key and without making a single API call**.

An API key is only needed if a request is *not* in the cache - for example because you
changed the wording of a question, or deleted the cache to re-price everything yourself.
The key is read from the ``TYPESAFE_API_KEY`` environment variable (or from the file
``~/.typesafe_api_key``). It is never logged and never written to the cache.

Two details that matter for reproducibility
-------------------------------------------
* The model version is **pinned** (``jev-1.13.0``). The alias ``jev-latest`` can move to a
  new model at any time, which would silently change a backtest.
* The model is **not deterministic**: identical requests differ by about +-0.02. The cache
  freezes one answer per request so results can be reproduced exactly. A call made with
  ``use_cache=False`` neither reads nor writes the cache.
"""

import asyncio
import hashlib
import json
import os
import sqlite3
import threading
import time

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"

REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.environ.get("JEV_CACHE", os.path.join(REPOSITORY_ROOT, "data", "jev_cache.sqlite"))

# The vendor allows 1,200 requests per minute. We stay under it.
REQUESTS_PER_SECOND = 17.0
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


def request_hash(state, questions, model=MODEL):
    """The cache key: a SHA-256 of the model name, the state and the questions."""
    canonical = json.dumps({"m": model, "s": state, "q": questions}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def read_api_key():
    """Return the API key, or stop with a clear message if there is none."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    key_file = os.path.expanduser("~/.typesafe_api_key")
    if os.path.exists(key_file):
        with open(key_file) as handle:
            return handle.read().strip()
    raise SystemExit(
        "A request is not in the cache and no API key was found.\n"
        "Set TYPESAFE_API_KEY (or create ~/.typesafe_api_key) to price new requests."
    )


class ResponseCache:
    """SQLite table ``responses``: one row per distinct request."""

    def __init__(self, path=CACHE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.connection.execute(
            "create table if not exists responses ("
            " request_hash text primary key,"
            " model text,"
            " answers_json text,"
            " input_tokens integer,"
            " latency_ms real,"
            " created_at real)"
        )
        self.connection.commit()

    def get(self, key):
        with self.lock:
            row = self.connection.execute(
                "select answers_json, input_tokens, latency_ms from responses where request_hash = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "answers": json.loads(row[0]),
            "input_tokens": row[1],
            "latency_ms": row[2],
            "from_cache": True,
        }

    def put(self, key, model, answers, input_tokens, latency_ms):
        with self.lock:
            self.connection.execute(
                "insert or replace into responses values (?, ?, ?, ?, ?, ?)",
                (key, model, json.dumps(answers), input_tokens, latency_ms, time.time()),
            )

    def commit(self):
        with self.lock:
            self.connection.commit()


class RequestPacer:
    """Spaces request starts evenly so we never exceed the vendor's rate limit."""

    def __init__(self, requests_per_second):
        self.interval = 1.0 / requests_per_second
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait_for_turn(self):
        async with self.lock:
            now = time.monotonic()
            self.next_start = max(self.next_start, now) + self.interval
            delay = self.next_start - self.interval - now
        if delay > 0:
            await asyncio.sleep(delay)


async def _call_api(session, limiter, pacer, cache, state, questions, model, attempts=6):
    """One API call with retries. A successful answer is written to the cache.

    ``cache`` is ``None`` when the caller asked to bypass the cache: a deliberate fresh call
    (for example the determinism probe) must never overwrite a frozen, published answer.
    """
    import aiohttp

    body = {"state": state, "model": model, "questions": questions}
    await pacer.wait_for_turn()
    async with limiter:
        for attempt in range(attempts):
            started = time.perf_counter()
            try:
                async with session.post(API_URL, json=body) as response:
                    text = await response.text()
                    latency_ms = 1000 * (time.perf_counter() - started)

                    if response.status == 200:
                        payload = json.loads(text)
                        input_tokens = int(payload.get("usage", {}).get("input_tokens", 0))
                        if cache is not None:
                            cache.put(
                                request_hash(state, questions, model),
                                payload.get("model", model),
                                payload["answers"],
                                input_tokens,
                                latency_ms,
                            )
                        return {
                            "answers": payload["answers"],
                            "input_tokens": input_tokens,
                            "latency_ms": latency_ms,
                            "from_cache": False,
                        }

                    if response.status in RETRYABLE_STATUS_CODES:
                        retry_after = response.headers.get("retry-after")
                        await asyncio.sleep(float(retry_after) if retry_after else min(30, 1.5**attempt))
                        continue

                    return {"error": f"HTTP {response.status}: {text[:300]}"}

            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(min(30, 1.5**attempt))

        return {"error": f"gave up after {attempts} attempts"}


async def _ask_many(jobs, model, max_parallel, use_cache, progress_every):
    cache = ResponseCache()
    results = [None] * len(jobs)

    # 1. Serve everything we can from the cache.
    missing = []
    for position, (state, questions) in enumerate(jobs):
        cached = cache.get(request_hash(state, questions, model)) if use_cache else None
        if cached is None:
            missing.append(position)
        else:
            results[position] = cached
    if not missing:
        return results

    # 2. Call the API for the rest (this is the only place a key is needed).
    import aiohttp

    if use_cache:
        print(
            f"   {len(missing):,} of {len(jobs):,} requests are not in the cache -> calling the API",
            flush=True,
        )
    writable_cache = cache if use_cache else None
    limiter = asyncio.Semaphore(max_parallel)
    pacer = RequestPacer(REQUESTS_PER_SECOND)
    headers = {"Authorization": f"Bearer {read_api_key()}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=90)
    connector = aiohttp.TCPConnector(limit=max_parallel)
    completed = 0
    started = time.time()

    async with aiohttp.ClientSession(headers=headers, timeout=timeout, connector=connector) as session:

        async def run(position):
            nonlocal completed
            state, questions = jobs[position]
            results[position] = await _call_api(
                session, limiter, pacer, writable_cache, state, questions, model
            )
            completed += 1
            if progress_every and completed % progress_every == 0:
                cache.commit()
                print(f"   {completed:,}/{len(missing):,}  {time.time() - started:6.0f}s", flush=True)

        await asyncio.gather(*(run(position) for position in missing))

    cache.commit()
    return results


def ask_many(jobs, model=MODEL, max_parallel=12, use_cache=True, progress_every=0):
    """Answer many requests.

    ``jobs`` is a list of ``(state, questions)`` pairs. Returns one dict per job, either
    ``{"answers": ..., "from_cache": bool, ...}`` or ``{"error": "..."}``.
    """
    return asyncio.run(_ask_many(jobs, model, max_parallel, use_cache, progress_every))


def ask(state, questions, model=MODEL, use_cache=True):
    """Answer a single request."""
    return ask_many([(state, questions)], model=model, max_parallel=1, use_cache=use_cache)[0]
