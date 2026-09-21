"""Minimal TypeSafe / Jev client.  POST /v1/systemone, no SDK.

* EVERY response is cached in data/jev_cache.sqlite, keyed by sha256(model, state, questions).
  The shipped cache holds every response used in this repo, so all scripts run WITHOUT an API
  key. A key is only needed on a cache miss (i.e. if you change the wording, or delete the cache
  to re-price everything yourself: about $0.50).
* The key is read from $TYPESAFE_API_KEY or ~/.typesafe_api_key. It is never logged or cached.
* The model is PINNED. `jev-latest` is an alias that can move under a backtest, and the model
  is not deterministic (identical requests differ by about +-0.02), so: pin and cache.
"""
import os, json, time, hashlib, sqlite3, asyncio, threading

URL = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-1.13.0'
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DB = os.environ.get('JEV_CACHE', os.path.join(ROOT, 'data', 'jev_cache.sqlite'))


def _key():
    k = os.environ.get('TYPESAFE_API_KEY')
    if not k:
        p = os.path.expanduser('~/.typesafe_api_key')
        if not os.path.exists(p):
            raise SystemExit('Cache miss and no API key. Set TYPESAFE_API_KEY (or ~/.typesafe_api_key) to price new states.')
        k = open(p).read().strip()
    return k


def req_hash(state, questions, model=MODEL):
    blob = json.dumps({'m': model, 's': state, 'q': questions}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class Cache:
    def __init__(self, path=CACHE_DB):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False); self.lock = threading.Lock()
        self.db.execute('create table if not exists r (h text primary key, model text, answers text, in_tok int, ms real, ts real)')
        self.db.commit()

    def get(self, h):
        with self.lock:
            row = self.db.execute('select answers,in_tok,ms from r where h=?', (h,)).fetchone()
        return None if row is None else dict(answers=json.loads(row[0]), in_tok=row[1], ms=row[2], cached=True)

    def put(self, h, model, answers, in_tok, ms):
        with self.lock:
            self.db.execute('insert or replace into r values (?,?,?,?,?,?)', (h, model, json.dumps(answers), in_tok, ms, time.time()))

    def commit(self):
        with self.lock:
            self.db.commit()

    def stats(self):
        with self.lock:
            return self.db.execute('select count(*), coalesce(sum(in_tok),0) from r').fetchone()


class _Pace:
    """Evenly spaced request starts (the vendor limit is 1,200 requests/minute)."""
    def __init__(self, rps): self.dt = 1.0 / rps; self.next = 0.0; self.lock = asyncio.Lock()

    async def wait(self):
        async with self.lock:
            now = time.monotonic(); self.next = max(self.next, now) + self.dt; delay = self.next - self.dt - now
        if delay > 0:
            await asyncio.sleep(delay)


async def _fetch(sess, sem, pace, cache, state, questions, model, retries=6):
    import aiohttp
    body = {'state': state, 'model': model, 'questions': questions}; await pace.wait()
    async with sem:
        for att in range(retries):
            t0 = time.perf_counter()
            try:
                async with sess.post(URL, json=body) as r:
                    txt = await r.text(); ms = 1e3 * (time.perf_counter() - t0)
                    if r.status == 200:
                        d = json.loads(txt); it = int(d.get('usage', {}).get('input_tokens', 0))
                        cache.put(req_hash(state, questions, model), d.get('model', model), d['answers'], it, ms)
                        return dict(answers=d['answers'], in_tok=it, ms=ms, cached=False)
                    if r.status in (429, 500, 502, 503, 504):
                        ra = r.headers.get('retry-after'); await asyncio.sleep(float(ra) if ra else min(30, 1.5 ** att)); continue
                    return dict(error=f'HTTP {r.status}: {txt[:300]}')
            except (aiohttp.ClientError, asyncio.TimeoutError):
                await asyncio.sleep(min(30, 1.5 ** att))
        return dict(error=f'gave up after {retries} attempts')


async def _many(jobs, model, conc, use_cache, progress, rps):
    cache = Cache(); out = [None] * len(jobs); miss = []
    for i, (st, qs) in enumerate(jobs):
        c = cache.get(req_hash(st, qs, model)) if use_cache else None
        if c is None: miss.append(i)
        else: out[i] = c
    if not miss:
        return out
    import aiohttp
    print(f'   {len(miss)} of {len(jobs)} requests are not in the cache -> calling the API', flush=True)
    sem = asyncio.Semaphore(conc); pace = _Pace(rps); done = 0; t0 = time.time()
    hdr = {'Authorization': f'Bearer {_key()}', 'Content-Type': 'application/json'}
    async with aiohttp.ClientSession(headers=hdr, timeout=aiohttp.ClientTimeout(total=90), connector=aiohttp.TCPConnector(limit=conc)) as sess:
        async def run(i):
            nonlocal done
            out[i] = await _fetch(sess, sem, pace, cache, jobs[i][0], jobs[i][1], model); done += 1
            if progress and done % progress == 0:
                cache.commit(); print(f'   {done}/{len(miss)}  {time.time()-t0:6.0f}s', flush=True)
        await asyncio.gather(*(run(i) for i in miss))
    cache.commit(); return out


def ask_many(jobs, model=MODEL, conc=12, use_cache=True, progress=0, rps=17.0):
    """jobs: list of (state, questions). Returns a list of dicts: {'answers': ...} or {'error': ...}."""
    return asyncio.run(_many(jobs, model, conc, use_cache, progress, rps))


def ask(state, questions, model=MODEL, use_cache=True):
    return ask_many([(state, questions)], model=model, conc=1, use_cache=use_cache)[0]
