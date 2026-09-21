#!/usr/bin/env python3
"""NEEDS AN API KEY (about $0.01). Sequential warm-connection latency, determinism, and how the
model's answer moves when ONE feature is changed and the rest stay neutral. No market data involved."""
import time, asyncio, statistics as S, aiohttp
from jev.jev_client import ask, ask_many, URL, MODEL, _key
from jev.states import T3, T3_Q, build_state

async def latency(n=40):
    hdr = {'Authorization': f'Bearer {_key()}', 'Content-Type': 'application/json'}; ms = []; tok = []
    async with aiohttp.ClientSession(headers=hdr) as s:
        for i in range(n):
            lv = [(i * (j + 1)) % 5 for j in range(6)]; body = {'state': build_state(T3, lv), 'model': MODEL, 'questions': T3_Q}; t0 = time.perf_counter()
            async with s.post(URL, json=body) as r:
                d = await r.json(); ms.append(1e3 * (time.perf_counter() - t0)); tok.append(d['usage']['input_tokens'])
    m = sorted(ms[1:]); q = lambda p: m[min(len(m) - 1, int(p * len(m)))]
    print(f"(a) LATENCY n={n} sequential, warm connection, 3 questions, ~{int(S.mean(tok))} input tokens: first (cold TLS) {ms[0]:.0f} ms | p50 {q(.5):.0f} | p90 {q(.9):.0f} | p99 {q(.99):.0f}")

asyncio.run(latency())
st = build_state(T3, (4, 3, 4, 4, 4, 4)); v = []
for _ in range(5):
    a = ask(st, T3_Q, use_cache=False)['answers']; v.append((a['up15']['noul'], a['up60']['noul'], a['action']['probabilities']['buy']))
print(f"(b) DETERMINISM: 5 identical uncached requests -> {len(set(v))} distinct answers: {sorted(set(v))}")
base = [2] * 6; jobs = []; idx = []
for fi, f in enumerate(T3):
    for l in range(5):
        lv = list(base); lv[fi] = l; jobs.append((build_state(T3, lv), T3_Q)); idx.append(f)
res = ask_many(jobs)
print("(c) RESPONSE SURFACE: one feature moved through levels 0..4, all others neutral")
for name, fn in (('P(up 15m)', lambda a: a['up15']['noul']), ('P(buy)-P(sell)', lambda a: a['action']['probabilities']['buy'] - a['action']['probabilities']['sell']), ('P(wait)', lambda a: a['action']['probabilities']['wait'])):
    print(f"   [{name}]")
    for f in T3: print(f"      {f:22s} " + '  '.join(f"{fn(r['answers']):6.3f}" for ff, r in zip(idx, res) if ff == f))
