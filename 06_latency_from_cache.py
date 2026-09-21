#!/usr/bin/env python3
"""Every cached response carries the wall-clock latency of the call that produced it
(measured from the EU, 12-14 concurrent connections, ~17 requests/second)."""
import os, sqlite3, numpy as np
db = sqlite3.connect(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'jev_cache.sqlite'))
ms = np.array([r[0] for r in db.execute('select ms from r')]); tok = np.array([r[0] for r in db.execute('select in_tok from r')])
print(f"{len(ms):,} API calls   latency under load: p50 {np.percentile(ms,50):.0f} ms   p90 {np.percentile(ms,90):.0f}   p99 {np.percentile(ms,99):.0f}   max {ms.max():.0f}   share under 100 ms: {100*(ms<100).mean():.2f}%")
print(f"input tokens {tok.sum():,}  ->  ${tok.sum()*0.042/1e6:.2f} at $0.042 per million input tokens")
