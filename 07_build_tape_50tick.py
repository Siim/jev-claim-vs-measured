#!/usr/bin/env python3
"""Rebuild the TAPE inputs from Binance's public archive: stream raw aggTrades (BTCUSDT + ETHUSDT, five days,
~180 MB) -> bars of 50 trades -> causal feature LEVELS and forward returns. One day at a time, in memory;
nothing raw is written to disk. No API key. Writes data/rebuilt_t5_tape_levels.parquet and compares it with
the shipped file; then `python 09_eval_tape.py --rebuilt` re-runs the test on what you built."""
import io, os, sys, zipfile, requests, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = f"{HERE}/data"
SYMS = ['BTCUSDT', 'ETHUSDT']; DAYS = ['2026-09-14', '2026-09-15', '2026-09-16', '2026-09-17', '2026-09-18']; N = 50; W = 2000
def plev(p): return pd.Series(np.select([p <= .1, p <= .3, p < .7, p < .9, p >= .9], [0, 1, 2, 3, 4], default=np.nan), index=p.index).where(p.notna())
def one(sym, day):
    r = requests.get(f'https://data.binance.vision/data/futures/um/daily/aggTrades/{sym}/{sym}-aggTrades-{day}.zip', timeout=180); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content)); x = pd.read_csv(z.open(z.namelist()[0]), header=None, low_memory=False)
    if not str(x.iloc[0, 0]).replace('.', '').isdigit(): x = x.iloc[1:]
    x = x[[1, 2, 5, 6]]; x.columns = ['price', 'qty', 't', 'bm']; x = x.astype({'price': float, 'qty': float, 't': 'int64'})
    x['bm'] = x['bm'].astype(str).str.lower().isin(['true', '1']); x = x.sort_values('t', kind='stable').reset_index(drop=True); n_tr = len(x)
    x['bar'] = np.arange(len(x)) // N; x = x[x.bar < len(x) // N]                                   # drop the last partial bar
    sv = np.where(x.bm, -x.qty, x.qty)                                                                # buyer is maker -> the aggressor SOLD
    g = x.assign(sv=sv).groupby('bar'); b = pd.DataFrame({'close': g.price.last(), 't0': g.t.first(), 't1': g.t.last(), 'vol': g.qty.sum(), 'sv': g.sv.sum()})
    lc = np.log(b.close); r1 = lc.diff(); rk = lambda s: s.rolling(W, min_periods=1000).rank(pct=True)
    dur5 = ((b.t1 - b.t0.shift(4)) / 1000.0).clip(lower=1e-3)
    f = pd.DataFrame({'flow_last_50_trades': plev(rk(b.sv / b.vol)), 'flow_last_1000_trades': plev(rk(b.sv.rolling(20).sum() / b.vol.rolling(20).sum())),
                      'price_last_50_trades': plev(rk(r1)), 'price_last_few_minutes': plev(rk(lc - lc.shift(100))),
                      'tape_speed': plev(rk(5.0 / dur5)), 'volatility': plev(rk(r1.rolling(20).std()))})
    for Lg in (0, 1):
        for h in (1, 10, 50): f[f'f{Lg}_{h}'] = ((lc.shift(-(Lg + h)) - lc.shift(-Lg)) * 1e4).astype('float32')
    f['sec'] = ((b.t1 - b.t0) / 1000.0).astype('float32'); f['ts'] = pd.to_datetime(b.t1, unit='ms'); f['sym'] = sym[:-4]; f['day'] = day
    K = list(f.columns[:6]); f = f.dropna(subset=K + ['f1_50']); f[K] = f[K].astype('int8')
    print(f'  {sym} {day}: trades {n_tr:,} -> bars {len(b):,} -> usable {len(f):,}; median bar {b.t1.sub(b.t0).median()/1000:.2f}s', flush=True); return f
if __name__ == '__main__':
    d = pd.concat([one(s, day) for s in SYMS for day in DAYS], ignore_index=True); d.to_parquet(f'{OUT}/rebuilt_t5_tape_levels.parquet', index=False, compression='zstd')
    ship = pd.read_parquet(f'{OUT}/t5_tape_levels.parquet'); same = len(ship) == len(d) and bool((ship[list(d.columns[:6])].values == d[list(d.columns[:6])].values).all()) and bool(np.allclose(ship['f1_10'].values, d['f1_10'].values, equal_nan=True))
    print(f'rebuilt file identical to the shipped one (levels and outcomes): {same}')
    K = list(d.columns[:6]); print(f"\nusable decision bars {len(d):,}; distinct states {d[K].drop_duplicates().shape[0]:,} of 15,625; median bar length {d.sec.median():.2f}s; bars/day/symbol {len(d)/10:,.0f}")
    print('level frequencies (%):'); print((d[K].apply(lambda s: s.value_counts(normalize=True).sort_index()) * 100).round(1).T.to_string())
