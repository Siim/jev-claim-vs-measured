#!/usr/bin/env python3
"""Rebuild the intraday inputs FROM BINANCE'S PUBLIC ARCHIVE, compare them with the shipped files,
and write data/rebuilt_t3_levels.parquet + data/rebuilt_t3_outcomes.parquet so that
`python 04_eval_t3_scalping.py --rebuilt` re-runs the whole test on data you built yourself.
No API key. Streams ~5,800 small daily zips (~170 MB total); nothing raw is written to disk. 10-20 minutes.

KNOWN WRINKLE, reported by this script rather than hidden: the shipped files were built from the
archive as downloaded on 2026-07-29. Binance has since RELABELLED the timestamps of rows from about
April 2025 onward by -5 minutes (identical values, moved labels; every column moves together). A
fresh rebuild therefore samples a 15-minute grid that is one bar off in that period: row-identical
for 2024, statistically equivalent but not identical afterwards. The comparison below checks each
shipped row against the rebuild at the same stamp AND at the stamp five minutes earlier.

Source: https://data.binance.vision/data/futures/um/daily/metrics/<SYMBOL>/<SYMBOL>-metrics-<DATE>.zip
(5-minute rows: open interest, open-interest value, top-trader and all-account long/short ratios,
taker buy/sell volume ratio). Price = open-interest value / open interest.

Everything is causal: each level uses only rows at or before its own timestamp, and the outcome is
measured from ONE BAR LATER (entry at t+5m; exits at t+20m and t+65m)."""
import io, os, json, zipfile, numpy as np, pandas as pd, requests, threading
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__)); D = f'{HERE}/data'
SYMS = json.load(open(f'{D}/t3_symbols.json')); DAYS = pd.date_range('2023-11-15', '2026-06-30', freq='D')
COLS = ['create_time', 'sum_open_interest', 'sum_open_interest_value', 'count_toptrader_long_short_ratio', 'sum_taker_long_short_vol_ratio']
tl = threading.local()

def one(args):
    s, d = args; url = f'https://data.binance.vision/data/futures/um/daily/metrics/{s}USDT/{s}USDT-metrics-{d.date()}.zip'
    if not hasattr(tl, 's'): tl.s = requests.Session()
    for _ in range(4):
        try:
            r = tl.s.get(url, timeout=60)
            if r.status_code == 404: return None
            if r.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(r.content)); x = pd.read_csv(z.open(z.namelist()[0]), usecols=COLS); x['sym'] = s; return x
        except Exception:
            pass
    return None

def plev(p): return pd.Series(np.select([p <= .1, p <= .3, p < .7, p < .9, p >= .9], [0, 1, 2, 3, 4], default=np.nan), index=p.index).where(p.notna())

def build(s, d):
    d = d.copy(); d['create_time'] = pd.to_datetime(d['create_time']).dt.floor('5min')
    d = d.drop_duplicates('create_time').set_index('create_time').sort_index(); d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq='5min'))
    px = d['sum_open_interest_value'] / d['sum_open_interest']; lp = np.log(px); rk = lambda x: x.rolling(2016, min_periods=1000).rank(pct=True)      # percentile vs own trailing 7 days
    tk = np.log(d['sum_taker_long_short_vol_ratio'].clip(lower=1e-6)).rolling(6, min_periods=4).mean(); tt = d['count_toptrader_long_short_ratio']
    z = (tt - tt.rolling(8640, min_periods=4000).mean()) / tt.rolling(8640, min_periods=4000).std()                                                    # z vs own trailing 30 days
    lo, hi = px.rolling(288, min_periods=200).min(), px.rolling(288, min_periods=200).max(); rp = (px - lo) / (hi - lo).replace(0, np.nan)
    f = pd.DataFrame({'price_15m': plev(rk(lp - lp.shift(3))), 'price_4h': plev(rk(lp - lp.shift(48))), 'taker_flow_30m': plev(rk(tk)),
        'open_interest_1h': plev(rk(np.log(d['sum_open_interest']) - np.log(d['sum_open_interest'].shift(12)))),
        'top_trader_accounts': pd.Series(np.select([z <= -1.5, z <= -0.5, z < 0.5, z < 1.5, z >= 1.5], [0, 1, 2, 3, 4], default=np.nan), index=z.index).where(z.notna()),
        'range_24h': pd.Series(np.select([rp <= .10, rp <= .35, rp < .65, rp < .90, rp >= .90], [0, 1, 2, 3, 4], default=np.nan), index=rp.index).where(rp.notna())})
    f = f[f.index >= '2023-12-31'].dropna().astype('int8'); f['sym'] = s; f.index.name = 'ts'            # full 5-minute grid; the 15-minute decision clock is applied below
    d2 = d[d.index >= '2023-12-25']; lp2 = np.log(d2['sum_open_interest_value'] / d2['sum_open_interest'])
    o = pd.DataFrame({'ts': lp2.index, 'sym': s, 'fwd15': (lp2.shift(-4) - lp2.shift(-1)).values * 1e4, 'fwd60': (lp2.shift(-13) - lp2.shift(-1)).values * 1e4})
    return f.reset_index(), o

if __name__ == '__main__':
    jobs = [(s, d) for s in SYMS for d in DAYS]; print(f'{len(jobs):,} daily files to stream', flush=True); parts = []
    with ThreadPoolExecutor(12) as ex:
        for i, x in enumerate(ex.map(one, jobs)):
            if x is not None: parts.append(x)
            if (i + 1) % 1000 == 0: print(f'   {i+1:,}/{len(jobs):,}', flush=True)
    raw = pd.concat(parts, ignore_index=True); print(f'rows {len(raw):,}; files missing in the archive: {len(jobs)-len(parts)}', flush=True)
    L, O = zip(*[build(s, raw[raw.sym == s].drop(columns='sym')) for s in SYMS]); L5, O = pd.concat(L, ignore_index=True), pd.concat(O, ignore_index=True)
    K = [c for c in L5.columns if c not in ('ts', 'sym')]
    L = L5[(L5.ts >= '2024-01-01') & (L5.ts.dt.minute % 15 == 0)].reset_index(drop=True)
    L.to_parquet(f'{D}/rebuilt_t3_levels.parquet', index=False); L[['ts', 'sym']].merge(O, on=['ts', 'sym'], how='left').to_parquet(f'{D}/rebuilt_t3_outcomes.parquet', index=False)
    print(f'wrote rebuilt levels {len(L):,} rows and outcomes -> now run: python 04_eval_t3_scalping.py --rebuilt', flush=True)
    ship = pd.read_parquet(f'{D}/t3_levels.parquet'); so = pd.read_parquet(f'{D}/t3_outcomes.parquet'); ship = ship.merge(so, on=['ts', 'sym'])
    def at(offset_min):
        x = L5.merge(O, on=['ts', 'sym'], how='left'); x['ts'] = x['ts'] + pd.Timedelta(minutes=offset_min); m = ship.merge(x, on=['ts', 'sym'], how='left', suffixes=('', '_new'))
        lev = (m[K].values == m[[f'{k}_new' for k in K]].values).all(axis=1); out = ((m.fwd15 - m.fwd15_new).abs() < 1e-6) | (m.fwd15.isna() & m.fwd15_new.isna())
        return pd.Series(lev, index=m.ts), pd.Series(out.values, index=m.ts)
    l0, o0 = at(0); l5, o5 = at(5)
    q = pd.DataFrame({'outcome same stamp %': o0, 'outcome stamp-5min %': o5, 'outcome either %': o0 | o5, 'levels same stamp %': l0, 'levels stamp-5min %': l5, 'levels either %': l0 | l5}).groupby(pd.Grouper(freq='QS')).mean() * 100
    print('\nshipped rows reproduced by the rebuild, by quarter (a shipped row stamped t is compared with the rebuild at t and at t-5min):'); print(q.round(2).to_string())
    print(f"\nOVERALL  outcomes reproduced {100*(o0|o5).mean():.3f}%   all six levels reproduced {100*(l0|l5).mean():.3f}%   (levels can differ for a few weeks after the relabelling date, where trailing windows straddle it)")
