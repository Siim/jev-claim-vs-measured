#!/usr/bin/env python3
"""The intraday / 'near real-time scalping' test. Needs no API key and no downloads.

Inputs (all in data/):  t3_levels.parquet  the six bucketed features at every 15-minute decision stamp
                        t3_table.parquet   the model's answer for every possible state
                        t3_outcomes.parquet  what the price did next (bp), entry ONE 5m BAR AFTER the stamp
Frozen trade rule: act when |P(buy) - P(sell)| >= 0.25, in that direction; hold 15 or 60 minutes.
Cost floor (Binance VIP0 round trip): ~8.4 bp maker incl. adverse selection, ~10.5 bp taker."""
import os, sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from jev.states import T3
HERE = os.path.dirname(os.path.abspath(__file__)); D = f'{HERE}/data'; K = list(T3); FLOOR_M, FLOOR_T = 8.42, 10.5
rng = np.random.default_rng(20260920)
PRE = 'rebuilt_' if '--rebuilt' in sys.argv else ''          # --rebuilt: use the files written by 02_build_t3_from_binance.py
print(f"inputs: data/{PRE}t3_levels.parquet + data/{PRE}t3_outcomes.parquet")
lv = pd.read_parquet(f'{D}/{PRE}t3_levels.parquet'); tb = pd.read_parquet(f'{D}/t3_table.parquet'); syms = json.load(open(f'{D}/t3_symbols.json'))
m = lv.merge(tb, on=K, how='left'); m['score'] = m['buy'] - m['sell']
m = m.merge(pd.read_parquet(f'{D}/{PRE}t3_outcomes.parquet'), on=['ts', 'sym'], how='left').dropna(subset=['fwd15', 'fwd60']); m['day'] = m['ts'].dt.normalize()
print(f"contracts {syms}   decision stamps {len(m):,}   {m.ts.min()} .. {m.ts.max()}")
print(f"|15m move| mean {m.fwd15.abs().mean():.1f} bp   |60m move| mean {m.fwd60.abs().mean():.1f} bp")
print(f"model's mean P(up 15m) {m.up15.mean():.3f}, P(up 60m) {m.up60.mean():.3f}   <->   price actually rose {100*(m.fwd15>0).mean():.1f}% / {100*(m.fwd60>0).mean():.1f}% of the time")
print("\n=== discrimination: AUC vs realised direction (0.500 = coin flip) ===")
for sc, h in (('up15', 'fwd15'), ('up60', 'fwd60'), ('score', 'fwd15'), ('score', 'fwd60')):
    x = m[m[h] != 0]; print(f"  {sc:>6s} -> {h}: AUC {roc_auc_score((x[h] > 0).astype(int), x[sc]):.4f}")

def trade_stats(x, h):
    t = x[x['score'].abs() >= 0.25]; pnl = np.sign(t['score']) * t[h]; dm = pnl.groupby(t['day']).mean()
    return len(t), float(pnl.mean()), float(dm.mean() / dm.std(ddof=1) * np.sqrt(len(dm))), float((pnl > 0).mean())
mid = pd.Timestamp('2025-04-01'); print("\n=== the trade rule ===")
for h in ('fwd15', 'fwd60'):
    for tag, x in (('all', m), ('first half', m[m.ts < mid]), ('second half', m[m.ts >= mid])):
        n, g, t, hit = trade_stats(x, h)
        print(f"  {h} {tag:<12s} trades {n:7,d}  hit rate {100*hit:5.2f}%  GROSS {g:+6.3f} bp/trade (day-clustered t {t:+5.2f})  net of maker costs {g-FLOOR_M:+6.2f}  net of taker costs {g-FLOOR_T:+6.2f}")
    real = trade_stats(m, h)[1]; nul = []
    for _ in range(500):                                   # null: shift each contract's score series in time (keeps its persistence, breaks the alignment)
        parts = []
        for s, x in m.groupby('sym'):
            k = int(rng.integers(96, len(x) - 96)); y = x[[h, 'day']].copy(); y['score'] = np.roll(x['score'].values, k); parts.append(y)
        nul.append(trade_stats(pd.concat(parts), h)[1])
    print(f"  {h} time-shift null: {np.mean(nul):+.3f} +/- {np.std(nul):.3f} bp   real {real:+.3f}   z {(real-np.mean(nul))/np.std(nul):+.2f}")
print("\n=== ceiling check: can ANY function of these six features scalp? (XGBoost fitted on the first half, read on the second) ===")
import xgboost as xgb
tr, te = m[m.ts < mid], m[m.ts >= mid]
for h in ('fwd15', 'fwd60'):
    mdl = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.7, colsample_bytree=0.8, min_child_weight=200, n_jobs=2, verbosity=0)
    mdl.fit(tr[K].values, tr[h].clip(-200, 200).values); p = mdl.predict(te[K].values)
    hi = p >= np.quantile(p, 0.9); lo = p <= np.quantile(p, 0.1); g = (te[h].values[hi].sum() - te[h].values[lo].sum()) / (hi.sum() + lo.sum())
    print(f"  {h}: trade the top/bottom 10% of predictions -> GROSS {g:+.3f} bp/trade   out-of-sample corr(pred, realised) {np.corrcoef(p, te[h].values)[0,1]:+.4f}   (cost floor {FLOOR_M} bp)")
