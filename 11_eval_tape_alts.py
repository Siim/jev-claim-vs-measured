#!/usr/bin/env python3
"""The tape test on ALTS (mid-price proxy). Symbols were picked by a volume-rank rule, not by hand (see METHOD).
No API key, no downloads: it reuses the 15,625-row table of the tape test, so nothing was tuned to these coins.
All Jev/NAIVE/XGB numbers are at entry L1 (one bar late) unless marked L0. Hit rate over non-flat outcomes."""
import os, sys, numpy as np, pandas as pd, xgboost as xgb

from jev.states_tape import T5
HERE = os.path.dirname(os.path.abspath(__file__)); J = f'{HERE}/data'; K = list(T5); rng = np.random.default_rng(20260921)
PRE = 'rebuilt_' if '--rebuilt' in sys.argv else ''
d = pd.read_parquet(f'{J}/{PRE}t6_tape_alts_levels.parquet').merge(pd.read_parquet(f'{J}/t5_table.parquet'), on=K, how='left'); d['score'] = d.buy - d.sell
d = d.sort_values(['sym', 'ts']).reset_index(drop=True); days = sorted(d.day.unique()); tr = d.day.isin(days[:3]).values; te = ~tr
d['jev'] = np.where(d.score >= .25, 1, np.where(d.score <= -.25, -1, 0)); d['naive'] = np.where(d.flow_last_50_trades == 4, 1, np.where(d.flow_last_50_trades == 0, -1, 0))
for h in (1, 10, 50):                                        # one pooled fitted model per horizon (levels are percentiles, so pooling is natural)
    m = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8, min_child_weight=300, n_jobs=2, verbosity=0)
    y = d[f'f1_{h}'].values.astype(float); lim = np.nanpercentile(np.abs(y[tr]), 99); m.fit(d.loc[tr, K].values, np.clip(y[tr], -lim, lim)); d[f'xp_{h}'] = m.predict(d[K].values)
def st(sig, y):
    k = sig != 0; p = sig[k] * y[k]; nz = p != 0
    return (int(k.sum()), float(p.mean()), float((p[nz] > 0).mean())) if k.any() and nz.any() else (int(k.sum()), np.nan, np.nan)
def zshift(x, col, n=300):
    sig = x.jev.values; y = x[col].values.astype(float); real = st(sig, y)[1]; idx = [np.where(x.day.values == g)[0] for g in x.day.unique()]; out = []
    for _ in range(n):
        s = sig.copy()
        for ix in idx:
            if len(ix) > 1200: s[ix] = np.roll(sig[ix], int(rng.integers(500, len(ix) - 500)))
        out.append(st(s, y)[1])
    return (real - np.nanmean(out)) / np.nanstd(out), (1 + sum(v >= real for v in out)) / (n + 1)
order = ['BTC', 'ETH', 'ZEC', 'SOL', 'AKE', 'XRP', 'ONDO', 'ZIL', 'FLOCK', 'GUN']; rows = []
print(f"{'sym':<6}{'bars':>8}{'bar s':>7}{'spread':>7} | ORACLE |move| bp 1/10/50 bars | {'JEV hit% 1bar':>13}{' gross bp 1/10/50':>22}{'  z(1)':>7}{'  z(10)':>7} | NAIVE hit% 1bar / gross 1/10/50 | XGB gross 1/10/50 (d4-5) | taker cost | oracle>taker at")
for s in order:
    x = d[d.sym == s]
    if len(x) < 3000: print(f"{s:<6}{len(x):>8}   too few usable bars under the pre-registered procedure -> not evaluated"); continue
    o = [float(np.abs(x[f'f1_{h}']).mean()) for h in (1, 10, 50)]; J_ = [st(x.jev.values, x[f'f1_{h}'].values.astype(float)) for h in (1, 10, 50)]; N_ = [st(x.naive.values, x[f'f1_{h}'].values.astype(float)) for h in (1, 10, 50)]
    xt = x[x.day.isin(days[3:])]; X_ = []
    for h in (1, 10, 50):
        q = np.quantile(xt[f'xp_{h}'], [.1, .9]); sig = np.where(xt[f'xp_{h}'] >= q[1], 1, np.where(xt[f'xp_{h}'] <= q[0], -1, 0)); X_.append(st(sig, xt[f'f1_{h}'].values.astype(float))[1])
    z1, p1 = zshift(x, 'f1_1'); z10, p10 = zshift(x, 'f1_10'); spr = float(x.spr.median()); tk = 10 + spr; ot = [f'{h}b' for h, v in zip((1, 10, 50), o) if v > tk]
    print(f"{s:<6}{len(x):>8,}{x.sec.median():>7.1f}{spr:>7.2f} | {o[0]:>8.2f}{o[1]:>8.2f}{o[2]:>8.2f}        | {100*J_[0][2]:>13.2f}{J_[0][1]:>+8.3f}{J_[1][1]:>+7.3f}{J_[2][1]:>+7.3f}{z1:>+7.1f}{z10:>+7.1f} | {100*N_[0][2]:>7.2f} / {N_[0][1]:>+6.3f}{N_[1][1]:>+7.3f}{N_[2][1]:>+7.3f}   | {X_[0]:>+7.3f}{X_[1]:>+7.3f}{X_[2]:>+7.3f}  | {tk:>7.1f} bp | {','.join(ot) or 'never'}")
    rows.append(dict(sym=s, bars=len(x), sec=float(x.sec.median()), spread=spr, o1=o[0], o10=o[1], o50=o[2], jev_hit1=J_[0][2], jev_g1=J_[0][1], jev_g10=J_[1][1], jev_g50=J_[2][1], z1=z1, p1=p1, z10=z10, p10=p10,
                     naive_hit1=N_[0][2], naive_g1=N_[0][1], naive_g10=N_[1][1], naive_g50=N_[2][1], xgb_g1=X_[0], xgb_g10=X_[1], xgb_g50=X_[2], jev_hit1_L0=st(x.jev.values, x['f0_1'].values.astype(float))[2], jev_g1_L0=st(x.jev.values, x['f0_1'].values.astype(float))[1]))
r = pd.DataFrame(rows); a = r[~r.sym.isin(['BTC', 'ETH'])]
print(f"\nzero-latency (L0) reference, Jev hit% / gross at 1 bar: " + '  '.join(f"{x.sym} {100*x.jev_hit1_L0:.1f}%/{x.jev_g1_L0:+.2f}" for x in r.itertuples()))
print(f"\nALTS evaluated: {len(a)}.  Jev best L1 gross per alt (bp): " + '  '.join(f"{x.sym} {max(x.jev_g1, x.jev_g10, x.jev_g50):+.2f}" for x in a.itertuples()))
npass = int(((a[['jev_g1', 'jev_g10', 'jev_g50']].max(axis=1) > 4.0)).sum())
print(f"NAIVE beats JEV on gross at 1 bar on {int((r.naive_g1 > r.jev_g1).sum())} of {len(r)} symbols; at 10 bars on {int((r.naive_g10 > r.jev_g10).sum())} of {len(r)}.")
print(f"VERDICT (rule fixed in advance: Jev L1 gross > 4.0 bp, p < 0.01, on >= 4 of 8 alts): alts clearing 4 bp = {npass} -> {'PASS' if npass >= 4 else 'FAIL'}")
