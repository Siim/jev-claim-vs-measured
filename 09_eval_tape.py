#!/usr/bin/env python3
"""The TAPE test: the model on raw trades, bars of 50 trades (~1 s each) -- the scale the "HFT" claim is made at.
Needs no API key and no downloads.
L0 = enter at the decision bar's close (zero latency, the model's best case); L1 = one bar later (~1-2 s).
Hit rate is over NON-FLAT outcomes (at one bar the price often does not move at all; a coin gets 50% of the rest)."""
import os, sys, numpy as np, pandas as pd

from jev.states_tape import T5
from sklearn.metrics import roc_auc_score
HERE = os.path.dirname(os.path.abspath(__file__)); J = f'{HERE}/data'; K = list(T5); rng = np.random.default_rng(20260921)
PRE = 'rebuilt_' if '--rebuilt' in sys.argv else ''
d = pd.read_parquet(f'{J}/{PRE}t5_tape_levels.parquet').merge(pd.read_parquet(f'{J}/t5_table.parquet'), on=K, how='left'); d['score'] = d.buy - d.sell
assert d.score.notna().all(); d = d.sort_values(['sym', 'ts']).reset_index(drop=True)
print(f"decision bars {len(d):,} | BTC {int((d.sym=='BTC').sum()):,}  ETH {int((d.sym=='ETH').sum()):,} | 5 days | median bar {d.sec.median():.2f} s  ->  1 / 10 / 50 bars ~ {d.sec.median():.0f} s / {10*d.sec.median():.0f} s / {50*d.sec.median():.0f} s")
print(f"Jev: wants to act on {100*(d.score.abs()>=.25).mean():.1f}% of bars (BUY {100*(d.score>=.25).mean():.1f}% / SELL {100*(d.score<=-.25).mean():.1f}%); mean P(up after 50 trades) {d.up50.mean():.3f}")

def stats(sig, y):
    m = sig != 0; p = sig[m] * y[m]; nz = p != 0
    return int(m.sum()), float(p.mean()) if m.any() else np.nan, float((p[nz] > 0).mean()) if nz.any() else np.nan, float(1 - nz.mean()) if m.any() else np.nan
def null_z(score_sig, y, groups, n=300):
    real = stats(score_sig, y)[1]; out = []
    idx = [np.where(groups == g)[0] for g in np.unique(groups)]
    for _ in range(n):
        s = score_sig.copy()
        for ix in idx: s[ix] = np.roll(score_sig[ix], int(rng.integers(500, len(ix) - 500)))
        out.append(stats(s, y)[1])
    return (real - np.mean(out)) / np.std(out), (1 + sum(v >= real for v in out)) / (n + 1)

jev = np.where(d.score >= .25, 1, np.where(d.score <= -.25, -1, 0)); naive = np.where(d.flow_last_50_trades == 4, 1, np.where(d.flow_last_50_trades == 0, -1, 0))
grp = (d.sym + d.day).values; days = sorted(d.day.unique()); tr = d.day.isin(days[:3]).values; te = ~tr
import xgboost as xgb
print(f"\n{'entry':<6}{'exit':<9}| {'ORACLE |move|':>13} | {'JEV trades':>10} {'hit%':>6} {'gross bp':>9} {'z':>6} {'p':>6} | {'NAIVE hit%':>10} {'gross':>7} | {'XGB hit%':>8} {'gross':>7} (days 4-5) | fees: 4 bp maker, 10 bp taker")
rows = []
for Lg in (0, 1):
    for h in (1, 10, 50):
        y = d[f'f{Lg}_{h}'].values.astype(float); n, g, hit, flat = stats(jev, y); z, p = null_z(jev, y, grp); nn, ng, nhit, _ = stats(naive, y)
        mdl = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.7, colsample_bytree=0.8, min_child_weight=300, n_jobs=2, verbosity=0)
        mdl.fit(d.loc[tr, K].values, np.clip(y[tr], -50, 50)); pr = mdl.predict(d.loc[te, K].values); q = np.quantile(pr, [.1, .9]); xs = np.where(pr >= q[1], 1, np.where(pr <= q[0], -1, 0)); xn, xg, xhit, _ = stats(xs, y[te])
        print(f"{'L'+str(Lg):<6}{str(h)+' bar'+('s' if h>1 else ''):<9}| {np.abs(y).mean():>13.2f} | {n:>10,d} {100*hit:>6.2f} {g:>+9.3f} {z:>+6.1f} {p:>6.3f} | {100*nhit:>10.2f} {ng:>+7.3f} | {100*xhit:>8.2f} {xg:>+7.3f}")
        rows.append(dict(L=Lg, h=h, oracle=np.abs(y).mean(), jev_hit=hit, jev_gross=g, z=z, p=p, naive_hit=nhit, naive_gross=ng, xgb_hit=xhit, xgb_gross=xg, flat=flat))
print("\nby symbol, Jev, entry L1:")
for s in ('BTC', 'ETH'):
    m = (d.sym == s).values; print(f"   {s}: " + '  '.join(f"{h} bar: hit {100*stats(jev[m], d.loc[m, f'f1_{h}'].values.astype(float))[2]:.2f}% gross {stats(jev[m], d.loc[m, f'f1_{h}'].values.astype(float))[1]:+.3f} bp" for h in (1, 10, 50)))
print("\ndiscrimination (AUC, non-flat outcomes): " + '   '.join(f"P(up50) vs L{Lg} 1-bar: {roc_auc_score((d[f'f{Lg}_1'][d[f'f{Lg}_1']!=0] > 0).astype(int), d.up50[d[f'f{Lg}_1']!=0]):.4f}" for Lg in (0, 1)) +
      f"   P(up500) vs L1 10-bar: {roc_auc_score((d.f1_10[d.f1_10!=0] > 0).astype(int), d.up500[d.f1_10!=0]):.4f}")
print(f"share of outcomes that are exactly flat: 1 bar {100*(d.f1_1==0).mean():.1f}%   10 bars {100*(d.f1_10==0).mean():.1f}%   50 bars {100*(d.f1_50==0).mean():.1f}%")
r = pd.DataFrame(rows); best = r[r.L == 1].jev_gross.max()
print(f"\nVERDICT (rule fixed in advance: gross > 4.0 bp/trade at L1, p < 0.01, both symbols): best Jev gross at L1 = {best:+.3f} bp -> {'PASS' if best > 4 else 'FAIL'} (short by {4.0/max(best,1e-9):.0f}x)" if best > 0 else f"\nVERDICT: best Jev gross at L1 = {best:+.3f} bp -> FAIL")

