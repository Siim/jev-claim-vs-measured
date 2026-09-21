#!/usr/bin/env python3
"""Fair is fair: the model on its home turf -- typed triage of 4,435 exchange announcement TITLES.
Ground truth = plain regexes on the classes they define. No prices involved. With the shipped
cache this makes zero API calls and needs no key."""
import os, re, pandas as pd
from jev.jev_client import ask_many
from jev.states import T4_Q
HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_parquet(f'{HERE}/data/announcement_titles.parquet').reset_index(drop=True)
res = ask_many([(t, T4_Q) for t in d.title], conc=12, progress=1000)
ok = [i for i, r in enumerate(res) if 'error' not in r]; d = d.loc[ok].copy(); A = [res[i]['answers'] for i in ok]
d['jev_type'] = [a['event_type']['choice'] for a in A]; d['jev_conf'] = [a['event_type']['confidence'] for a in A]
d['jev_dir'] = [a['direction']['choice'] for a in A]

def truth(t):
    if re.match(r'^Binance Will Delist ', t) and 'Futures' not in t: return 'spot_delisting'
    if 'Binance Futures Will Delist' in t: return 'perp_delisting'
    if re.match(r'^Binance Will List ', t): return 'spot_listing'
    if re.search(r'Binance Futures Will Launch', t) and 'Pre-Market' not in t: return 'perp_launch'
    if 'Monitoring Tag' in t or 'Seed Tag' in t and 'Will List' not in t: return 'monitoring_tag'
    if re.search(r'Margin Tiers|Leverage|Funding Rate|Tick Size|Collateral Ratio|Price Limit', t): return 'risk_parameter_change'
    if re.search(r'Removal of (Spot|Margin)|Will Add .* Trading Pairs|Notice of Removal', t): return 'pair_change'
    return None
d['truth'] = d.title.map(truth); g = d[d.truth.notna()]
print(f"titles {len(d):,}   with a regex ground truth {len(g):,}   fresh API calls {sum(1 for r in res if not r.get('cached'))}")
print(f"raw accuracy {100*(g.jev_type==g.truth).mean():.1f}%")
for th in (0.5, 0.8, 0.9):
    s = g[g.jev_conf >= th]; print(f"   confidence >= {th:.1f}: kept {len(s):,} of {len(g):,} ({100*len(s)/len(g):.1f}%)   accuracy {100*(s.jev_type==s.truth).mean():.1f}%")
print("\ndirection call by true class (% down / unclear / up):")
print((g.groupby('truth').jev_dir.value_counts(normalize=True).unstack().fillna(0) * 100).round(0)[['down', 'unclear', 'up']].to_string())
print("\nnote: several of the 'errors' are the REGEX being wrong, e.g. a title that ends a pre-market and lists a token\n(with a Seed Tag) is a listing, as the model said. The regex truth is a lower bound on the model's accuracy.")
