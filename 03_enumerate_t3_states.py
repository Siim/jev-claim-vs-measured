#!/usr/bin/env python3
"""Price all 5**6 = 15,625 intraday states ONCE -> data/t3_table.parquet (the model's complete
decision function). With the shipped cache this makes zero API calls and needs no key."""
import itertools, os, pandas as pd
from jev.jev_client import ask_many, Cache
from jev.states import T3, T3_Q, build_state
HERE = os.path.dirname(os.path.abspath(__file__))
levels = list(itertools.product(range(5), repeat=len(T3)))
res = ask_many([(build_state(T3, lv), T3_Q) for lv in levels], conc=14, progress=2000)
bad = [r for r in res if 'error' in r]
rows = [dict(zip(T3, lv), up15=a['up15']['noul'], up60=a['up60']['noul'], buy=a['action']['probabilities']['buy'], sell=a['action']['probabilities']['sell'],
             wait=a['action']['probabilities']['wait'], act_conf=a['action']['confidence']) for lv, r in zip(levels, res) if 'error' not in r for a in [r['answers']]]
new = pd.DataFrame(rows); path = f'{HERE}/data/t3_table.parquet'
if os.path.exists(path):
    old = pd.read_parquet(path); same = old[list(new.columns)].reset_index(drop=True).equals(new)
    print(f'rebuilt table identical to the shipped one: {same}')
new.to_parquet(path, index=False)
score = new['buy'] - new['sell']
print(f'{len(new)} states priced, {len(bad)} failed, fresh API calls {sum(1 for r in res if not r.get("cached"))}')
print(f'states where the model wants to act (|P(buy)-P(sell)| >= 0.25): {int((score.abs()>=.25).sum())}  ->  BUY {int((score>=.25).sum())}  SELL {int((score<=-.25).sum())}')
print(f'mean P(up 15m) over all states {new.up15.mean():.3f}   mean P(up 60m) {new.up60.mean():.3f}')
