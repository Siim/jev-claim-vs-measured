#!/usr/bin/env python3
"""Price all 15,625 TAPE states once -> data/t5_table.parquet. With the shipped cache: zero API calls, no key."""
import itertools, os, pandas as pd
from jev.jev_client import ask_many
from jev.states_tape import T5, T5_Q, build_state
HERE = os.path.dirname(os.path.abspath(__file__)); L = list(itertools.product(range(5), repeat=6))
res = ask_many([(build_state(lv), T5_Q) for lv in L], conc=14, progress=2500)
rows = [dict(zip(T5, lv), up50=a['up50']['noul'], up500=a['up500']['noul'], buy=a['action']['probabilities']['buy'], sell=a['action']['probabilities']['sell'],
             wait=a['action']['probabilities']['wait'], act_conf=a['action']['confidence']) for lv, r in zip(L, res) if 'error' not in r for a in [r['answers']]]
new = pd.DataFrame(rows); path = f'{HERE}/data/t5_table.parquet'
if os.path.exists(path): print('rebuilt table identical to the shipped one:', pd.read_parquet(path)[list(new.columns)].reset_index(drop=True).equals(new))
new.to_parquet(path, index=False); s = new.buy - new.sell
print(f'{len(new)} states priced, fresh API calls {sum(1 for r in res if not r.get("cached"))}')
print(f'wants to act in {int((s.abs()>=.25).sum())} states: BUY {int((s>=.25).sum())}  SELL {int((s<=-.25).sum())}   mean P(up after 50 trades) {new.up50.mean():.3f}')
