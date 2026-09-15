"""Sensitivity of P(personnel injury) to the arbitrary parent-cap constant.

Zhang's recipe caps parents at 12 ("maxElements"). Nothing justifies 12.
We hold data, node set, and CPT rule fixed and vary ONLY the cap.
"""
import json, numpy as np, pandas as pd

BASE = '/mnt/user-data/uploads/export_for_review/'
df = pd.read_csv(BASE + 'accident_variable_matrix.csv')
meta = json.load(open(BASE + 'frozen_bn.json'))['meta']

SEV = 'personnel injury'
STATES = meta['outcome_states'][SEV]
parents12 = meta['severity_parents']
# rank by frequency, exactly as "selected by frequency"
freq = df[parents12].sum().sort_values(ascending=False)
ranked = list(freq.index)

y = df[SEV].values
prior = np.array([(y == s).mean() for s in STATES])          # marginal fallback
M = 5.0                                                       # support-weight

def cpt_and_query(k, smoothing=M):
    """Empirical CPT over top-k parents, support-weighted toward the marginal."""
    ps = ranked[:k]
    key = df[ps].astype(str).agg('|'.join, axis=1)
    table = {}
    for cfg, idx in key.groupby(key).groups.items():
        sub = y[df.index.isin(idx)]
        n = len(sub)
        emp = np.array([(sub == s).sum() for s in STATES], dtype=float)
        table[cfg] = (emp + smoothing * prior) / (n + smoothing)
    return ps, key, table

rows, coverage = {}, {}
KS = [2, 4, 6, 8, 10, 12]
for k in KS:
    ps, key, table = cpt_and_query(k)
    rows[k] = np.array([table[c][STATES.index('fatal injury')] for c in key])
    coverage[k] = (len(table), 2 ** k)

print('=' * 78)
print('P(personnel injury = fatal) — same data, same rule, only the parent cap changes')
print('=' * 78)
print(f'{"cap":>4} {"configs seen":>13} {"of possible":>12} {"coverage":>9}   {"mean P(fatal)":>13}')
for k in KS:
    seen, poss = coverage[k]
    print(f'{k:>4} {seen:>13,} {poss:>12,} {seen/poss:>8.1%}   {rows[k].mean():>13.4f}')

spread = np.abs(rows[12] - rows[8])
big = np.argsort(-spread)[:12]
print('\n' + '=' * 78)
print('Per-accident: P(fatal) under each cap  (12 accidents with largest 12-vs-8 gap)')
print('=' * 78)
hdr = f'{"ev_id":>16} ' + ' '.join(f'k={k:<6}' for k in KS) + f' {"actual":>16}'
print(hdr)
for i in big:
    vals = ' '.join(f'{rows[k][i]:<8.4f}' for k in KS)
    print(f'{str(df.ev_id.iloc[i]):>16} {vals} {df[SEV].iloc[i]:>16}')

print('\n' + '=' * 78)
print('Spread statistics across all 1,742 accidents')
print('=' * 78)
allv = np.vstack([rows[k] for k in KS])
rng = allv.max(axis=0) - allv.min(axis=0)
print(f'  max range over caps : {rng.max():.4f}')
print(f'  mean range          : {rng.mean():.4f}')
print(f'  median range        : {np.median(rng):.4f}')
print(f'  accidents where P(fatal) changes by >0.05 : {(rng > .05).sum():,} / {len(rng):,}')
print(f'  accidents where P(fatal) changes by >0.10 : {(rng > .10).sum():,} / {len(rng):,}')
print(f'  accidents where P(fatal) at least doubles : {(allv.max(0) > 2*np.maximum(allv.min(0),1e-9)).sum():,} / {len(rng):,}')

np.save('/home/claude/work/ntsb/rows.npy', allv)
df[['ev_id', SEV]].to_csv('/home/claude/work/ntsb/ids.csv', index=False)
