"""Would the graph be the same every time?

Resample the 1,742 accidents with replacement, re-rank the severity parents by
frequency (exactly as the recipe does), keep the top 8, rebuild the CPT, and
re-query. Nothing about the method changes -- only which 1,742 accidents the
build happened to see.
"""
import json, numpy as np, pandas as pd
from collections import Counter

BASE = '/mnt/user-data/uploads/export_for_review/'
df = pd.read_csv(BASE + 'accident_variable_matrix.csv')
meta = json.load(open(BASE + 'frozen_bn.json'))['meta']
SEV, K, B = 'personnel injury', 8, 300
STATES = meta['outcome_states'][SEV]
cand = meta['severity_parents']
FATAL = STATES.index('fatal injury')
rng = np.random.default_rng(0)
M = 5.0

P = df[cand].values.astype(np.int8)
y = df[SEV].values
prior_full = np.array([(y == s).mean() for s in STATES])

# three real accidents to track
track = {'20001213X31759': None, '20020917X01907': None, '20001211X10056': None}
for t in track:
    track[t] = int(np.where(df.ev_id.astype(str).values == t)[0][0])

sel_counter, post = Counter(), {t: [] for t in track}
top8_sets = []
for b in range(B):
    idx = rng.integers(0, len(df), len(df))
    Pb, yb = P[idx], y[idx]
    order = np.argsort(-Pb.sum(axis=0))[:K]
    chosen = tuple(sorted(cand[i] for i in order))
    top8_sets.append(chosen)
    for c in chosen:
        sel_counter[c] += 1
    keyb = [tuple(r) for r in Pb[:, order]]
    tbl = {}
    for cfg in set(keyb):
        m = np.array([k == cfg for k in keyb])
        sub, n = yb[m], m.sum()
        emp = np.array([(sub == s).sum() for s in STATES], float)
        tbl[cfg] = (emp + M * prior_full) / (n + M)
    for t, i in track.items():
        cfg = tuple(P[i, order])
        post[t].append(tbl.get(cfg, prior_full)[FATAL])

print('=' * 78)
print(f'{B} bootstrap rebuilds — cap fixed at {K}, only the sample varies')
print('=' * 78)
print(f'\ndistinct top-{K} parent sets selected: {len(set(top8_sets))}')
print(f'most common set chosen in {Counter(top8_sets).most_common(1)[0][1]}/{B} rebuilds\n')
print(f'{"how often each parent made the top-8":<62}{"":>6}')
for c, n in sel_counter.most_common():
    bar = '#' * int(40 * n / B)
    print(f'  {c[:52]:<52} {n/B:>6.1%} {bar}')

print('\n' + '=' * 78)
print('P(fatal) for three real accidents across the 300 rebuilds')
print('=' * 78)
print(f'{"ev_id":>16} {"actual":>16} {"min":>8} {"median":>8} {"max":>8} {"p5":>8} {"p95":>8}')
for t, v in post.items():
    v = np.array(v); a = df[SEV].iloc[track[t]]
    print(f'{t:>16} {a:>16} {v.min():>8.4f} {np.median(v):>8.4f} {v.max():>8.4f} '
          f'{np.percentile(v,5):>8.4f} {np.percentile(v,95):>8.4f}')

np.save('/home/claude/work/ntsb/boot.npy',
        np.array([post[t] for t in ['20001213X31759','20020917X01907','20001211X10056']]))
