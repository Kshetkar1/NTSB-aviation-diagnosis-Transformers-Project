"""Parent-cap sensitivity using Zhang's EXACT CPT rules.

Severity node : support-weighted mixture of per-parent empirical severity
                distributions (bn_upgraded.py, FIX 2).
Boolean nodes : Beta-CDF, per-child Nelder-Mead calibration, floored by
                max active ratio (bn_build_ours.py, Zhang Eqs 10-14).

Both rules' parameters are recovered exactly from frozen_bn.json:
  * single-active CPT cell of a severity node  == that parent's empirical dist
  * single-active CPT cell of a Boolean node   == that parent's edge ratio
  * two-active cells give the support weights by least squares
"""
import json, numpy as np, pandas as pd, scipy.stats
from scipy.optimize import minimize

BASE = '/mnt/user-data/uploads/export_for_review/'
D = json.load(open(BASE + 'frozen_bn.json'))
N = {x['name']: x for x in D['nodes']}
df = pd.read_csv(BASE + 'accident_variable_matrix.csv')
SEV, FATAL = 'personnel injury', 0
STATES = D['meta']['outcome_states'][SEV]

# ---------- recover severity parameters ---------------------------------------
s = N[SEV]; P = s['parents']; K = len(P)
arr = np.array(s['cpt_flat']).reshape(s['cpt_shape'])
def cell(active):                      # active = list of parent indices set Yes
    idx = [1]*K
    for i in active: idx[i] = 0
    return arr[tuple(idx)]
dists = np.array([cell([i]) for i in range(K)])
default = cell([])                                        # all-inactive state
w = np.ones(K)                                            # relative supports
for j in range(1, K):
    d0, dj, dij = dists[0], dists[j], cell([0, j])
    denom = np.dot(d0 - dj, d0 - dj)
    r = float(np.dot(dij - dj, d0 - dj) / denom) if denom > 1e-12 else .5
    r = min(max(r, 1e-6), 1 - 1e-6)
    w[j] = w[0] * (1 - r) / r
order = np.argsort(-w)                                    # by support, as the code does
print('recovered severity parent supports (relative, ranked):')
for i in order:
    print(f'   {w[i]/w[order[0]]:7.3f}  {P[i][:58]}')

def severity_cpt_lookup(active_idx, keep):
    """Zhang/your exact rule on the parent subset `keep`."""
    a = [i for i in active_idx if i in keep]
    if not a: return default
    ww = w[a]; dd = dists[a]
    d = (dd * ww[:, None]).sum(0) / ww.sum()
    return d / d.sum()

present = df[P].values.astype(bool)
KS = [2, 4, 6, 8, 10, 12]
rows = {}
for k in KS:
    keep = set(order[:k].tolist())
    rows[k] = np.array([severity_cpt_lookup(np.where(r)[0], keep)[FATAL] for r in present])

print('\n' + '='*80)
print('SEVERITY NODE — exact support-weighted mixture rule, only the cap varies')
print('='*80)
print(f'{"cap":>4}  {"mean P(fatal)":>13}  {"distinct values":>15}')
for k in KS:
    print(f'{k:>4}  {rows[k].mean():>13.4f}  {len(np.unique(np.round(rows[k],6))):>15}')

allv = np.vstack([rows[k] for k in KS])
rng = allv.max(0) - allv.min(0)
print(f'\n  max range over caps                       : {rng.max():.4f}')
print(f'  mean range                                : {rng.mean():.4f}')
print(f'  accidents changing by > 0.05              : {(rng>.05).sum():,} / {len(rng):,}')
print(f'  accidents changing by > 0.10              : {(rng>.10).sum():,} / {len(rng):,}')
print(f'  accidents where P(fatal) at least doubles : {(allv.max(0) > 2*np.maximum(allv.min(0),1e-9)).sum():,} / {len(rng):,}')

big = np.argsort(-rng)[:8]
print(f'\n{"ev_id":>16} {"actual":>16} ' + ' '.join(f'k={k:<6}' for k in KS))
for i in big:
    print(f'{str(df.ev_id.iloc[i]):>16} {df[SEV].iloc[i]:>16} ' +
          ' '.join(f'{rows[k][i]:<8.4f}' for k in KS))

np.save('/home/claude/work/ntsb/rows_exact.npy', allv)

# ---------- Boolean node, exact Beta-CDF --------------------------------------
def calibrate_beta(ratios):
    norm = ratios / ratios.sum()
    f = lambda x: float(np.mean((scipy.stats.beta.cdf(norm, a=x[0], b=x[1]) - ratios)**2))
    return tuple(minimize(f, [2.0, 1.0], method='Nelder-Mead', tol=1e-7).x)

def cpt_yes(active_mask, ratios, ab):
    k = int(active_mask.sum())
    if k == 0: return 0.0
    if k == 1: return float(ratios[active_mask][0])
    c = float(ratios[active_mask].sum() / ratios.sum())
    return max(float(scipy.stats.beta.cdf(c, a=ab[0], b=ab[1])), float(ratios[active_mask].max()))

cands = [x for x in D['nodes'] if len(x['parents']) >= 10 and x['states'] == ['Yes','No']
         and x['name'] in df.columns and all(p in df.columns for p in x['parents'])]
node = max(cands, key=lambda x: df[x['name']].sum())
bp = node['parents']; Kb = len(bp)
barr = np.array(node['cpt_flat']).reshape(node['cpt_shape'])
def bcell(i):
    idx = [1]*Kb; idx[i] = 0
    return barr[tuple(idx)][0]
ratios_full = np.array([bcell(i) for i in range(Kb)])
bo = np.argsort(-ratios_full)
bpres = df[bp].values.astype(bool)

print('\n' + '='*80)
print(f'BOOLEAN NODE — exact Zhang Beta-CDF rule (Eqs 10-14), only the cap varies')
print(f'target: "{node["name"]}"  ({Kb} parents)')
print('='*80)
BKS = [2, 4, 6, 8, 10, Kb]
brows = {}
for k in BKS:
    keep = bo[:k]
    r = ratios_full[keep]
    ab = calibrate_beta(r) if len(r) >= 2 else (1., 1.)
    brows[k] = np.array([cpt_yes(row[keep], r, ab) for row in bpres])
print(f'{"cap":>4}  {"mean P(Yes)":>12}')
for k in BKS: print(f'{k:>4}  {brows[k].mean():>12.4f}')
ball = np.vstack([brows[k] for k in BKS]); brng = ball.max(0) - ball.min(0)
print(f'\n  max range over caps          : {brng.max():.4f}')
print(f'  accidents changing by > 0.05 : {(brng>.05).sum():,} / {len(brng):,}')
print(f'  accidents changing by > 0.10 : {(brng>.10).sum():,} / {len(brng):,}')
