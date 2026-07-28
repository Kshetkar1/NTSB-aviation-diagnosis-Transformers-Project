"""Three ways to estimate a forward conditional CPT P(outcome | cause(s)) on the
corrected 1982-2006 data, for sparse-cell comparison:

  1. raw_count   -- count(all causes & outcome) / count(all causes)   [noisy when sparse]
  2. beta_cdf    -- Zhang's method: beta.cdf(contribution, ALPHA, BETA) [parametric smoothing]
  3. semantic    -- YOUR novel method: among the K incidents most semantically
                    similar to the cause description, the fraction that had the
                    outcome [data-driven / neighbor smoothing]

ALPHA/BETA are Zhang's globally-calibrated values (reproduced to 9 decimals).
"""
from __future__ import annotations

import json
from pathlib import Path

import scipy.stats

import zhang_diagnosis as zd
from config import DATA_DIR

ALPHA = 1.04645351
BETA = 2.02591394

DP = DATA_DIR


def _outcome_cause_counts(outcome: str, ds: dict):
    """count[cause] = # outcome-accidents with cause->outcome edge; + total."""
    targets = zd.OUTCOME_ALIASES.get(outcome, {outcome})
    count: dict[str, int] = {}
    n_outcome = 0
    for inc in ds.values():
        causes = zd._causes_into_outcome(inc, targets)
        if not causes:
            continue
        n_outcome += 1
        for c in causes:
            count[c] = count.get(c, 0) + 1
    return count, sum(count.values()), n_outcome


def beta_cdf_cpt(cause_labels, count, total_contribution):
    contrib = sum(count.get(c, 0) for c in cause_labels) / total_contribution
    return contrib, float(scipy.stats.beta.cdf(contrib, a=ALPHA, b=BETA))


def raw_count_cpt(cause_labels, outcome, ds):
    targets = zd.OUTCOME_ALIASES.get(outcome, {outcome})

    def labels(inc):
        s = set()
        for f in inc.get("findings", []):
            s.add((f.get("finding_description") or "").strip().lower())
        for ev in inc.get("sequence_of_events", []):
            s.add((ev.get("Occurrence_Description") or "").strip().lower())
        return s

    def has_outcome(inc):
        return any((ev.get("Occurrence_Description") or "").strip().lower() in targets
                   for ev in inc.get("sequence_of_events", []))

    want = {c.strip().lower() for c in cause_labels}
    with_causes = [inc for inc in ds.values() if want.issubset(labels(inc))]
    if not with_causes:
        return 0.0, 0, 0
    n_out = sum(1 for inc in with_causes if has_outcome(inc))
    return n_out / len(with_causes), n_out, len(with_causes)


def semantic_cpt(query, outcome, k=50, main_app=None):
    """YOUR method: fraction of the k nearest (semantic) incidents that had outcome."""
    if main_app is None:
        import main_app as _m
        main_app = _m
    targets = zd.OUTCOME_ALIASES.get(outcome, {outcome})
    q = main_app.get_embedding(query)
    _, matches = main_app.find_top_matches(q)
    seen, ev_ids = set(), []
    for m in matches:
        if m.get("source") != "incident":
            continue
        ev = m.get("ev_id")
        if ev and ev not in seen:
            seen.add(ev)
            ev_ids.append(ev)
        if len(ev_ids) >= k:
            break
    ds = main_app.refined_dataset
    n_out = 0
    for ev in ev_ids:
        inc = ds.get(ev, {})
        if any((s.get("Occurrence_Description") or "").strip().lower() in targets
               for s in inc.get("sequence_of_events", [])):
            n_out += 1
    return (n_out / len(ev_ids) if ev_ids else 0.0), n_out, len(ev_ids)
