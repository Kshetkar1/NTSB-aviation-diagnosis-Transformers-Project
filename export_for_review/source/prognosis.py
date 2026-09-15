"""PROGNOSIS (forward escalation) estimators for the NTSB Bayesian-Network reproduction.

A forward cell is P(outcome | evidence-cause(s)) -- "given this cause, how likely is the
aircraft to escalate to this outcome". This module implements THREE estimators for such a
cell so they can be compared head-to-head against Zhang (2026) Table 9:

  1. zhang_baseline_cpt  -- Zhang's EXACT rule, replicated on our data (the BASELINE).
        single parent : ratio = |events with edge cause->outcome| / |events where cause
                        appears as a graph node (from OR to)|, and if that ratio == 1.0 it
                        is multiplied by 0.95 (Zhang's hardcoded cap).
        >=2 parents   : Beta-CDF path -- beta.cdf(contribution, ALPHA, BETA) floored by the
                        max active-parent ratio, where contribution = (sum active-parent
                        ratios) / (sum of ALL parent ratios of the outcome node).
        See `Zhang's Approach 2026/main.py` lines ~751-916 (dictElement ratio+cap,
        constructCPT). ALPHA/BETA are Zhang's global calibrated values (from sparse_cpt).

  2. honest_forward_cpt  -- count(outcome & cause) / count(cause), NO cap, ALWAYS returns N
        (the denominator) so sparsity is visible. This is the principled conditional.

  3. semantic_forward_cpt -- among the K incidents most semantically similar to the
        evidence/cause description, the fraction that escalate to the outcome (neighbour
        smoothing via the embedding retrieval index). Returns K and N.

Nothing here mutates Zhang's reference files; it re-derives the same quantities from
`data/processed/refined_dataset_1982_2006.json`.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import scipy.stats

import zhang_diagnosis as zd
from config import WINDOW_DATA_PATH
from sparse_cpt import ALPHA, BETA

DATASET = WINDOW_DATA_PATH

# Zhang's leaf-node label maps (damage / injury). Used so the forward graph contains the
# same terminal nodes Zhang draws (last occurrence -> aircraft damage, -> injury level).
DAMAGE_LABELS = {
    "DEST": "destroyed aircraft",
    "SUBS": "substantial damage",
    "MINR": "minor damage",
    "NONE": "no damage",
    "UNK": "unknown damage",
}
INJURY_LABELS = {
    "FATL": "fatal injury",
    "SERS": "serious injury",
    "MINR": "minor injury",
    "NONE": "no injury",
}


def load_dataset(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or DATASET).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------------------
# Label helpers (data hygiene: every field cast str(x or "") -> never NaN/float crashes).
# --------------------------------------------------------------------------------------
def _s(x) -> str:
    return str(x or "").strip()


def _ordered_occurrences(inc: dict):
    """(descs, occ_nos) sorted by Occurrence_No, lowercased descriptions, blanks dropped."""
    seq = sorted(
        inc.get("sequence_of_events", []),
        key=lambda e: int(_s(e.get("Occurrence_No")) or 0),
    )
    descs, nos = [], []
    for e in seq:
        d = _s(e.get("Occurrence_Description")).lower()
        if d:
            descs.append(d)
            nos.append(_s(e.get("Occurrence_No")))
    return descs, nos


def _occ_by_no(inc: dict) -> dict:
    out = {}
    for e in inc.get("sequence_of_events", []):
        d = _s(e.get("Occurrence_Description")).lower()
        if d:
            out[_s(e.get("Occurrence_No"))] = d
    return out


def _incident_node_labels(inc: dict) -> set:
    """All finding + occurrence labels present in an incident (lowercased)."""
    s = set()
    for f in inc.get("findings", []):
        d = _s(f.get("finding_description")).lower()
        if d:
            s.add(d)
    for e in inc.get("sequence_of_events", []):
        d = _s(e.get("Occurrence_Description")).lower()
        if d:
            s.add(d)
    return s


# --------------------------------------------------------------------------------------
# Zhang-faithful directed graph (buildOneGraphRep, re-expressed over the refined JSON).
# --------------------------------------------------------------------------------------
def build_edges(inc: dict) -> set:
    """Directed edges (from_label, to_label) for ONE incident, matching Zhang's graph:

      * finding(subject) -> occurrence   (finding attached by matching Occurrence_No)
      * occurrence[i]    -> occurrence[i+1]   (consecutive in the sequence)
      * last occurrence  -> aircraft-damage node
      * last occurrence  -> injury node

    Labels are lowercased; edges are de-duplicated within the incident (Zhang counts an
    edge once per eventId)."""
    edges: set = set()
    occ_by_no = _occ_by_no(inc)

    # finding -> occurrence (same Occurrence_No)
    for f in inc.get("findings", []):
        fd = _s(f.get("finding_description")).lower()
        tgt = occ_by_no.get(_s(f.get("Occurrence_No")))
        if fd and tgt:
            edges.add((fd, tgt))

    descs, _nos = _ordered_occurrences(inc)
    for i in range(len(descs) - 1):
        if descs[i] != descs[i + 1]:
            edges.add((descs[i], descs[i + 1]))

    if descs:
        last = descs[-1]
        dmg = DAMAGE_LABELS.get(_s(inc.get("damage")).upper())
        if dmg:
            edges.add((last, dmg))
        inj = INJURY_LABELS.get(_s(inc.get("ev_highest_injury")).upper())
        if inj:
            edges.add((last, inj))
    return edges


def build_graph(ds: dict):
    """Returns (edge_events, node_events):

      edge_events[(from,to)] = set of ev_ids containing that edge
      node_events[node]      = set of ev_ids where node appears as `from` OR `to`
    """
    edge_events: dict = defaultdict(set)
    node_events: dict = defaultdict(set)
    for ev, inc in ds.items():
        if not inc.get("sequence_of_events"):
            continue
        for (a, b) in build_edges(inc):
            edge_events[(a, b)].add(ev)
            node_events[a].add(ev)
            node_events[b].add(ev)
    return edge_events, node_events


def resolve_outcome_targets(outcome: str, ds: dict) -> set:
    """LOEP-family aware outcome label set, via zhang_diagnosis.detect_outcome (so 'loss of
    engine power' expands to all 5 coded variants, the same node Zhang's Table 9 column is)."""
    det = zd.detect_outcome(outcome, dataset=ds)
    if det is not None:
        return {t.lower() for t in det[1]}
    return {outcome.lower()}


# --------------------------------------------------------------------------------------
# Estimator 1: ZHANG BASELINE (faithful).
# --------------------------------------------------------------------------------------
def zhang_baseline_cpt(cause: str, outcome_targets: set, edge_events, node_events,
                       cap: bool = True):
    """Zhang's exact SINGLE-parent cell value.

    Returns dict(value, joint_n, denom_n, raw_ratio, capped). value is None if the cause
    label never appears in the graph."""
    cause = cause.lower()
    joint = set()
    for (a, b), evs in edge_events.items():
        if a == cause and b in outcome_targets:
            joint |= evs
    denom = node_events.get(cause, set())
    if not denom:
        return {"value": None, "joint_n": 0, "denom_n": 0, "raw_ratio": None,
                "capped": False}
    raw = len(joint) / len(denom)
    val = raw
    capped = False
    if cap and raw == 1.0:
        val = raw * 0.95
        capped = True
    return {"value": val, "joint_n": len(joint), "denom_n": len(denom),
            "raw_ratio": raw, "capped": capped}


def parent_ratios(outcome_targets: set, edge_events, node_events) -> dict:
    """All parents of the outcome node -> their Zhang (capped) ratio. This is Zhang's
    parentInfo `count` column (which is the dictElement ratio, not a raw count)."""
    causes = {a for (a, b) in edge_events if b in outcome_targets}
    out = {}
    for c in causes:
        r = zhang_baseline_cpt(c, outcome_targets, edge_events, node_events, cap=True)
        if r["value"] is not None:
            out[c] = r["value"]
    return out


def zhang_baseline_multiparent(causes, outcome_targets: set, edge_events, node_events):
    """Zhang's Beta-CDF path for >=2 simultaneously-active parents (constructCPT).

    contribution = (sum active-parent ratios) / (sum of ALL parent ratios of the node);
    Yes = beta.cdf(contribution, ALPHA, BETA), then floored by max(active ratios)."""
    causes = [c.lower() for c in causes]
    parents = parent_ratios(outcome_targets, edge_events, node_events)
    total = sum(parents.values())
    active = [parents[c] for c in causes if c in parents]
    missing = [c for c in causes if c not in parents]
    if len(active) < 2 or total <= 0:
        return {"value": None, "contribution": None, "active_ratios": active,
                "missing": missing, "n_parents": len(parents)}
    contrib = sum(active) / total
    yes = float(scipy.stats.beta.cdf(contrib, a=ALPHA, b=BETA))
    floor = max(active)
    yes = max(yes, floor)
    return {"value": yes, "contribution": contrib, "active_ratios": active,
            "missing": missing, "n_parents": len(parents), "floor": floor,
            "beta_raw": float(scipy.stats.beta.cdf(contrib, a=ALPHA, b=BETA))}


# --------------------------------------------------------------------------------------
# Estimator 2: HONEST forward conditional (no cap, N always reported).
# --------------------------------------------------------------------------------------
def honest_forward_cpt(cause_labels, outcome_targets: set, ds: dict,
                       require_all: bool = False):
    """P(outcome occurs | cause present) = count(cause & outcome) / count(cause), NO cap.

    cause_labels: one label, or a family of labels (e.g. all 'engine instruments*').
    require_all: if True an incident must contain ALL cause_labels; if False (default for
    a family) it must contain ANY. Returns dict(value, n_outcome, n_cause)."""
    want = {c.lower() for c in cause_labels}

    def has_cause(inc) -> bool:
        labs = _incident_node_labels(inc)
        return want.issubset(labs) if require_all else bool(want & labs)

    def has_outcome(inc) -> bool:
        descs, _ = _ordered_occurrences(inc)
        return any(d in outcome_targets for d in descs)

    with_cause = [inc for inc in ds.values()
                  if inc.get("sequence_of_events") and has_cause(inc)]
    n_cause = len(with_cause)
    if n_cause == 0:
        return {"value": None, "n_outcome": 0, "n_cause": 0}
    n_out = sum(1 for inc in with_cause if has_outcome(inc))
    return {"value": n_out / n_cause, "n_outcome": n_out, "n_cause": n_cause}


# --------------------------------------------------------------------------------------
# Estimator 3: IMPROVED semantic-neighbour estimate.
# --------------------------------------------------------------------------------------
def semantic_forward_cpt(query: str, outcome_targets: set, k: int = 50,
                         main_app=None, outcome_predicate=None):
    """Among the K nearest (semantic) incidents to `query`, the fraction that escalate to
    the outcome. Returns dict(value, n_outcome, k). Requires the embedding index + network.

    outcome_predicate(inc) -> bool overrides occurrence-label matching (used for damage /
    injury leaf outcomes whose presence is read from the incident fields, not the seq)."""
    if main_app is None:
        import main_app as _m
        main_app = _m

    if outcome_predicate is None:
        def outcome_predicate(inc):
            descs, _ = _ordered_occurrences(inc)
            return any(d in outcome_targets for d in descs)

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
    n_out = sum(1 for ev in ev_ids if outcome_predicate(ds.get(ev, {})))
    return {"value": (n_out / len(ev_ids) if ev_ids else None),
            "n_outcome": n_out, "k": len(ev_ids)}


# --------------------------------------------------------------------------------------
# Downstream / leaf outcome predicates + honest downstream reachability.
# --------------------------------------------------------------------------------------
def zhang_injury_code(inc: dict) -> str:
    """Zhang's per-accident injury level (FATL/SERS/MINR/NONE), recovered verbatim
    from his released main.py (calculate_injury_level): sum the per-person injury
    rows (injury.xlsx), take the WORST level present, and default to NONE when the
    accident has no usable injury rows. This is NOT events.ev_highest_injury --
    the two disagree on accidents whose events row is blank/unmapped."""
    tot = {"FATL": 0.0, "SERS": 0.0, "MINR": 0.0, "NONE": 0.0, "TOTL": 0.0}
    for r in inc.get("injuries") or []:
        lvl = _s(r.get("injury_level")).upper()
        if lvl in tot:
            try:
                tot[lvl] += float(r.get("inj_person_count") or 0)
            except (TypeError, ValueError):
                pass
    if tot["TOTL"] != 0:
        for code in ("FATL", "SERS", "MINR"):
            if tot[code] > 0:
                return code
    return "NONE"


def make_outcome_predicate(spec: dict):
    """Build an incident->bool predicate for a downstream/leaf outcome.

    spec kinds:
      {'kind':'occurrence', 'targets': set(...)}  -> occurrence label present in sequence
      {'kind':'damage', 'code':'DEST'}            -> incident damage code matches
      {'kind':'injury', 'code':'SERS'}            -> incident highest-injury code matches
                                                     (events.ev_highest_injury field)
      {'kind':'injury_zhang', 'code':'SERS'}      -> Zhang's per-person derivation
                                                     (zhang_injury_code) matches
    """
    kind = spec["kind"]
    if kind == "occurrence":
        targets = {t.lower() for t in spec["targets"]}

        def pred(inc):
            descs, _ = _ordered_occurrences(inc)
            return any(d in targets for d in descs)
        return pred
    if kind == "damage":
        code = spec["code"].upper()
        return lambda inc: _s(inc.get("damage")).upper() == code
    if kind == "injury":
        code = spec["code"].upper()
        return lambda inc: _s(inc.get("ev_highest_injury")).upper() == code
    if kind == "injury_zhang":
        code = spec["code"].upper()
        return lambda inc: zhang_injury_code(inc) == code
    raise ValueError(f"unknown outcome spec kind: {kind!r}")


def honest_downstream(cause_labels, outcome_predicate, ds: dict,
                      require_all: bool = False):
    """Honest reachability: among incidents containing the evidence cause, the fraction
    that ALSO reach the (downstream/leaf) outcome. This is an empirical conditional, NOT a
    full BN posterior -- it is reported with N so the (often severe) sparsity is explicit."""
    want = {c.lower() for c in cause_labels}

    def has_cause(inc) -> bool:
        labs = _incident_node_labels(inc)
        return want.issubset(labs) if require_all else bool(want & labs)

    with_cause = [inc for inc in ds.values()
                  if inc.get("sequence_of_events") and has_cause(inc)]
    n_cause = len(with_cause)
    if n_cause == 0:
        return {"value": None, "n_outcome": 0, "n_cause": 0}
    n_out = sum(1 for inc in with_cause if outcome_predicate(inc))
    return {"value": n_out / n_cause, "n_outcome": n_out, "n_cause": n_cause}


# --------------------------------------------------------------------------------------
# MULTI-STEP ESCALATION CHAINS (additive -- the "small issue -> bigger issue -> beyond"
# story the advisor asked for). Built from CONSECUTIVE Occurrence_No-ordered occurrence
# pairs, identical counting semantics to main_app._transition_probabilities but computed
# GLOBALLY over the whole dataset (not just a retrieved neighbourhood). Every step carries
# its support N so sparsity is always visible; cumulative = product of per-hop steps.
# --------------------------------------------------------------------------------------
def build_transition_counts(ds: dict):
    """One forward Markov step over occurrence sequences, GLOBAL across all incidents.

    Returns dict with:
      pair_counts[(a,b)]   = # of consecutive a->b transitions observed (numerator)
      from_counts[a]       = # of transitions leaving a (denominator for P(b|a))
      pair_incidents[(a,b)]= # of DISTINCT incidents containing the a->b step (conservative)
      from_incidents[a]    = # of DISTINCT incidents where a has a following event

    Data hygiene: every field cast via str(x or ""), sequence ordered by Occurrence_No,
    blank descriptions dropped, incidents with <2 usable events skipped.
    """
    pair_counts: dict = defaultdict(int)
    from_counts: dict = defaultdict(int)
    pair_incidents: dict = defaultdict(set)
    from_incidents: dict = defaultdict(set)
    n_used = 0
    for ev, inc in ds.items():
        descs, _nos = _ordered_occurrences(inc)
        if len(descs) < 2:
            continue
        n_used += 1
        for i in range(len(descs) - 1):
            a, b = descs[i], descs[i + 1]
            pair_counts[(a, b)] += 1
            from_counts[a] += 1
            pair_incidents[(a, b)].add(ev)
            from_incidents[a].add(ev)
    return {
        "pair_counts": dict(pair_counts),
        "from_counts": dict(from_counts),
        "pair_incidents": {k: v for k, v in pair_incidents.items()},
        "from_incidents": {k: v for k, v in from_incidents.items()},
        "n_incidents_used": n_used,
    }


def transition_step(a: str, tc: dict):
    """All next-events of `a`, each with raw transition estimate + support, sorted by N.

    Returns list of dict(to, p, n, denom, n_incidents). p = P(to | a) in [0,1]."""
    a = a.lower()
    denom = tc["from_counts"].get(a, 0)
    out = []
    if not denom:
        return out
    for (x, b), n in tc["pair_counts"].items():
        if x != a:
            continue
        out.append({
            "to": b,
            "p": n / denom,
            "n": n,
            "denom": denom,
            "n_incidents": len(tc["pair_incidents"].get((a, b), ())),
        })
    out.sort(key=lambda r: (r["n"], r["p"]), reverse=True)
    return out


def rank_multistep_chains(tc: dict, hops: int = 2, top: int = 20,
                          min_step_n: int = 1, drop_self_loops: bool = True):
    """Enumerate all length-`hops` chains and rank by their WEAKEST step support
    (min N along the path) -- the honest measure of how well the data backs a chain.

    Returns list of dict(events=[...], steps=[{from,to,p,n,...}], min_n, cumulative).
    """
    # adjacency: a -> list of (b, n)
    adj: dict = defaultdict(list)
    for (a, b), n in tc["pair_counts"].items():
        if drop_self_loops and a == b:
            continue
        if n >= min_step_n:
            adj[a].append((b, n))

    results = []

    def extend(path_events, path_steps):
        if len(path_steps) == hops:
            min_n = min(s["n"] for s in path_steps)
            cumulative = 1.0
            for s in path_steps:
                cumulative *= s["p"]
            results.append({
                "events": list(path_events),
                "steps": list(path_steps),
                "min_n": min_n,
                "cumulative": cumulative,
            })
            return
        last = path_events[-1]
        denom = tc["from_counts"].get(last, 0)
        if not denom:
            return
        for (b, n) in adj.get(last, ()):
            if b in path_events:  # acyclic chains only
                continue
            step = {
                "from": last, "to": b, "p": n / denom, "n": n, "denom": denom,
                "n_incidents": len(tc["pair_incidents"].get((last, b), ())),
            }
            extend(path_events + [b], path_steps + [step])

    for a in list(tc["from_counts"]):
        extend([a], [])

    results.sort(key=lambda r: (r["min_n"], r["cumulative"]), reverse=True)
    return results[:top]


def multistep_chain(seed: str, tc: dict, hops: int = 3, min_step_n: int = 1,
                    drop_self_loops: bool = True, avoid=None):
    """Greedily follow the best-SUPPORTED (highest N) transition out of `seed`, `hops`
    times, building one concrete escalation chain. Acyclic. Returns dict(seed, steps,
    cumulative, events) where steps carry per-hop p / n / denom / n_incidents.

    `avoid` -- optional set of labels to never step into (e.g. uninformative buckets)."""
    seed = seed.lower()
    avoid = {a.lower() for a in (avoid or set())}
    visited = {seed}
    events = [seed]
    steps = []
    cumulative = 1.0
    cur = seed
    for _ in range(hops):
        cands = [s for s in transition_step(cur, tc)
                 if s["to"] not in visited
                 and not (drop_self_loops and s["to"] == cur)
                 and s["to"] not in avoid
                 and s["n"] >= min_step_n]
        if not cands:
            break
        nxt = dict(cands[0])
        nxt["from"] = cur
        steps.append(nxt)
        cumulative *= nxt["p"]
        visited.add(nxt["to"])
        events.append(nxt["to"])
        cur = nxt["to"]
    return {"seed": seed, "events": events, "steps": steps, "cumulative": cumulative}


def chain_from_events(events, tc: dict):
    """Score an EXPLICIT, curated escalation path (list of event labels) against the data.

    For each consecutive (a,b) hop returns the raw transition estimate p=P(b|a), its
    support N and distinct-incident support, then the cumulative product of steps. Lets us
    present a readable, data-backed chain without letting a generic high-frequency bucket
    hijack a greedy walk. Returns dict(events, steps, cumulative)."""
    events = [e.lower() for e in events]
    steps = []
    cumulative = 1.0
    for a, b in zip(events, events[1:]):
        denom = tc["from_counts"].get(a, 0)
        n = tc["pair_counts"].get((a, b), 0)
        p = (n / denom) if denom else 0.0
        cumulative *= p
        steps.append({
            "from": a, "to": b, "p": p, "n": n, "denom": denom,
            "n_incidents": len(tc["pair_incidents"].get((a, b), ())),
        })
    return {"events": events, "steps": steps, "cumulative": cumulative}


def semantic_step_estimate(current_event: str, next_event: str, k: int = 50,
                           main_app=None):
    """Semantic-neighbour SMOOTHED estimate of a single hop, for when the raw transition
    N is too small to trust. Among the K incidents most similar to `current_event`, the
    fraction whose sequence contains `next_event`. Requires the embedding index + network.

    Returns dict(value, n_outcome, k) -- same shape as semantic_forward_cpt."""
    return semantic_forward_cpt(current_event, {next_event.lower()}, k=k,
                                main_app=main_app)
