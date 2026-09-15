"""BRANCHING TREES for NTSB Bayesian-Network reproduction (diagnosis + prognosis).

The advisor (Maha) clarified that BOTH diagnosis and prognosis are TREES, not
linear chains: from a single event MULTIPLE outcomes can branch, each with its own
probability, and each branch can expand further. This module generalizes the two
existing engines into branching trees:

  * DIAGNOSIS TREE  -- root = the OUTCOME observed in the user's query (e.g. "fire");
      level-1 children = candidate causes ranked by P(cause | outcome); level-2
      children = co-occurring sub-causes ranked by P(cause2 | outcome AND cause1).
      Reuses zhang_diagnosis.empirical_cause_distribution (Zhang's Table-7 edge
      counting + denominator) at every branch.

  * PROGNOSIS TREE  -- root = the INITIAL event from the user's query; children =
      possible NEXT events with forward transition probability P(next | current);
      recurse to depth D, pruning low-probability / low-support branches. Reuses
      prognosis.build_transition_counts / transition_step (the same Markov hop the
      linear chain in prognosis.py used) but expands the top-B branches at each
      node instead of greedily following one.

QUERY-FIRST (hard requirement): every tree is generated STARTING FROM the user's
free-text narrative query -- embed the query (main_app.get_embedding), retrieve the
most similar incidents (main_app.find_top_matches), and build the tree over that
query-relevant population. Structured/Zhang components are layered on top, but the
entry point is always the narrative query.

This module IMPORTS its building blocks and does NOT edit them:
  zhang_diagnosis : detect_outcome, empirical_cause_distribution, GENERIC_CAUSES
  prognosis       : load_dataset, build_transition_counts, transition_step,
                    semantic_step_estimate, _ordered_occurrences, _incident_node_labels
  main_app        : get_embedding, find_top_matches  (query-first retrieval)

Run the demo:  tests/tree_demo.py
"""
from __future__ import annotations

import json
from pathlib import Path

import zhang_diagnosis as zd
import prognosis as pg

ROOT = Path(__file__).resolve().parent


# =====================================================================================
# Node / edge schema
# =====================================================================================
# A tree is a plain nested dict so it JSON-exports trivially (the Streamlit app will
# reuse the JSON). Every node has the SAME shape for diagnosis and prognosis:
#
#   {
#     "id":        unique string id (breadth-first: "n0", "n1", ...),
#     "label":     human-readable event / cause / outcome label,
#     "kind":      "outcome" | "cause" | "event"  (root kind differs per tree),
#     "edge_prob": P(this node | its parent context) in [0,1]; root = 1.0,
#     "path_prob": product of edge_prob along the path from the root (root = 1.0),
#     "n":         support count (numerator) backing edge_prob,
#     "denom":     denominator backing edge_prob,
#     "depth":     0 at the root,
#     "children":  list of child nodes (possibly empty),
#   }
#
# Edge-probability semantics:
#   DIAGNOSIS  edge_prob = P(cause | outcome [AND ancestor causes])  -- Zhang counting.
#              path_prob down the tree approximates P(all causes on path | outcome).
#   PROGNOSIS  edge_prob = P(next event | current event)             -- Markov hop.
#              path_prob down the tree = probability of that escalation path.
# =====================================================================================


def _new_node(label, kind, edge_prob, path_prob, n, denom, depth):
    return {
        "id": None,  # assigned in a final breadth-first pass
        "label": label,
        "kind": kind,
        "edge_prob": float(edge_prob),
        "path_prob": float(path_prob),
        "n": int(n) if n is not None else None,
        "denom": int(denom) if denom is not None else None,
        "depth": int(depth),
        "children": [],
    }


def _assign_ids(root):
    """Breadth-first id assignment so ids are stable + visualization-friendly."""
    queue = [root]
    i = 0
    while queue:
        node = queue.pop(0)
        node["id"] = f"n{i}"
        i += 1
        queue.extend(node["children"])
    return root


def _count_nodes(root):
    return 1 + sum(_count_nodes(c) for c in root["children"])


# =====================================================================================
# Query-first retrieval (shared)
# =====================================================================================
def _resolve_main_app(main_app):
    if main_app is not None:
        return main_app
    import main_app as _m
    return _m


def retrieve_pool(query, top_n_incidents=200, main_app=None):
    """QUERY-FIRST: embed the narrative query, retrieve the most similar incidents,
    return their ev_ids (incident rows only, de-duplicated, in similarity order).

    This is the entry point that makes every tree start from the user's free text.
    Requires the embedding index + network (main_app.get_embedding)."""
    m = _resolve_main_app(main_app)
    q_emb = m.get_embedding(query)
    _, matches = m.find_top_matches(q_emb)
    seen, ev_ids = set(), []
    for h in matches:
        if h.get("source") != "incident":
            continue
        ev = h.get("ev_id")
        if ev and ev not in seen:
            seen.add(ev)
            ev_ids.append(ev)
        if len(ev_ids) >= top_n_incidents:
            break
    return ev_ids


# =====================================================================================
# DIAGNOSIS TREE
# =====================================================================================
def outcome_position_audit(targets, ds, restrict_ev_ids=None):
    """Data-driven cause-vs-consequence rule (the principled rule Maha asked for,
    replacing the hand-written response blocklist).

    For every accident containing the outcome, locate the FIRST outcome occurrence
    in the Occurrence_No-ordered sequence, then classify every other label by
    where it sits relative to it:

      * an OCCURRENCE label counts 'before' if its first appearance is at or
        before the outcome occurrence, else 'after';
      * a FINDING label inherits the position of the occurrence it is attached
        to via Occurrence_No (Zhang's own finding->occurrence edge semantics).

    Returns {label: {'before': b, 'after': a, 'frac_after': a/(a+b)}}. A label
    whose appearances are mostly AFTER the outcome is a consequence (evacuation,
    emergency procedure, ...) and does not belong in a diagnosis tree.
    """
    allow = set(restrict_ev_ids) if restrict_ev_ids is not None else None
    stats: dict = {}
    for ev, inc in ds.items():
        if allow is not None and ev not in allow:
            continue
        descs, nos = pg._ordered_occurrences(inc)
        out_idx = next((i for i, d in enumerate(descs) if d in targets), None)
        if out_idx is None:
            continue
        idx_by_no = {no: i for i, no in enumerate(nos)}
        # first position of every non-outcome occurrence label
        first_occ: dict = {}
        for i, d in enumerate(descs):
            if d in targets:
                continue
            if d not in first_occ:
                first_occ[d] = i
        # first attached position of every finding label
        first_find: dict = {}
        for f in inc.get("findings", []):
            d = pg._s(f.get("finding_description")).lower()
            i = idx_by_no.get(pg._s(f.get("Occurrence_No")))
            if not d or i is None:
                continue
            if d not in first_find or i < first_find[d]:
                first_find[d] = i
        for d, i in list(first_occ.items()) + list(first_find.items()):
            b = stats.setdefault(d, [0, 0])
            b[0 if i <= out_idx else 1] += 1
    return {d: {"before": b, "after": a, "frac_after": a / (a + b)}
            for d, (b, a) in stats.items() if (a + b) > 0}


def downstream_labels_from_audit(audit, frac_after_min=0.5, min_support=2):
    """Labels the position audit classifies as CONSEQUENCES of the outcome:
    seen at least `min_support` times, appearing after the outcome more than
    `frac_after_min` of the time."""
    return {d for d, s in audit.items()
            if (s["before"] + s["after"]) >= min_support
            and s["frac_after"] > frac_after_min}


def _label_is_downstream(label: str, downstream: set) -> bool:
    """True if `label` matches a sequence-position consequence audit key.

    Audit keys come from occurrence/finding text (often shorter). Zhang
    cause-factor labels may add parenthetical qualifiers, e.g. audit key
    ``loss of engine power`` vs cause label
    ``Loss of engine power (total) - mechanical failure/malfunction``."""
    low = str(label or "").lower().strip()
    if not low or not downstream:
        return False
    for d in downstream:
        if low == d or low.startswith(d + " ") or low.startswith(d + "("):
            return True
    return False


def _causes_list(name, targets, ds, restrict_ev_ids, drop_labels, drop_generic,
                 cause_factor_only=False, exclude_responses=False,
                 downstream=None):
    """One ranked P(cause | outcome [AND ...]) distribution via Zhang counting,
    with optional labels removed (the parent cause, already-used labels) and the
    two generic catch-all buckets optionally dropped.

    cause_factor_only forwards Zhang's faithful Table-7 mode (only Cause/Factor
    findings count; nan/empty -> "Unknown quantity"; denominator = count(outcome));
    with it ON the level-1 edges reproduce the PUBLISHED Table 7 cell values exactly.

    exclude_responses drops operational post-incident labels (evacuation, emergency
    procedure, …) so the diagnosis tree shows upstream causes, not downstream actions.

    downstream (optional set) is the DATA-DRIVEN consequence set from
    outcome_position_audit: labels that sit after the outcome in the event
    sequence are dropped -- the principled sequence-position rule."""
    res = zd.empirical_cause_distribution(
        name, targets=targets, dataset=ds, restrict_ev_ids=restrict_ev_ids,
        cause_factor_only=cause_factor_only,
    )
    denom = res.get("outcome_count", 0)
    drop = {d.lower() for d in (drop_labels or set())}
    downstream = downstream or set()
    out = []
    for c in res.get("causes", []):
        lab = str(c.get("cause") or "")
        low = lab.lower()
        if low in drop:
            continue
        if drop_generic and low in zd.GENERIC_CAUSES:
            continue
        if exclude_responses and (low in zd.DIAGNOSIS_RESPONSE_LABELS
                                  or _label_is_downstream(low, downstream)):
            continue
        out.append({"label": lab, "prob": c["probability"], "n": c["n"], "denom": denom})
    return out, denom


def build_diagnosis_tree(query, top_n_incidents=200, branching=4, depth=2,
                         min_prob=0.03, drop_generic=True, dataset=None,
                         main_app=None, cause_factor_only=True,
                         exclude_responses=True, full_population=False,
                         min_n=1):
    """Build a branching DIAGNOSIS tree from a free-text narrative query.

    Pipeline (query-first):
      1. Embed the query + retrieve the most similar incidents (the pool).
      2. detect_outcome(query) -> the observed OUTCOME = tree root (e.g. "fire").
      3. Level 1: P(cause | outcome) over the pool (Zhang Table-7 counting) ->
         top-`branching` causes above `min_prob` become the root's children.
      4. Level >=2: for each cause node, re-rank P(cause2 | outcome AND cause1)
         by restricting the pool to incidents that also contain cause1, dropping
         cause labels already used on the path. Recurse until `depth`.

    Params:
      top_n_incidents : retrieval breadth (population the tree is built over).
      branching       : max children per node (top-B by probability).
      depth           : max cause levels below the outcome root (1 = causes only).
      min_prob        : prune branches whose edge probability < this.
      drop_generic    : drop the two generic catch-all cause buckets.
      cause_factor_only : faithful Zhang Table-7 mode (default True).
          When True only Cause/Factor findings count as contributory factors and
          the denominator is count(outcome) -- so the level-1 edges reproduce the
          PUBLISHED Table 7 cell values exactly (see docs/TREES_VALIDATION_REPORT.md).
          Set False only for legacy "all findings + occurrences" counting.
      exclude_responses : drop operational post-incident labels (evacuation,
          emergency procedure, …) from diagnosis branches (default True). Keeps
          the diagnosis tree upstream (causes) separate from prognosis (consequences).
      min_n           : minimum accident support for edges BELOW level 1
          (default 1 = keep all). Level-1 edges are never filtered by this so
          the Table 7 reproduction is untouched; deeper cohorts are small
          (n=8-14), so min_n=2 hides one-accident branches that read as noise.
      full_population : Zhang-reproduction mode (additive, default False). When
          True the tree is counted over ALL outcome accidents in the dataset
          (no query conditioning, denominator = count(outcome) over the whole
          window), so combined with cause_factor_only=True the level-1 edges
          reproduce the PUBLISHED Table 7 cells exactly (airframe = 32/102 =
          0.31373, denom 102). The query is still used only to detect the
          OUTCOME root. Default False keeps the query-first behaviour (counting
          over the retrieved similar-incident pool). This path needs no
          embedding/network call.

    Returns dict(meta=..., tree=root_node). Raises nothing for "no outcome": it
    returns meta['error'] instead so the caller/UI can show a message.
    """
    m = _resolve_main_app(main_app)
    ds = dataset if dataset is not None else m.refined_dataset

    det = zd.detect_outcome(query, dataset=ds)
    if det is None:
        return {"meta": {"query": query, "error": "no recognized outcome in query",
                         "kind": "diagnosis"}, "tree": None}
    name, targets = det
    targets = {t.lower() for t in targets}

    if full_population:
        # Zhang-reproduction mode: count over the WHOLE window (restricting to all
        # ev_ids == no restriction) so level-1 reproduces the published Table 7.
        pool = list(ds.keys())
    else:
        pool = retrieve_pool(query, top_n_incidents=top_n_incidents, main_app=m)

    # Principled cause-vs-consequence rule: sequence-position audit over the FULL
    # dataset (positions are a property of the outcome, not of the query pool).
    downstream = set()
    audit = {}
    if exclude_responses:
        audit = outcome_position_audit(targets, ds)
        downstream = downstream_labels_from_audit(audit)

    # Root = the observed outcome. denom = # outcome accidents in the pool.
    root_dist, root_denom = _causes_list(name, targets, ds, pool, set(), False,
                                         cause_factor_only=cause_factor_only,
                                         exclude_responses=exclude_responses,
                                         downstream=downstream)
    root = _new_node(name, "outcome", 1.0, 1.0, root_denom, root_denom, 0)

    def expand(node, used_labels, eligible_ev_ids):
        if node["depth"] >= depth:
            return
        if node["depth"] == 0:
            ranked = root_dist  # already computed over the pool
        else:
            ranked, _ = _causes_list(name, targets, ds, eligible_ev_ids,
                                     used_labels, drop_generic,
                                     cause_factor_only=cause_factor_only,
                                     exclude_responses=exclude_responses,
                                     downstream=downstream)
        kept = [c for c in ranked
                if c["prob"] >= min_prob
                and (node["depth"] == 0 or c["n"] >= min_n)][:branching]
        for c in kept:
            child = _new_node(c["label"], "cause", c["prob"],
                              node["path_prob"] * c["prob"], c["n"], c["denom"],
                              node["depth"] + 1)
            node["children"].append(child)
            # Restrict population for this child's sub-causes: pool incidents that
            # ALSO contain this cause label (anywhere in the incident).
            child_label = c["label"].lower()
            sub_pool = [ev for ev in eligible_ev_ids
                        if child_label in pg._incident_node_labels(ds.get(ev, {}))]
            expand(child, used_labels | {child_label}, sub_pool)

    # Apply min_prob/branching/drop_generic to level 1 too.
    root_dist = [c for c in root_dist
                 if (not drop_generic or c["label"].lower() not in zd.GENERIC_CAUSES)]
    expand(root, set(), pool)
    _assign_ids(root)

    meta = {
        "query": query,
        "kind": "diagnosis",
        "outcome": name,
        "outcome_labels": sorted(targets),
        "outcome_count_in_pool": root_denom,
        "retrieved_incidents": len(pool),
        "full_population": full_population,
        "population_mode": "all-incidents (Zhang reproduction)" if full_population
                           else "query-retrieved pool",
        "params": {"top_n_incidents": top_n_incidents, "branching": branching,
                   "depth": depth, "min_prob": min_prob, "drop_generic": drop_generic,
                   "cause_factor_only": cause_factor_only,
                   "exclude_responses": exclude_responses,
                   "full_population": full_population, "min_n": min_n},
        # data-driven consequences (sequence-position rule), with their audit
        # counts so the exclusion is fully auditable in the UI.
        "position_excluded": sorted(
            [{"label": d, **audit[d]} for d in downstream if d in audit],
            key=lambda r: (-r["frac_after"], -(r["before"] + r["after"]))),
        "n_nodes": _count_nodes(root),
        "prob_semantics": "edge_prob = P(cause | outcome AND ancestor causes); "
                          "path_prob = product down the path."
                          + (" [faithful Table-7 cause/factor mode]"
                             if cause_factor_only else ""),
    }
    return {"meta": meta, "tree": root}


# =====================================================================================
# PROGNOSIS TREE  (generalizes prognosis.py's linear chain into branching)
# =====================================================================================
def _seed_event(query, ds, tc):
    """QUERY-FIRST seed for the prognosis root: map the query to a concrete
    occurrence label that actually has forward transitions.

    Uses zhang_diagnosis.detect_outcome to map the free text to an occurrence
    family, then picks the family member with the most outgoing transitions in
    `tc` (so the tree can actually branch). Falls back to the raw detected name."""
    det = zd.detect_outcome(query, dataset=ds)
    if det is None:
        return None, None
    name, targets = det
    cands = [(t.lower(), tc["from_counts"].get(t.lower(), 0)) for t in targets]
    cands = [c for c in cands if c[1] > 0]
    if cands:
        cands.sort(key=lambda x: x[1], reverse=True)
        return cands[0][0], name
    # fallback: the detected name itself if it has transitions
    if tc["from_counts"].get(name.lower(), 0) > 0:
        return name.lower(), name
    return name.lower(), name


# Default terminal (leaf) outcomes -- Zhang's BN leaves: aircraft DAMAGE and
# personnel INJURY read off the structured incident fields (NOT the occurrence
# sequence), via prognosis.make_outcome_predicate. These are the "aircraft damage /
# injury" nodes Zhang draws at the end of every escalation path (Table 9 leaves,
# Fig. 11/13). (label, spec) pairs; order = render order.
# Injury uses ZHANG'S OWN per-person derivation (recovered from his released
# main.py: worst level across injury.xlsx rows, default no-injury), not
# events.ev_highest_injury -- so the leaves match the upgraded BN's semantics.
DEFAULT_LEAF_SPECS = [
    ("substantial aircraft damage", {"kind": "damage", "code": "SUBS"}),
    ("destroyed aircraft", {"kind": "damage", "code": "DEST"}),
    ("serious injury", {"kind": "injury_zhang", "code": "SERS"}),
    ("fatal injury", {"kind": "injury_zhang", "code": "FATL"}),
]


def _attach_leaf_outcomes(node, ds, leaf_specs, leaf_min_prob):
    """Attach Zhang terminal damage/injury leaves to a leaf event `node`.

    edge_prob = honest empirical reachability P(terminal outcome | the leaf event
    is present in the incident) = count(event & outcome) / count(event), over the
    GLOBAL dataset (these terminal rates are sparse, so the global population is the
    honest support). Each leaf carries n/denom and source='leaf-outcome-empirical'.
    This is an empirical conditional, NOT a full BN posterior -- the difference vs
    Zhang's diluted network posteriors is documented in the validation report."""
    label = node["label"]
    for out_label, spec in leaf_specs:
        pred = pg.make_outcome_predicate(spec)
        r = pg.honest_downstream([label], pred, ds)
        val = r.get("value")
        if val is None or val < leaf_min_prob:
            continue
        leaf = _new_node(out_label, "outcome", val, node["path_prob"] * val,
                         r["n_outcome"], r["n_cause"], node["depth"] + 1)
        leaf["source"] = "leaf-outcome-empirical"
        leaf["outcome_spec"] = spec
        node["children"].append(leaf)


def build_prognosis_tree(query, top_n_incidents=200, branching=3, depth=3,
                         min_prob=0.05, min_n=2, query_relevant=True,
                         drop_self_loops=True, dataset=None, main_app=None,
                         semantic_smoothing=False, sparse_n=15, semantic_k=50,
                         add_outcome_leaves=False, leaf_specs=None,
                         leaf_min_prob=0.0, deep_backoff=False,
                         bn_posteriors=False, drop_generic=False):
    """Build a branching PROGNOSIS tree from a free-text narrative query.

    Generalizes prognosis.multistep_chain (a single greedy path) into a TREE: at
    every node we expand the top-`branching` next-events instead of only the best.

    Pipeline (query-first):
      1. Embed the query + retrieve the most similar incidents (the pool).
      2. Build forward transition counts P(next | current) over the pool
         (query_relevant=True) or globally (False), via prognosis.build_transition_counts.
      3. Seed the root = the initial event detected from the query.
      4. Recurse: at each node take prognosis.transition_step(label) -> next-events
         with P and support N; keep the top-`branching` with P >= min_prob and
         N >= min_n; avoid revisiting labels on the path (acyclic). Stop at `depth`.

    Params mirror the diagnosis tree, plus:
      min_n             : prune hops backed by fewer than N transitions (sparsity gate).
      query_relevant    : build transitions over the retrieved pool (True, query-first)
                          or the whole dataset (False).
      semantic_smoothing: for sparse hops (N < sparse_n) attach a semantic-neighbour
                          estimate via prognosis.semantic_step_estimate (needs network).

    NEW (all additive, defaults preserve the historical tree):
      add_outcome_leaves: terminate leaf branches in Zhang's terminal DAMAGE/INJURY
                          outcomes (prognosis.make_outcome_predicate), each with an
                          empirical reachability probability. kind='outcome'.
      leaf_specs        : list of (label, spec) terminal outcomes (default
                          DEFAULT_LEAF_SPECS); leaf_min_prob prunes tiny leaves.
      deep_backoff      : when the query-relevant pool can't supply `branching`
                          well-supported hops at a node, BACK OFF to the global
                          transition model to fill the remaining slots so deep
                          branches don't bottom out at meaningless N=1 edges. Every
                          edge records source='pool' | 'global-backoff' (visible).
      bn_posteriors     : at branch points with >=2 active parent events on the path,
                          optionally use Zhang's Beta-CDF multiparent posterior
                          (prognosis.zhang_baseline_multiparent) for the edge instead
                          of the plain Markov hop. Edges that switch record
                          prob_source='bn-beta-cdf' and keep markov_p for comparison.
      drop_generic      : skip the two generic catch-all occurrence buckets
                          (GENERIC_CAUSES) as next-events, for cleaner escalation.

    Returns dict(meta=..., tree=root_node) (meta['error'] set if no seed/transitions).
    """
    m = _resolve_main_app(main_app) if (query_relevant or semantic_smoothing) else None
    if dataset is not None:
        ds = dataset
    elif m is not None:
        ds = m.refined_dataset
    else:
        ds = pg.load_dataset()

    # Query-first population for the transition model.
    pool = None
    if query_relevant:
        pool = retrieve_pool(query, top_n_incidents=top_n_incidents, main_app=m)
        sub_ds = {ev: ds[ev] for ev in pool if ev in ds}
    else:
        sub_ds = ds
    tc = pg.build_transition_counts(sub_ds)

    # Global transition model for the deep-chain sparsity backoff (built lazily).
    tc_global = None
    if deep_backoff:
        tc_global = tc if not query_relevant else pg.build_transition_counts(ds)

    # Zhang-faithful forward graph for the optional Beta-CDF multiparent path.
    edge_events = node_events = None
    if bn_posteriors:
        edge_events, node_events = pg.build_graph(ds)

    seed, family = _seed_event(query, ds, tc)
    # If the query-relevant pool is too sparse to host the seed, fall back to global.
    if (seed is None or tc["from_counts"].get(seed, 0) == 0) and query_relevant:
        tc = pg.build_transition_counts(ds)
        seed, family = _seed_event(query, ds, tc)

    if seed is None:
        return {"meta": {"query": query, "error": "no recognized initial event",
                         "kind": "prognosis"}, "tree": None}

    leaf_specs = leaf_specs if leaf_specs is not None else DEFAULT_LEAF_SPECS

    root = _new_node(seed, "event", 1.0, 1.0, None,
                     tc["from_counts"].get(seed, 0), 0)

    def _candidate_steps(label, visited, source_tc):
        """Ranked, filtered next-events from a given transition model."""
        out = []
        for s in pg.transition_step(label, source_tc):
            if drop_self_loops and s["to"] == label:
                continue
            if s["to"] in visited:
                continue
            if drop_generic and s["to"] in zd.GENERIC_CAUSES:
                continue
            if s["p"] < min_prob or s["n"] < min_n:
                continue
            out.append(s)
        return out

    def expand(node, visited, parents):
        if node["depth"] >= depth:
            return
        # Primary lane: the query-relevant (or global) pool transition model.
        kept = []
        for s in _candidate_steps(node["label"], visited, tc):
            s = dict(s); s["source"] = "pool" if pool is not None else "global"
            kept.append(s)
            if len(kept) >= branching:
                break
        # Deep-chain sparsity backoff: top up from the GLOBAL model when the pool
        # could not supply enough well-supported hops at this node.
        if deep_backoff and tc_global is not None and len(kept) < branching:
            have = {s["to"] for s in kept}
            for s in _candidate_steps(node["label"], visited | have, tc_global):
                s = dict(s); s["source"] = "global-backoff"
                kept.append(s)
                if len(kept) >= branching:
                    break
        for s in kept:
            edge_p, n, denom = s["p"], s["n"], s["denom"]
            prob_source = "markov"
            markov_p = edge_p
            # Optional Zhang Beta-CDF multiparent posterior at >=2-parent branches.
            if bn_posteriors and len(parents) >= 2:
                mp = pg.zhang_baseline_multiparent(
                    list(parents), {s["to"]}, edge_events, node_events)
                if mp.get("value") is not None:
                    edge_p = mp["value"]
                    prob_source = "bn-beta-cdf"
            child = _new_node(s["to"], "event", edge_p,
                              node["path_prob"] * edge_p, n, denom,
                              node["depth"] + 1)
            child["n_incidents"] = s.get("n_incidents")
            child["source"] = s["source"]
            child["prob_source"] = prob_source
            if prob_source == "bn-beta-cdf":
                child["markov_p"] = markov_p
            if semantic_smoothing and n < sparse_n and m is not None:
                try:
                    est = pg.semantic_step_estimate(node["label"], s["to"],
                                                    k=semantic_k, main_app=m)
                    child["semantic"] = est
                except Exception as exc:  # network/index issues are non-fatal
                    child["semantic_error"] = f"{type(exc).__name__}: {exc}"
            node["children"].append(child)
            expand(child, visited | {s["to"]}, parents | {s["to"]})

    expand(root, {seed}, {seed})

    # Terminate leaf branches in Zhang's terminal damage/injury outcomes. Any
    # childless EVENT node gets the leaves -- including the root itself when the
    # seed is a terminal occurrence with no onward transitions (e.g. a gear-collapse
    # query), so such a query still yields a useful damage/injury prognosis instead
    # of a bare one-node tree.
    if add_outcome_leaves:
        def _walk_leaves(node):
            if not node["children"]:
                if node["kind"] == "event":
                    _attach_leaf_outcomes(node, ds, leaf_specs, leaf_min_prob)
                return
            for c in list(node["children"]):
                _walk_leaves(c)
        _walk_leaves(root)

    _assign_ids(root)

    meta = {
        "query": query,
        "kind": "prognosis",
        "seed_event": seed,
        "seed_family": family,
        "transition_population": ("query-relevant pool" if query_relevant
                                  else "global dataset"),
        "incidents_in_population": (len(pool) if pool is not None
                                    else tc["n_incidents_used"]),
        "params": {"top_n_incidents": top_n_incidents, "branching": branching,
                   "depth": depth, "min_prob": min_prob, "min_n": min_n,
                   "query_relevant": query_relevant,
                   "semantic_smoothing": semantic_smoothing,
                   "add_outcome_leaves": add_outcome_leaves,
                   "deep_backoff": deep_backoff, "bn_posteriors": bn_posteriors,
                   "drop_generic": drop_generic},
        "n_nodes": _count_nodes(root),
        "prob_semantics": "edge_prob = P(next event | current event) Markov hop "
                          "(or Zhang Beta-CDF posterior where prob_source="
                          "'bn-beta-cdf'); leaf outcome edges = empirical "
                          "reachability P(damage/injury | leaf event); "
                          "path_prob = product down the path.",
    }
    return {"meta": meta, "tree": root}


# =====================================================================================
# Renderers
# =====================================================================================
def _short(label, width=58):
    label = str(label)
    return label if len(label) <= width else label[: width - 1] + "\u2026"


def render_tree(result, max_width=58):
    """Readable ASCII rendering of a diagnosis/prognosis tree result dict.

    Diagnosis edges read "<- P" (cause inferred FROM the outcome); prognosis edges
    read "-> P" (forward escalation). Each line shows edge prob, support (n/denom),
    and the cumulative path probability."""
    meta = result.get("meta", {})
    tree = result.get("tree")
    lines = []
    kind = meta.get("kind", "tree")
    if meta.get("error"):
        return f"[{kind}] ERROR: {meta['error']}  (query={meta.get('query')!r})"

    if kind == "diagnosis":
        lines.append(f"DIAGNOSIS TREE  query={meta.get('query')!r}")
        lines.append(f"  outcome = {meta.get('outcome')!r}  "
                     f"(outcome accidents in pool = {meta.get('outcome_count_in_pool')}, "
                     f"retrieved = {meta.get('retrieved_incidents')})")
        arrow = "<-"  # cause inferred from outcome
    else:
        lines.append(f"PROGNOSIS TREE  query={meta.get('query')!r}")
        lines.append(f"  seed event = {meta.get('seed_event')!r}  "
                     f"(population = {meta.get('transition_population')}, "
                     f"incidents = {meta.get('incidents_in_population')})")
        arrow = "->"  # forward escalation

    root = tree
    lines.append(f"\n[{root['kind'].upper()}] {root['label']}"
                 + (f"  (N={root['denom']})" if root.get("denom") else ""))

    def walk(node, prefix):
        kids = node["children"]
        for i, c in enumerate(kids):
            last = (i == len(kids) - 1)
            branch = "└─ " if last else "├─ "
            nb = f"N={c['n']}/{c['denom']}" if c.get("n") is not None else f"N≈{c['denom']}"
            tags = ""
            if "semantic" in c and c["semantic"].get("value") is not None:
                tags += f"  [sem~{c['semantic']['value']:.2f} k={c['semantic'].get('k')}]"
            if c.get("prob_source") == "bn-beta-cdf":
                tags += f"  [BN-betaCDF; markov={c.get('markov_p', 0):.3f}]"
            if c.get("source") == "global-backoff":
                tags += "  [global-backoff]"
            # leaf terminal damage/injury outcome marker
            a = "=>" if c.get("kind") == "outcome" and node["kind"] == "event" else arrow
            lines.append(
                f"{prefix}{branch}{a} p={c['edge_prob']:.3f}  ({nb})  "
                f"path={c['path_prob']:.3f}  {_short(c['label'], max_width)}{tags}"
            )
            ext = "   " if last else "│  "
            walk(c, prefix + ext)

    walk(root, "")
    lines.append(f"\n  nodes = {meta.get('n_nodes')}  | params = {meta.get('params')}")
    return "\n".join(lines)


# =====================================================================================
# JSON export (reused by the Streamlit visualization later)
# =====================================================================================
def to_json(result, indent=2):
    return json.dumps(result, indent=indent, ensure_ascii=False)


def export_json(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(result), encoding="utf-8")
    return path


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "engine caught fire during takeoff"
    print(render_tree(build_diagnosis_tree(q)))
    print()
    print(render_tree(build_prognosis_tree(q)))
