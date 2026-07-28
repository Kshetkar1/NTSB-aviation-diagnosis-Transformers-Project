# Branching Trees: Diagnosis & Prognosis (Design)

> Module: `trees.py` · Demo: `tests/tree_demo.py` · JSON export: `outputs/*.json`

## 0. Why this exists

The advisor (Maha) clarified that **both diagnosis and prognosis are TREES, not
linear multi-step chains**. From a single event, **multiple** different outcomes can
branch, each with its own probability, and each branch can expand further.

The existing engine had:

- `zhang_diagnosis.py` — a *flat* ranked list of causes: `P(cause | outcome)`.
- `prognosis.py` — a *linear* escalation chain (`multistep_chain`, `chain_from_events`):
  one greedy path `P(next | current)` per hop.

`trees.py` generalizes both into **branching trees** while reusing those engines
unchanged (it imports them; it does not edit them). The diagnosis tree is implemented
and validated end-to-end; the prognosis tree is also implemented (a generalization of
`prognosis.py`'s chain into top-B branching).

## 1. Hard requirement: QUERY-FIRST

Every tree is generated **starting from the user's free-text narrative query**:

```
narrative query  ──embed──▶  retrieve similar incidents  ──▶  build tree over that pool
                main_app.get_embedding   main_app.find_top_matches
```

This is `trees.retrieve_pool(query, top_n_incidents, main_app)` and it is the entry
point for **both** trees. Zhang/structured counting is layered *on top* of this
query-relevant population — the entry point is always the narrative.

## 2. Node / edge schema

A tree is a plain nested `dict` (so it JSON-exports trivially for the Streamlit app).
Every node — diagnosis or prognosis — has the **same shape**:

```jsonc
{
  "id":        "n0",            // breadth-first id, stable for visualization
  "label":     "fire",         // event / cause / outcome label
  "kind":      "outcome" | "cause" | "event",
  "edge_prob": 1.0,             // P(this node | its parent context); root = 1.0
  "path_prob": 1.0,             // product of edge_prob from root to here; root = 1.0
  "n":         69,              // support numerator backing edge_prob (null at prognosis root)
  "denom":     69,              // support denominator backing edge_prob
  "depth":     0,               // 0 at root
  "children":  [ ... ]          // recursive
}
```

The whole result is `{"meta": {...}, "tree": <root node>}`. `meta` carries the query,
tree kind, root info, params, node count, and a `prob_semantics` string.

Prognosis nodes additionally carry `n_incidents` (distinct-incident support) and,
when semantic smoothing is on, `semantic` / `semantic_error`.

### Edge-probability semantics

| Tree | `edge_prob` | `path_prob` down the tree |
|---|---|---|
| **Diagnosis** | `P(cause \| outcome AND ancestor causes)` — Zhang Table-7 counting | ≈ `P(all causes on the path \| outcome)` |
| **Prognosis** | `P(next event \| current event)` — Markov forward hop | probability of that escalation path |

## 3. Diagnosis tree

`trees.build_diagnosis_tree(query, top_n_incidents=200, branching=4, depth=2, min_prob=0.03, drop_generic=True)`

1. **Query-first:** embed query → retrieve pool of similar incidents.
2. **Root** = the observed outcome from `zhang_diagnosis.detect_outcome(query)`
   (e.g. `"fire"`). `denom` = number of outcome accidents within the pool.
3. **Level 1** = `P(cause | outcome)` over the pool, computed by
   `zhang_diagnosis.empirical_cause_distribution(name, targets, ds, restrict_ev_ids=pool)`.
   This is **exactly Zhang's edge logic** (a cause is a finding attached to the outcome
   occurrence, or the occurrence immediately preceding it) with Zhang's denominator
   (count of outcome accidents), restricted to the query-relevant pool. Keep the top
   `branching` causes with `edge_prob ≥ min_prob`.
4. **Level ≥ 2** = for each cause node, re-rank `P(cause2 | outcome AND cause1)` by
   restricting the pool to incidents that **also contain cause1**
   (`prognosis._incident_node_labels`), dropping cause labels already used on the path,
   then re-running `empirical_cause_distribution` over that subset. Recurse to `depth`.
5. The two generic catch-all buckets (`zhang_diagnosis.GENERIC_CAUSES`:
   *airframe/component/system failure/malfunction* and *miscellaneous/other*) are
   dropped when `drop_generic=True`, so branches are specific mechanisms.

### Probability math at each diagnosis branch

```
edge_prob(cause | context) = count(cause-edge AND outcome AND context) / count(outcome AND context)
```

where *context* = the ancestor causes on the path (empty at level 1), and the counting
is Zhang's `_causes_into_outcome` edge logic via `empirical_cause_distribution`.
`path_prob` is the running product, an empirical estimate of the joint
`P(cause1, cause2, … | outcome)` along the branch.

### Faithful Table-7 mode (`cause_factor_only`)

`build_diagnosis_tree(..., cause_factor_only=False)` (additive, default off). With it
**on**, level-1 edges use Zhang's *contributory-factor* counting (only Cause/Factor
findings, nan/empty → "Unknown quantity", denominator = count of outcome accidents),
which reproduces the **published Table 7 cell values exactly** (e.g. electric wiring
9/102 = 0.08823, APU 5/102 = 0.04901, emergency procedure 1/102 = 0.00980). The default
(off) keeps the historical all-findings+occurrences edges, which match Zhang's **Eq.9**
anchor `P(fire | airframe) = 32/102 = 0.31373` but over-attribute occurrence-level
labels (e.g. emergency procedure 11/102) relative to the published table. See
`docs/TREES_VALIDATION_REPORT.md §2`.

## 4. Prognosis tree

`trees.build_prognosis_tree(query, top_n_incidents=200, branching=3, depth=3, min_prob=0.05, min_n=2, query_relevant=True, semantic_smoothing=False, add_outcome_leaves=False, leaf_specs=None, leaf_min_prob=0.0, deep_backoff=False, bn_posteriors=False, drop_generic=False)`

1. **Query-first:** embed query → retrieve pool. Build forward transition counts
   `P(next | current)` over the pool (`query_relevant=True`) — or over the whole
   dataset (`False`, richer support) — using `prognosis.build_transition_counts`
   (consecutive `Occurrence_No`-ordered occurrence pairs; the same counting the linear
   chain used).
2. **Root seed** = the initial event detected from the query (`_seed_event`): map the
   query to an occurrence family with `detect_outcome`, then pick the family member
   with the most outgoing transitions (so the tree can branch). Falls back to the
   global transition model if the query-relevant pool is too sparse to host the seed.
3. **Branch:** at each node, `prognosis.transition_step(label, tc)` gives all
   next-events with `p = P(next | current)` and support `N`. Keep the top `branching`
   with `p ≥ min_prob` **and** `N ≥ min_n` (sparsity gate), skipping self-loops and any
   label already on the path (acyclic). Recurse to `depth`.
4. **Sparse hops** (`N < sparse_n`) optionally get a semantic-neighbour estimate via
   `prognosis.semantic_step_estimate` (the same K-NN smoothing as the chain demo).

### New prognosis capabilities (all additive; defaults preserve the old tree)

| Param | Effect | Node fields added |
|---|---|---|
| `add_outcome_leaves` | Terminate leaf event branches in Zhang's terminal **damage / injury** outcomes via `prognosis.make_outcome_predicate`; `edge_prob` = empirical reachability `P(outcome \| leaf event present)` (`prognosis.honest_downstream`, global support). Attaches to a **childless root** too, so a terminal seed (e.g. *gear collapsed*) still yields a damage/injury prognosis. | leaf nodes have `kind="outcome"`, `source="leaf-outcome-empirical"`, `outcome_spec` |
| `leaf_specs` / `leaf_min_prob` | Configure which terminal outcomes (default `DEFAULT_LEAF_SPECS` = substantial/destroyed damage, serious/fatal injury) and prune tiny leaves. | — |
| `deep_backoff` | When the query-relevant pool can't supply `branching` hops passing `min_prob`/`min_n` at a node, **back off to the GLOBAL transition model** to fill the remaining slots, so 3-hop branches don't bottom out at meaningless `N=1` pool edges. | each child carries `source="pool"` \| `"global-backoff"` \| `"global"` |
| `bn_posteriors` | At branch points with **≥2 active parent events** on the path, use Zhang's **Beta-CDF multiparent** posterior (`prognosis.zhang_baseline_multiparent`) for the edge instead of the plain Markov hop (falls back to Markov when the path events aren't graph-parents of the child). | switched edges carry `prob_source="bn-beta-cdf"` + `markov_p`; else `prob_source="markov"` |
| `drop_generic` | Skip the two generic catch-all occurrence buckets (`zhang_diagnosis.GENERIC_CAUSES`) as next-events for cleaner escalation chains. | — |

The node schema stays **additive** (only new optional fields), so the Streamlit app's
`{id,label,kind,edge_prob,path_prob,n,denom,depth,children}` consumer is unaffected.

### Probability math at each prognosis branch

```
edge_prob(b | a) = count(consecutive a→b transitions) / count(transitions out of a)
```

over the chosen population. `path_prob` = product of hops = probability of the
escalation path (identical semantics to `prognosis.chain_from_events`'s `cumulative`,
just expanded into a tree).

This is the **direct generalization** of `prognosis.multistep_chain` (which followed
one greedy hop per step) into a branching tree (top-B hops per node).

## 5. Mapping to Zhang's paper

- **Diagnosis** reproduces Zhang's **Table 7** quantity `P(cause | outcome)` at every
  branch (same edge logic + denominator as `zhang_diagnosis`, see
  `docs/ZHANG_REPRODUCTION_REPORT.md §3–10`). The tree just makes the inference
  *recursive*: causes of the outcome, then co-occurring sub-causes of each cause.
- **Prognosis** reproduces the **forward / escalation** direction of Zhang's
  **Table 9 / §11** (`P(outcome | cause)` read forward). Zhang's full BN does
  multi-parent Beta-CDF posterior propagation; our tree uses the honest empirical
  Markov hop `P(next | current)` with visible support `N` at every edge — consistent
  with the report's "honest forward conditional" framing (`§11.3–11.7`). The branching
  tree is the natural multi-outcome generalization of the report's §11.7 escalation
  chains.

## 6. Sample rendered diagnosis tree (fire query)

Query: `"engine caught fire during takeoff"` (Zhang fire example), `branching=4, depth=2`,
generic buckets dropped, over a 300-incident query-relevant pool (69 fire accidents):

```
[OUTCOME] fire  (N=69)
├─ <- p=0.145  (N=10/69)  path=0.145  Emergency procedure
│  ├─ <- p=0.167  (N=2/12)  path=0.024  Fuel system, line fitting
│  ├─ <- p=0.167  (N=2/12)  path=0.024  Fluid, fuel
│  └─ <- p=0.083  (N=1/12)  path=0.012  Fuel system, drain
├─ <- p=0.116  (N=8/69)  path=0.116  Loss of engine power (total) - mechanical failure/malfunc…
│  ├─ <- p=0.125  (N=1/8)  path=0.014  Combustion assembly, outer casing
│  └─ <- p=0.125  (N=1/8)  path=0.014  Engine compartment
├─ <- p=0.087  (N=6/69)  path=0.087  Evacuation
│  ├─ <- p=0.250  (N=2/8)  path=0.022  Auxiliary power unit (APU)
│  └─ <- p=0.125  (N=1/8)  path=0.011  Aborted takeoff
└─ <- p=0.087  (N=6/69)  path=0.087  Fluid, fuel
   ├─ <- p=0.286  (N=2/7)  path=0.025  Emergency procedure
   └─ <- p=0.143  (N=1/7)  path=0.012  Fuel system, line
```

`<-` means "cause inferred from the outcome"; prognosis renders with `->` (forward
escalation). Each edge shows `p` = edge probability, `N=n/denom` = support, and `path`
= cumulative path probability.

## 7. JSON export (for Streamlit)

`trees.export_json(result, path)` writes the full `{meta, tree}` dict. The demo writes
`outputs/diagnosis_tree.json` and `outputs/prognosis_tree.json`. The nested `children`
structure with `id`, `label`, `edge_prob`, `path_prob`, `n`, `denom`, `depth` maps
directly onto a graph/tree visualization (e.g. a Streamlit `graphviz`/`pyvis` view).

## 8. Decisions & assumptions

- **Build on top, no edits.** `trees.py` imports `zhang_diagnosis`, `prognosis`,
  `main_app`. It uses public functions plus the explicitly-sanctioned helpers
  `prognosis._ordered_occurrences` / `_incident_node_labels` / `load_dataset`. No file
  owned by the concurrent agent is touched.
- **Diagnosis "cause" includes occurrence labels** (e.g. *Emergency procedure*,
  *Evacuation*) because Zhang's edge logic treats the immediately-preceding occurrence
  as a cause-edge. This is intentional fidelity to Zhang, not a bug.
- **Level-2 conditioning** uses "cause1 present anywhere in the incident"
  (`_incident_node_labels`), matching `zhang_diagnosis.diagnose_conditional` semantics.
- **Prognosis seed** is derived from the query via `detect_outcome` then snapped to the
  best-supported occurrence label, so the root can actually branch.
- **Query-relevant vs global prognosis.** The query-relevant pool keeps the tree
  faithful to the narrative but can be sparse; `query_relevant=False` gives richer
  branching support. The demo shows both.
- **Pruning** is by `min_prob` (and `min_n` for prognosis) plus a `branching` cap and
  `depth` limit — all parameters, no hardcoded structure.

## 9. Status & what remains

| Deliverable | Status |
|---|---|
| Diagnosis tree (query-first, end-to-end) | ✅ Implemented + validated (demo PASS) |
| Diagnosis faithful Table-7 mode (`cause_factor_only`) | ✅ reproduces published Table 7 cells exactly (16/16) |
| Diagnosis robustness across outcomes (fire, LOEP, gear, nose gear, LOC, …) | ✅ 4 level-1 branches each, no empty/degenerate trees |
| Prognosis tree (branching generalization of the chain) | ✅ Implemented + validated (demo PASS) |
| Prognosis **leaf outcomes** (damage/injury, incl. terminal-seed root) | ✅ `add_outcome_leaves` |
| Prognosis **deep-chain global backoff** (visible `source` flag) | ✅ `deep_backoff` |
| Prognosis **Zhang Beta-CDF multiparent BN posteriors** | ✅ `bn_posteriors` (gated, graceful fallback) |
| Zhang validation gates in the demo (Table 7 + Table 9) | ✅ all PASS, exit 0 |
| ASCII renderer (shows leaves `=>`, `[global-backoff]`, `[BN-betaCDF]`) | ✅ |
| JSON export (Streamlit-ready, additive schema) | ✅ `outputs/*.json` |
| Demo (`tests/tree_demo.py`, framework Py 3.11) | ✅ runs clean, exit 0 |

**Honest remaining gaps (see `docs/TREES_VALIDATION_REPORT.md §5`):**

1. **Markov hop ≠ Zhang BN posterior.** Tree edges are honest empirical conditionals
   `P(next \| current)` (and `P(damage/injury \| leaf event)` for leaves); Zhang's
   Table-9 downstream cells are **full multi-hop BN posteriors** (diluted across the
   740-node network). The single-parent forward *edge ratios* match exactly (Table 9
   anchors), but the deep leaf magnitudes differ by construction — documented, not a bug.
2. **`detect_outcome` coverage.** A few phrasings don't map to an occurrence label
   (e.g. "overran" vs the label "overrun"); this lives in `zhang_diagnosis` (not edited).
3. **Deep query-relevant support is still thin.** `deep_backoff` repairs structure
   from the global model, but the deepest hops remain low-`N`; treat 3-hop magnitudes
   as illustrative (every edge still reports its `N` and `source`).
4. **True BN propagation** (GeNIe/pySMILE 100k-sample inference) remains out of scope;
   `bn_posteriors` gives Zhang's Beta-CDF *cell* value at multi-parent branch points,
   not full network marginalization.
