# Export for Review — Frozen Bayesian Network + Narrative Evidence

Self-contained bundle for reproducing a single Bayesian network inference and
inspecting how NTSB accident narratives are coded into node states.

**Total size:** ~17 MB (all plain text / JSON / XML / CSV)

---

## Contents

| File / Folder | What it is |
|---|---|
| `frozen_bn.bifxml` | The frozen 785-node Bayesian network in pyAgrum's BIFXML format (nodes, states, edges, CPTs). Loadable by `pyAgrum.loadBN()`. |
| `frozen_bn.json` | Same network in a portable JSON format: every node, its states, parent list, and full CPT as a flat array + shape. No library needed to read. |
| `nodes.json` | Reference list of all 785 nodes with states and short descriptions. |
| `narratives/` | 10 raw NTSB accident narratives (2007–2019 held-out set, ≥200 chars), plain text. |
| `coded_fields.csv` | For those 10 accidents: coded injury/damage levels, NTSB coded labels, and the pipeline's hard + soft evidence mapping. |
| `accident_variable_matrix.csv` | The coded dataset the frozen BN was learned from. One row per accident (1,742), one column per node (783 Boolean + 2 severity). See "Dataset" below. |
| `source/` | Read-only copies of the coding logic (see below). |
| `run_inference.py` | Standalone inference script (see "Running" below). |
| `build_export.py` | The script that generated this bundle (for reproducibility). |

---

## The Bayesian Network

- **785 nodes**, **1,438 arcs**
- Built from Zhang & Mahadevan's recipe on the **1982–2006 NTSB refined dataset** (1,742 accidents)
- **Frozen**: structure and CPTs are never retrained on the test set (2007–2019)
- Two outcome nodes added on top of the faithful Zhang replication:
  - `personnel injury` — 4 states: fatal / serious / minor / no injury
  - `aircraft damage` — 4 states: destroyed / substantial / minor / no damage
  - Each has **12 parents** (last-occurrence nodes, selected by frequency)
  - CPTs: support-weighted mixture of empirical severity distributions

All other ~783 event/finding/person nodes are **Boolean** (Yes/No) with Beta-CDF
CPTs following Zhang's Eqs. 10–14.

### Sparsity note

785 nodes learned from 1,742 accidents — roughly **one accident per variable**.
The matrix is 99.3% zeros (0.71% fill rate; average 5.5 active nodes per accident).
Structure learned from data this sparse is inherently unstable: small changes in
the input can move many edges. This is not a criticism of the method — it is
evidence that the frozen structure should not be treated as certain, and it
motivates the structure-variance analysis.

---

## The Coded Dataset (`accident_variable_matrix.csv`)

The accident × variable matrix from which the frozen BN was learned.

- **1,742 rows** (one per accident, 1982–2006 build window)
- **786 columns**: `ev_id` + 783 Boolean event/finding/person nodes (1 = active,
  0 = inactive) + `personnel injury` (4-state label) + `aircraft damage` (4-state label)
- **Size:** 2.8 MB (CSV, plain text)
- **Fill rate:** 0.71% (9,662 active cells out of 1,363,986 Boolean cells)

This is the data you need to learn alternative network structures. The frozen BN's
BIFXML encodes the CPTs (statistical summaries), but not which accidents activated
which nodes — you need this matrix for that.

### Why BIFXML instead of BIF?

Some NTSB node names start with digits (e.g. `1 engine`), which violates the BIF
specification. The BIFXML format handles arbitrary names correctly.

---

## Source Files (read-only reference)

| File | Role |
|---|---|
| `source/query_to_bn.py` | **The coding logic.** Turns a narrative into evidence: phrase matching → hard evidence, retrieval → soft evidence (Jeffrey conditioning via likelihood ratios). Contains everything listed below. |
| `source/bn_upgraded.py` | Builds the upgraded network: person→finding edges + 4-state severity nodes on top of the Zhang base. |
| `source/bn_build_ours.py` | The faithful Zhang-recipe builder: graph construction, Beta-CDF CPTs, parent capping, cycle removal. |
| `source/prognosis.py` | Graph utilities: edge construction, occurrence ordering, dataset loading. |
| `source/zhang_diagnosis.py` | Outcome/event parser: detects occurrences and findings from text. |

### What's inside `source/query_to_bn.py` (the key file)

| Component | Location | What it does |
|---|---|---|
| **Phrase-matching vocabulary** | `_PERSON_ALIASES` (line ~54), `_STOP_CORES`, `_STOPWORDS` | Maps "pilot error" → `person: pilot-in-command`, etc. |
| **Assertion guard** | `_NEG_CUES`, `_IRREALIS_CUES`, `_CLAUSE_SPLIT`, `_asserted()` | Prevents negated/hypothetical mentions from becoming evidence |
| **Leak guard** | `_REDACT_ONLY` (line ~279), `_DMG_STATED`, `_INJ_STATED`, `redact_severity_phrases()` | Strips outcome-stating phrases before embedding/parsing |
| **Severity statement patterns** | `_DMG_STATED`, `_INJ_STATED` (line ~262) | Regex patterns for "was destroyed", "substantial damage", etc. |
| **Retrieval thresholds** | `retrieval_facts()` defaults (line ~181) | `top_k=100`, `min_fq=0.15`, `min_lift=3.0`, `top_m=3`, `max_conf=0.95` |
| **Jeffrey conditioning** | `jeffrey_likelihood()`, `apply_evidence()` | Converts soft confidence → likelihood ratio against the node's prior |
| **Severity confusion matrices** | `severity_likelihoods()` (line ~379) | L[stated=i \| coded=j] counted from training narratives with Laplace smoothing |

**No LLM prompt templates** live in this file — the pipeline is retrieval-based,
not LLM-based. The existing LLM arm (`code/llm_evidence.py`, not exported) is a
separate experiment that scored 68.2%/51.7% and lost.

**These are copies** — the originals live in the main repo under
`Frozen-BN-Narrative-Evidence-2026-07-20/code/` and `tests/`.
They import each other and the full dataset, so they won't run standalone.
They are included for code review, not execution.

---

## Running

### Quick inspection (no dependencies beyond Python 3.10+)

```bash
cd export_for_review/
python3 run_inference.py
```

Reads `frozen_bn.json` and shows: network metadata, severity CPT slices, and
how demo evidence maps to nodes. No pyAgrum or pgmpy needed.

### Full exact inference (needs pyAgrum + ~4 GB RAM)

```bash
pip install pyAgrum
python3 run_inference.py --full
python3 run_inference.py --full "fire" "fuel starvation"   # custom evidence
```

Loads `frozen_bn.bifxml`, compiles the junction tree, and prints posteriors over
both severity nodes. The 785-node junction tree requires ~4 GB free RAM.

---

## coded_fields.csv columns

| Column | Meaning |
|---|---|
| `ev_id` | NTSB event ID |
| `injury_coded` / `damage_coded` | Ground-truth severity from NTSB coded data |
| `narr_length` | Character length of the narrative |
| `ntsb_coded_labels` | Selected NTSB coded events/findings (from structured data) |
| `pipeline_hard_evidence` | Nodes set to "Yes" by phrase matching (confidence 1.0) |
| `pipeline_soft_evidence` | Nodes set via retrieval with confidence scores |
| `n_hard` / `n_soft` | Counts |

---

## What could NOT be exported

- **Embedding vectors**: The pipeline uses OpenAI `text-embedding-3-small` for retrieval-based soft evidence. These are API-generated and cached in `.npz` files (~50 MB). Not included because: (a) they require an API key to regenerate, (b) they exceed the size budget.
- **Full JSON dataset**: The raw `refined_dataset_1982_2006.json` (13 MB) with narratives, weather, metadata, etc. is not included — but the **accident × variable matrix** (2.8 MB) extracts what's needed for structure learning.
- **Zhang's released NTSB.xdsl**: His original GeNIe model is in the repo at `Zhang-Replication-Foundation-2026-06-04/reference/NTSB.xdsl` but is not ours to redistribute.
- **pyAgrum / pgmpy libraries**: Install separately via pip.

---

## Regenerating this bundle

From the repo root:

```bash
python3 export_for_review/build_export.py
```

Requires the full repo with all dependencies installed (pyAgrum, numpy, etc.)
and the refined dataset at `shared/data/processed/refined_dataset*.json`.
