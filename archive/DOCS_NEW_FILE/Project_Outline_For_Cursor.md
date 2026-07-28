# NTSB diagnosis / evaluation project — outline for Cursor (start → end)

Give this document to Cursor as **project context** so implementation stays aligned. If something is missing, **defaults** are in §10.

**Repo context:** Aviation NTSB accident data merged into **`merged_dataset.json`** (`config.MERGED_DATA_PATH`); semantic embeddings + similarity retrieval; diagnosis via similarity-weighted and/or **law of total probability** over incident clusters; optional Streamlit UI; prognosis on event sequences.

---

## 1. North star

Build a **defensible evaluation** of the embedding-based diagnosis pipeline:

- **Ground truth (primary):** structured **findings** (`finding_description`, `Cause_Factor`).
- **Metrics:** **Recall@K**, **MRR** (and Recall@1 as “top-1 accuracy”) comparing **ranked predicted causes** to reference findings under a **fixed embedding similarity threshold τ**.
- **Strict setting:** **Train-only** merged data + **train-only** embeddings / diagnostic augmentation + **test** incidents scored with **narrative-only** queries; **exclude self** from retrieval; **never** pass findings into the model input.
- **Later:** optional **structural / rule-based baseline** (codes, dictionary lookups) using the **same** metrics; optional **full-corpus** artifacts for deployment.

Secondary narrative-based metrics are optional (lower priority than findings).

---

## 2. What already exists in this codebase

| Area | Role |
|------|------|
| `config.py` | Paths, `OPENAI_API_KEY`, `EMBEDDING_MODEL`, `data/processed` targets; **`NTSB_USE_TRAIN_INDEX`** switches **`ACTIVE_*`** paths to **train** merged + train embeddings. |
| `main_app.py` | Loads **`ACTIVE_INCIDENT_DATA_PATH`** / **`ACTIVE_EMBEDDINGS_PATH`** / **`ACTIVE_EMBEDDINGS_MAP_PATH`** (full corpus by default; **train-only** when `NTSB_USE_TRAIN_INDEX=1`; see [`evaluation/README.md`](../evaluation/README.md)). `find_top_matches`, `get_embedding`, clustering, diagnosis (similarity-weighted + conditional / “chain rule” **internal** naming), prognosis helpers, Mermaid diagrams. |
| `streamlit_app.py` | UI over `main_app`. |
| `data/preprocessing/01_*.py` | Merge raw NTSB tables → `merged_dataset.json`. |
| `2_generate_embeddings.py` | Build embedding matrix + `embeddings_map.json`. |
| `2b_precompute_diagnostic_data.py` | Augment map with `diagnostic_data` (causes from findings + narrative extraction). |
| `3_precompute_clusters.py` | K-means + labels → `cluster_label` on incidents (when used). |
| `data/raw/*` | Source xlsx/txt for `01`. |

**Presentation / theory fixes** from faculty are tracked with status in [`Faculty_Feedback_And_Fixes.md`](Faculty_Feedback_And_Fixes.md) (living table + agent prompts). Summary themes: **LOTP** naming in UI, **ties**, **Mermaid** captions, **prognosis** anchor, **exclude-self** for eval.

### Data model (`merged_dataset.json`)

- **Shape:** top-level JSON **object** keyed by **`ev_id`** (string). Each value is one incident record (`orient='index'` from the merge in `01_create_refined_dataset.py`).
- **Common keys:** event metadata (`ev_date`, location, weather, injury level, …), `narr_accp`, `narr_accf`, `narr_cause`, nested lists `findings`, `sequence_of_events`, `injuries`, `engines` (empty list if none).
- **Findings:** each element includes at least `finding_no`, `finding_description`, `Cause_Factor` (use these for **labels**; scoring loads findings **after** prediction only).
- **Optional later:** `cluster_label` when `3_precompute_clusters.py` has been run on that merged file.
- **`embeddings_map.json`:** list **parallel** to rows of `embeddings.npy`. Incident rows include `source` (`'incident'` or `'dictionary'`), `ev_id`, and `type` (`'causal_narrative'`, `'factual_narrative'`, `'fallback_narrative'`, `'finding'`, …). **Eval** uses a **narrative-only** query and a **train-only** index; do not use finding-type rows as the query text.

### Similarity metric, τ, and matching for metrics

- **Metric:** **cosine similarity** between query embedding and each row of the embedding matrix. With OpenAI embeddings (L2-normalized), **`np.dot(embeddings, query_vec)`** equals cosine similarity (see `find_top_matches` in `main_app.py`).
- **τ:** threshold on that **same** similarity score (typically in **[-1, 1]**; for normalized vectors often **~0.8–0.99** for “close” neighbors). Use τ only as documented in §5 (tune on **dev/train**, not test). Exact use in code (e.g. **filter neighbors with score ≥ τ** before aggregation vs. post-filter ranked causes) should match whatever `evaluate_diagnosis.py` implements and **must be reported** in eval exports.
- **Reference labels (findings):** restrict to `Cause_Factor` per §10 (default **`C` only**). Build a set/list of normalized `finding_description` strings per test `ev_id`.
- **Predictions:** each method returns ranked cause dicts with **`cause`** (string) and **`probability`**. `diagnose_with_conditional_probabilities` exposes them as **`weighted_causes`**. `diagnose_root_causes` wraps the same list as **`all_causes`** / **`top_causes`** (slice). A **hit** for Recall@K / MRR: the **normalized** predicted `cause` **equals** any reference `finding_description` after the same normalization (or a documented substring/rule). Implement normalization once in the eval script and **record the rule** in results JSON.

### `main_app.py` entry points (today)

| Function | Role |
|----------|------|
| `get_embedding(text)` | Single-query embedding (OpenAI `EMBEDDING_MODEL`). |
| `find_top_matches(query_embedding)` | All incidents sorted by descending similarity; returns `(top_scores, top_matches)`. **No `exclude_ev_ids` yet** — add for strict eval. |
| `diagnose_root_causes(query, top_n=10)` | Similarity-weighted causes: embed → `find_top_matches` → `calculate_similarity_weighted_diagnosis`. |
| `diagnose_with_conditional_probabilities(query, …)` | Mixture-of-clusters path: embed → matches → `cluster_incidents_by_type` → `calculate_chain_rule_diagnosis` (user-facing **law of total probability** wording in UI). |

For offline eval, prefer: **slice test narrative** (§10 char cap) → `get_embedding` → `find_top_matches` **over the train index only** (set **`NTSB_USE_TRAIN_INDEX=1`** before importing `main_app`, or load train `.npy`/map explicitly) → **drop self `ev_id`** → then call the same cause aggregation the app uses (or a thin wrapper), then score vs findings (labels may be read from the **full** merge for test `ev_id`s).

### Not in repo yet (implement per Phases 1–3)

- **Splits + train-only merged:** script or documented one-off to write `evaluation/splits/train_ev_ids.txt`, `test_ev_ids.txt`, and `merged_dataset_train.json` (train keys only).
- **`evaluate_diagnosis.py` (or equivalent):** loop test IDs, narrative-only query, train-only index, exclude-self, τ handling, Recall@K / MRR, export JSON/CSV.
- **`find_top_matches` extension:** e.g. `exclude_ev_ids: set[str]` (or filter immediately after retrieval).

**Implemented in repo (Track B):** `2_generate_embeddings.py`, `2b_precompute_diagnostic_data.py`, and `3_precompute_clusters.py` support **`--train`**: read `TRAIN_MERGED_DATA_PATH`, write `embeddings_train.npy`, `embeddings_map_train.json`, `cause_statistics_train.json` (2b), optional `cluster_label` on **`merged_dataset_train.json` only** (3). Paths: `config.EMBEDDINGS_TRAIN_PATH`, `EMBEDDINGS_MAP_TRAIN_PATH`, etc. See [`agent_tasks/02_Track_Preprocessing_Hooks.md`](agent_tasks/02_Track_Preprocessing_Hooks.md).

---

## 3. Target workflow (new or cleaned repo)

```text
Phase 0  Bootstrap: venv, deps, .env, copy code + raw, Cursor rules.
Phase 1  Full merged JSON (or copy) + eligibility + train/test ID files + merged_dataset_train.json.
Phase 2  Train-only embeddings + 2b (+ 3 if needed) → *_train artifacts.
Phase 3  Eval script: per test ev_id → narrative query → diagnosis(train index) → match to findings → Recall@K / MRR → results JSON/CSV.
Phase 4  Meeting fixes: LOTP copy, ties, Mermaid caption, exclude-self, prognosis anchor, etc.
Phase 5  Optional: structural baseline row + optional full-corpus rebuild for production.
```

---

## 4. Data splitting (summary)

- **Split unit:** `ev_id` (never random rows inside one incident).
- **Eligibility:** e.g. ≥1 finding with `Cause_Factor in {'C'}` (and/or `'F'` per paper) + narrative length floor for query field.
- **Train / test:** disjoint lists on disk: `evaluation/splits/train_ev_ids.txt`, `test_ev_ids.txt`; fixed **seed** and **fraction** documented.
- **Multi-label:** one incident can have **multiple** reference findings; a prediction **hits** if it matches **any** reference under τ.
- **Train-only merged:** `merged_dataset_train.json` = subset of full `merged_dataset.json` whose keys ⊆ train IDs only.
- **Detailed procedure:** [`docs/Train_Test_Split_Plan.md`](../docs/Train_Test_Split_Plan.md).

---

## 5. Evaluation protocol (non-negotiables)

| Rule | Detail |
|------|--------|
| Query input | Narrative snippet only (e.g. first N chars of `narr_accp`); **no** `finding_description` in the string passed to diagnosis. |
| Labels | Load findings **after** prediction, for scoring only. |
| Self-retrieval | Exclude same `ev_id` from neighbor list when evaluating that incident (or ensure test not in train index). |
| τ | Chosen on **dev** or **train** subsample; **not** by maximizing test Recall. |
| Report | Split seed, query rule, Cause_Factor set, τ, embedding model id, N test scored vs skipped. |

---

## 6. Verification / tests (intent)

Automated checks (pytest or one runner) should enforce:

- Train ∩ test = ∅; train-only JSON has no test keys.
- `embeddings_train.shape[0] == len(embeddings_map_train)` (when train index exists).
- Incident rows in train map have `ev_id` not in test set.
- `exclude_self` behavior on `find_top_matches`.
- Synthetic **Recall@K / MRR** unit test (hand-verified small example).

**Full test table:** if the Cursor chat/plan **“NTSB eval project from zero”** is not checked into the repo yet, treat §6 here as the contract; when you add it, store it under `docs/` or link from `.cursor/rules/` (e.g. `ntsb_eval_protocol.mdc`) so paths stay stable.

---

## 7. Artifacts and paths (conventions)

| Artifact | Typical path (strict eval) |
|----------|----------------------------|
| Full merged | `data/processed/merged_dataset.json` (`MERGED_DATA_PATH`) |
| Train-only merged | `data/processed/merged_dataset_train.json` (`TRAIN_MERGED_DATA_PATH`) |
| Full embeddings | `data/processed/embeddings.npy`, `embeddings_map.json` |
| Train embeddings | `data/processed/embeddings_train.npy` (`EMBEDDINGS_TRAIN_PATH`), `embeddings_map_train.json` (`EMBEDDINGS_MAP_TRAIN_PATH`), `cause_statistics_train.json` (`CAUSE_STATS_TRAIN_PATH`) |
| Splits | `evaluation/splits/*.txt` |
| Eval results | `evaluation/results/*.json`, `*.csv` |

Preprocessing: use **`--train`** on `2` / `2b` / `3` for strict-eval artifacts; defaults without flags remain **full corpus**.

---

## 8. Bundle / migration

- Portable copy list: [`NTSB_shift_to_new_project_approach_2026-03-25.md`](../NTSB_shift_to_new_project_approach_2026-03-25.md).
- Folder snapshot: `NTSB_new_project_bundle_2026-03-25/` (code + raw + `cursor_3` rename).
- Cursor rules: install from `cursor 3/` → `.cursor/rules` and `.cursor/commands`; optional eval protocol rule `ntsb_eval_protocol.mdc`.

---

## 9. Out of scope (unless user expands)

- Full **Bayesian network** structure learning.
- **t-SNE/UMAP** as primary clustering for production (maybe exploratory only).
- **Calibration** of displayed percentages (optional research).
- **Evidence theory / Dempster–Shafer** for unknown causes (future work).
- Real-time incremental index updates without rebuild.

---

## 10. Defaults when something is unspecified

Use these unless the user overrides:

| Topic | Default |
|-------|---------|
| Random split seed | `42` |
| Test fraction | `0.30` |
| Query field | `narr_accp` |
| Max query chars | `400` |
| Min narrative len (eligibility) | `80` |
| Cause_Factor for labels | `C` only |
| Similarity threshold τ | `0.82` (tune on dev, not test) |
| Primary metric headline | Recall@5 + MRR |
| Strict eval | Train-only index + exclude-self |

---

## 11. Open questions (answer only if user wants to change defaults)

1. **Time-based vs random** split for the paper?  
2. **Stratified** by `cluster_label`?  
3. **Pilot test size** (e.g. 200) before full test run for API cost?  
4. **Single repo** vs bundle-only new repo? (Code can live either place; protocol is the same.)

---

## 12. How Cursor should behave

- **Prefer executing** terminal commands and writing code over only suggesting commands (per user preference).
- **Match existing style** in `main_app.py` / `streamlit_app.py` when editing.
- **Small diffs:** only what the task needs; no unrelated refactors.
- **After structural changes:** run or add tests described in §6 when possible.
- **Planning:** for large design changes, follow project’s `cursor 3` planning rules if installed; for “implement X” with clear spec from this doc, implement directly.

---

## 13. Key file index

- Train/test detail (still under `docs/`): [`docs/Train_Test_Split_Plan.md`](../docs/Train_Test_Split_Plan.md)  
- Understand the project (plain language): [`Project_Primer_For_Agents.md`](Project_Primer_For_Agents.md)  
- Review any plan for gaps: [`Plan_Review_Checklist.md`](Plan_Review_Checklist.md)  
- Professor feedback & fixes: [`Faculty_Feedback_And_Fixes.md`](Faculty_Feedback_And_Fixes.md)  
- Rationale / what worked: [`What_Worked_And_Why.md`](What_Worked_And_Why.md)  
- Parallel Cursor chats + integrator: [`Multi_Agent_Workflow.md`](Multi_Agent_Workflow.md) and [`agent_tasks/README.md`](agent_tasks/README.md)  
- Copy list / bundle: [`NTSB_shift_to_new_project_approach_2026-03-25.md`](../NTSB_shift_to_new_project_approach_2026-03-25.md)  
- Core logic: `main_app.py`, `config.py`  
- Preprocessing: `data/preprocessing/*.py`  
- UI: `streamlit_app.py`  

**Run UI (typical):** from repo root, with `OPENAI_API_KEY` set (and processed data paths valid in `config.py`): `streamlit run streamlit_app.py`.

---

_End of outline — point Cursor here with `@DOCS_NEW_FILE/Project_Outline_For_Cursor.md` or copy into an always-apply rule._
