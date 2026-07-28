# Track A — Splits & train-only merged JSON

## Goal

Produce **leak-safe split artifacts** and a **train-only merged incident dataset** file for strict evaluation.

> **Golden rule:** For **strict** Recall@K / MRR (and any “no leakage” claim), **retrieval and diagnosis must load `merged_dataset_train.json` + train embeddings** (`config.TRAIN_MERGED_DATA_PATH`, `EMBEDDINGS_TRAIN_PATH`, `EMBEDDINGS_MAP_TRAIN_PATH`) — **not** the full `merged_dataset.json` as the search index. The full merge file stays on disk as the **master copy** (all `ev_id`s) and to **build** the train JSON; it is **wrong** to point `main_app` or `evaluate_diagnosis.py` at the full file for strict eval. In this repo, set **`NTSB_USE_TRAIN_INDEX=1`** before importing `main_app` (see [`evaluation/README.md`](../../evaluation/README.md)).

**Terminology:** The consolidated JSON is the **merged dataset** (one record per `ev_id`). **Canonical paths:** full corpus `data/processed/merged_dataset.json`, train-only `data/processed/merged_dataset_train.json` (see `config.MERGED_DATA_PATH` / `TRAIN_MERGED_DATA_PATH`). Legacy `refined_dataset.json` may exist in older copies—rename to `merged_dataset.json` or re-run `01_create_refined_dataset.py`.

### Train vs test (who uses what, when)

- **`train_ev_ids.txt` + train-only merged JSON** feed **Tracks B–C** to build the **train-only retrieval index**. Do **not** put test incidents in the train-only JSON.
- **`test_ev_ids.txt`** is the **held-out** list for **later** scoring (Recall@K / MRR in Track C): narrative-only query, compare predictions to findings. Creating this file in Track A is **not** the same as running metrics—you only **reserve** those IDs here.
- **Pytest** in this track checks **artifact correctness**, not benchmark scores.

### Methodology note — leak-safe eval vs research trade-offs

There is not one “best” protocol for every goal. For **honest, leak-safe evaluation** (what these docs describe), the shape you are using is sound:

- **Train-only merged JSON + disjoint train/test IDs** — the index must not contain test incidents.
- **Splitting by `ev_id`** — correct so narratives/findings for the same accident do not leak across boundaries.
- **Narrative-only query + labels from findings** — right concept for “do not leak labels into the query” (enforced in Track C scoring, not here).

**Tight eligibility** (e.g. `Cause_Factor` in an allowed set, minimum narrative length) is defensible when the metric is explicitly *predict official cause findings*; it **shrinks N**, which is a research-design consequence, not an engineering mistake.

| If your priority is… | This pipeline… |
|----------------------|----------------|
| Credible metrics, no leakage | Aligned. |
| Larger train/test sets | May be too strict unless most eligible rows really satisfy the rule; any **wider label rule** (e.g. multiple allowed factors) belongs in a **documented secondary** protocol—never mixed with strict numbers. |
| Representative / rare causes | Small eligible sets hurt power; consider stratification or reporting metrics **per factor / cause bucket** (and uncertainty), not only global Recall@K. |
| Fast iteration | Small eligible N keeps embedding/index cheap; scale volume when the protocol is frozen. |

**Practical rule:** keep **this** pipeline as the **strict baseline**. If you relax eligibility or label rules, do it **explicitly**—new run, new README counts, and preferably a **separate** split folder or filename pattern—so papers, demos, and comparisons never conflate “strict” vs “loose” results.

## Scope (you own)

- `evaluation/splits/train_ev_ids.txt` — one `ev_id` per line, **UTF-8**; optional **sorted** lines for stable git diffs.
- `evaluation/splits/test_ev_ids.txt` — one `ev_id` per line, **UTF-8**; optional sorted.
- **Train-only merged JSON** — `data/processed/merged_dataset_train.json` (per outline §7 / `config.TRAIN_MERGED_DATA_PATH`): object whose keys are **only** train `ev_id`s; values must match **`merged_dataset.json`** for those keys.

Optional:

- `evaluation/splits/README.md` — seed, fraction, eligibility rule, row counts.
- Script: e.g. `evaluation/build_splits.py` (document commands in README).

## Do not touch (other tracks)

- `main_app.py` retrieval logic (Track C).
- `streamlit_app.py` (Track D).
- Embedding generation internals (Track B), except you may **read** the full merged dataset path from `config.py`.

## Requirements (from project outline)

- Split unit: **incident** (`ev_id`), never random rows inside an incident.
- **`ev_id` as text:** write the same string in split files as JSON uses for keys (usually stringified `ev_id`); avoid int vs string mismatches between `train_ev_ids.txt` and `merged_dataset_train.json`.
- Train ∩ test = ∅.
- **Coverage / audit:** After eligibility filtering, every considered `ev_id` must land in **exactly one** of: `train_ev_ids.txt`, `test_ev_ids.txt`, or a **documented exclude** (skipped count + reason in README or script logs)—no silent drops, no double assignment.
- Eligibility: e.g. ≥1 finding with `Cause_Factor` in allowed set, min narrative length on query field—**match** [`docs/Train_Test_Split_Plan.md`](../../docs/Train_Test_Split_Plan.md) unless user overrides.
- Defaults when unspecified: [`Project_Outline_For_Cursor.md`](../Project_Outline_For_Cursor.md) §10.

## After creation — test that it was built correctly

Do **not** stop at generating files. Run **checks below** (and extend pytest) so one bad run does not poison Tracks B–C.

### Automated (pytest, multiple cases)

- **Module:** [`tests/test_merged_dataset_split_properties.py`](../../tests/test_merged_dataset_split_properties.py)
- **Always run (no corpus required):** property tests use `@pytest.mark.parametrize` (disjoint train/test, train-only JSON ⊆ full merged, parsing `ev_id` lines, duplicate detection). These should stay green in CI.

  ```bash
  pytest tests/test_merged_dataset_split_properties.py -q -m "not integration"
  ```

- **After artifacts exist on disk:** run the full module (includes **`integration`**): disjoint splits, no duplicate lines in list files, train-only JSON is a key-for-key subset of full merged with **no test `ev_id` keys**, and (when `train_ev_ids.txt` exists) train JSON keys **equal** that list. Implement new checks in the same file if you add eligibility filters or new filenames.

  ```bash
  pytest tests/test_merged_dataset_split_properties.py -q
  ```

- When you add **eligibility** or **split builder** logic, add **more parametrized unit tests** (e.g. empty narrative, no `C` finding)—keep the fast suite meaningful without relying on one manual scenario.

### Manual quick checks (optional backup)

`comm` below is **macOS/Linux** (Git Bash/WSL on Windows).

```bash
# Train and test ID lists must not overlap
comm -12 <(sort evaluation/splits/train_ev_ids.txt) <(sort evaluation/splits/test_ev_ids.txt)
# expect no output

# Counts (sanity)
wc -l evaluation/splits/train_ev_ids.txt evaluation/splits/test_ev_ids.txt
```

If you use Python one-liners, verify: every key in `merged_dataset_train.json` is in `merged_dataset.json`, no key in train file is in `test_ev_ids.txt`, and payloads are byte-identical (or deep-equal) to the full merged file.

## Definition of done

- [ ] Both ID files exist and are disjoint.
- [ ] Train-only merged JSON: no test IDs as keys; each value matches full merged for that `ev_id`.
- [ ] Documented: random seed, test fraction, eligibility filters, paths, **train / test / skipped** counts and why skipped (if any).
- [ ] **`pytest tests/test_merged_dataset_split_properties.py` passes** (fast suite at minimum; full suite including `integration` once files exist).
- [ ] Short handoff: paths to train-only merged JSON + splits for Tracks B and C.

## End-of-chat report (paste for integrator)

- Files added/changed:
- Paths for train-only merged JSON and splits:
- Eligibility formula and **counts (train / test / skipped)** with skip reasons:
- Pytest command(s) run and result:
