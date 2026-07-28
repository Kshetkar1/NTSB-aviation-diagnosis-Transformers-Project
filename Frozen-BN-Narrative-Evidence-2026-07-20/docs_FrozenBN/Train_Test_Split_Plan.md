# Plan: Turn NTSB data into training and testing sets

This document is a **standalone playbook** for splitting incidents into **train** and **test** so you can build **train-only** merged data and embeddings, then score on **test** without treating labels as model inputs.

**Last updated:** 2026-03-25 (v2 — 10/10 upgrades: stratified split, toy example, findings semantics, prerequisites, script contract).

---

## 1. Goals

1. **Freeze** two disjoint lists of incident IDs: `train_ev_ids` and `test_ev_ids`.
2. Use **train** only to build **merged incident JSON / embeddings / diagnostic stats** when you want a leak-safe evaluation setup.
3. Use **test** only for **queries + scoring** (findings are labels, not features in the query).
4. Record every choice (seed, eligibility rules, fields) so runs are **reproducible**.

---

## 2. Definitions

| Term | Meaning |
|------|---------|
| **`ev_id`** | Primary key for one NTSB accident record in `merged_dataset.json`. |
| **Eligible incident** | An `ev_id` that meets your rules for being in the split pool (see §3). |
| **Train set** | IDs used to fit / index the retrieval world (strict mode: **only** these rows in `embeddings_train` + augmented map). |
| **Test set** | IDs used to **evaluate**: build a **narrative-only** query per ID, run diagnosis against **train** index, compare to **structured findings**. |
| **Leakage (avoid)** | (a) Test incident in the embedding index, (b) test findings inside the query string, (c) tuning on test, (d) **self-retrieval** (query from incident X retrieves X). |

---

## 3. Eligibility rules (lock these in `evaluation/README.md`)

Before splitting, decide **who enters the pool**. Recommended defaults:

1. **Structured findings:** At least one `findings[]` row with `Cause_Factor == 'C'` (probable cause), or include `'F'` if your paper includes contributing factors—**pick one rule and keep it**.
2. **Multi-label ground truth:** One incident may have **several** rows with `Cause_Factor == 'C'`. For evaluation, the **reference set** is the list of **all** matching `finding_description` strings (dedupe after strip if needed). Metrics (Recall@K) treat a prediction as a **hit** if it matches **any** reference string under your τ rule—not “exactly one” gold cause.
3. **Narrative for query:** Non-empty `narr_accp` **or** `narr_cause` (pick which field feeds the eval query) with length ≥ **N** characters (e.g. N = 80). Skip empty or trivial rows.
4. **Optional:** Require `sequence_of_events` non-empty if prognosis is in scope later.

**Output:** a Python `set` of eligible `ev_id`s from the full `merged_dataset.json` (keys of the top-level dict).

---

## 4. Inputs and prerequisites

| Input | Location | Note |
|------|----------|------|
| Full merged incidents | `data/processed/merged_dataset.json` | Build via `01_create_refined_dataset.py` or copy from existing project. |
| Split lists (to be created) | `evaluation/splits/train_ev_ids.txt`, `test_ev_ids.txt` | One `ev_id` per line, no headers. |
| Train-only merged (to be created) | `data/processed/merged_dataset_train.json` | Subset of full `merged_dataset.json` by train IDs only. |

**Config / raw alignment (before Step A):** In this repo, `01_create_refined_dataset.py` reads raw tables from `config.DATA_DIR` (`data/processed`). Your bundle may keep raw under `data/raw/`. Before running `01`, either (1) copy/symlink raw xlsx/txt into the path the script expects, or (2) change `DATA_DIR` / script paths once and **record that choice** in `evaluation/README.md`. A failed merge is usually a **path** issue, not eligibility logic.

**Edge case:** If `n_eligible` is very small, `round(0.30 * n)` can be `0`. Enforce `n_test = max(1, int(round(frac * n)))` **or** require `n_eligible >= 50` to split; document the rule.

---

## 5. Split strategy

### 5.1 Random split (default)

1. Collect all **eligible** `ev_id`s into a list `E`.
2. Sort `E` **deterministically** (e.g. sorted alphabetically) so the same seed order is stable across machines.
3. Set `random.seed(42)` (or another fixed integer—**record the number**).
4. Shuffle a **copy** of `E` using `random.shuffle`.
5. Let `n = len(E)`, `n_test = int(round(0.30 * n))` for 70/30 (or `0.2` for 80/20—**document the fraction**).
6. `test_ev_ids = E_shuffled[:n_test]`, `train_ev_ids = E_shuffled[n_test:]`.

**Checks:**

- `set(train) ∩ set(test) == ∅`
- `set(train) ∪ set(test) == set(E)` (every eligible appears exactly once)

### 5.2 Time-based split (optional)

1. Parse `ev_date` (or sort key) per incident from merged data.
2. Assign all incidents with `date < T_cut` to **train**, `≥ T_cut` to **test** (or the reverse—document).
3. Still apply **eligibility** so you do not test on incidents with no findings.

**Trade-off:** more realistic for “future” generalization; random is simpler and often enough for a first paper.

### 5.3 Stratified split (optional — pushes plan to “reviewer-ready”)

**When:** Rare `cluster_label` (or aircraft category) modes might **vanish** from test under pure random split.

**Idea:** Group eligible IDs by a **stratum** key (e.g. `rec.get("cluster_label") or "unknown"`). Within each stratum, assign the same **test fraction** with a **per-stratum seeded** shuffle (or allocate at least 1 test ID per stratum with enough n). **Constraint:** if a stratum has fewer than 2 IDs, put both in train or document exclusion.

**Fallback:** If stratification is too sparse, report **random split** as primary and stratified as supplementary sensitivity analysis.

### 5.4 Worked toy example (sanity check)

Eligible list after sort:  
`E = [A01, A02, A03, A04, A05, B01, B02, B03, B04, B05]` (10 incidents).

- `frac_test = 0.30` → `n_test = round(3) = 3`.
- After `seed(42)` + `shuffle`: suppose `E' = [B03, A01, B01, A04, B05, A02, B02, A05, B04, A03]`.
- **Test** = first 3: `[B03, A01, B01]`. **Train** = remaining 7.

Check: disjoint, union = `E`, counts 7/3. This is the exact logic your script should implement.

---

## 6. Step-by-step procedure

### Step A — Build or obtain full merged data

- Run `data/preprocessing/01_create_refined_dataset.py` with raw files under the path your `config`/script expects, **or**
- Copy `merged_dataset.json` from the parent project into `data/processed/`.

### Step B — Compute eligible IDs

- Load JSON: `data = json.load(open(REFINED_DATA_PATH))`.
- Loop `for ev_id, rec in data.items():` apply §3 filters.
- Save optional debug: `evaluation/splits/eligible_ev_ids.txt` (sorted).

### Step C — Split and write files

- Run §5.1 or §5.2.
- Write:

  - `evaluation/splits/train_ev_ids.txt`
  - `evaluation/splits/test_ev_ids.txt`

- First line comment alternative: use `# seed=42 frac_test=0.30` as first line **only if** your loader strips lines starting with `#`.

### Step D — Build train-only merged JSON

- Load full merged dict.
- `train_only = { k: data[k] for k in train_ids if k in data }`
- Assert: no key in `train_only` is in `test_ids`.
- `json.dump(train_only, open("data/processed/merged_dataset_train.json", "w"), ...)`

**Do not** embed test incidents in `merged_dataset_train.json` when building a strict train index.

### Step E — (Later) Regenerate train-only embeddings

- Point embedding script at `merged_dataset_train.json` → produce `embeddings_train.npy`, `embeddings_map_train.json`.
- Run `2b` on **train-only** map + train-only merged JSON.

### Step F — Evaluation loop (test IDs only)

For each `ev_id` in `test_ev_ids.txt`:

1. Build **query** = first **400** chars of `narr_accp` (example—**fix N and field in README**).
2. Load **ground truth** = list of `finding_description` for chosen `Cause_Factor` values from **full** merged row for that `ev_id` (scoring only).
3. Run diagnosis using **train-only** knowledge base.
4. **Exclude** `ev_id` from retrieved neighbors if the index ever contains it (ideally it should not).
5. Compute Recall@K / MRR vs findings.

---

## 7. Verification checklist (pass / fail)

Run these before you trust “train/test done”:

| # | Check | Pass condition |
|---|--------|------------------|
| 1 | Disjoint | `train_ids & test_ids == ∅` |
| 2 | Coverage | Every eligible ID is in train ∪ test |
| 3 | Train-only merged | All keys of `merged_dataset_train.json` ⊆ train_ids |
| 4 | No test in train JSON | No key from test_ids in `merged_dataset_train.json` |
| 5 | Counts | `len(train) + len(test) == len(eligible)` |
| 6 | Reproducible | Same seed + same eligibility → same split files (byte-compare or hash lists) |

Automate in `evaluation/tests/` when the new repo is ready (see main eval-from-zero plan).

---

## 8. Constants to document once (fill in)

| Constant | Your value (fill) |
|----------|-------------------|
| Random seed | e.g. `42` |
| Test fraction | e.g. `0.30` |
| Query field | `narr_accp` / `narr_cause` |
| Max query chars | e.g. `400` |
| Finding Cause_Factor(s) | e.g. `C` only or `C` + `F` |
| Min narrative length | e.g. `80` |

---

## 9. Common mistakes

1. **Shuffling without fixed seed** → irreproducible papers.
2. **Putting test rows in train-only merged JSON** then claiming strict eval.
3. **Using findings text in the query** → label leakage.
4. **Same incident in top-K** as self when query is from that narrative → inflate metrics; **exclude self** or ensure test not in index.
5. **Tuning τ on test** → pick τ on dev/train subset only.

---

## 10. Files this plan produces

```
evaluation/
  README.md                 # copy §8 constants here
  splits/
    train_ev_ids.txt
    test_ev_ids.txt
    eligible_ev_ids.txt    # optional
data/processed/
  merged_dataset_train.json
  # later: embeddings_train.npy, embeddings_map_train.json
```

---

## 11. Relation to other docs

- **Copy/layout:** [`NTSB_shift_to_new_project_approach_2026-03-25.md`](../NTSB_shift_to_new_project_approach_2026-03-25.md) (repo root).
- **Full eval + fixes + tests:** Cursor plan **“NTSB eval project from zero”** (`ntsb_eval_project_from_zero_3c24eb85.plan.md` in `.cursor/plans/`).

When implementation is done, a short script `evaluation/scripts/build_splits.py` can encode §6 B–D so you do not hand-edit IDs.

---

## 12. Script contract (`evaluation/scripts/build_splits.py`) — implements §6 B–D

Suggested **CLI** (adjust names as you like):

```text
python evaluation/scripts/build_splits.py \
  --merged data/processed/merged_dataset.json \
  --out-dir evaluation/splits \
  --train-merged-out data/processed/merged_dataset_train.json \
  --seed 42 \
  --test-fraction 0.30 \
  --min-narrative-len 80 \
  --query-field narr_accp \
  --cause-factors C
```

**Behavior:**

1. Load merged JSON.  
2. Build `eligible` per §3 (including multi-label note: eligibility only needs **≥1** qualifying finding).  
3. Write `eligible_ev_ids.txt` (sorted).  
4. Split per §5.1 (or §5.3 if `--stratify-by cluster_label`).  
5. Assert §7 checks 1–2, 5.  
6. Write `train_ev_ids.txt`, `test_ev_ids.txt`.  
7. Write `merged_dataset_train.json` per §6 D; assert checks 3–4.  
8. Exit **non-zero** if any assertion fails.  
9. Print summary: `|eligible|, |train|, |test|`, path to outputs.

Having this script (even v0) is what turns the playbook from “excellent” into **operationally closed**: less room for hand error.

---

## 13. “10/10” self-audit (tick before you freeze the split)

- [ ] Constants in §8 filled and copied to `evaluation/README.md`.  
- [ ] `01` / raw paths verified; `merged_dataset.json` loads.  
- [ ] Eligibility code reviewed for off-by-one (empty findings list).  
- [ ] `n_test >= 1` (or documented minimum n).  
- [ ] Stratified or random—**one** primary method named in README.  
- [ ] Multi-reference findings understood by metric code.  
- [ ] Split files committed or checksummed (SHA256) for reproducibility.  
- [ ] `build_splits.py` run fresh produces identical IDs to frozen checksum (CI optional).
