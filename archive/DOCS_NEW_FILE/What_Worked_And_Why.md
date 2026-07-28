# What worked — decisions and rationale

**Purpose:** Short institutional memory so future you (or an agent) knows **why** something was chosen, not only **what** the code does. When something works well in a meeting or paper, add a row. When you **reverse** a decision, add an entry explaining the pivot.

This is **not** a duplicate of [`Project_Outline_For_Cursor.md`](Project_Outline_For_Cursor.md); the outline is the contract. This file is the **story** behind it.

---

## How to add an entry

Use a new row in the table or a new `###` subsection for longer narratives.

| When | Topic | What we did | Why it worked (evidence / intuition) | Caveats |
|------|-------|-------------|--------------------------------------|---------|
| Example | Retrieval | Cosine via dot on normalized embeddings | Same as sklearn cosine; fast `np.dot` on full matrix | Assumes embedding API returns L2-normalized vectors |
| Example | Labels | Findings with `Cause_Factor == C` | Matches NTSB “cause” notion for primary eval | Excludes contributory factors unless we expand scope |

---

## Log (seeded from current project direction)

| When | Topic | What we did | Why it worked | Caveats |
|------|-------|-------------|---------------|---------|
| Project direction | Strict eval | Train-only index + test narrative queries | Stops trivial leakage and “remembering” test findings | More moving parts: train-only merged JSON + train embeddings (`--train`) |
| Project direction | Ground truth | Findings text vs model junk | Auditable vs free-form narrative “accuracy” | String match needs **shared normalization** |
| Project direction | Metrics | Recall@K + MRR | Standard IR-style measures for ranked causes | τ and neighbor set must be **reported** with numbers |
| Implementation note | Similarity | `np.dot(embeddings, q)` in `find_top_matches` | OpenAI embeddings normalized → cosine | If model changes, re-check normalization |
| Protocol | Strict index | Train merged + train embeddings only for search | Full `merged_dataset.json` would put test in the neighbor pool | `NTSB_USE_TRAIN_INDEX=1` before `import main_app`; see `evaluation/README.md` |

---

## Prompt for an agent: “capture what worked”

```text
Read the changes in [branch / commit / file list]. Update @DOCS_NEW_FILE/What_Worked_And_Why.md:
- Add one table row per non-obvious decision.
- "Why it worked" must cite behavior (tests, metric, or faculty acceptance)—not vibes.
- If something was tried and reverted, log under Caveats or a short bullet.
```

---

## Prompt: “explain to my professor why X”

```text
Using @DOCS_NEW_FILE/What_Worked_And_Why.md and @DOCS_NEW_FILE/Project_Outline_For_Cursor.md, write 3–5 tight bullets I can paste into an email defending [X].
No code. Cite evaluation protocol if relevant.
```
