# Testing_Structural_Mapping_Slides

Isolated copy of `Testing_Structural_Mapping/` that swaps the structural
similarity scoring for the **exact formulas shown in the slide deck**
(`docs/structural_mapping_presentation.pptx-2.pdf`, Formulas 1-4).

The original `Testing_Structural_Mapping/` folder is **not modified**.
Run both, compare, then decide which version stays in the paper.

## What changed vs. the original

Only `scripts/struct_score_v2.py` was rewritten. Everything else
(`extract_struct_v2.py`, `eval_diagnosis_structural.py`, `struct_hooks.py`,
`reweight.py`, the schema, the cached extractions) is byte-identical to
the original folder, so the only thing varying between the two runs is
the scoring math.

| Term                    | Original `struct_score_v2.py`        | Slide formulas (this folder)             |
| ----------------------- | ------------------------------------ | ---------------------------------------- |
| F1 weights (role / mech / sys) | 0.45 / 0.20 / 0.35            | **0.50 / 0.30 / 0.20**                   |
| F2 gap penalty          | -0.30                                | **-0.05**                                |
| F3 strong threshold     | sim >= 0.75                          | **sim >= 0.70**                          |
| F3 partial threshold    | 0.45 <= sim < 0.75                   | **0.35 <= sim < 0.70**                   |
| F3 denominator          | average chain length                 | **total = len(A) + len(B)**              |
| F3 unmapped penalty     | 0.15 * (unmapped/avg_len)            | **0.10 * (unmapped/total)**              |
| Factor overlap term     | +0.15 weight                         | **removed**                              |
| Failure-pattern bonus   | +0.20 if both share pattern          | **removed**                              |

Sanity check (slide 9): chain pair with strong=1, partial=4, unmapped=1,
total=10 should give `(1 + 0.5*4)/10 - 0.10*(1/10) = 0.30 - 0.01 = 0.29`.
The new module reproduces this.

## How to run

All commands assume the project root as cwd.

### 1. Run A2 with slide formulas (full test set, ~minutes)

The script keeps the `--struct-version v2` flag, but in *this* folder the
v2 implementation **is** the slide version, so no other changes needed:

```bash
cd Testing_Structural_Mapping_Slides/scripts
../../.venv/bin/python eval_diagnosis_structural.py \
    --n all \
    --structural \
    --struct-version v2 \
    --alpha 2.0 \
    --output-stem eval_diagnosis_A2 \
    | tee ../outputs/eval_diagnosis_A2_run.log
```

This produces:
- `Testing_Structural_Mapping_Slides/outputs/eval_diagnosis_A2.csv`
- `Testing_Structural_Mapping_Slides/outputs/eval_diagnosis_A2.json`
- `Testing_Structural_Mapping_Slides/outputs/eval_diagnosis_A2_summary.json`

The original `Testing_Structural_Mapping/outputs/eval_diagnosis_A2.csv`
is left untouched (still the numbers reported in the current paper).

### 2. Compare the two A2 variants against A0

```bash
.venv/bin/python Testing_Structural_Mapping_Slides/scripts/compare_v2_vs_slides.py
```

This prints Top-1 / Recall@5 / MRR / Avg match % with Wilson CIs for
all three runs (A0, A2-v2, A2-slides), then bootstrap CIs on the paired
deltas vs A0, then McNemar p-values, then a recommendation summary.

It also writes `outputs/analysis/v2_vs_slides_summary.json` for the
record.

### 3. Pick the variant for the paper

Use the comparison output to decide:

- If **A2-slides** wins on >= 2 of {Top-1, Recall@5, MRR} with paired CIs
  excluding zero, switch the paper formulas + reported numbers to the
  slide variant. The win story stays the same and the math now matches
  the slides verbatim.
- If **A2-v2** still wins, keep the current paper numbers. Update the
  paper to explicitly call out that the production scoring extends the
  slide formulas with a factor-overlap term and a failure-pattern bonus,
  and explain why those terms help.

Either way, end state is paper, slides, and code all in agreement.

## What was archived

`outputs/_carried_over_from_original/` holds the A2 csv/json/log files
that came in with the copy. They reflect the **original** scoring and
should not be used for the slide-formula evaluation - the new run
overwrites the top-level filenames.
