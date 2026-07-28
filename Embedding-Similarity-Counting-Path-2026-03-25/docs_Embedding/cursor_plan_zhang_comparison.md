# Cursor Plan: Compare My Probabilities vs Zhang's BN

## Goal

Run Zhang's exact examples (Table 9, Figure 11, Figure 12 from his paper) through our A2 model and produce side-by-side Excel spreadsheets: Zhang's Bayesian Network probabilities vs our A0 vs our A2.

Zhang's published numbers are already hardcoded in the scripts. We are NOT re-implementing his BN — we are running his same scenarios through our retrieval-based system and comparing the outputs.

Do NOT modify `main_app.py`.

---

## Prerequisites

The first plan (`cursor_plan_get_my_probabilities.md`) should be finished or at least running. These Zhang comparison scripts are independent — they don't read from the 77-incident eval files. They run their own small set of queries (5, 4, and 2 queries respectively).

You DO need:
- `openpyxl` installed (should already be from the first plan)
- The structural cache at `Testing_Structural_Mapping_Slides/cache/struct_train_v2.jsonl` (already exists)

---

## STEP 0: Verify prerequisites

```bash
cd ~/Desktop/Vanderbilt_University/Internships/RRR/Current_Projects/NTSB/NTSB_Shivy

# Check structural cache exists
ls -la Testing_Structural_Mapping_Slides/cache/struct_train_v2.jsonl

# Check openpyxl is installed
python -c "import openpyxl; print('openpyxl OK')"
```

If openpyxl is missing: `pip install openpyxl`

If the structural cache is missing, you need to run the first plan first — that cache is built during the A2 eval.

---

## STEP 1: Table 9 — Loss of Engine Power

```bash
cd ~/Desktop/Vanderbilt_University/Internships/RRR/Current_Projects/NTSB/NTSB_Shivy

python Testing_Structural_Mapping_Slides/scripts/compare_table9_zhang.py
```

### What this does

Zhang's Table 9 is his main worked example. He takes "Loss of Engine Power" and tests 5 different evidence combinations:

1. Inoperative engine instruments
2. Combustion liner failure
3. Improper oil usage
4. Inop. instruments + Improper oil (combined)
5. All three causes (combined)

For each evidence set, the script:
- Runs **diagnosis** through A0 and A2 → maps to Zhang's 54 occurrence codes → finds P(loss of engine power)
- Runs **prognosis** through A0 and A2 → matches against Zhang's target labels using embedding similarity
- Compares against Zhang's published BN probabilities

### What you get

| Row | Type | What it shows |
|-----|------|--------------|
| Loss of engine power | DIAGNOSIS | P(this cause) given the evidence — Zhang vs A0 vs A2 |
| Forced landing | prognosis | P(forced landing) — Zhang vs A0 vs A2 |
| Ditching | prognosis | P(ditching) — Zhang vs A0 vs A2 |
| Gear collapsed | prognosis | P(gear collapsed) — Zhang vs A0 vs A2 |
| Other gear collapsed | prognosis | P(other gear collapsed) — Zhang vs A0 vs A2 |
| Destroyed aircraft | prognosis | P(destroyed) — Zhang vs A0 vs A2 |
| Substantial aircraft damage | prognosis | P(substantial damage) — Zhang vs A0 vs A2 |
| Minor aircraft damage | prognosis | P(minor damage) — Zhang vs A0 vs A2 |
| Serious injury | prognosis | P(serious injury) — Zhang vs A0 vs A2 |
| No injury | prognosis | P(no injury) — Zhang vs A0 vs A2 |

Each row has 3 columns per evidence set: Zhang BN value, our A0 value, our A2 value.

**Runtime:** ~5-10 minutes (5 queries × A0 + A2 = 10 model runs)

**Output:**
- `Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_table9.xlsx`
- `Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_table9.json`

### Known limitation

The "No injury" row will show a mismatch. Zhang's BN gives ~94-99% for "No injury" because it models absence states. Our retrieval system predicts events that *happened* in historical incidents — it can't produce high probabilities for things that *didn't* happen. This is an architectural difference, not a bug. Flag it for Jesse/Maha but don't try to fix it.

---

## STEP 2: Figure 11 — Main Gear Collapse Outcomes

```bash
python Testing_Structural_Mapping_Slides/scripts/compare_fig11_zhang.py
```

### What this does

Zhang's Figure 11 shows what happens after different types of landing gear failures. 4 evidence conditions:

1. Landing main gear strut failure
2. Landing gear emergency extension assembly failure
3. Landing gear locking mechanism failure
4. Landing main gear attachment failure

For each, the script runs prognosis through A0 and A2 and compares 3 outcomes against Zhang:
- Destroyed aircraft damage
- Minor aircraft damage
- Minor personnel injury

**Runtime:** ~5 minutes (4 queries × A0 + A2 = 8 model runs, prognosis only)

**Output:**
- `Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_fig11.xlsx`
- `Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_fig11.json`

---

## STEP 3: Figure 12 — Pilot Error Influence Propagation

```bash
python Testing_Structural_Mapping_Slides/scripts/compare_fig12_zhang.py
```

### What this does

Zhang's Figure 12 shows how probabilities change when pilot error is observed, and then when unstable approach is also observed. 2 conditions:

1. After pilot error is observed
2. After pilot error + unstable approach are both observed

For each condition, the script runs BOTH diagnosis and prognosis through A0 and A2, then takes the best match (max of diagnosis vs prognosis) for 8 downstream nodes:

| Node | Type |
|------|------|
| Pilot error | cause (diagnosis) |
| Unstable approach | cause/event |
| Hard landing | event (prognosis) |
| Improper flare | event (prognosis) |
| Dragged wing/tail on runway | event (prognosis) |
| Substantial aircraft damage | damage severity |
| No injury | absence state |
| Minor injury | injury severity |

**Runtime:** ~5 minutes (2 queries × A0 + A2 = 4 model runs, but each runs both diagnosis and prognosis)

**Output:**
- `Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_fig12.xlsx`
- `Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_fig12.json`

### Known limitation

Same "No injury" issue as Table 9. Zhang's BN gives 97-61% for "No injury" depending on the condition. Our model will likely give ~0% because it predicts events that occurred, not absence of events.

---

## STEP 4: Verify all outputs exist

```bash
ls -la Testing_Structural_Mapping_Slides/outputs/zhang_comparison/

# Expected files:
#   comparison_table9.xlsx
#   comparison_table9.json
#   comparison_fig11.xlsx
#   comparison_fig11.json
#   comparison_fig12.xlsx
#   comparison_fig12.json
```

---

## Summary

| Step | Script | Queries | Runtime | Output |
|------|--------|---------|---------|--------|
| 1 | compare_table9_zhang.py | 5 evidence sets × 10 rows | ~5-10 min | comparison_table9.xlsx |
| 2 | compare_fig11_zhang.py | 4 evidence conditions × 3 outcomes | ~5 min | comparison_fig11.xlsx |
| 3 | compare_fig12_zhang.py | 2 conditions × 8 nodes | ~5 min | comparison_fig12.xlsx |

**Total runtime:** ~15-20 minutes

All outputs go to: `Testing_Structural_Mapping_Slides/outputs/zhang_comparison/`

Each Excel file is color-coded:
- **Blue columns** = Zhang's BN values (from his paper)
- **Green columns** = our A0 (embedding-only baseline)
- **Orange columns** = our A2 (embeddings + structural mapping)

This gives you a direct apples-to-apples comparison: Zhang ran these exact scenarios through his Bayesian Network, and now you've run them through your retrieval system. The spreadsheets show where you agree, where you differ, and by how much.
