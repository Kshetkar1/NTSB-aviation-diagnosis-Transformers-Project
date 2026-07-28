# Cursor Plan: Get My Probabilities (A2)

## Goal

Build a formatted Excel spreadsheet showing A2's full probability outputs for all 77 test incidents — diagnosis (full distribution over Zhang's 54 codes) and prognosis (predicted downstream events) — with ground truth alongside.

Do NOT modify `main_app.py`.

---

## STEP 0: Check that eval files exist

The eval scripts have already been run. Verify the output files are there:

```bash
cd ~/Desktop/Vanderbilt_University/Internships/RRR/Current_Projects/NTSB/NTSB_Shivy

ls -la Testing_Structural_Mapping_Slides/outputs/eval_A2_structural.json
```

If the file exists, proceed to Step 1.

If it does NOT exist, run the eval first:
```bash
python Testing_Structural_Mapping_Slides/scripts/eval_diagnosis_structural.py \
  --n all --structural --struct-version v2 --output-stem eval_A2_structural
```
This takes ~30-45 min. Has a `--resume` flag if it crashes.

---

## STEP 1: Build the spreadsheet

```bash
pip install openpyxl
python Testing_Structural_Mapping_Slides/scripts/build_three_model_comparison.py
```

This reads the existing A2 eval JSON, and for each of the 77 test incidents:

1. Looks up the ground truth (what actually caused the incident)
2. Maps A2's cause predictions to Zhang's 54 occurrence codes using embedding similarity
3. Builds a full probability vector across all 54 codes + UNMAPPED column
4. Runs A2 prognosis (reuses cached structural extractions — no duplicate LLM calls)
5. Writes everything to a 3-sheet Excel file

**Runtime:** ~15-20 minutes

**Resume-safe:** If the script crashes or you Ctrl-C it, just re-run the same command. It saves a checkpoint after every incident (`checkpoint.jsonl`) and skips already-completed ones on restart. It also saves a partial Excel every 5 incidents, so you can open the `.xlsx` at any time to see progress.

**Output:** `Testing_Structural_Mapping_Slides/outputs/three_model_comparison/three_model_comparison.xlsx`

**Checkpoint:** `Testing_Structural_Mapping_Slides/outputs/three_model_comparison/checkpoint.jsonl` (delete this to start fresh)

---

## What you get

| Sheet | What it shows |
|-------|--------------|
| Diagnosis — Full Distributions | 77 incidents × full probability over all 54 Zhang codes for A2 + ground truth |
| Prognosis | 77 incidents × top-5 predicted downstream events for A2 |
| Summary | Per-incident accuracy, entropy, aggregate stats |

This is the raw probability output Jesse and Maha asked for — A2's full distributions in Zhang's coding system.
