# NTSB Improvements — June 2026 (self-contained)

Comparison of an NTSB incident-probability method against **Zhang & Mahadevan (2021)** —
**without** building our own Bayesian network and **without** agents. This folder is
**fully self-contained**: every input is vendored under `data/`, and no code reaches
outside the folder.

## The two-lane framing (read this first)

| Lane | What it is | vs Zhang | Honest expectation |
|------|-----------|----------|--------------------|
| **Lane 1 — Empirical CPT (counting)** | Count P(outcome \| evidence) directly from cases — same *kind* of estimator as Zhang's CPTs | same **band** on data-rich cells | near Zhang, **not exact** |
| **Lane 2 — Retrieval + structural** | Narrative → diagnosis/prognosis on **unseen** incidents (future: LLM token probs) | **different question** | won't match — does what the BN **can't** |

Two findings anchor the story:
1. **Table 4 is pedagogical.** In Zhang's built `NTSB.xdsl`, `Fire`'s only parent is the
   anti-ice node (0.95) — not brake/overheat. Table 4 was a Fig-2 teaching CPT, never
   estimated from counts. Kept as the honest "why it can't match" exhibit.
2. **Counting lands in Zhang's band, not exact** — coding era (eADMS vs legacy) +
   Beta-CDF smoothing + labeling sensitivity. Table 5: 0.73 vs 0.92 (within 0.20).

## Run

```bash
cd NTSB_improvements_june_19_2026
python scripts/run_all.py
# → outputs/maha_probability_comparison.html  (open in a browser)
```

## Layout

```
data/                       vendored inputs (self-contained)
  refined_dataset.json        NTSB incidents (gitignored — copy locally)
  struct_cache_v2.jsonl       LLM causal-chain structures (177 incidents)
  zhang_table9_ground_truth.json   Zhang BN posteriors (repro vs published)
  engine_vs_zhang_table9.json      retrieval engine A0/A2 vs Zhang (May 2026)
src/                        self-contained code (no parent imports)
scripts/                    pipeline steps + report builder
docs/                       ROADMAP_TO_MAHA.md, label_protocol.md
outputs/                    generated artifacts
```

## Pipeline

| Step | Script | Output | Lane |
|------|--------|--------|------|
| 1 | `step01_label_incidents.py` | `incident_labels.csv` | — |
| 2 | `step02_table4_cpt.py` | `table4_cpt_full.*` (contrast) | 1 |
| 5 | `step05_table4_cpt_restricted.py` | `table4_cpt_restricted.*` | 1 |
| 3 | `step03_table5_cpt.py` | `table5_cpt.csv` (0.73 vs 0.92) | 1 |
| 4 | `step04_audit_sample.py` | `table4_label_audit.csv` | 1 |
| 6 | `step06_keyword_vs_struct.py` | `step06_keyword_vs_struct.*` | 2 |
| 7 | `step07_diagnosis_vs_zhang.py` | `step07_diagnosis_vs_zhang.*` | 2 |
| 8 | `step08_doubt_register.py` | `step08_doubt_register.*` | — |
| — | `build_html_report.py` | `maha_probability_comparison.html` | — |

## Docs

- `docs/ROADMAP_TO_MAHA.md` — staged plan, two-lane framing, doubt register, definition-of-done
- `docs/label_protocol.md` — labeling rules

## Note on `data/refined_dataset.json`

This 13 MB file is gitignored. To re-vendor it:
```bash
cp ../data/processed/refined_dataset.json data/refined_dataset.json
```
