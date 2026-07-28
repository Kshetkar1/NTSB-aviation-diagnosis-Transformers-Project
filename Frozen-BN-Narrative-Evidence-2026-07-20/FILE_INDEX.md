# File index — Frozen BN + Narrative Evidence

| File | Purpose |
|------|---------|
| `code/query_to_bn.py` | Narrative → evidence → BN inference |
| `code/llm_evidence.py` | Tier-2 LLM parser (fallback) |
| `code/bn_upgraded.py` | Build frozen upgraded BN |
| `code/prognosis.py` | Prognosis / forward inference |
| `code/sparse_cpt.py` | Sparse CPT / Beta-CDF helpers |
| `code/trees.py` | Diagnosis + prognosis trees |
| `tests/frozenbn_heldout_narrative_bn_eval.py` | Main 296 held-out eval |
| `tests/heldout_significance.py` | Significance table writer |
| `tests/reproduce_all_examples.py` | Zhang reproduction bundle |
| `tests/zhang_conditional_method.py` | Section 4.3 conditional checks |
| `tests/frozenbn_maha_narrative_evidence_ladder_demo.py` | Maha ladder demo |
| `tests/frozenbn_tiered_parser_validation_11_scenarios.py` | Parser scoreboard |
| `outputs/heldout_significance.md` | Hero numbers for paper |
| `presentations_FrozenBN/jesse_jul28_update.pptx` | Jesse update deck |
| `paper_drafts/Draft_NTSB_Paper_7_26_26.docx` | Latest draft (copied from Desktop Jul 26) |
| `paper_drafts/Draft_NTSB_Paper_RESTRUCTURED_7_22_26.docx` | Restructured draft Jul 22 |
| `paper_drafts/NTSB_Paper_6:4:26.docx` | Early Jun 4 draft |
| `REPRODUCE.md` | How to run everything |

Run from repo root with:

```bash
export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code"
```
