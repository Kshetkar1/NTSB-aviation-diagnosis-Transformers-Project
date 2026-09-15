# Apps Parity — Implementation Plan

New folder (`apps-parity/`) so the original demo in `apps/` stays untouched.

**Goal:** Separate Diagnosis and Prognosis modes. Any narrative runs. Every
probability is labeled. Zhang published values shown side-by-side where they exist.

---

## Phase 1 — Scaffold (this PR)

- [x] **Step 1:** Create `apps-parity/` with `PLAN.md`, `README.md`, `run_app.sh`
- [x] **Step 2:** `zhang_reference.py` — Table 7 fire lookup, Table 9 grid, evidence→column matcher
- [x] **Step 3:** `evidence_bridge.py` — parse narrative; outcome/seed fallback pickers (no `st.stop`)
- [x] **Step 4:** `demo_common.py` — shared retrieval, BN inference, tree rendering, Zhang audit panel
- [x] **Step 5:** `diagnosis_view.py` — upstream causes + Table 7 vs Zhang columns + tree
- [x] **Step 6:** `prognosis_view.py` — BN downstream (primary, Zhang-comparable) + Markov tree (secondary)
- [x] **Step 7:** `streamlit_app.py` — sidebar mode switch (Diagnosis | Prognosis), shared evidence bar

## Phase 2 — Paper parity (this update)

- [x] **Step 8:** Leak-safe query prep (`[:4000]` + `inference_query`) before embed/parse
- [x] **Step 9:** Default retrieval index = Zhang window 1982-2006 (`run_app.sh`)
- [x] **Step 10:** bn-sev panel (k=100 virtual severity → frozen BN) in prognosis mode
- [ ] **Step 11:** Run `tests/bn_full_comparison.py`; ensure audit JSON exists
- [ ] **Step 12:** Maha review: sign off on “close” tolerance for 12 known BN cell deltas

## Phase 3 — Advisor sign-off (meeting, not code)

- [ ] **Step 11:** Demo query 1 — fire diagnosis (Table 7 exact)
- [ ] **Step 12:** Demo query 2 — engine instrument prognosis (Table 9 anchors exact)
- [ ] **Step 13:** Maha + Jesse initial estimand table in this plan’s README

---

## Probability labels (locked semantics)

| Mode | Panel | Estimand | Zhang comparable? |
|------|-------|----------|-------------------|
| Diagnosis | Table 7 ranking | P(cause \| outcome) counting | Yes — 85/85 fire |
| Diagnosis | LTP / retrieval | P(cause \| Q) tilt | Neutral = exact Table 7 |
| Diagnosis | Tree level 1 | Same as Table 7 | Yes |
| Diagnosis | Tree level 2+ | P(cause \| outcome ∧ parent) | No benchmark |
| Prognosis | **bn-sev (primary)** | P(severity \| narrative) via k-NN virtual evidence | Yes — 90.9 / 77.4% held-out |
| Prognosis | **BN readout (Table 9)** | BN posterior P(target \| evidence) | Yes — Table 9 / Fig 12 |
| Prognosis | Markov tree | P(next event \| current) | No — ours, labeled |
| Prognosis | Tree leaves | Empirical P(outcome \| event) | No — labeled |

---

## Launch

```bash
cd Frozen-BN-Narrative-Evidence-2026-07-20/apps-parity
./run_app.sh
```

Original demo unchanged:

```bash
streamlit run Frozen-BN-Narrative-Evidence-2026-07-20/apps/frozenbn_streamlit_diagnosis_prognosis_demo.py
```
