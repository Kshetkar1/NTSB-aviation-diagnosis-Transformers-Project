# Apps Parity — NTSB Diagnosis / Prognosis Demo

Separate **Diagnosis** and **Prognosis** Streamlit modes with labeled probabilities
and side-by-side **Zhang (published)** comparison. The original demo in `../apps/`
is untouched.

## What changed vs the original app

| Feature | Original (`apps/`) | This folder (`apps-parity/`) |
|---------|-------------------|--------------------------------|
| Missing outcome | Hard stop (`st.stop`) | Manual outcome picker |
| **Leak-safe pipeline** | Yes (redact + truncate) | **Yes** — `inference_query`, 4000-char cap |
| **Retrieval index default** | Zhang window | **Zhang window 1982-2006** (`run_app.sh`) |
| **bn-sev severity panel** | Yes (90.9 / 77.4%) | **Yes** — primary prognosis readout |
| Prognosis BN readout | Hidden behind toggle | **Always visible** (Table 9 scenarios) |
| Zhang comparison | Partial / offline epilogue | **Ours \| Zhang \| Δ \| Method** on every table |
| Markov tree | Primary prognosis view | Collapsed expander, labeled "ours, not Zhang BN" |
| Diagnosis / Prognosis | Separate modes | Still separate (sidebar radio) |

## Probability labels

See `PLAN.md` for the full estimand table. Short version:

- **Diagnosis Table 7:** `P(cause | outcome)` — Zhang counting
- **Diagnosis tree level 2+:** exploratory, no Zhang benchmark
- **Prognosis bn-sev (primary):** k=100 severity virtual evidence → frozen BN (paper §5)
- **Prognosis BN panel:** `P(target | evidence)` — frozen BN propagation (Table 9)
- **Prognosis Markov tree:** `P(next | current)` — our sequence model

## Run

```bash
cd Frozen-BN-Narrative-Evidence-2026-07-20/apps-parity
chmod +x run_app.sh
./run_app.sh
```

Or manually:

```bash
export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code:${PYTHONPATH:-}"
streamlit run Frozen-BN-Narrative-Evidence-2026-07-20/apps-parity/streamlit_app.py
```

## Demo queries (Maha sign-off)

1. **Diagnosis:** `engine caught fire during takeoff` → Table 7 should match Zhang (fire 85/85)
2. **Prognosis:** `trouble with an engine instrument during the flight` → P(LOEP)=0.95, P(forced landing)≈0.136

## Files

| File | Role |
|------|------|
| `streamlit_app.py` | Entry point, sidebar, shared evidence bar |
| `diagnosis_view.py` | Upstream causes + tree + optional BN |
| `prognosis_view.py` | BN downstream (primary) + Markov tree (secondary) |
| `demo_common.py` | Retrieval, BN inference, shared UI |
| `evidence_bridge.py` | Narrative parsing, manual pickers |
| `zhang_reference.py` | Table 9 / Fig 12 published numbers |
| `PLAN.md` | Implementation checklist |

## Offline Zhang audit

Regenerate full 93-value comparison:

```bash
python Frozen-BN-Narrative-Evidence-2026-07-20/tests/bn_full_comparison.py
```

The epilogue panel reads `outputs/bn_full_comparison.json`.
