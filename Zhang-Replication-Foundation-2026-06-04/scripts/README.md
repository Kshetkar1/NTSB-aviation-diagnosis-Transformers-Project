# Zhang Replication Runner

Faithful reproduction of the numerical artifacts in:

> Zhang, X., & Mahadevan, S. (2021). *Bayesian network modeling of accident
> investigation reports for aviation safety assessment.* Reliability
> Engineering & System Safety, 209, 107371.

This folder contains **our** scripts. The verbatim mirror of Zhang's GitHub
repository sits at the sibling folder `../Zhang's Approach 2026/` and is
**never modified** — that's the "crime-scene" snapshot the thesis cites.

## What gets reproduced

| # | Artifact | Section | Script | Status |
|---|---|---|---|---|
| 1 | Sensitivity table (12 rows × strut → main/gear collapsed) | Table 8, §5.2 | `01_table8_sensitivity.py` | ✅ 8/12 OK at 50K samples |
| 2 | Multi-evidence accumulation (4 evidences → 3 outcomes) | Fig 11, §5.2 | `02_fig11_multi_evidence.py` | ✅ trend correct |
| 3 | Pilot error scenario (3 phases × 4 outcomes) | Fig 12 / §5.3.1 | `03_fig12_pilot_error.py` | ✅ wing/tail/rotor exact, no-injury fails (absence state) |
| 4 | Loss of engine power table (5 evidences × 10 outcomes) | Table 9, §5.4 | `04_table9_engine_power.py` | ✅ 36/50 OK across the 4 non-rare evidence sets |
| 5 | Consolidated validation report | (this) | `05_validation_report.py` | 76/86 cells match (88%) |
| 6 | Full per-incident posterior dump (cross-model harness) | (n/a) | `06_full_posterior_table.py` | ✅ ready for downstream comparison |
| 7 | Multi-seed noise-band validation | (defensibility) | `07_seed_band.py` + `run_seed_band.sh` | runs after 99M completes |
| 8 | α=1.04645, β=2.02591 calibration | Table 7, §4 | `08_table7_alpha_beta.py` | ✅ EXACT MATCH to 9 decimal places |
| 9 | Full 740-node posteriors at every paper scenario | (research artifact) | `09_full_table_paper_scenarios.py` | ✅ 26 scenarios × 740 nodes × 2 states = 38,480 rows |
| 10 | Pivoted Yes-state matrix (node × scenario) | (browseable view) | `10_pivot_yes_matrix.py` | ✅ 740 × 26 + interesting-nodes sheet (230 active) |

Latest aggregate: see [`outputs/VALIDATION.md`](outputs/VALIDATION.md) and
[`outputs/validation_report.xlsx`](outputs/validation_report.xlsx).
Full Zhang BN probability table:
[`outputs/zhang_full_probability_table.xlsx`](outputs/zhang_full_probability_table.xlsx) +
[`outputs/zhang_yes_matrix.xlsx`](outputs/zhang_yes_matrix.xlsx).

## Setup (one-time)

### 1. Conda env (Python 3.12)

```bash
conda create -n zhang2021 python=3.12 -y
conda activate zhang2021
pip install --index-url https://support.bayesfusion.com/pysmile-A/ pysmile
pip install pandas numpy scipy openpyxl matplotlib
```

### 2. Academic license

Place your BayesFusion academic license at `pysmile_license.py` in this
folder. The file is gitignored — never commit it.

You can request a free 6-month academic license at
<https://download.bayesfusion.com/> (sign up with `@vanderbilt.edu`, click
"Get 6-month academic key" under SMILE).

### 3. Verify

```bash
conda activate zhang2021
cd Zhang_Replication_Runner
python 00_smoke_test.py
```

Expected: prints "loaded NTSB.xdsl" + 740 nodes, then a sample posterior.

## Run all replication targets

### Fast pass (default — ~30 seconds total)

```bash
conda activate zhang2021
python 01_table8_sensitivity.py    --samples 50000  --seed 42
python 02_fig11_multi_evidence.py  --samples 100000 --seed 42
python 03_fig12_pilot_error.py     --samples 100000 --seed 42
python 04_table9_engine_power.py   --samples 100000 --seed 42
python 05_validation_report.py
```

### Full-fidelity pass (matches Zhang's XDSL: 99,999,999 samples — ~7-8 hours)

```bash
bash run_high_sample.sh
```

This runs the same four scripts at Zhang's exact declared sample count
(`numsamples=99999999` from his `NTSB.xdsl`) and writes outputs with a
`_99M` suffix so the fast-pass outputs are not overwritten. Progress is
appended live to `outputs/run_99M.log`. Resume-friendly: re-launching the
script skips any step whose `*_99M.json` already exists.

The high-sample run is the canonical "as close to Zhang as we can get
without his random seed" output. The fast pass is for development /
sanity checks.

### Full Zhang probability table at every paper scenario (~5 min)

Captures Zhang's BN posteriors for **every node × every scenario in his
paper** -- 26 scenarios × 740 nodes × 2 states = 38,480 rows. This is the
artifact you inspect when you want to see "all of Zhang's probabilities,
not just the cells he printed".

```bash
python 09_full_table_paper_scenarios.py --samples 1000000 --seed 42
python 10_pivot_yes_matrix.py
```

Outputs:
- `outputs/zhang_full_probability_table.{parquet,xlsx}` -- long-form table
- `outputs/zhang_yes_matrix.xlsx` -- pivoted node × scenario matrix of P(node=Yes)
  with sorted "interesting" sheet (230 nodes whose probabilities move
  meaningfully across scenarios) and per-scenario-kind top-80 sheets.

### Optional: arbitrary per-incident posterior dump (cross-model harness)

```bash
# every node × state, no evidence, fast (~10 s)
python 06_full_posterior_table.py --no-evidence --samples 1000000 --out-suffix _prior

# per-test-case posteriors (input: JSON list of {case_id, evidence})
python 06_full_posterior_table.py \
    --evidence-json my_test_cases.json \
    --samples 1000000 \
    --out-suffix _mycases
```

This is the file you join against your own model's outputs when you build
the per-incident comparison harness.

Each script writes JSON + Excel to `outputs/`. The validation report at the
end reads those JSONs and produces the consolidated comparison table.

## Why pysmile (and not GeNIe GUI / pyAgrum / pgmpy)

- **pysmile**: same SMILE engine as Zhang (he used `set_bayesian_algorithm(3)`
  in his `Scenario analysis.ipynb`). Bit-identical numbers within sampling
  noise. Scriptable.
- **GeNIe GUI**: same engine but manual point-and-click. Not reproducible.
- **pyAgrum / pgmpy**: open source, reads `.xdsl`, but uses different
  inference implementations → ε-level numerical drift on the same network.
  Not "exactly how he did it."

## Caveats and known limits

See [`outputs/VALIDATION.md`](outputs/VALIDATION.md) for the long-form
explanation. Short version:

- Tiny priors (1e-7 to 1e-9) need ~1e7 samples to escape the noise floor.
  Zhang's XDSL declares `numsamples=99,999,999`; we run at 50K-100K for
  speed and accept that the bottom rows of Table 8 read as 0.
- "No injury" is an absence/complement state encoded such that
  likelihood-weighted sampling drives it to 0 when other evidence is
  present. This is structural, not a bug.
- Setting evidence directly on Loss-of-engine-power (a low-prior child)
  causes likelihood weighting to degenerate; we fall back to EPIS_SAMPLING
  automatically, but accuracy still suffers in column 5 of Table 9.
- "Substantial aircraft damage" shows a consistent ~+0.02 overshoot vs.
  the paper across multiple scenarios. Most likely explanation: minor
  network revision between paper publication and the GitHub `NTSB.xdsl`
  commit. The committed file is what we replicate against.

## File tour

```
Zhang_Replication_Runner/
├── README.md                     # this file
├── pysmile_license.py            # YOUR license (gitignored)
├── PYSMILE_README.txt            # BayesFusion's notes (gitignored)
├── pysmile.so                    # native binary (gitignored — installed via pip)
├── 00_smoke_test.py              # load .xdsl, run inference, sanity check
├── 01_table8_sensitivity.py      # Table 8 reproduction
├── 02_fig11_multi_evidence.py    # Fig 11 reproduction
├── 03_fig12_pilot_error.py       # Fig 12 reproduction
├── 04_table9_engine_power.py     # Table 9 reproduction
├── 05_validation_report.py       # consolidate all four into validation_report{,_99M}.xlsx + VALIDATION{,_99M}.md
├── 06_full_posterior_table.py    # arbitrary per-incident posteriors (for cross-model comparison)
├── 07_seed_band.py               # multi-seed noise-band aggregator
├── 08_table7_alpha_beta.py       # Table 7 alpha/beta calibration
├── 09_full_table_paper_scenarios.py  # full 740-node posteriors at every paper scenario
├── 10_pivot_yes_matrix.py        # pivot full table into node × scenario Yes-matrix
├── run_high_sample.sh            # launcher for the 99M-sample pass
├── run_seed_band.sh              # launcher for the 5-seed × 1M band run
├── status.sh                     # live progress dashboard for the 99M run
└── outputs/
    ├── table8_sensitivity{,_99M,_seedN}.{xlsx,json}
    ├── fig11_multi_evidence{,_99M,_seedN}.{xlsx,json}
    ├── fig12_pilot_error{,_99M,_seedN}.{xlsx,json}
    ├── table9_engine_power{,_99M,_seedN}.{xlsx,json}
    ├── validation_report{,_99M}.xlsx + VALIDATION{,_99M}.md
    ├── seed_band_report.xlsx + SEED_BAND.md
    ├── table7_alpha_beta.json
    ├── zhang_full_probability_table.{parquet,xlsx} + _meta.json
    ├── zhang_yes_matrix.{parquet,xlsx}
    ├── full_posterior{,_*}.parquet
    └── run_99M.log + run_seed_band.log + run_full_table.log
```
