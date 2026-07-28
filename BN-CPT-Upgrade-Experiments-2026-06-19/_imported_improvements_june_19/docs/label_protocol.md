# Label protocol — Table 4 variables

Three-state labels: **yes**, **no**, **unknown**. CPT cells use only incidents where both parents are **yes** or **no** (unknown excluded from that analysis).

## x1 — Landing gear normal brake system wear

| State | Layer 1 rule |
|-------|----------------|
| **yes** | Corpus matches `BRAKE_WEAR_YES` patterns (brake + wear/fail/degraded) |
| **no** | Corpus mentions landing gear or brake AND matches `BRAKE_WEAR_NO` (serviceable, no wear, normal brake) OR brake topic with no wear/failure signal |
| **unknown** | Neither yes nor no patterns match |

## x2 — Electrical system wiring overheating

| State | Layer 1 rule |
|-------|----------------|
| **yes** | `ELECTRICAL_OVERHEAT_YES` (overheat, arc, hot wire, electrical smoke/fire origin) |
| **no** | Electrical/wiring mentioned with `ELECTRICAL_OVERHEAT_NO` (normal, no fault) |
| **unknown** | Otherwise |

## x3 — Fire (Table 4 outcome)

| State | Layer 1 rule |
|-------|----------------|
| **yes** | Sequence/findings: fire or explosion occurrence; or `acft_fire` ∈ {IFLT, GRD, BOTH}; or fire/smoke (non-impact) in sequence text |
| **no** | `acft_fire` = NONE/empty AND no fire/smoke/explosion in sequence/findings |

## x4 — Downstream outcome (Table 5)

Configurable proxy for Zhang’s **x4**: **substantial or destroyed aircraft damage**.

| State | Rule |
|-------|------|
| **yes** | damage field SUBS/DEST or sequence/findings substantial/destroyed damage |
| **no** | damage NONE/MIN or no substantial/destroyed signal |

## Corpus fields (in order)

1. `sequence_of_events[].Occurrence_Description`
2. `sequence_of_events[].Occurrence_Code`
3. `findings[].finding_description`
4. `narr_cause` (first 2000 chars)
5. `damage`, `acft_fire`, `ev_highest_injury`

Each label stores `*_method` = `rule` and `*_snippet` = matched text (audit).
