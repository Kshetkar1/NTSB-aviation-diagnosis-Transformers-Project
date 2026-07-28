# What is Zhang's "Table 4"? — Settling the Question

**Paper:** Zhang, X. & Mahadevan, S. (2021). *Bayesian network modeling of accident
investigation reports for aviation safety assessment.* Reliability Engineering and
System Safety 209, 107371. (`docs/BN-NTSB RESS 2021.pdf`)

---

## VERDICT (up front)

**Table 4 is an ILLUSTRATIVE / pedagogical example, NOT a quantity produced by
Zhang's real estimator and NOT something that can be "recreated" from the NTSB
data.**

It is the conditional probability table (CPT) of node `x3` in the *toy* four‑node
Bayesian network of **Fig. 2** ("A simple Bayesian network illustrating the
occurrence of an aviation accident"), which Zhang uses in **Section 3 (background
on Bayesian networks)** purely to teach the reader what a CPT is. Its numbers
(`0.99, 0.93, 0.95, 2e‑9`) are hand‑picked round values, not counts from the
database, and they are mathematically inconsistent with the estimator Zhang
actually defines in Section 4.

➡️ **The earlier "illustrative" analysis is correct. Maha's expectation that
Table 4 should be reproduced from data does not hold for Table 4 itself.** What
*can* (and should) be reproduced is the *real* estimator in Section 4.3 (Eqs. 8–11)
and its worked fire example (Eq. 9) — which the repo's `empirical_cause_distribution`
already reproduces exactly (see §5 below).

---

## 1. Where Table 4 lives and what surrounds it

Table 4 appears on **page 5**; the text that introduces it is on **page 4**. Both
pages are inside **Section 3, "Bayesian networks"** — the tutorial/background
section, *before* Section 4 "Proposed methodology" begins (Section 4 starts on
page 5, after Table 5).

### Section context (page 3 → 4, the heading that owns Table 4)

> **3. Bayesian networks**
> A Bayesian network is a directed acyclic graph (DAG) ... **Fig. 2 shows a
> four-node Bayesian network**, where the random variables x1 and x2 are the
> parent nodes of x3 ... x3 is the parent node of x4 ...

**Fig. 2 caption (page 5):** *"A simple Bayesian network illustrating the
occurrence of an aviation accident."*

### Lead-in text immediately before the tables (page 4)

> "For the Bayesian network shown in Fig. 2, the marginal distributions of root
> nodes can be inferred from historical data. With respect to the event electrical
> system wiring overheating, we can count how many times overheating occurred in
> the wiring out of the total number of flights. **For the sake of demonstration,
> assume that random variables x1 and x2 have the marginal distributions shown in
> Table 3.**"
>
> "Another important quantity is the conditional probability table (CPT) for the
> child nodes. The CPT measures the probability of each value of one variable if
> we know the values taken by the other variables. **Table 4 shows the CPT of the
> random variable x3**, we observe ..." (continues on page 5) "... **that if the
> landing gear normal brake system is worn out and the electric wiring is
> overheated, then there is a probability of 0.99 for the fire occurrence.**"

The phrase **"For the sake of demonstration, assume ..."** governs Table 3, and
Table 4 is the CPT of the *same* demonstration network (`x3` is the child of `x1`,
`x2` from Table 3). So Table 4 inherits the "assume / for demonstration" framing.

### Table 4 exact caption and contents (page 5)

> **Table 4 — Conditional probability table (CPT) of random variable x3 shown in Fig. 2.**

| | Landing gear normal brake system wear = **Yes** | | Landing gear ... = **No** | |
|---|---|---|---|---|
| Electrical system wiring overheating → | **Yes** | **No** | **Yes** | **No** |
| **P(x3 = 1)** | 0.99 | 0.93 | 0.95 | 2e−9 |
| **P(x3 = 0)** | 0.01 | 0.07 | 0.05 | 1−(2e−9) |

(The row label "P(x1 = 0)" for the second row is a typo in the paper; from context
it is P(x3 = 0). The same typo appears in Tables 3 and 5.)

For reference, its sibling tables:

- **Table 3 (page 5):** marginals `P(x1=1)=0.0001`, `P(x2=1)=0.0002` — also explicitly
  "for the sake of demonstration".
- **Table 5 (page 5):** CPT of `x4` (aircraft damage given fire): `P(x4=1|fire)=0.92`,
  `P(x4=1|no fire)=0`. Same toy network, same illustrative status.

### What x1…x4 mean in the toy network

- `x1` = landing gear normal brake system wear
- `x2` = electrical system wiring overheating
- `x3` = **fire** occurrence
- `x4` = **aircraft damage**

These four made-up nodes exist only to demonstrate inference. Fig. 3 (page 6) then
uses these same illustrative numbers to *demonstrate belief updating* ("if the fire
happened ... 67% probability that the wiring is overheated, 33% that the landing
gear is worn"). That whole worked inference is built on the Table 3/4/5 toy numbers.

---

## 2. What Table 4 represents, and how Zhang says he produced it

- **Represents:** the CPT `P(x3 | x1, x2)` — i.e., P(fire | landing-gear-wear,
  wiring-overheating) for the 2×2 parent combinations — in the *illustrative*
  Fig. 2 network.
- **How produced:** Zhang does *not* say he estimated it. The governing verb is
  **"assume … for the sake of demonstration."** The numbers are presented as given
  inputs to motivate the inference demo in Fig. 3, not as outputs of an estimator.

This stands in deliberate contrast to **Section 4**, where Zhang lays out the
estimator he *actually* uses on the data:

- **Prior probabilities** (§4.2, Eq. 6): `P(e) = T(e) / T_sf`, with the denominator
  `T_sf = 184,517,128` total performed flights (1982–2006), e.g. `P(fire) = 102 /
  184,517,128 ≈ 5.52e‑7` (page 6).
- **Conditional probabilities** (§4.3, Eqs. 7–11): single‑cause contribution
  `P(ω | e_i) = P(ω, e_i)/P(e_i)` (Eq. 8); multi‑cause via a noisy‑max lower bound
  `P(ω | e_1..e_m) ≥ max_i P(ω | e_i)` (Eqs. 10–11).
- **Worked fire example** (§4.3, Eq. 9, page 8):
  `P(fire | fuel control leak) = 1/102 = 9.80e‑3`,
  `P(fire | electrical wiring overheating) = 1/102 = 9.80e‑3`,
  `P(fire | airframe component malfunction) = 32/102 = 31.37e‑2`.

Note the values in Eq. (9) are tiny/odd fractions of integer counts — nothing like
Table 4's `0.99/0.93/0.95`.

---

## 3. Does Table 4 appear in Zhang's code? (`main.py`, `NTSB.xdsl`)

**No.** The toy CPT is nowhere in the code or the generated model.

- `Zhang's Approach 2026/main.py` — **no standalone `0.99`** anywhere (`rg "0\.99"`
  → no matches). The only `0.95` is an unrelated guard that caps a *computed*
  conditional probability when it equals 1.0:

```762:769:Zhang's Approach 2026/main.py
        if key not in dictElement.keys():
            dictElement[key] = len(jointEvents)/len(denominator)
            
            if dictElement[key] == 1:
                dictElement[key] = dictElement[key] * 0.95
```

  i.e. CPT entries are `count(joint)/count(denominator)` from data — the Eq. (8)
  estimator — never the literals in Table 4.

- `Zhang's Approach 2026/NTSB.xdsl` — the generated GeNIe model. Its CPT
  `<probabilities>` are **tiny computed priors** (e.g. `5.41955e‑08
  0.99999994…`), produced by the Eq. (6)/(8) machinery. There is **no toy CPT** of
  the form `0.99 / 0.93 / 0.95 / 2e‑9`, and **no node CPT** for "landing gear normal
  brake system wear" × "wiring overheating" → fire with Table 4's values. (The
  strings "landing gear", "brake", "overheating" do appear as *real* NTSB cause
  node names elsewhere in the network, but carry data‑estimated probabilities, not
  the round toy numbers.)

So Table 4's numbers are neither computed nor hardcoded by Zhang's program. They
exist only in the prose/figures of the Section 3 tutorial.

---

## 4. Why this is decisive (the proof that it's illustrative)

1. **Location & framing.** Table 4 is in **Section 3 (background)**, bound to Fig. 2,
   the "simple / four-node" demonstration network. Its sibling Table 3 is explicitly
   prefaced "**For the sake of demonstration, assume …**". The real estimator is a
   *separate* later section (Section 4).
2. **Hand-picked numbers.** `0.99, 0.93, 0.95` are round teaching values; `2e‑9` is
   an arbitrary near-zero baseline. None is a ratio of integer counts over `102` or
   `184,517,128` (Zhang's actual denominators).
3. **Inconsistent with Zhang's own estimator.** His multi-cause rule (Eqs. 10–11)
   forces `P(fire | both causes) ≥ max_i P(fire | e_i)`. From his *own* Eq. (9), the
   single-cause fire numbers are ≈ `0.0098` (wiring) and at most `0.314` (airframe).
   A "both-present → **0.99**" cell is unreachable by his estimator — the estimator
   would top out around `0.31`, never `0.99`. Table 4 therefore cannot be an output
   of the method the paper proposes.
4. **Absent from code.** As shown in §3, neither `main.py` nor `NTSB.xdsl` contains
   Table 4's values.

Any one of these is suggestive; together they are conclusive.

---

## 5. The nearby quantity that IS recreatable — and a live recreation

What *is* reproducible from the data is Zhang's **real** conditional-probability
estimator (Eqs. 8–9) and its companion Table 7‑style `P(cause | outcome)` counts.
The repo already implements this in `zhang_diagnosis.empirical_cause_distribution`
(`P(cause|outcome) = count(cause & outcome)/count(outcome)`).

Running it on the user's `data/processed/refined_dataset_1982_2006.json` (1,742
incidents) for the **fire** outcome:

| Quantity | Zhang (paper) | Recreated (this repo) | Match |
|---|---|---|---|
| `T(fire)` total fire accidents 1982–2006 | **102** (page 6, Eq. 9) | **102** | ✅ exact |
| `P(fire ↔ airframe component malfunction)` | **32/102 = 31.37e‑2** (Eq. 9) | **32/102 = 0.314** | ✅ exact |
| electrical wiring present in fire accidents | 1 (Eq. 9, single co-occurrence cited) | 11 of 102 (0.108) | partial* |

\* Zhang's Eq. (9) cites a single very specific finding string for wiring; the repo's
label set (`Electrical system, electric wiring`) is broader, hence 11 vs 1. The
headline anchors — `T(fire)=102` and `airframe = 32/102 = 0.314` — reproduce Zhang
**exactly**, confirming the repo reproduces his *real* estimator.

Top recreated `P(cause | fire)` (real, data-derived):

```
0.314  (32) Airframe/component/system failure/malfunction
0.108  (11) Electrical system, electric wiring
0.108  (11) Emergency procedure
0.088  ( 9) Loss of engine power (total) - mechanical failure/malfunction
0.069  ( 7) Fluid, fuel
```

This is the genuine, reproducible CPT-style content. Table 4 is **not** part of it.

---

## 6. Recommended message for the advisor (Maha)

> Maha — I traced Table 4 through the paper and Zhang's released code. Table 4 is
> **not** an estimated quantity, so there is nothing in it to "recreate from data."
> It is the CPT of node x3 in the *toy* four-node network in **Fig. 2**, which lives
> in **Section 3 (the Bayesian-network background section)**. Zhang introduces it
> with "**for the sake of demonstration, assume …**" (page 4), and uses its
> round numbers (0.99 / 0.93 / 0.95 / 2e‑9) only to walk the reader through belief
> updating in Fig. 3. The 0.99 cannot come from his estimator: his own multi-cause
> rule (Eqs. 10–11) caps the fire CPT at the max single-cause value, which from his
> Eq. (9) is ≈ 0.31, never 0.99. The values also appear nowhere in `main.py` or
> `NTSB.xdsl`.
>
> The quantities that *are* reproducible — and that I think you actually want — are
> Zhang's **real** prior (Eq. 6) and conditional (Eqs. 8–9) estimators. We already
> reproduce those exactly: total fire accidents `T(fire)=102` and
> `P(fire ↔ airframe component malfunction) = 32/102 = 0.314`, both matching Zhang's
> Eq. (9) to the digit. If it would help, I can build a **"real" CPT for an actual
> child node** (e.g. fire) populated entirely from the data via Eqs. 8–11 — i.e. a
> data-derived analogue of Table 4 for a genuine network node, rather than the
> textbook toy table.

**Bottom line:** Table 4 = illustrative/pedagogical (correct per the earlier
analysis). The recreatable target is Section 4.3's estimator (Eqs. 8–9), already
reproduced. Offer Maha a data-built CPT for a real node if she wants a "populated
Table 4"-style artifact.

---

### Source index

- Table 4 caption & contents: **page 5**.
- Table 4 lead-in ("for the sake of demonstration", "Table 4 shows the CPT of x3",
  "probability of 0.99 for the fire occurrence"): **pages 4–5**.
- Section 3 heading + Fig. 2 ("four-node" / "simple … illustrating"): **pages 3–5**.
- Real estimator — priors Eq. (6), `T_sf=184,517,128`, `P(fire)=102/…`: **page 6**.
- Real estimator — conditionals Eqs. (7)–(11), fire example Eq. (9): **pages 7–8**.
- Code: `Zhang's Approach 2026/main.py` lines ~762–769 (Eq. 8 + 0.95 cap; no 0.99);
  `Zhang's Approach 2026/NTSB.xdsl` (computed priors, no toy CPT).
- Recreation: `zhang_diagnosis.empirical_cause_distribution("fire", …)` on
  `data/processed/refined_dataset_1982_2006.json`.
