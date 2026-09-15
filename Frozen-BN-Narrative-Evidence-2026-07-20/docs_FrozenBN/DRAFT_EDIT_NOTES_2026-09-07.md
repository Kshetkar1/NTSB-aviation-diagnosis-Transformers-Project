# Draft edit notes — 2026-09-07

All numbers below were regenerated today. Originals backed up in
`outputs/_prerun_backup_2026-09-07/`. Re-ran: `frozenbn_heldout_narrative_bn_eval.py`,
`heldout_significance.py`, `diagnosis_heldout_eval.py`, `bn_upgraded_full.py`.

---

## A. Corrections to make in the draft

**A1 — Separate the two reproduction scoreboards.**
Right now the draft implies one. There are two, and Table 7 is not in the 93.
State explicitly: Table 7 is its own scoreboard, 85/85 exact. The 93 published
BN quantities are Tables 8-9, Figs 11-13, and §5.2 text. Then give the build:
strict Zhang-recipe rebuild differs on **26**; after the person-node and
4-state severity upgrades, **12** remain. Never write "12" without naming which
build. Source: `outputs/BN_COMPARISON_REPORT.md`, `outputs/bn_upgraded_full.json`.

**A2 — Delete the random-tie-break attribution.**
`SECTION_8_DISCUSSION.md` line 15 blames residual disagreement on "tie-breaking
randomness in parent selection." Our parent selection has no randomness. Rewrite
to say: parents are sorted by edge ratio descending, then edge support descending,
then node name — fully deterministic, which is why no seed is set. The random
jitter is in the original construction code, not ours. Source:
`tests/bn_build_ours.py` lines 94-102.

**A3 — Recount the 93-item breakdown.**
Manuscript part 7 says 48 exact / 29 close / 12 differ. That sums to 89, and with
the 8 qualitative items it reaches 97 against a total of 93. Regenerate the counts
from `outputs/bn_upgraded_full.json` rather than restating them.

**A4 — Update the events-only row.**
Injury moved by one accident out of 296 on re-run. Events-only (`hard+soft`) is
now **82.1%**, not 82.4%. The `hard` row is **63.9%**, not 64.2%. Damage 50.7%
unchanged. Everything else reproduces exactly: 58.4 / 42.6 prior, 90.9 / 77.4
severity, 84.2 / 57.7 diagnosis.

**A5 — Split the baseline claim by metric.**
"Ties strong supervised baselines while leading on damage Macro-F1" is doing
quiet work. True on injury *top-1 accuracy* (McNemar p = 0.289, Holm 0.867 — a
genuine tie). Not true on injury Macro-F1: tfidf-lr 0.603 against our 0.470, a
gap with no CI attached. Damage is a clean lead on both (77.4 / 0.697 against
73.3 / 0.621). Rewrite so the reader can't carry "ties" across both metrics.

**A6 — Fix the retrieval description.**
Not "fraction of neighbours carrying the label." Each of the 100 neighbours
contributes its cosine similarity as a weight; the result is Laplace-smoothed
(alpha = 0.5) and converted to a likelihood ratio against the network's own
severity prior, not entered as a probability. Source: `code/query_to_bn.py`
lines 429-438 and 486-496.

**A7 — Scope the edge rules.**
The three rules (finding to occurrence, occurrence to next occurrence, last
occurrence to injury and damage) describe the Zhang-faithful build only. The
upgraded network actually evaluated **drops** the last-occurrence-to-severity
rule, adds person-to-finding edges, and gives the 4-state severity nodes the
12 most frequent event labels as parents. Say which network each claim is about.
Source: `code/bn_upgraded.py` lines 67-83 and 129.

**A8 — Distinguish the two fire numbers.**
102 / 184,517,128 = 5.5e-7 is the *formula*. The network's actual marginal on
the fire node is **7.72e-8**, because fire has parents and the marginal comes
from propagation. Don't let one stand in for the other.

**A9 — State the narrative filter, and correct the retrieval pool size.**
1,288 is the count of non-empty `narr_accf` in the 1982-2006 window; at the
>= 100-character threshold it is 1,286. But neither is the pool the code
actually searches. `embeddings_1982_2006.npy` holds **1,703** vectors, all of
them window accidents, and `severity_retrieval_distributions` drops a
neighbour only when its `ev_id` is absent from the dataset — which never
happens here. So every k = 100 readout in the paper is drawn from 1,703
candidates, not 1,286.

The gap is index staleness: only 1,363 window accidents currently have any
narrative field at all, so roughly 340 vectors were embedded from text that is
no longer in the processed dataset. Those accidents still contribute their
coded severity to the neighbour vote, so nothing is *wrong* with the readout —
but the reported pool size is, and the index cannot be regenerated from the
current data. Either state 1,703 and note the provenance, or rebuild the index
from the current narratives and re-run. Do not leave 1,288 standing next to a
k-NN result drawn from 1,703. Verified 2026-09-09.

**A10 — Correct why the 43 diagnosis accidents were dropped.**
The draft says all 43 lack cause codes. That is true of 10 of them (6 in 2008,
one each in 2009, 2010, 2011, 2014). The other **33 are all from 2007 and do
have cause codes** — `Occurrences.txt` and `seq_of_events.txt` cover the year,
but they were never decoded into the processed dataset. Root cause, if asked:
the original preprocessing
(`archive/.../01_create_refined_dataset.py` line 30) reads `Occurrences.txt`
with `sep='\t'`, and that file is comma-separated, so it parses to a single
column and merges nothing. The repair scripts `rebuild_1982_2006.py` and
`add_findings_1982_2006.py` use the right delimiter but are scoped to the
network window, because that is where the failure was visible (the network
cannot be built without it; fire == 102 was the check that caught it). 2007 is
too late for the window repair and too early for the modern tables, which begin
in 2008, so it fell between them and nothing failed loudly.
Decoding them with the same `metaData.xlsx` dictionary yields usable cause
categories for 31 of the 33. Rewrite the exclusion sentence to name both
reasons; keep 253 as the headline cohort. Do not silently grow it to 284: 2007
truth would come from the legacy keyword rules while 2008+ truth comes from the
CICTT first-segment rule, and on those accidents the BN arm maps nodes to
categories through the same legacy rules that would define the truth. Source:
`tests/coded_evidence_baseline.py`.

**A11 — CORRECTED 2026-09-09. The test set was never used for tuning, and the
selection procedure disagrees with what shipped.**
An earlier version of this note said the sensitivity sweeps ran on the 296
held-out accidents. That was wrong. `hyperparam_sensitivity.md` states its own
scope in line 3: 236 window accidents from 2002-2006, pool restricted to
1982-2001, "2007-2019 test never touched." Every hyperparameter in this paper
was chosen without the test set. That is cleaner than most published pipelines
and the paper should say so in one sentence.

The real problem is different and sharper. The internal split does not endorse
the shipped setting:

| top_k | injury top-1 | damage top-1 |
|---|---|---|
| 25 | **86.0%** | **66.5%** |
| 50 | 83.5% | 66.5% |
| 100 (production) | 83.5% | 61.9% |
| 200 | 83.1% | 59.3% |

k = 25 wins on both targets, by 2.5 and 4.6 points, on the split that exists
precisely to make this choice. A reviewer who reads the table will ask why the
paper ships 100. There is a real coupling argument — the soft-fact gates
(`min_fq` 0.15, `min_lift` 3.0) were calibrated on a top-100 pool, so k and the
gates were not selected independently — but it is post-hoc unless stated up
front. Two acceptable resolutions, both honest: (i) report held-out results at
k = 25 alongside k = 100 as a robustness check, declaring k = 100 as the
pre-specified setting; or (ii) state the gate coupling explicitly as the reason
100 was fixed. Do NOT silently switch to k = 25 after having seen the k = 100
held-out numbers — the internal-split evidence is legitimate, but the decision
to revisit it would not be, and the clean provenance above is worth more than
a few points.

Note also that held-out accuracy (90.9 / 77.4) far exceeds internal-split
accuracy at the same k (83.5 / 61.9). Likely cause: the internal split searches
a pool of 1982-2001 only, while held-out queries search the full 1982-2006
index. Worth one sentence so the gap does not read as a leak.

**A12 — Soft evidence does not land where the paper says it lands.**
`apply_evidence` documents Jeffrey semantics — "the narrative says the fact
holds with probability c" — but implements Pearl virtual evidence with a
likelihood ratio fixed against the UNCONDITIONAL prior,
LR = [c/(1-c)] / [p0/(1-p0)]. That reaches c only when a fact is the sole soft
fact. Measured over 180 soft facts on the held-out cohort:

| condition | mean \|reached - intended\| | off by > 0.05 |
|---|---|---|
| soft facts alone | 0.246 | 41% |
| soft + hard, as the pipeline runs | 0.277 | 47% |

**45% of soft facts end at a posterior above 0.999** — entered at roughly 0.25
to 0.45 confidence, realised as certainties. Note the first row: this is soft
facts interfering with each other, not hard evidence overriding them.

Mechanism: event-node priors are per-flight and tiny, so the ratio needed to
lift one fact to c is enormous (p0 ~ 1e-5, c = 0.25 gives LR ~ 3e4). Several
such ratios interacting through shared structure saturate one another.

This is not the same defect as BII-9 — nothing here contradicts its own
inputs, and with multiple soft facts no single-pass likelihood assignment can
hit every target marginal (that needs iterative proportional fitting). So the
required fix is to the CLAIM, not necessarily the code: stop writing that the
updated belief equals the retrieval frequency. Say the ratio is calibrated so
the fact alone would reach c, and report that with several facts the realised
beliefs diverge, often to certainty. Verified 2026-09-09; see D7 for the
experiment that would let you keep the stronger claim.

**A13 — FIXED 2026-09-09, and the fix turned an objection into a result.**
The guard was incomplete; it now is not (D9). Re-scored with the leaked wording
removed, the headline is **unchanged at 90.9% / 77.4%**. Write this up as an
experiment: the residual text was worth 12 points to a keyword-only classifier
and zero to the pipeline. That is a much stronger statement than the original
leak-safe assurance, because it is measured on both sides. The diagnosis that
produced it follows.

*Original finding:* the leak guard is incomplete, and the residual is measurable.
Audited 2026-09-09 over all 296 held-out narratives at the 4000-char
truncation. `redact_severity_phrases` removes every CANONICAL graded phrase —
`serious injury` 35 to 0, `minor injury` 28 to 0, `substantial damage` 38 to 0,
`no injuries` 39 to 0, `destroyed` 2 to 0, `fatal` 1 to 0, `hospitalized` 2 to
0. It does not remove bare statements that harm occurred: `injur*` survives in
62 narratives, `burn` in 15, `damage` in 66. Examples that pass the guard —
"the standing passenger fell to the floor and was injured", "resulting in
second-degree burns".

Those survivors are predictive:

| | injury != none | n |
|---|---|---|
| base rate | 41.6% | 296 |
| survivor present (`injur*` or `burn`) | 74.0% | 77 |
| survivor absent | 30.1% | 219 |

A classifier reading NOTHING but whether a survivor appears scores **70.3% on
injury**, against the 58.4% prior. So about 12 points of the injury headroom is
reachable from residual leaked text. On damage the residual is minor: 49.3%
keyword-only against a 42.6% prior.

Do not overstate this in either direction. The pipeline's 90.9% is twenty
points above the keyword rule, so it is plainly doing much more than reading a
leaked word. But the paper currently asserts a leak-safe protocol without
qualification, and this check is one a reviewer can run in ten minutes.

Two acceptable resolutions, and the first is better. (i) Extend the guard to
strip bare outcome statements about persons, re-run the headline, and report
the new number — if 90.9% holds, the objection is closed permanently. (ii) Keep
the guard and disclose this table as a measured bound on residual leakage.
Silence is not an option. `fracture` survives 64/64 but is almost entirely
metallurgical ("sent for fracture analysis") and is not the concern; `burn` is.
See D9.

---

## B. New section to write — diagnosis of the 12 remaining cells

Currently the 12 are counted but never explained. Nobody has asked why any
individual cell misses. Write this up; it is the strongest new material.

**B1 — Two cells are not wrong answers, they are no answer.**
`minor dmg | comb liner` and `minor dmg | oil` both read 5.62291e-07, which is
exactly the untouched prior P(minor damage). The evidence never reaches the node.

**B2 — The mechanism is label fragmentation.**
The network holds four distinct nodes meaning "engine lost power." Only
`loss of engine power (total) - mechanical failure/malfunction` is a parent of
the severity nodes. Combustion liner and oil grade both point to the plain
`loss of engine power` node, which has no edge to damage — the path dead-ends.
The EPR gauge finding also reaches `in flight collision with terrain/water`,
which *is* a severity parent, which is why the eng-instr cells move and these
two do not. Fix is to merge the LOEP node family;
`prognosis.resolve_outcome_targets` already does that expansion elsewhere.
**Do not apply the fix before the meeting** — it would move every number
regenerated today.

**B3 — Report the asymmetry, it is the sharpest thing here.**
The same dead end produces two of the "resolved" wins. `no injury | comb liner`
and `no injury | oil` both read 0.99999939, which is exactly the untouched prior
P(no injury). They grade EXACT only because the published value (0.9978) sits
within 1% of 1.0. Identical non-event, opposite grades. Two of the 14 resolved
cells are a node that never moved.

**B4 — Group the rest by cause, not by count.**
Four cells overshoot at 5.22x-5.50x — too uniform to be four independent
problems, so attribute it to one systematic property of how the severity CPT
allocates mass once any parent fires. Two cells (1.27x, 1.29x) miss only because
the CLOSE threshold is 25%; they would pass at 30%. One cell (T8 main gear at
base prior) is where the released model file agrees with us and not with the
published table.

**B5 — Frame the comparison as calibration, never as "better."**
There is no ground truth in Table 9 — it is a model output, so a different
number is different, not better. What is defensible: on the cells where evidence
reaches the severity nodes, our posteriors sit closer to the empirical rates in
the shared 1982-2006 data. Empirical, among 147 accidents with loss of engine
power: minor damage 0.537, substantial 0.218, destroyed 0.061, serious injury
0.068. On all four, ours is closer than the published value. On the two
dead-end cells, the published value is closer than ours — say so.

**B6 — Report that both networks are far from the data.**
Data says 53.7% of engine-power-loss accidents had minor damage. Published says
0.38%. Ours says 2.1%. Closer is not close. Attribute to the per-flight
denominator inherited from Eq. 6 and flag it as an open question rather than
asserting the cause.

**B7 — Note the support behind the showcase scenario.**
The combustion-liner finding appears in 2 accidents in the whole window; oil
grade in 1. Both combustion-liner accidents had minor damage. Every edge in that
scenario has n = 1 support. This is a methodological observation independent of
whose posterior is closer, and it belongs in the paper.

---

## B-II. New section to write — why a narrative route is necessary, not merely convenient

Generated 2026-09-09 by `tests/coded_evidence_baseline.py`; full tables in
`outputs/coded_evidence_baseline.md`. This is the strongest motivation the
paper has and it is currently absent.

**BII-1 — The coded-evidence route cannot be run after 2006.**
The NTSB replaced its occurrence and finding taxonomy. The network's nodes are
legacy labels (`airframe/component/system failure/malfunction`); accidents from
2008 on are coded in the replacement scheme (`personnel issues-action/decision-
action-incomplete action-ground crew - f`). Across the 263 held-out accidents
from 2008 onward the two vocabularies share **zero** labels, so the original
evidence-setting procedure has no inputs and every posterior falls back to the
prior: 57.8% injury, 42.6% damage against the narrative pipeline's 91.6% and
78.3%. The narrative is not an alternative interface to this network on modern
data; it is the only one.

**BII-2 — Where both routes do run, they tie.**
2007 is the last legacy year. After decoding its codes (see A10), all 33
accidents set real evidence on the network, giving the only head-to-head
comparison available. Injury: coded 81.8%, narrative 84.8%. Damage: coded
66.7%, narrative 69.7%. Exact McNemar p = 1.0 on both. Write this as "no
detectable difference," never as "narratives are better" — n = 33 and the
intervals are wide. Note that 2007 is the year most favourable to the coded
route, which strengthens the tie rather than weakening it.

**BII-3 — This reframes the whole contribution.**
The paper currently argues narratives are a good substitute for coded evidence,
which invites the reply that coded evidence was already there. The defensible
claim is stronger: a 2006-vintage network is unusable on post-2006 accidents by
its own procedure, and the narrative pipeline is what restores it. That is a
reason for the work to exist that does not depend on beating a baseline.

**BII-5 — The network structure does carry signal, and this is the proof.**
On the 33 accidents, human-coded occurrences entered on the *event* nodes and
propagated through the graph lift injury from the prior's 63.6% to **81.8%**
and damage from 42.4% to **66.7%**. Nothing else in the project demonstrates
this as cleanly. Use it whenever the "what is the BN buying us" question comes
up (C5).

**BII-6 — The null BN result is about where evidence enters, not about the
network.** The headline path enters k-NN severity distributions directly on the
injury and damage nodes, where Jeffrey conditioning forces the posterior to
equal the input; that is why `bn-sev` and `retrieval-sev` agree to three
decimals, and it is a consequence of the injection point, not evidence that the
graph is inert. BII-5 is the same network doing real work when evidence enters
upstream. Say it that way — it is both more accurate and more defensible than
"the BN adds nothing."

**BII-7 — The cost of automation, measured, and it is asymmetric.**
Comparing at the same interface on the 33 accidents (evidence on event nodes,
propagated), narrative-parsed evidence versus the human-coded record:

| Target | Human-coded | Narrative-parsed (`hard+soft`) | Exact McNemar |
|---|---|---|---|
| Injury | 81.8% | 81.8% | p = 1.000 |
| Damage | 66.7% | 45.5% | p = 0.039 * |

The parser matches human coding on injury and loses significantly on damage.
This diagnoses the weak `hard+soft` damage number (50.7% on the full cohort):
the deficit is in the narrative-to-event-label step, not in the network. State
it as preliminary — n = 33, one year, p just under 0.05 — and it becomes an
honest limitation plus a concrete next step rather than an unexplained hole.

**BII-8 — The fire cross-inference verdict is too strong and must be reworded.**
`outputs/fire_node_cross_inference.md` concludes the network "carries NO usable
information about the unobserved fire node." That experiment enters event
evidence only and, by design, never observes severity — but in the frozen DAG
`fire` has exactly two children, `aircraft damage` and `personnel injury`, so
that choice closes the only strong path into the node. Re-scored with the k-NN
severity virtual evidence the main pipeline already uses, on the same cohort
and the same masked text (`tests/fire_cross_inference_diagnosis.py`):

| evidence entered | AUC, coded-field | AUC, occurrence |
|---|---|---|
| prior | 0.500 | 0.500 |
| events only (published arm) | 0.380 | 0.422 |
| severity only (collider open) | **0.728** [0.616, 0.829] | **0.801** [0.735, 0.861] |
| events + severity | 0.548 | 0.610 |
| retrieval on the same masked text | 0.936 | 0.960 |

Three things follow. (i) The network *does* perform genuine cross-node
inference — both severity intervals exclude 0.5 — so "no usable information" is
false as written. (ii) The event-evidence failure is real and is topological:
`fire` has 13 ancestors out of 783, six barred by the guard, and the parser
only ever reaches the generic `person: flightcrew` (143/296 accidents), so
event evidence is almost never d-connected to `fire` in a fire-specific way.
(iii) Retrieval still ranks fire better than the network does, and that should
be stated plainly. Rewrite the verdict as: cross-inference works where the
topology leaves a path open, this network is thin above the occurrence layer,
and on this query neighbour voting is the stronger predictor.

**BII-9 — FIXED 2026-09-09. The evidence-combination bug, and what replaced it.**

*The bug.* The severity likelihood was built as L(j) = f_q(j) / p0(j) with p0
the UNCONDITIONAL severity prior. That is correct only when no other evidence
is entered. With event evidence present the posterior becomes f_q(j) *
p1(j)/p0(j), and on Zhang's per-flight priors — p0(none) = 1.000, p0(fatal) =
1.4e-7 — event evidence moves the rare states by 3 to 5 orders of magnitude
while barely touching `none`, so that factor annihilates the majority state.
Measured directly: an accident where retrieval said none = 0.831 and the
network given events said none = 0.989 came out of the fused arm with none =
0.000. Across the cohort `bn-fused` recalled `none` in 1 of 173 injury cases
and 1 of 126 damage cases, scoring 38.5% / 41.9% against a 58.4% / 42.6% prior.

*The fix.* `qb.jeffrey_likelihood` takes the ratio against the marginal that
holds at inference time, L(j) = f_q(j)/p1(j) with p1 = P(node | event
evidence). Jeffrey conditioning asserts a marginal; realised as a Pearl
likelihood vector the denominator must be the current marginal, not the
unconditional one. Verified to reproduce the target distribution exactly.
`bn-fused` and `bn-fused-t` now read 90.9% / 77.4%. Every other number in
`heldout_significance.md` is byte-identical — 58.4 / 42.6 prior, 90.9 / 77.4
severity — so nothing else in the paper moves.

*The finding that replaces the broken number.* Corrected, `bn-fused` equals
`bn-sev` exactly (McNemar 0 / 0 on all 296). That is not a coincidence: once
you ASSERT a node's marginal, no other evidence can speak to that node, so
Jeffrey conditioning can never be a fusion. Genuine fusion needs f_q as a
likelihood rather than an assertion, and at the right scale — f_q counts
ACCIDENTS while p0 is per-FLIGHT, so the two denominators must not be mixed.
Scored with the per-accident base rate q0 (`tests/severity_fusion_fix.py`):
injury 90.5% vs retrieval's 90.9% (p = 1.000, a tie), damage 70.9% vs 77.4%
(p = 0.0013, significantly worse). So: **event evidence adds nothing to the
retrieval severity signal under any correct combination rule.** State it that
way. It is a clean negative result with three rules tested, and it is far
stronger than the broken arm it replaces, which proved nothing.

*Still true and unrelated:* event evidence degrades the fire query (0.801 to
0.619 even after the fix). That is not the fusion bug — event evidence is
genuinely anti-correlated with fire, at AUC 0.380 on its own, largely a
leak-guard artifact. Two separate problems that happened to look alike.

**BII-10 — This is the motivation for Phase 2.** The cross-inference capacity
of the network is a property of its topology, and the topology came from one
set of construction choices applied once. That is exactly the structural
uncertainty question the next project proposes to measure. Use BII-8 as the
empirical hook rather than motivating Phase 2 abstractly.

**BII-4 — It does not answer the supervised baselines.**
TF-IDF never touches the network, so none of this bears on it. What changes is
the framing: TF-IDF returns a label, the pipeline returns a label plus a
posterior over 783 event nodes in the taxonomy the field reasons in. Keep the
two arguments separate; do not let BII-1 look like a rebuttal to A5.

**BII-11 — 2026-09-09. The head-to-head at scale: n = 500, not n = 33.**

Generated by `tests/coded_vs_narrative_window.py`; tables in
`outputs/coded_vs_narrative_window.md`. BII-2 and BII-7 rest entirely on the 33
accidents of 2007, and that is the paper's most-quoted claim. The build window
itself supports the same comparison at forty times the sample: of its 1,742
accidents, **1,281** carry both a factual narrative (>= 100 characters) and
coded labels that are network nodes. 500 were sampled (`random.seed(0)`) and
scored through the same frozen upgraded network, leave-one-out — the query
`ev_id` excluded from its own retrieval pool, verified absent in 500 of 500
checks. The guard is not a formality: probed with it off, the query was its own
**#1** neighbour in 25 of 25 accidents.

| Arm | Evidence enters at | Injury | Damage |
|---|---|---|---|
| `prior` | — | 67.4% | 39.0% |
| `coded` (own coded record, hard) | event nodes | 77.2% | 56.6% |
| `narr-events` (parsed narrative) | event nodes | 79.2% | 46.0% |
| `narr-sev` (k-NN severity) | severity nodes | 80.8% | 66.6% |

Exact McNemar, two-sided, `coded` against each narrative arm:

| Comparison | Target | coded only right | narrative only right | p |
|---|---|---|---|---|
| `coded` vs `narr-events` | Injury | 15 | 25 | 0.154 |
| `coded` vs `narr-sev` | Injury | 6 | 24 | 0.0014 * |
| `coded` vs `narr-events` | Damage | 106 | 53 | 3.2e-05 * |
| `coded` vs `narr-sev` | Damage | 42 | 92 | 1.9e-05 * |

*The caveat, and it must travel with every one of these numbers.* The network's
structure and its CPTs — including the injury and damage CPTs — were estimated
from these same 1982-2006 accidents. The `coded` arm therefore enters labels
the network was fitted to and is graded on the outcomes those labels were
fitted against. It is scored in-sample with no correction, so it is an **upper
bound on the coded route, not a peer**. The narrative arms carry no such
advantage: redacted text, and a retrieval pool with the query removed. The
handicap runs one way only. A narrative arm that ties here has done better than
tie; a narrative arm that wins here wins by at least the margin shown. Never
write `coded` as a baseline.

*What it settles.* BII-2's tie on injury holds at n = 500 (p = 0.154) and
BII-7's damage deficit holds and is now decisive (p = 3.2e-05) rather than
marginal. The 2007 result was not a small-sample accident. Replace "n = 33, one
year, p just under 0.05" in BII-7 with this; keep 2007 as the out-of-sample
corroboration and lead with the window.

*What it adds.* The deployed path, `narr-sev`, clears the inflated coded
ceiling on both targets. And the split diagnoses itself: the same narratives
that lose on damage through the parser (46.0%) win on damage through retrieval
(66.6%). The damage information is in the free text; the phrase-matching
interface is what fails to reach it. That is an engineering deficit, not a
limit on what narratives carry, and it is the concrete next step BII-7 asked
for.

*What it does not license.* Not "narratives beat coded fields." `coded` is
in-sample and `narr-sev` bypasses the graph, so this is not like-for-like — the
defensible sentence is that the narrative route clears an inflated coded
ceiling. And it says nothing about post-2006 accidents, where per BII-1 the
coded route has no inputs and there is no comparison to run.

---

## C. How to deliver this in the meeting

**C1 — He is the co-author.** The paper is Zhang & Mahadevan, RESS 209, 2021.
Do not say "Zhang's paper" all meeting; it is his too. Every finding below is
about work he put his name on.

**C2 — Convert findings into questions.** Same content, opposite reception.
Ask whether the n=1 and n=2 edges were chosen as illustrative scenarios or
should be treated as low-confidence. Ask whether the closer-to-empirical result
means something was fixed or something was deviated from. Ask whether the
published Table 8 came from a different build than the released `.xdsl`.

**C3 — Ask about the per-flight denominator directly.** It is his modeling
choice, he is the best person alive to explain it, and the answer may dissolve
B6 entirely. Ask before it goes in the paper, not after.

**C4 — Lead with the self-audit, not the presentation.** Open with the
corrections in section A. That is the one thing in the meeting that cannot be
generated rather than found.

**C5 — Have the "what is the BN buying us" answer loaded.** SUPERSEDED on
2026-09-09; the old version of this note cited the fire AUC of 0.39-0.46 as
evidence against the network and that reading no longer holds (BII-8). The
current answer is two-sided and better. The network does real work: coded event
evidence lifts injury 63.6% to 81.8% and damage 42.4% to 66.7% (BII-5), and
opening the severity collider ranks fire at AUC 0.80 against a 0.500 prior
(BII-8). It also does not beat the simpler alternatives: retrieval ranks fire
at 0.96, and the headline severity path bypasses the graph entirely (BII-6).
So: the graph carries signal, its reach is bounded by a topology that is thin
above the occurrence layer, and on these particular queries simpler methods
win. That is a finding, not a concession — and it is the empirical hook for
Phase 2 (BII-10). Say it before he assembles the pattern himself.

**C6 — Say you regenerated everything today.** Two headline artifacts were
stale relative to their code before this. They are not now.

---

## D. Code work, in priority order

**D1 — Evidence fusion defect (BII-9). The only true bug.**
Combining event evidence with severity virtual evidence degrades both targets.
Fire falls 0.801 -> 0.610 AUC; `bn-fused` scores 38.5% / 41.9% against a
58.4% / 42.6% prior. Start in `frozenbn_heldout_narrative_bn_eval.py` where the
fused arm is built and in `qb.apply_evidence`: a likelihood ratio on a severity
node and hard evidence on event nodes are being multiplied into one inference
whose normalisation is almost certainly wrong. The per-target inference used by
`posteriors()` exists precisely because entering both severity vectors at once
distorts them — the same reasoning was never applied to the event+severity
combination. Fix, then re-run both the severity eval and
`fire_cross_inference_diagnosis.py`.

**D2 — TESTED AND REFUTED 2026-09-09. Frequency-selected parents beat
information-selected ones, and the reason is worth a paragraph.**

The complaint was sound on its face: `bn_upgraded.py` gives injury and damage
the SAME twelve parents from `support.most_common()[:12]`, and mixes them by
that same frequency. Frequency is not informativeness. So each target's parents
were re-selected by mutual information against that target, computed on the
1982-2006 window only, and the held-out cohort re-scored
(`tests/severity_parent_selection_ablation.py`, results in
`outputs/severity_parent_ablation.md`).

| scheme | arm | injury | damage |
|---|---|---|---|
| A, shipped frequency parents | soft-priority | 90.2% | **55.7%** |
| A, shipped frequency parents | soft-only | 89.2% | 55.4% |
| B, mutual-information parents | soft-priority | 90.2% | **52.7%** |
| B, mutual-information parents | soft-only | 89.2% | 55.4% |

Exact McNemar, B against A: injury 0/0, p = 1.000 — the two networks make
**identical** injury predictions on all 296. Damage under soft-priority: A right
on 10 that B misses, B right on 1 that A misses, **p = 0.0117**. The change is
significantly WORSE.

The explanation is the useful part. MI selection hands damage labels like
`nose gear collapsed` and `abrupt maneuver` — genuinely more informative about
damage, and almost never observed by the pipeline. D3 measured recall on the
FREQUENCY-selected parents at 63.2%; the MI set is chosen for association
without regard to whether a narrative can be made to trigger it. So the node
gets better parents and less evidence, and loses.

Write it this way: **parent selection cannot be decoupled from observability.**
Frequency-based selection looks naive but is accidentally well-matched to a
pipeline whose evidence arrives from retrieval over common labels. That is a
real finding about building BNs meant to be driven by text, and it belongs in
the discussion rather than being buried as a failed ablation.

Consequence for the paper's argument: combined with D3, two independent attempts
to close the damage gap through the graph have now failed — better extraction
tops out at 66.7% (BII-7), better parents make it worse. The gap between 55.7%
through the graph and 77.4% for the bypass is not an evidence problem and not a
parent-selection problem. Say that plainly; it is a stronger and more honest
claim than promising future work that this ablation already refutes.

Scheme C (MI parents plus MI mixture weights) was dropped: with injury and
damage on disjoint parent sets the moralized graph produces a junction-tree
clique that exhausts memory, and the run had to be killed. That is itself worth
one line — the shipped design's shared parent set is what keeps exact inference
tractable here.

*Harness check:* scheme A reproduces the main eval exactly (soft-priority
90.2% / 55.7%, soft-only 89.2% / 55.4%), so the ablation is scoring the same
thing the headline does.

**D3 — REVISED 2026-09-09. The parser is NOT the damage bottleneck, and the
soft-evidence mechanism now has its empirical justification.**
Measured on 300 window accidents with narratives and human codes, leave-one-out
retrieval, scored on the twelve parents of the damage node:

| damage parent | in human codes | phrase match | + retrieval |
|---|---|---|---|
| airframe/component/system failure/malfunction | 94 | 0.0% | 66.0% |
| in flight encounter with weather | 56 | 1.8% | 91.1% |
| miscellaneous/other | 51 | 0.0% | 25.5% |
| on ground/water collision with object | 33 | 0.0% | 63.6% |
| fire | 30 | 90.0% | 100.0% |
| in flight collision with terrain/water | 17 | 0.0% | 29.4% |
| loss of engine power (total) - mechanical | 15 | 0.0% | 100.0% |
| near collision between aircraft | 13 | 0.0% | 84.6% |
| **overall** | **353** | **9.3%** | **63.2%** |
| precision | | 57.9% | 71.8% |

Two things to write into the paper.

*(i) This is the motivation for soft evidence, stated empirically for the first
time.* Deterministic phrase matching recovers 9.3% of damage-relevant labels
because it can only fire on labels that are ordinary English — `fire` at 90%,
`hard landing` at 46%, and zero for every taxonomy category name. No
investigator writes "airframe/component/system failure/malfunction"; they write
"the left main gear collapsed." Retrieval never needs the label text to appear,
so it recovers 63.2% — and precision RISES, 57.9% to 71.8%. Stop asserting that
retrieval adds coverage and give this table.

*(ii) The parser has a ceiling below the bypass, so stop treating it as the
fix.* The chain: 9.3% recall gives unusable damage; 63.2% recall gives 50.7%;
100% recall (human coding, BII-7) gives 66.7%; skipping the graph gives 77.4%.
Perfect extraction still loses to the bypass by ~11 points. The binding
constraint is the damage node, not the reader. The earlier version of this note
called parser work the highest-value fix; that was wrong.

UPDATE: D2 was then tested and also refuted — re-selecting the damage node's
parents by mutual information makes damage significantly WORSE (55.7% to 52.7%,
p = 0.0117), because informative labels are ones the pipeline rarely observes.
So neither better extraction nor better parents closes the damage gap. Report
the two together as a bounded negative result rather than as future work.

**D4 — Merge the LOEP node family (B2).**
Four nodes mean "engine lost power"; only one is a severity parent, which is
why two of the twelve cells never move.
`prognosis.resolve_outcome_targets` already does this expansion elsewhere.
Do NOT do this before the meeting — it moves every regenerated number.

**D5 — The model never predicts fatal or minor.**
Structural, and the coded-evidence arm has the same blind spot, so it is not
the parser's fault. k = 100 neighbours almost certainly washes out classes with
3 and 16 instances. Worth trying a smaller k or a sharper similarity
temperature for the minority classes, but treat it as a limitation to report
rather than a defect to hide.

**D7 — WITHDRAWN 2026-09-11 by D13. Do not implement. Cap the soft-evidence likelihood ratio, then re-score (see A12).**

> Superseded. The cap was tested against the real cohort and made the targeting
> worse, not better (mean |posterior − c| 0.269 → 0.458). A12 was closed
> instead by taking the likelihood ratio against the conditional prior
> p1 = P(node | hard evidence). See D13 for the measurements. The reasoning
> below is kept only to show why the cap looked right at the time.
45% of soft facts saturate above 0.999 because the ratio needed to lift a
per-flight prior to c is ~1e4, and several of those interacting saturate each
other. Cheap test: bound the ratio in `apply_evidence` (LR <= 100 is a
reasonable first cut, chosen on the internal split, never on the held-out set)
and re-score `soft-only`, `hard+soft` and `soft-priority`. If accuracy holds or
improves, the semantics get closer to what the paper claims at no cost and A12
can keep the stronger wording. If accuracy drops, the saturation is doing the
work and that itself is worth reporting honestly. Either outcome is publishable;
the current state — a documented claim the code does not deliver — is not.

**D8 — The LLM arms were never affected by the BII-9 defect.**
`llm-tier` and `llm-first` enter event evidence only, and the `+stated` arms
use `severity_virtual_evidence`, which is a MEASURED confusion likelihood
P(stated | true) rather than the f_q/p0 ratio. No re-run is needed and none of
the LLM numbers move. For the record they remain well behind: `llm-tier` scores
68.2% injury and 51.7% damage against retrieval's 90.9% / 77.4%, on a tier
split of 176 deterministic / 120 LLM. The "LLMs did not beat the pipeline"
conclusion stands and was never in doubt from the bug.

**D9 — RESOLVED 2026-09-09. Guard closed, headline re-scored, nothing moved.**

The test set was scored ONCE after the guard change. Result:

| predictor | injury before -> after | damage before -> after |
|---|---|---|
| prior | 58.4% -> 58.4% | 42.6% -> 42.6% |
| hard | 64.2% -> 64.2% | 44.6% -> 44.6% |
| hard+soft | 82.4% -> 82.8% | 50.7% -> 51.0% |
| soft-only | 89.2% -> 89.2% | 55.1% -> 55.4% |
| soft-priority | 89.9% -> 90.2% | 55.4% -> 55.7% |
| **retrieval-sev / bn-sev / bn-fused** | **90.9% -> 90.9%** | **77.4% -> 77.4%** |

The headline is bit-identical. Nothing else moves by more than 0.3 points, which
is a single accident out of 296, and the movement is upward. `severity_stated`
is now 0 of 296 on both targets, where the guard previously let stated phrases
through.

This is the strongest possible outcome and it should be written into the paper
as an experiment, not buried as a methods detail. The objection "your narratives
still say who got hurt" is now answered with a measurement rather than an
assurance: the leaked wording was worth 12 points to a keyword-only classifier
(70.3% against a 58.4% floor, A13) and worth **zero** to the pipeline. That
separates the two claims cleanly — the residual text did carry signal, and the
method was not using it.

Note the `bn-fused` row in `git HEAD` still shows 38.5% / 41.9%; that jump to
90.9% / 77.4% is the BII-9 fusion fix from earlier the same day, not the leak
guard. Do not attribute it here.

*Guard construction and verification follow.*
Added to `_REDACT_ONLY` in `query_to_bn.py`: every `injur*` token, degree-burns
and burns-to-persons phrasings, and on-scene medical response (`paramedic`,
`ambulance`, `medical personnel`, `first responders`). Plain `burned`/`burning`
is deliberately kept — all surviving mentions are aircraft fire damage ("burned
through the shroud", "fire burned thru the cabin floor"), and `fire` is a
legitimate event node.

Patterns were written from the vocabulary and verified on the 1,286 BUILD-WINDOW
narratives, never tuned against held-out accuracy. Window verification:
`injur` 248 to 0, `paramedic`/`ambulance`/`medical personnel` to 0, `fatal` to
0, with 98.5% of narrative text retained so event description survives intact.

Held-out verification: injury survivors **0 of 296** (was 77), and the
keyword-only classifier falls to **58.4%**, exactly the majority-class floor —
zero residual information. Report this as the audit, it is the strongest form
of the claim.

*Original diagnosis follows.*
This is now the most important open code item: it is the only one that bears on
whether 90.9% is the true number. Extend `redact_severity_phrases` to remove
bare person-outcome statements (`was injured`, `sustained injuries`, `suffered
burns`, `second-degree burns`, and the like) while leaving event description
intact — "the passenger fell to the floor" stays, "and was injured" goes. Then
re-run `frozenbn_heldout_narrative_bn_eval.py` and report the delta.

Note the retrieval arm is the one at risk, not the phrase parser: retrieval
embeds the whole redacted narrative, so a surviving "was injured" is inside the
vector that selects neighbours. That is precisely the path producing 90.9%.

Do NOT tune the new patterns against held-out accuracy. Write the patterns from
the vocabulary, verify on the window narratives, then score the test set once.

**D10 — LLM as semantic label mapper CLOSES the event-node damage gap (2026-09-10).**
`tests/llm_label_mapper.py`, gpt-4o-mini, n=300 build-window accidents,
outputs in `outputs/llm_label_mapper.md` and `..._results.json`.

Scope is deliberately narrow: the LLM is given the 12 severity-parent labels
verbatim and asked which apply. It is NOT asked to parse the whole network.
This is the distinction from the earlier llm-tier arm (68.2%/51.7%), which
tried to replace the entire parser across all 783 nodes and lost.

Label level, against human-coded labels on the same 12 nodes:

| method | precision | recall | F1 |
|---|---|---|---|
| phrase parser | 46.4% | 8.9% | 14.9% |
| LLM mapper | 45.4% | **74.7%** | 56.5% |

Recall is 8.4x higher at unchanged precision. The label that was driving the
damage gap — `airframe/component/system failure/malfunction`, 75 of 300
accidents, the most frequent parent — goes from unreachable to **93.3%**
recall (70/75). Weakest: `miscellaneous/other` 23.1% (12/52) and
`on ground/water collision with terrain/water` 20.0% (2/10).

End to end through the BN event nodes, same 300 accidents:

| arm | injury | damage |
|---|---|---|
| prior | 62.3% | 37.3% |
| phrase parser | 64.0% | 40.3% |
| **LLM mapper** | **72.7%** | **61.7%** |
| human coded | 72.0% | 59.7% |

vs phrase parser: injury +8.7 pp (26 accidents), damage +21.3 pp (64
accidents) — both large. vs human coded: injury +0.7 pp (2 accidents),
damage +2.0 pp (6 accidents) — **noise, do not claim a win**. The defensible
claim is *statistically indistinguishable from human coding*, which is the
thesis anyway.

This supersedes the D3 framing that damage was structurally unreachable
through the event-node interface. It was reachable; the phrase parser could
not name the labels.

**HELD-OUT RUN COMPLETE (2026-09-10).** `--heldout`, same 296 cohort as
`frozenbn_heldout_narrative_bn_eval.py`, outputs in
`outputs/llm_label_mapper_heldout.md`.

| arm | injury | damage |
|---|---|---|
| prior | 58.4% | 42.6% |
| phrase parser | 62.5% | 45.9% |
| **LLM mapper** | **83.4%** | **66.9%** |
| human coded | *not computable* | *not computable* |

+20.9 pp on both against the phrase parser (62 accidents each). The prior row
reproduces the documented 58.4%/42.6% majority floors exactly, confirming the
cohort. Two things to state carefully:

- The coded arm is **unavailable, not weak**. Verified 2026-09-10: **0 of 296**
  held-out accidents carry any label matching the 12 severity parents, against
  232 of 300 in the window. The CICTT recoding is total. So on post-2007
  accidents the LLM mapper is the *only* working route into the event-node
  interface — the coded route does not exist and the phrase parser sits 4 pp
  above the prior. This is a stronger framing than "closes the damage gap."
- 83.4%/66.9% remains **below** the 90.9%/77.4% headline, which comes from the
  severity bypass. The LLM repairs the event-node route; it does not beat the
  bypass. Do not present these as competing with the headline.

**Open before this can go in the paper:**
1. ~~Run on the 296 held-out accidents.~~ DONE, see above.
2. Disclose the memorization risk. gpt-4o-mini was trained on public web text
   that plausibly includes these NTSB reports. Unlike the phrase parser and
   retrieval, this is not auditable. A held-out-only run does not fix it,
   because the held-out accidents are equally public.
3. Note that 45% precision means over half of emitted labels are wrong; the
   arm works because the mixture CPT tolerates extra parents. Test whether
   filtering low-confidence labels helps or hurts.
4. State the reproducibility cost: every inference now needs an API call,
   which breaks the "frozen, no external dependency" property.

**D11 — The BN value demos are weaker than they look. Do not present Demo 1
or 3 as causal reasoning (2026-09-10).**
`tests/bn_value_demos.py` runs three demonstrations (counterfactual, joint,
evidence attribution) intended to show what the network adds beyond retrieval.
Demo 1 reports that removing `fire` from {fire, engine power loss, terrain
collision} *raises* P(fatal) from 0.169 to 0.293.

That is arithmetic, not causality. `bn_upgraded.build_upgraded` builds the
severity CPT for k active parents as a support-weighted average of those
parents' empirical marginal severity distributions:

    dist = (dists[active] * w[:, None]).sum(axis=0) / w.sum()

So adding any parent whose marginal is milder than the current mixture
mechanically pulls the posterior milder, and removing it pulls it harsher.
The "counterfactual" measures whether fire's marginal severity distribution
is above or below the running average. It is not fire's causal contribution.
Demo 3's attribution walk has the same defect.

Demo 2 is honest but thin: the injury/damage dependence ratio is 1.024x,
i.e. very close to independent, which is expected because both nodes are
mixtures over the *same* 12 parents with the same construction.

Maha co-authored the CPT recipe and will recognize this immediately.
Presenting these as causal counterfactuals is the single largest own-goal
risk currently in the project.

What the network can still legitimately be claimed to add: a full calibrated
distribution over all four states rather than a vote, a coherent joint over
injury and damage, and the ability to condition on arbitrary evidence
patterns including ones absent from the corpus. Those are real. "Isolating a
variable's causal effect" is not, under this CPT.

**D12 — Replacement argument for "what does the BN add", measured
(2026-09-10).** `tests/bn_structural_value.py`, outputs in
`outputs/bn_structural_value.json`. These claims do NOT depend on the severity
CPT behaving causally, so they survive the D11 objection. Use these instead of
the counterfactual/attribution demos.

**Claim 1 — compositional coverage. DEFENSIBLE, strongest of the three.**

| | |
|---|---|
| severity-parent pattern space | 2^12 = 4,096 |
| distinct patterns observed in 1,742 accidents | **59 (1.44%)** |
| unobserved | 4,037 (98.56%) |
| most parents ever co-occurring | **3** — never 4+ |
| accidents with <=1 active parent | 1,541 (88.5%) |
| observed patterns resting on one accident | 20 of 59 (34%) |

By pattern size: 1-active 12/12 observed, 2-active 33/66 (50%), 3-active
13/220 (5.9%), **4-or-more 0 of 3,797**. Retrieval answers by copying from
accidents that exist; for any 4+-parent pattern there is nothing to copy.

**State this caveat or a reviewer will supply it:** the network's answer for an
unobserved pattern is a support-weighted mixture of the active parents'
marginals. It is coherent and defined, but its *correctness* on unobserved
patterns is untestable — there is no ground truth there. Claim "a principled
answer where retrieval has no support," not "the right answer."

**Claim 2 — joint over injury and damage. REAL BUT MODEST, one sentence only.**
The two severity nodes share all 12 parents and have no edge between them, so
they are conditionally independent given the full parent set; residual
dependence comes only from uncertainty over unobserved parents. Measured over
the 24 most common patterns: max total variation from independence **0.039**,
mean 0.0089, max joint/product ratio 89.9x. The large ratio sits on tiny
probability mass — quote the total variation, not the ratio. Dependence is
largest for single-parent patterns, where 11 parents remain uncertain.

**Claim 3 — taxonomy stability. Strongest capability claim, see D10 held-out.**
0 of 296 post-2007 accidents carry a label in the network's vocabulary. The
frozen network plus the LLM mapper is the only route that spans the 2007 CICTT
recoding; the coded route does not exist on modern accidents. Purely a
capability argument, independent of the CPT.

None of the three claim better accuracy. All three claim something retrieval
structurally cannot do. That is the defensible position.

**D13 — 2026-09-11. A12 closed. The D7 cap was the wrong fix; the right one is free.**

Generated by `tests/a12_soft_evidence_audit.py`; results in
`outputs/a12_soft_evidence_audit.json`. A12 said the documented soft-evidence
mechanism did not match the implemented one. Confirmed, and quantified on the
held-out cohort (253 accidents, 599 soft facts — the run is OOM-killed near
accident 280 by a single large-clique case, so the audit is 85% of the cohort,
not all of it; say 253 in the paper, not 296).

The cause is the reference prior. `apply_evidence` built the likelihood ratio
against the node's UNCONDITIONAL prior p0, which is exact only when the soft
fact is the sole evidence. With hard event evidence also entered, the prior
that actually holds is p1 = P(node | hard evidence), and p1 >> p0 on
per-flight rates, so the LR overshoots — the observed maximum was 4.2e7.

| mechanism | saturated | mean &#124;posterior − c&#124; | median | max LR |
|---|---|---|---|---|
| unconditional p0 (shipped before 2026-09-11) | 29.7% | 0.269 | 0.054 | 4.2e7 |
| D7 likelihood-ratio cap at 100 | 3.0% | **0.458** | **0.405** | 100 |
| conditional p1 (now shipped) | 3.2% | **0.034** | ~1e-16 | 2.5e7 |

"Saturated" = a fact entered below c = 0.5 that ends above 0.999 posterior.

Two findings, and the first one reverses a standing plan:

**The D7 cap must not be used.** It suppresses saturation but destroys the
targeting it was supposed to protect — mean error more than doubles (0.269 to
0.458) and the median goes to 0.405, meaning the typical soft fact no longer
lands anywhere near its stated confidence. A cap treats the symptom. Do not
implement D7; cite this table as the reason.

**The correct fix is to take the LR against p1**, exactly the correction
already applied to the severity nodes in `jeffrey_likelihood`. Median error
falls to machine epsilon: soft facts now land *precisely* at their stated
confidence, which is what Jeffrey conditioning claims and what the paper
describes. Implemented in `code/query_to_bn.py` (`_node_priors_given`,
`SOFT_EVIDENCE_REFERENCE`, default `conditional`; set the env var to
`unconditional` to reproduce the old behaviour). Residual 3.2% saturation is
soft-fact-on-soft-fact interaction, since p1 conditions on hard evidence only.

**The headline numbers do not move.** Re-ran
`tests/frozenbn_heldout_narrative_bn_eval.py` on all 296. `narrative-evidence`
changed **0 of 296** injury predictions and **0 of 296** damage predictions;
90.9% / 77.4% and Brier 0.1756 / 0.3744 are identical before and after. The
reason is structural and worth one sentence in the paper: the headline
predictor reaches severity through the k-NN bypass, so graph-routed soft
evidence never touches it. Only the ablation rows move — `soft-priority`
injury 90.2 → 85.8 and damage 55.7 → 45.9, `hard+soft` 82.8 → 79.4 and
51.0 → 47.0. Those rows are now *correct* where they were previously flattered
by evidence that had silently hardened; regenerate any table containing them.

Net: the method description and the code now agree, and the claim the paper
rests on cost nothing to make honest. State the fix, the 0/296, and the table.

**D14 — 2026-09-11. Temporal split. The 2020–2024 confirmatory run is impossible.**

Generated by `tests/heldout_temporal_split_and_ci.py`; results in
`outputs/heldout_temporal_split_and_ci.{md,json}`. The notes carried a
commitment to a fresh 2020–2024 set. The corpus ends in **2018** (2017: 12,
2018: 1), so that set does not exist and the commitment must be struck.

What replaces it is a split inside the held-out window. Both halves are scored
by the same frozen 1982–2006 CPTs, and the late half sits 7–12 years further
from the build window, so decay would show as an early-to-late drop.

| cohort | n | injury acc (95% CI) | damage acc (95% CI) |
|---|---|---|---|
| 2007–2012 | 206 | 90.8% [0.860, 0.940] | 76.7% [0.705, 0.820] |
| 2013–2018 | 90 | 91.1% [0.834, 0.954] | 78.9% [0.694, 0.860] |
| all | 296 | 90.9% [0.871, 0.937] | 77.4% [0.723, 0.818] |

No drift: +0.3 pp injury, +2.2 pp damage, both intervals overlapping. This is
a genuine result — the frozen structure does not decay across a 12-year
horizon — but be exact about its limits. It is **not** a confirmatory test.
Every one of these 296 was seen during development, and the late half is only
90 accidents. Claim stability under temporal distance; do not claim a fresh
held-out validation, because one is not available in this corpus.

**D15 — 2026-09-11. A5 closed, and it cuts both ways.**

Same script. 10,000 paired bootstrap resamples against `tfidf-lr`, n = 296.

| task | metric | ours | tfidf-lr | diff | 95% CI | one-sided p |
|---|---|---|---|---|---|---|
| injury | top-1 acc | 0.909 | 0.922 | −0.014 | — | tie (McNemar 0.289) |
| injury | Macro-F1 | 0.470 | 0.603 | **−0.133** | [−0.260, +0.000] | **0.026** |
| damage | top-1 acc | 0.774 | 0.733 | +0.041 | — | — |
| damage | Macro-F1 | 0.697 | 0.621 | +0.076 | [−0.166, +0.296] | 0.297 |

A5 asked for an interval on the injury Macro-F1 gap. The gap is **real**:
97.4% of resamples favour tfidf-lr, p = 0.026. Note the percentile interval
touches zero at its upper bound purely because Macro-F1 is discrete on 296
items — read the tail mass, not the boundary. Report the deficit outright. The
explanation is not a hedge: a discriminative model fit on this distribution
*should* beat a frozen generative one on rare-class recall, and that gap is the
price of the freeze the paper is defending.

The unplanned finding is the other row. A5 described damage as "a clean lead on
both" metrics. On Macro-F1 that is **not supported** — CI [−0.166, +0.296],
only 70% of resamples favour us, p = 0.297. Soften every damage Macro-F1
claim to top-1 accuracy only. Better to cut this myself than have Maha find a
+0.076 lead reported inside a ±0.23 interval.

### Audits that came back CLEAN (2026-09-09) — state these in the paper

- **No test-set contamination of the retrieval index.** All 1,703 vectors in
  `embeddings_1982_2006.npy` are build-window accidents; zero held-out ev_ids
  appear. The A9 count discrepancy is index staleness, not leakage.
- **Cohorts are disjoint.** Window (1,742) and held-out (296) share no ev_id.
- **Graded severity phrases are fully removed** by the guard (A13 table).
- **No hyperparameter ever saw the test set** (A11).

**D6 — Optional: persist the 2007 decode.**
Currently done in memory inside `coded_evidence_baseline.py` by design. Only
write it into `refined_dataset.json` if the diagnosis cohort is ever grown to
284, which A10 argues against.

### Still open, not code

- Explain the uniform 5.3x overshoot (B4). Currently unattributed.
- Resolve the per-flight vs per-accident scale question (B6, C3).
- ~~Attach a CI or drop the comparison on injury Macro-F1 (A5).~~ CLOSED by
  D15 — the deficit is real (p = 0.026); report it. D15 also found the damage
  Macro-F1 "lead" is not separable and must be softened.
- ~~Fix or re-describe the soft-evidence mechanism (A12).~~ CLOSED by D13 —
  fixed, not re-described. Headline numbers unchanged (0 of 296 predictions
  moved). The D7 cap is withdrawn; do not implement it.
- ~~Run a fresh 2020–2024 confirmatory set.~~ WITHDRAWN by D14 — the corpus
  ends in 2018. Replaced by the within-window temporal split, which shows no
  drift but is not a confirmatory test.
