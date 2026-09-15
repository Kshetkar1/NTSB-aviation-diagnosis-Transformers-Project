# Draft 3 section-by-section audit — 2026-09-07

Source: `Desktop/NTSB Paper Process/Draft 3- My edits for 8:25:26.docx`
(the version Maha is grading). Every number below was regenerated today.

Severity key: **[BN]** = a sentence a Bayesian-network expert reads as a
misunderstanding. He told you on 9/1 that your BN understanding is weak — these
are the sentences that will confirm it. **[NUM]** = a number that no longer
matches the repo. **[GAP]** = something a reviewer needs that isn't there.

---

## Abstract

**A-1 [GAP]** Never says how similarity is computed. Add the encoder, the metric,
and the truncation in one clause.

**A-2** Consider adding one hedge. You report 90.9 / 77.4 / 84.2 with no mention
that a TF-IDF baseline scores higher on injury. It's in §8.1, so the abstract
isn't lying, but leading with only your own numbers invites "did you compare?"

---

## §1 Introduction

**1-1** "We extended the network, so the reproduction runs and matches Zhang and
Mahadevan's published numbers." Overclaims — 12 of 93 still differ. Rewrite to
the stronger and truer version: the extensions are *what make the published
severity rows reproducible at all*. See §5.7 below.

---

## §2.3 Hard, Soft and Virtual Evidence

**2-1 [BN]** The section names three evidence types but never defines what
virtual evidence *is*. This is a core Pearl concept and its absence is exactly
the gap he flagged. Define it: a likelihood vector attached to a node, not a
probability — it multiplies into the node's existing belief rather than
replacing it. Cite Pearl [16], which you already have.

**2-2** "enter neighbor fractions on event nodes" — not fractions. See 6-2.

---

## §3 Data and Temporal Split

**3-1 [NUM]** Table 1 says 1,288 narratives for training. That's the count of
*non-empty* narratives. The held-out cohort uses a >= 100-character filter, and
at that threshold the training count is 1,286. State which filter each row uses,
or the reader can't reconcile 1,288 against a 296 cohort built on a different rule.

---

## §4.1 Bayesian Network background

**4-1 [BN]** "These edges also have arrowheads that point to nodes (x1 -> x2 is
different than x3 -> x4)." The contrast is garbled. The point is that
x1 -> x2 differs from **x2 -> x1**.

**4-2 [BN]** "Root nodes are the prior probabilities." Root nodes *have* priors.
Small wording, but it's the kind of slip that accumulates.

**4-3 [BN][GAP]** **Conditional independence and d-separation appear nowhere in
the paper.** This is the single most important concept in Bayesian networks and
the reason the graph buys you anything over a joint table. Its absence is
probably a large part of why he said the understanding is weak. Add a short
paragraph: the graph encodes which variables are independent given which others,
which is what makes the joint factorize into the CPTs, which is what makes
inference tractable.

**4-4 [GAP]** Never says why a network instead of one big joint distribution.
One sentence on parameter savings from factorization would fix it.

---

## §4.2 Three-node example

**4-5 [BN]** "Setting fire = yes also updates brake wear ... which is higher than
the prior probability alone." **Compute it.** With your own Tables 2 and 3:
P(fire=yes) = 0.10(0.80) + 0.90(0.05) = 0.125, so
P(brake wear=yes | fire=yes) = 0.08 / 0.125 = **0.64**, up from the 0.10 prior.
Putting the number in shows you can do the inference; asserting "higher" shows
you know it moves. He will notice which one you did.

**4-6 [BN]** "P(injury = serious | fire = yes) = 0.40 ... equals the posterior on
injury." True, but say *why*: injury is conditionally independent of brake wear
given fire, so the chain blocks and no other evidence reaches injury. That single
clause is a d-separation demonstration on your own example and it directly
answers the weakness he named.

**4-7** "(x1 -> x2)" labels a three-node chain. Use x1 -> x2 -> x3.

---

## §5.2 Graph construction

**5-1** The three edge rules are correct **for the faithful rebuild only**. §5.7
replaces rule 3. Say so here, or the methods contradict each other.

---

## §5.3 Prior probabilities

**5-2** "This equation takes and divides by the Statistics (BTS) departure from
the same training period" — sentence is broken, rewrite.

**5-3 [GAP]** State plainly that this is a **per-flight** scale, and that it
makes root priors ~1e-7. It's the reason every downstream severity posterior is
tiny, and naming it here pre-empts the question rather than leaving it to be
discovered in §9.

---

## §5.4 Parent Selection — HIGHEST PRIORITY

**5-4 [BN]** "When two candidate parents have the same strength, the tie is
broken at random." **Your code does not do this.** It sorts by edge ratio
descending, then edge support descending, then node name — fully deterministic,
which is why no seed is set. Random jitter is in the *original* construction
code, not yours. Source: `tests/bn_build_ours.py` lines 94-102. This error also
appears in §10.1 and it is load-bearing there.

**5-5** "The estimated ratios get capped at 0.95." Only when the raw ratio is
exactly 1.0 — it is not a general ceiling. Source: `bn_build_ours.py` line 92.

---

## §5.5 Beta-CDF smoothing

**5-6** "if no parent is there, the conditional probability returns 0" →
"when **no parent is active**." "If parent = 1" → "when **exactly one parent is
active**." As written they read as graph properties rather than CPT rows.

**5-7** Worth one sentence: 12 binary parents means 4,096 parent-state rows per
child, which is why the smoothing is needed rather than counting each cell.

---

## §5.6 Table 7

**5-8** Say explicitly that Table 7 is its **own** scoreboard, 85/85, and is not
among the 93 posterior quantities. Right now a reader can merge them and hear
"85 of 93."

---

## §5.7 Upgrades — REWRITE AS A FINDING

**5-9** Currently: "Both of these upgrades only help the Bayesian network." That
undersells what you actually found. The strict recipe **cannot reproduce the
published Table 9 severity rows at all**: P(no injury | combustion liner) comes
out **0.0040** against the published **0.9978**, and the prior P(no injury) comes
out **6.3e-7** against the published **0.9999** — because binary severity leaves
inherit a per-flight prior and can never approach 1. The four-state upgrade
brings both to **0.99999**, scoring EXACT. Five "no injury" rows, two "serious
injury" rows and two "substantial" rows all resolve this way.

Frame it as: the upgrade is not a feature we added, it is what makes the
published numbers reproducible. That is a reproduction result and it is the
strongest paragraph available to you.

---

## §6.3 Retrieval and soft evidence — HIGHEST PRIORITY

**6-1 [GAP]** **The paper never names the encoder, the metric, or the
truncation.** Add: OpenAI `text-embedding-3-small` (`shared/code/config.py`
line 48), cosine similarity on normalized vectors, narratives truncated to
**4,000 characters** (`apps-parity/demo_common.py` line 41), redaction applied
before embedding. Without this the method is not reproducible and it is the
first thing a reviewer will ask.

**6-2 [NUM]** "the evidence strength is the fraction of those neighbors ... if 71
of 100 neighbors have landing phase code, then 0.71 (71/100)." **Wrong.** Each
neighbour contributes its **cosine similarity as a weight**; the strength is a
similarity-weighted frequency, not a count over 100. Source:
`code/query_to_bn.py` lines 220-235.

**6-3 [GAP]** No mention that soft facts are **filtered** before entry: minimum
weighted frequency 0.15, minimum odds lift over base rate 3.0, at most 3 facts,
confidence capped at 0.95. Source: `query_to_bn.py` lines 180-183. Omitting the
filter makes the method sound more automatic than it is.

---

## §6.4 Severity virtual evidence — HIGHEST PRIORITY

**6-4 [NUM]** "We build a fraction per state, where we take the count and divide
by 100 ... 41 neighbors had no injury ... becomes 0.41." **Wrong twice.**
Similarity-weighted, and Laplace-smoothed with alpha = 0.5. The worked example
summing neatly to 100 reinforces the wrong picture. Source: `query_to_bn.py`
lines 429-438.

**6-5 [BN]** The weighted distribution is **not entered as a probability**. It is
converted to a likelihood vector L(j) = f_q(j) / p0(j) against the network's own
severity prior, so Jeffrey conditioning moves the posterior toward f_q. Source:
`query_to_bn.py` lines 486-496. This is the single most BN-technical step in your
whole method and the paper currently describes it as division. Fix this one first.

---

## §6.5 Propagation — HIGHEST PRIORITY

**6-6 [BN]** "We run the propagation update on the CPTs." **Propagation does not
update CPTs.** The CPTs are fixed; propagation computes posteriors. You say the
correct thing in three other places, which makes this sentence look like a slip
of understanding rather than of typing — and it is precisely the sentence that
confirms what he said on 9/1. Rewrite: evidence is loaded, one propagation runs,
every node gets a posterior, the CPTs never change.

**6-7** Good as-is: "the Bayesian network severity agrees with the raw neighbor
readout on all of our 296 accidents. The network propagates the signal but does
not add severity accuracy." Keep it. That honesty is an asset.

---

## §6.6 Table 5

**6-8 [NUM]** "Injury/damage virtual evidence | Neighbor counts / 100" — same
error as 6-4.

**6-9 [OPEN QUESTION]** The row "K = 100 neighbors | Fixed default | Before test
set scoring" needs to be true. The repo contains
`outputs/heldout_narrative_bn_eval_k99.json` and `heldout_eval_k25_run.log`,
which means k was varied at some point. **Work out before the meeting whether
those runs happened before or after k = 100 was fixed.** If k was chosen by
looking at test scores, the claim in this table is wrong and it is the most
serious methodological issue in the paper. If they were post-hoc sensitivity
checks, say that explicitly.

---

## §7 Test set Evaluation

**7-1** The section has no number in the document.

**7-2** The paragraph beginning "We use the test set many times" duplicates the
one under Table 5 almost verbatim. Cut one.

---

## §8.1 Severity results

**8-1 [NUM]** Table 6, "Parsed events only": **82.4% → 82.1%**, Macro-F1
**0.422 → 0.420**. One accident out of 296 flipped on re-run today. Damage 50.7%
and 0.309 unchanged. Prose under the table repeats 82.4% — fix both.

**8-2** "TF-IDF is 92.2% on injury vs our 90.9%." Add the test: McNemar gives
p = 0.289 (Holm 0.867), so on **accuracy** that is a genuine tie, not a loss.
But TF-IDF's injury Macro-F1 is **0.603 against your 0.470**, and that gap has
no CI attached. Split the sentence by metric so "ties" can't be carried across
both. Your damage lead (77.4 / 0.697 vs 73.3 / 0.621) is clean on both.

---

## §8.2 Diagnosis results

**8-3** "We also used CICTT, which is described in Section 8.3." Wrong pointer —
§8.3 is the LLM experiments.

**8-4** Table 8 shows supervised embedding LR at **88.1%**, which **beats your
retrieval readout at 84.2%**. Verified in `outputs/diagnosis_emb_lr.md`. The
prose says retrieval "is the primary diagnosis readout" without acknowledging
that a trained baseline scores higher. Say it before he does.

**8-5** Table numbering runs 6 → 8, because Table 7 is Zhang's. Then §9 has your
Tables 9 and 10 while also discussing his Tables 7 and 9. A reader cannot track
this. Renumber yours or prefix every reference with whose table it is.

---

## §9.2 Case study

**9-1** "Minor differences in low probability injury and damage cells reflect
multi state severity encoding in the upgraded network." This is a hand-wave, and
you now have the real answer — point to the §10.1 diagnosis instead.

---

## §10.1 Discussion — HIGHEST PRIORITY

**10-1 [BN]** "Most of that is random tie-breaking when parents get chosen."
Same error as 5-4, and here it is the entire explanation for your residual
disagreement. Replace with the real diagnosis: two cells sit at the untouched
prior because of LOEP label fragmentation, four overshoot uniformly at ~5.3x,
two miss only because the CLOSE threshold is 25%, one is the cell where his
released `.xdsl` agrees with you rather than with the table.

**10-2** "12 out of 93 after the small upgrades" — name the build. The strict
rebuild differs on **26**; the upgraded network on **12**.

**10-3** "His own released model file does not match every published cell
either." True and verified (his `.xdsl` gives 1.77e-7 where the paper says
1.21e-7). Keep it, but he is the co-author — phrase it as an observation about
build provenance, not as a defect.

---

## §10.2 Limitations — ADD THREE

**10-4** Embedding pretraining may include 2007-2019 NTSB text; redaction cannot
reach inside a pretrained encoder. This is already written in
`docs_FrozenBN/SECTION_8_DISCUSSION.md` line 35 but **is not in the draft**.
TF-IDF is your pretraining-free reference point.

**10-5** The per-flight prior means severity posteriors sit far below empirical
rates. Data: 53.7% of engine-power-loss accidents had minor damage; the network
says 2.1%.

**10-6** Two Table 9 cells return the prior unchanged because evidence cannot
reach the node — a known wiring limitation, with the fix identified.

---

## §10.3 Future work

**10-7** Already points at the agent direction, which matches what he asked for
on 9/1. Strengthen by naming **ground transportation** — he told you explicitly
to look there and not at maritime.

---

## Order to fix — ranked by damage to the paper

### Tier 1 — the paper is wrong or unreproducible as written

1. **6-2, 6-4, 6-8 — Section 6 does not describe what the code does.** The paper
   says the evidence strength is a fraction of 100 neighbours; the code computes
   a similarity-weighted, Laplace-smoothed distribution. This is a methods-
   integrity problem, not a wording problem, and it appears in three places.
2. **6-1 — the method is not reproducible.** Encoder, similarity metric and
   truncation appear nowhere in the paper. First thing a reviewer asks.
3. **5-4 and 10-1 — tie-break described as random; the code is deterministic.**
   Same error twice, and in §10.1 it is the *entire stated explanation* for the
   residual disagreement with the published numbers. Fixing 5-4 alone leaves the
   discussion resting on a false cause.
4. **6-9 — settle the k = 100 provenance.** If k was chosen after seeing test
   scores, the Table 5 claim is false and this outranks everything else here.
   Repo has k=25 and k=99 runs; you need to know when they ran.
5. **8-1 — 82.4 → 82.1** (and 0.422 → 0.420). Two places: table and prose.

### Tier 2 — attackable by a reviewer

6. **8-2 and 8-4 — baselines that beat you are under-reported.** TF-IDF injury
   Macro-F1 0.603 vs your 0.470, unhedged. Supervised embedding LR gets 88.1% on
   diagnosis against your 84.2% retrieval, shown in Table 8 but not acknowledged
   in the prose. Reads as selective reporting if you don't say it first.
7. **10-2 — "12 of 93" without naming the build.** Strict rebuild differs on 26.
8. **1-1 — "matches published numbers" overclaims** in the introduction.
9. **10-4, 10-5, 10-6 — three real limitations missing**, including the embedding
   pretraining caveat that is already written in your own `SECTION_8_DISCUSSION.md`
   but never made it into the draft.

### Tier 3 — adds value rather than repairing damage

10. **5-9 — rewrite §5.7 as a reproduction finding.** The strict recipe cannot
    reproduce the published severity rows; your upgrade is what makes them
    reproducible. Strongest available paragraph, currently undersold.
11. **New §10.1 content — the 12-cell diagnosis** replacing the tie-break story.

### Tier 4 — conceptual wording (matters for the conversation more than the page)

12. **6-6 — "propagation update on the CPTs."** One wrong sentence, but it is the
    method section describing the core operation incorrectly.
13. **6-5, 2-1 — virtual evidence never defined as a likelihood ratio.**
14. **4-3, 4-6 — conditional independence / d-separation absent from a BN paper.**

### Tier 5 — mechanics, fix in one pass

15. Table numbering runs 6 → 8, then your 9 and 10 collide with his 7 and 9 (8-5).
16. §8.2 points at §8.3 for CICTT; §8.3 is the LLM experiments (8-3).
17. §7 unnumbered; duplicated "scored the test set many times" paragraph (7-1, 7-2).
18. Broken sentences: 5-2, 4-1, 4-7, 5-6. Typo "mutli-label" in §2.2.
19. 3-1 — state which narrative filter Table 1 uses (1,288 vs 1,286).
