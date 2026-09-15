# Bayesian network fundamentals + design decision log

Written because of the 9/1 note: **"He also said my Bayesian network
understanding is weak."** Part 1 is the concepts he is most likely to probe,
answered on *your* network with *your* numbers. Part 2 is every design choice
with its justification, including the ones where the honest answer is
"we didn't test that."

---

# PART 1 — Bayesian network concepts

## 1.1 Why a network at all, instead of one big table

A joint distribution over 785 binary variables has 2^785 entries. The graph says
which variables are conditionally independent of which others, and that lets the
joint factorize into one small table per node: P(all nodes) = product over nodes
of P(node | its parents). Your largest CPT is a node with 12 binary parents —
4,096 rows instead of 2^785. **The graph is a claim about independence, and the
parameter savings are the consequence.** If he asks "why a Bayesian network,"
that is the answer, not "because it's interpretable."

## 1.2 Conditional independence and d-separation

This is the concept missing from your paper (§4 has none of it) and the most
likely thing behind his comment.

A node is independent of everything else given its **Markov blanket** — parents,
children, and children's other parents. Evidence flows along paths unless a path
is *blocked*. Three cases:

- **Chain** A -> B -> C: knowing B blocks A from C.
- **Fork** A <- B -> C: knowing B blocks A from C.
- **Collider** A -> B <- C: B *blocks* by default, and knowing B (or any
  descendant of B) **opens** it. This is the one people get wrong.

**Say it on your own example (§4.2).** brake wear -> fire -> injury is a chain.
Once fire is observed, injury is d-separated from brake wear, which is *why*
P(injury = serious | fire = yes) reads straight off Table 4 as 0.40 with no
contribution from the brake wear prior. Your draft asserts this result without
naming the reason. Naming it is the fix.

**Compute the backward direction too.** Same tables:
P(fire=yes) = 0.10(0.80) + 0.90(0.05) = 0.125
P(brake wear=yes | fire=yes) = 0.10(0.80) / 0.125 = **0.64**, up from 0.10.
Forward reasoning is prognosis, backward is diagnosis, one network does both.
That sentence is your whole paper in miniature.

**The collider case is in your leak protocol.** Your fire experiment barred the
severity nodes from the parser's namespace specifically so the collider
`fire -> aircraft damage` never opens. If you had entered damage evidence, it
would have opened a path from damage back to fire and contaminated the
cross-inference. Knowing why you did that is a d-separation answer.

## 1.3 Hard, soft, and virtual evidence — say these precisely

- **Hard**: clamp a node to one state. P(node = Yes) = 1. Used when phrase
  matching reads a fact directly out of the narrative.
- **Soft (Jeffrey conditioning)**: you want the posterior on a node to *land at*
  a specified value c. Implemented as a likelihood ratio against the node's own
  prior: LR = [c/(1-c)] / [p0/(1-p0)]. Your code does exactly this
  (`query_to_bn.py`, `apply_evidence`). The comment there explains why a naive
  [c, 1-c] likelihood fails: your priors are ~1e-7, so a flat likelihood would
  be swallowed and move nothing.
- **Virtual (Pearl)**: attach a likelihood **vector** to a node — not a
  probability. It multiplies into the existing belief instead of replacing it.
  Yours is L(j) = f_q(j) / p0(j) over the four severity states, where f_q is the
  similarity-weighted neighbour distribution and p0 is the network's own prior.

**If he asks the difference between soft and virtual:** soft evidence is a
statement about a node's marginal that you want honored; virtual evidence is a
statement about the *likelihood of an observation* given each state. Jeffrey
conditioning fixes the posterior; Pearl's virtual evidence fixes the likelihood
ratio and lets the network decide the posterior. Yours uses both.

## 1.4 Why the network adds no severity accuracy — the honest mechanism

`bn-sev` and `retrieval-sev` agree on all 296 accidents, to three decimals
including CIs. This is **not** a disappointing result, it is a d-separation
consequence, and you should present it that way.

You enter the neighbour severity distribution directly on the injury and damage
nodes. Those nodes are leaves. Nothing downstream of them competes, and the
event evidence you enter elsewhere reaches them only through paths that are weak
relative to a likelihood ratio applied at the node itself. So the network
**losslessly mediates** the retrieval signal — it neither adds nor destroys it.
The right claim is that the BN is a coherent container for the signal, not a
booster of it.

**Proof you did the work:** merging event evidence and neighbour severity in one
update *drops* accuracy to 38.5% / 41.9%, because the same narrative is counted
twice through two paths. You measured the double-counting rather than assuming
it away.

## 1.5 Why fire cross-inference failed — also d-separation

Best BN arm reaches ROC AUC 0.39-0.46; retrieval on the same text reaches 0.97.
The mechanism is structural, not statistical: **`fire` has only 13 ancestors out
of 783 nodes**, six are barred by the leak guard, and the only one the parser
ever actually enters is the generic `person: flightcrew`. Narrative evidence is
almost never d-connected to `fire` in a fire-specific way, so the posterior
barely moves off its 7.7e-8 prior. Knowing the ancestor count is the difference
between "it didn't work" and "here is why it couldn't have."

## 1.6 Why the priors are per-flight, and what it costs

Zhang's Eq. 6 divides an occurrence count by **184,517,128 BTS departures**, so
root priors are ~1e-7 and P(no damage) is 0.999999. Consequence you should
volunteer: conditioning on a single event node cannot drag a probability from
1e-6 up to an empirical rate. In the data, **53.7% of the 147 engine-power-loss
accidents had minor damage**; the published number is 0.38% and yours is 2.1%.
Both are far below the data. **This is his modeling choice, inherited — ask him
why per-flight rather than per-accident.** He may have a reason you don't know,
and you want it before it goes in the paper.

## 1.7 What the Beta-CDF is for

With 12 parents there are 4,096 parent-state rows per child, and the 1982-2006
data has nowhere near enough accidents to estimate each one by counting. The
rule: no parent active gives 0; exactly one active gives that parent's ratio;
two or more active gives Beta-CDF(sum of active ratios / sum of all ratios),
with the shape parameters fit per child by Nelder-Mead from x0 = [2, 1], and the
result floored at the largest active parent's ratio. **It is an interpolation
scheme for unobserved parent combinations, not a smoothing of observed counts.**

---

# PART 2 — Design decision log

Format: choice, why, what happens if asked whether you tested alternatives.
**Where the honest answer is "no", say no.** Guessing is worse.

| Choice | Value | Why | If asked about alternatives |
|---|---|---|---|
| Encoder | `text-embedding-3-small` | Fixed before test scoring; `config.py` line 48 | **We did not test `-3-large` or `ada-002`.** Say so. Note TF-IDF is in the paper as a pretraining-free reference (92.2 / 73.3) |
| Similarity | Cosine on normalized vectors | Standard for this encoder; dot product on unit vectors | Not benchmarked against alternatives |
| Truncation | 4,000 characters | `demo_common.py` line 41 | Not swept |
| k | 100 neighbours | Fixed default, not tuned on test | **Settle this before the meeting** — repo has k=25 and k=99 runs. If those ran after k=100 was fixed they're sensitivity checks; if before, the "not tuned" claim is wrong |
| Laplace alpha | 0.5 | Prevents zero mass on unseen severity states | Not swept |
| Soft-fact filter | f_q >= 0.15, lift >= 3.0, top 3, cap 0.95 | Keeps only facts that are both frequent among neighbours and elevated over base rate | Thresholds not swept; disclose them |
| Max parents | 12 | Zhang's `maxElements`, kept for fidelity | Deliberately not changed — fidelity is the point of Job 1 |
| Ratio cap | 0.95, only when raw ratio == 1.0 | Zhang's code cap; avoids a deterministic edge | Not a general ceiling |
| Tie-break | ratio desc, then edge support desc, then name | **Deterministic** — this is why no seed exists | Replaces the original's random jitter; ours is reproducible, that's the argument |
| Severity states | 4 ordered levels | Binary leaves inherit ~1e-7 priors and cannot reach the published 0.9978 | This is the reproduction fix, not a feature |
| Person nodes | Added | Zhang's pilot scenarios can't be posed without them | Needed to run his own published queries |
| Cause categories | 4 (Personnel, Aircraft, Environment, Organizational) | CICTT top level; rolls pre-2008 and post-2008 codes to one scheme | Keyword rules, not trained on test |
| Inference engine | pyAgrum, LazyPropagation | Open source; no GeNIe/pySMILE dependency | Exact on ancestral fragments; approximate runs are flagged UNVERIFIED in the outputs |
| Redaction | Outcome phrases stripped pre-embedding | Otherwise severity prediction is reading the answer | Ablation exists (`heldout_eval_leaky_ablation.log`) |

---

# PART 3 — Questions to ask him

Asking these is worth more than answering three of his. They are all real.

1. **Why per-flight rather than per-accident priors?** It's his Eq. 6, and it's
   why both networks sit far below empirical rates (1.6 above).
2. **Were the n=1 and n=2 support edges in the Table 9 scenario intended as
   illustrative?** Combustion liner appears in 2 accidents in the window, oil
   grade in 1. Both combustion-liner accidents had minor damage.
3. **Did the published Table 8 come from a different build than the released
   `NTSB.xdsl`?** His file gives 1.77e-7 where the paper reports 1.21e-7; your
   rebuild gives 1.75e-7, so you match the file rather than the table.
4. **Does he want the evidence extractor to stay rule-based?** If auditability is
   the point, that settles the BERT/learned-extractor question in the proposal.
5. **Ground transportation** — he said to look there, not maritime. Confirm the
   next dataset.
