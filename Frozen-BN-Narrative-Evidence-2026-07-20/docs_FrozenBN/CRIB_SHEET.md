# Crib sheet: every question Maha or Jesse can ask, answered in your own words

One page per question. No jargon you can't defend. Numbers as of 2026-07-29
(leak-safe protocol). Sources cited so you can pull up the file live.

---

## "How does the narrative get into the Bayesian network?" (Jesse's #1)

Three routes, all ending as evidence on named nodes of the frozen network:

1. **Direct vocabulary match (hard evidence).** The parser scans the
   narrative for NTSB's own node names ("loss of engine power", "fire",
   "landing gear, tire"). If the text names the fact, that node is set to
   Yes with probability 1. No model, no weights -- string matching against
   the network's node list.
2. **Retrieval facts (soft evidence).** Embed the narrative, find the 100
   most similar 1982-2006 accidents, and look at what NTSB coded on THEM.
   If 40 of the 100 have "carburetor icing" coded, the carburetor node gets
   soft evidence of strength 0.40. The strength is a measured fraction from
   data -- not an LLM guess, not a tuned weight.
3. **k-NN severity as virtual evidence (bn-sev, the primary predictor).**
   The injury/damage distribution of those same 100 neighbors enters the
   network's severity nodes as a probability distribution (Pearl's virtual
   evidence / Jeffrey conditioning: a likelihood vector on the node, the
   standard way to give a BN uncertain evidence). The network then does the
   inference.

Say it in one line: *the narrative is translated into evidence on Zhang's
own coded variables; every evidence strength is a fraction counted from the
1982-2006 data.*

## "What are the weights? Are they learned?" (Jesse)

There are no learned weights anywhere in the primary chain. Every number is
either (a) frozen -- Zhang's CPTs built from 1982-2006 coded data; (b) a
measured fraction -- soft-evidence strengths counted from retrieved
neighbors; or (c) a selected hyperparameter -- top_k=100, chosen on an
internal 2002-2006 validation split, never on the held-out years. Full
inventory: `docs_FrozenBN/FREE_PARAMETERS.md`. The only FITTED models in
the repo are the two supervised baselines (lr, emb-lr) we run against
ourselves for comparison.

## "So what are you training? Where's the validation set?" (Jesse)

Nothing is trained. 1982-2006 is the build window (network + retrieval
index), 2002-2006 is the internal validation slice used once to pick
top_k, and 2007-2019 is held out and touched only for final scoring. The
word to use is "frozen", never "trained". If asked why we even have a
held-out set: to *evaluate*, not to fit.

## "What exactly is that 93%... sensitivity? specificity?" (Jesse)

The old 93%/81% slide numbers are superseded; the leak-safe numbers are:

- **Accuracy** = top-1 of the 4-state distribution equals the NTSB-coded
  class. Injury 90.9%, damage 77.4% (n=296, bn-sev).
- Majority-class floor: injury 58.4% (always "none"), damage 42.6%. We
  report **Macro-F1** (0.470 / 0.697) because accuracy alone flatters
  majority guessing.
- **Severe-outcome screen** (fatal/serious injury vs not): sensitivity
  93.5%, specificity 96.8%. Severe damage: sensitivity 75.3%, specificity
  89.8%.
- All differences vs baselines carry McNemar exact tests and bootstrap
  CIs: `outputs/heldout_significance.md`.

## "Isn't the model reading the outcome in the text?" (Jesse's leakage flag)

It was a real risk and we closed it. Outcome phrases ("was destroyed",
"received serious injuries") are stripped by regex before any embedding or
parsing; stated-severity evidence is OFF by default. Two audits: a
redaction leak probe (a classifier on the redacted text keys on crash
mechanics, not outcome words -- `tests/redaction_leak_probe.py`) and a
held-out leak audit (`tests/heldout_leak_audit.py`). Numbers dropped from
93%/81% to 90.9%/77.4% when we fixed this -- we report the honest ones.

## "What is the BN contributing?" (both advisors)

Be direct: on held-out *accuracy*, the BN matches the retrieval readout by
construction (a self-test asserts the posterior equals the k-NN input);
the raw signal comes from the narratives. What the network adds is
everything accuracy doesn't measure: (1) a validated causal structure --
we reproduce Zhang's published tables to the digit; (2) joint reasoning --
combine "icing conditions" + "night" + retrieved severity in one coherent
inference, which counting one conditional at a time can't do; (3)
interrogability -- what-if queries, explanation of which evidence moved
which node. The retrieval numbers are the accuracy claim; the BN is the
reasoning-and-explanation claim. Don't oversell the reverse.

## "Do your numbers match Zhang's?" (Maha's publishing bar)

Yes, on everything Zhang published: fire occurrences 102 (was 38 before
the data fix), prior P(fire) = 102/184,517,128 = 5.527942e-7 exact,
Table 7 cause distribution 113/113 exact. Report:
`docs_FrozenBN/ZHANG_REPRODUCTION_REPORT.md`.

## "How do you evaluate diagnosis across the 2008 coding change?"

NTSB switched taxonomies in 2008 (legacy subject codes -> CICTT), so exact
code matching across the split is impossible by design. We roll BOTH eras
up to CICTT's four top-level cause categories (Personnel / Aircraft /
Environment / Organizational); legacy subjects map by auditable keyword
rules (98.3% coverage; hand-audit sample:
`outputs/mapping_audit_sample.csv`). A prediction is correct if its top
category is among the accident's coded cause categories. Results (n=253):
frequency baseline 45.8%, retrieval 83.8%, supervised emb-LR 88.1%, BN
event path 57.7%. All gaps significant by McNemar.

## "A linear model beats you -- why is your architecture justified?"

We ran that attack ourselves. Severity: emb-LR 91.6%/74.0% vs our
90.9%/77.4% -- no significant difference on injury (p=0.50), we're better
on damage severe-recall (75.3% vs 67.9% sensitivity). Diagnosis: emb-LR
88.1% does beat retrieval 83.8% (p=0.027) -- we disclose it. The answer:
the supervised model needs coded labels to train and outputs an opaque
score; our chain needs zero training and every prediction decomposes into
named evidence on a validated causal network. Close-to-parity with zero
parameters is the selling point, not raw victory.

## "Why did some numbers change from last week?"

One reason, stated once: we found and closed outcome-phrase leakage and
disabled stated-severity evidence by default. Old: 93%/81%. New, honest:
90.9%/77.4%. Every current number regenerates from the repo
(`REPRODUCE.md`); query embeddings are disk-cached so reruns are
deterministic.

## Terms you must not misuse (Maha called this out)

- "Posterior" -- only for P(node | evidence) computed by the network.
  The retrieval readout is a "weighted neighbor distribution", not a
  posterior.
- "Training" -- we do none. Say "frozen", "selected on validation",
  or "fitted" (baselines only).
- "Asserted" -- banned. Say "the parser mapped 'windshield' to the
  window-node with strength 0.14, measured from neighbors".
- "Virtual evidence / Jeffrey conditioning" -- fine, but be ready to say
  it plainly: "we tell the network the severity node's distribution
  instead of a single observed value."
