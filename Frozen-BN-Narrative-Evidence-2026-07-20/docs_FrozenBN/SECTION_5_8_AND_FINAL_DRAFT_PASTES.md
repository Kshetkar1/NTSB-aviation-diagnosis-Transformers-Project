# §5.8 + final-draft paste blocks (Draft 616d)

Paste into Word after **§5.7 Upgrades**, before **§6**. Then apply the replacement paragraphs below.

Sources: `DRAFT_EDIT_NOTES_2026-09-07.md` §A1, §A2, §B; `bn_build_ours.py`; `tests/bn_upgraded_full.py`.

---

## §5.8 Posterior scoreboard and the twelve remaining gaps

Zhang and Mahadevan report two kinds of quantities that are easy to merge by accident. **Table 7** is a counting table: **P(cause | fire)** on the **102** fire accidents in 1982–2006 (Section 5.6; **85/85** rows in Appendix A). It is **not** part of the **93** published Bayesian-network posterior checks in Tables 8–9 and the benchmark scenarios (engine instruments, combustion liner, oil grade, and downstream injury and damage readouts).

We score reproduction in two stages on the same coded window. **Stage 1 — strict recipe (Sections 5.1–5.5):** graph, Eq. (6) priors, Eq. (8) parent cap, and Beta-CDF CPTs from 1982–2006 only. Against the 93 benchmark quantities, this faithful build **disagrees on 26**. **Stage 2 — operational network (Section 5.7):** we add person-finding nodes so pilot-style evidence runs, and four-state personnel injury and aircraft damage so Table 9 severity rows are defined on the same nodes Zhang uses. That network is **frozen** for all narrative experiments in Sections 6–8. On the same 93-item scoreboard, **48** cells match exactly, **29** match within a **25%** relative tolerance (“close”), **12** still differ, and **4** are qualitative (different estimand or display, not a single scalar parity check). We never attribute the twelve to random parent tie-breaking: our parent ranking is **deterministic** (Eq. (8) ratio, then edge support count, then alphabetical parent name), with **no** random seed in the build code.

**Why twelve remain (grouped causes, not twelve independent bugs).**

*(1) Evidence never reaches the severity node (two cells).* Under evidence **combustion liner** or **improper oil usage**, **P(minor damage)** and sometimes injury stay at the **unupdated prior** (~5.6×10⁻⁷ on minor damage). The finding activates a **plain** “loss of engine power” occurrence node, but only the **total / mechanical** LOEP label is wired as a parent of four-state severity in this graph. The path **dead-ends**—label fragmentation across LOEP variants, not a counting error.

*(2) False “exact” on the same dead end (two cells).* **P(no injury | combustion liner)** and **P(no injury | oil)** also sit at the untouched **P(no injury)** prior (~0.999999). They grade **exact** only because Zhang’s published values (~0.998) lie within 1% of 1.0 although **no evidence propagated**—the same mechanism as (1), opposite direction on the scoreboard.

*(3) Systematic overshoot (~four cells).* Several downstream severity cells overshoot published values by a similar factor (~5.2×–5.5×) once multiple parents are active. We treat that as one **CPT / Beta-CDF** scaling effect on sparse multi-parent rows, not four unrelated mistakes.

*(4) Threshold and provenance (~three cells).* Two cells miss **close** only because our **25%** tolerance is tight (they would pass at ~30%). One **main-gear** cell matches Zhang’s **released model file** while the **paper table** differs—build provenance, not a narrative or test-set issue.

*(5) Calibration, not “ground truth.”* Table 9 cells are **model outputs**, not empirical labels. Where evidence **does** reach severity parents, our posteriors are sometimes **closer to 1982–2006 empirical rates** among LOEP accidents (e.g. minor damage ~53.7% in data vs ~0.4% published vs ~2% ours)—**closer is not close**; both networks use **per-flight** priors (Eq. 6) that keep absolute probabilities tiny. We state that as an open scale question, not a claim that we “beat” Zhang.

*(6) Sparse showcase support.* The combustion-liner and oil-grade scenario edges rest on **one or two** accidents in the window; any posterior is high-variance. That limits how much digit-level parity proves.

Zhang’s public **.xdsl** model also **does not** reproduce every published table cell. We claim **faithfulness to method, counting anchors (Table 7, 102 fires, Eq. 6), and forward Table 9 edges where evidence paths match**, not bitwise identity on all 93 cells. Narrative and hand-set evidence agree when parsing lands on the **same** nodes (Sections 8–9).

---

## Replace — Introduction (¶ with pyAgrum / 81 of 93)

**Delete:** “We match 81 of 93 of their published Bayesian networks cells. The explanation is in Section 5.”

**Paste:**

We query the frozen network in pyAgrum [10]. Appendix A reproduces **P(cause | fire)** on all **85** Table 7 rows. On Zhang’s **93** published benchmark posterior quantities (Tables 8–9 scenarios), our **upgraded** network (Section 5.7) matches **48** exactly and **29** within tolerance; **12** differ for traced reasons in **Section 5.8** (not random tie-breaking). Table 7 is a separate counting check, not part of those 93.

---

## Replace — §10.1 (reproduction paragraph only)

**Delete:** “Some full network cells still differ … random tie-breaking …”

**Paste:**

The reproduction holds on data-derived anchors: **102** fires, **P(fire) = 102/184,517,128**, and **85/85** Table 7 rows. On the **93** benchmark posteriors, **12** cells still differ on the **upgraded** network (**26** on the strict Sections 5.1–5.5 build alone). Section **5.8** groups those twelve by mechanism (unreachable evidence through LOEP labels, multi-parent CPT scaling, tolerance and released-model provenance). Parent selection in our code is **deterministic**, not random. Zhang’s released model file also disagrees with some published table entries. We trust the **method and counting tables**, not every single published posterior digit.

---

## Replace — Abstract (full paragraph)

**Paste:**

The National Transportation Safety Board (NTSB) publishes coded accident fields and free-text narratives for the same investigations. Zhang and Mahadevan (2021) built a Bayesian network for aviation diagnosis and prognosis from coded records (1982–2006); whether that model is reproducible and whether narratives can supply query-time evidence without retraining remain open. We reproduce the counting layer (85/85 Table 7 fire-cause rows, 102 fire occurrences, Eq. 6 prior), implement Zhang’s graph and Beta-CDF construction (Sections 5.1–5.5), extend the network minimally so published benchmark queries run (Section 5.7), and freeze all parameters before held-out evaluation. A leak-safe pipeline redacts explicit outcome phrases, maps redacted text to **hard evidence** and **virtual evidence** on the network’s variables (phrase matching and similarity retrieval over a 1982–2006 index), and runs inference on the fixed graph. We evaluate on accidents outside the build window (2007–2019, n = 296 for severity) without refitting CPTs or the retrieval index. We report where posteriors match Zhang’s benchmarks, where twelve of ninety-three cells differ for explained reasons (Section 5.8), and where narrative retrieval versus network readouts lead on held-out diagnosis and prognosis (Sections 7–8). The deliverable is an auditable path from investigation text to evidence on a reproduced network—not a claim that every published probability matches or that the network always beats simpler readouts.

---

## Replace — §10.2 (soft evidence sentence)

**Find:** “rely on soft evidence from the k-nearest neighboring accidents”

**Paste:** “receive **event virtual evidence (Section 6.3)** from the k = 100 most similar 1982–2006 accidents”

---

## Replace — §10.3 (both “soft evidence” mentions)

**Find:** “rely on the soft evidence” / “fewer queries rely on the soft evidence”

**Paste:** “depend on **event virtual evidence (Section 6.3)**” / “fewer queries depend on **event virtual evidence alone**”

**Optional:** After agent sentence, add: “Confirmatory scoring on **2020–2024** accidents never used in development is planned before journal submission.”

**Optional:** Name **ground transportation** (not maritime) if Maha asked for domain extension.

---

## Optional — Table 6 injury ablation

If repo re-run confirms **82.1%** (not 82.4%) for hard + event virtual injury row, update Table 6 and §8.1 prose together (`DRAFT_EDIT_NOTES` A4).

---

## Final grep (Word)

`soft evidence`, `virtual severity`, `random tie`, `81 of 93`, `verify all the published`, `floor` (§5.5 only if any left).

---

## Paste order checklist

1. §5.8 after §5.7  
2. Introduction replacement  
3. Abstract  
4. §10.1 reproduction paragraph  
5. §10.2–10.3 soft → event virtual  
6. Grep → PDF  
