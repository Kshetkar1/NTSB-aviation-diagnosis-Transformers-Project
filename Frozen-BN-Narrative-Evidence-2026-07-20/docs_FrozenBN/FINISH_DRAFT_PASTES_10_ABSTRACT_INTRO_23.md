# Finish draft — §10, Abstract, Intro, §2.3

Paste into Word (`616d` draft). Edit voice as needed. **Find/replace** items marked FIND.

---

## §10.1 Discussion of results (full section — or replace para 2 + fixes only)

**Paragraph 1** (keep yours; optional trim)

Zhang and Mahadevan set evidence on their Bayesian network from coded NTSB fields. We ask whether redacted narratives can supply that evidence without retraining, so the same fixed network returns comparable probabilities on Zhang’s published scenarios and on post-2006 accidents. On the test set, narrative retrieval leads diagnosis while the network carries inference (Section 8). We reproduce the network from 1982–2006 coded data, then test the narrative interface on Zhang’s benchmark queries and on 296 accidents from 2007–2019.

**Paragraph 2** (REPLACE your reproduction paragraph)

The reproduction holds on counting anchors: 102 fire accidents in 1982–2006, P(fire) = 102/184,517,128, and 85/85 Table 7 rows (Appendix A). Those Table 7 rows are not part of the 93 scenario posteriors in Tables 8 and 9. After the Section 5.7 extensions, twelve of those ninety-three published values still disagree with Zhang’s tables (twenty-six on the strict Section 5.1–5.5 build alone). Section 5.8 explains those twelve. Zhang’s released model file also disagrees with some paper cells. We reproduce Zhang’s construction and Table 7 counting; we do not match every posterior in Tables 8 and 9.

**Paragraph 3** (keep yours)

On his benchmark sentences, the narrative bridge does what we wanted. When the text names a fact Zhang would code by hand, phrase matching lands on the same network node and the posteriors match. The engine instrument case is the clearest: typed English gives P(loss of engine power) equivalent to 0.95, the same anchor as Table 9. That answers the main question for the scenarios he published: narratives can drive the same network to the same probabilities when the wording maps cleanly to his vocabulary.

**Paragraph 4** (REPLACE held-out paragraph)

The harder test is accidents the network never saw. We remove explicit outcome phrases, map each redacted narrative to evidence, and read prognosis as BN-SEV (Sections 6.4–6.5): virtual evidence on injury and damage, then inference on the fixed network. Injury accuracy rises from 58.4% (prior alone) to 90.9%; damage from 42.6% to 77.4%. Supervised text models trained only on 1982–2006 are in the same range on injury (Section 8.1). We lead on damage Macro-F1. For causes, narrative retrieval gives 84.2% four-category top-1; BN diagnosis with hard evidence (Section 6.2) and event virtual evidence (Section 6.3) reaches 57.7%. Retrieval is the stronger diagnosis readout on new reports. Supervised embedding logistic regression reaches 88.1% on the same four categories (Section 8.2).

**Paragraph 5** (keep; one FIND)

Some results look bad on paper, but they clarify what the network is for. On severity, BN-SEV matches k = 100 neighbors on all 296 accidents. The network is not adding top-1 accuracy beyond that neighbor readout; it propagates the signal and supports joint conditioning and what-if queries. If we merge event virtual evidence (Section 6.3) with injury/damage virtual evidence (Section 6.4) in one update, accuracy falls to 38.5% and 41.9%. A fire-node test shows parsed events do not predict fire well through the network; retrieval does. LLMs help map sentences to evidence on Zhang’s benchmarks and are usable but miscalibrated on real narratives. They cannot replace the coded record or the network when asked for probabilities directly.

**Paragraph 6** (keep)

Zhang’s network supplies the probabilities; the narratives supply the evidence. We showed that link on his published examples and on hundreds of new accidents, and we reported where it holds and where it does not.

---

## §10.2 Limitations (FIND → REPLACE only)

**FIND:** `rely on soft evidence from the k-nearest neighboring accidents`  
**REPLACE:** `receive event virtual evidence (Section 6.3) from the k = 100 most similar 1982–2006 accidents`

**ADD** one sentence at end of first paragraph (optional):

We do not claim cell-for-cell parity with every Table 8–9 posterior; Section 5.8 lists the twelve remaining gaps on the ninety-three-item scoreboard.

---

## §10.3 Future work (FIND → REPLACE)

**FIND:** `rely on the soft evidence`  
**REPLACE:** `depend on event virtual evidence (Section 6.3) alone`

**FIND:** `fewer queries rely on the soft evidence`  
**REPLACE:** `fewer queries depend on event virtual evidence alone`

**Optional ADD** (one sentence):

Confirmatory scoring on 2020–2024 accidents never used in development is planned before journal submission.

**Optional ADD** (if Maha asked):

Later work may extend the agent-style coding workflow to **ground transportation** records, not maritime cases.

---

## Abstract (REPLACE entire paragraph)

The National Transportation Safety Board (NTSB) publishes coded fields and investigation narratives for the same accidents. Zhang and Mahadevan (2021) built a Bayesian network for aviation diagnosis and prognosis from coded records (1982–2006); whether that model is reproducible and whether narratives can supply query-time evidence without retraining remain open. We reproduce Table 7 counting (85/85 rows, 102 fires), implement Zhang’s graph construction and Beta-CDF CPTs (Sections 5.1–5.5), add person-finding nodes and four-state injury and damage so published benchmark queries run (Section 5.7), and freeze the network before held-out evaluation. A leak-safe pipeline redacts explicit outcome phrases, maps redacted text to hard evidence and virtual evidence on the network’s variables (phrase matching and similarity retrieval over a 1982–2006 index), and runs inference on the fixed graph without refitting CPTs. We evaluate accidents from 2007–2019 (n = 296 for severity) as queries only. Primary prognosis uses BN-SEV (Sections 6.4–6.5); primary four-category diagnosis uses narrative retrieval (Section 8). We report where posteriors match Zhang’s Tables 8–9 scenarios, where twelve of ninety-three benchmark cells differ for reasons in Section 5.8, and where simpler readouts outperform the network on held-out tests. The deliverable is an auditable path from investigation text to evidence on a reproduced network.

---

## Introduction (REPLACE paragraph starting “We rebuilt Zhang…”)

We rebuilt Zhang and Mahadevan’s Bayesian network from NTSB coded data (1982–2006) and query the frozen network in pyAgrum [10]. Appendix A reproduces all 85 Table 7 P(cause | fire) rows. On Zhang’s ninety-three published scenario posteriors (Tables 8 and 9), our network after Section 5.7 matches forty-eight exactly and twenty-nine within tolerance; twelve differ (Section 5.8 explains why) and four are qualitative checks. Table 7 is separate from that ninety-three-item scoreboard.

**Keep** the next paragraph on person nodes and four-state severity as-is (or merge with Section 5.7 wording: “Section 5.7 extensions” instead of “We extend”).

---

## §2.3 Hard and virtual evidence

**Title in Word:** rename “Hard, Soft and Virtual Evidence” → **Hard and virtual evidence** (delete every “soft evidence” mention in this section).

**Paragraph 1 — hard evidence** (REPLACE if yours still mixes “soft”):

When a redacted narrative names a fact in Zhang’s vocabulary with high confidence, we set **hard evidence**: the node is clamped to one state before inference, as in standard Bayesian-network conditioning [16]. Section 6.2 maps phrases to network labels with deterministic rules (occurrence/finding/person patterns and LOEP-style labels). Hard evidence overrides event virtual evidence on the same node when both fire.

**Paragraphs 2–5 — virtual evidence** (REPLACE old “soft” / “71/100” text):

Not every phrase in a narrative is certain enough to clamp a node. For those cases we use **virtual evidence** [16]: we attach a **likelihood** on the node and let the network compute the posterior. The pipeline uses two retrieval-based virtual paths on the **fixed** network from Section 5.

**Event nodes (Yes/No).** Section 6.3 embeds the redacted query, retrieves the k = 100 most similar 1982–2006 narratives (cosine similarity on OpenAI text-embedding-3-small), and maps neighbor occurrences and findings to network labels. For each label that passes the gates in Section 6.3, **c** is its similarity-weighted share among neighbors that carry that label (not a raw count such as 71/100). Hard evidence from Section 6.2 wins when both apply. For a binary event node, let **p_ref** be the network’s marginal P(node = Yes) before this evidence. Virtual evidence uses a likelihood ratio so that, when this is the only soft fact on the node, the updated belief in Yes is **c** [16]:

LR = [c / (1 − c)] / [p_ref / (1 − p_ref)].

pyAgrum receives a likelihood vector proportional to [LR, 1] on {Yes, No}.

**Injury and damage (multi-state).** Section 6.4 uses the **same** k neighbors. It reads each neighbor’s coded injury and damage, forms similarity-weighted counts with Laplace smoothing (α = 0.5), and normalizes to a distribution **f_q** over the four levels (fatal, serious, minor, none for injury; destroyed, substantial, minor, none for damage). For each state j, let **p_ref(j)** be the network marginal on that node before neighbor evidence. Virtual evidence uses

L(j) = f_q(j) / p_ref(j),

(re-normalized over states for inference) so Jeffrey conditioning moves the posterior toward **f_q** without treating **f_q** as a hard clamp. This is the BN-SEV path used for prognosis in Sections 6.5 and 8.1.

**One update only.** Hard evidence, event virtual evidence, and injury/damage virtual evidence are combined in one inference pass for ablations; merging event and severity virtual evidence from the **same** narrative in one update double-counts the story and hurts accuracy (Section 8.1). Diagnosis on held-out accidents uses hard and event virtual evidence for the BN path and retrieval over neighbors as the primary category readout (Section 8.2).

**DELETE** from old §2.3 if redundant: “We don’t enter raw 71/100”; “We tried one update … but it failed” → replaced by clearer sentence above.

---

## Final grep (whole document)

`soft evidence`, `virtual severity`, `random tie`, `81 of 93`, `verify all the published`, `four step`, `Zhang's four`

---

## Done order (tired-friendly)

1. §10.1 ¶2 + ¶4 + ¶5 FIND (20 min)  
2. §10.2–10.3 FIND (5 min)  
3. Abstract + Intro one paragraph (10 min)  
4. §2.3 paste (15 min)  
5. Save; grep tomorrow if needed  
