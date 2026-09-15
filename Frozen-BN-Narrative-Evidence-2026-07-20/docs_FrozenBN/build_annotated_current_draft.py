#!/usr/bin/env python3
"""Annotate the CURRENT draft (July 26 2026 Draft NTSB Paper 72126.docx) with
corrective yellow GUIDE blocks, using the leak-safe results as of 2026-07-29
(docs_FrozenBN/RESULTS_SECTION.md) and the structural-mapping verdict from the
2026-07-28 Jesse meeting deck.

This does NOT rewrite any of the user's own prose. It only inserts
instruction blocks (same convention as build_restructured_draft.py) pointing
out: (a) content that is now stale/incorrect and must not be submitted as-is,
(b) a new results subsection that is missing entirely, (c) where the archived
old-draft material begins and why it should not be built on.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      docs_FrozenBN/build_annotated_current_draft.py
"""
from __future__ import annotations

import docx
from docx.enum.text import WD_COLOR_INDEX

SRC = "/Users/kanushetkar/Desktop/July 26 2026 Draft NTSB Paper 72126.docx"
OUT = ("/Users/kanushetkar/Desktop/Vanderbilt_University/Internships/RRR/"
       "Current_Projects/NTSB/NTSB_Shivy/Frozen-BN-Narrative-Evidence-"
       "2026-07-20/paper_drafts/Draft_NTSB_Paper_ANNOTATED_7_29_26.docx")

doc = docx.Document(SRC)
paras = doc.paragraphs


def guide_before(anchor_para, lines: list[str]):
    """Insert a yellow-highlighted instruction block immediately before
    anchor_para. First line bold. Returns nothing (mutates document)."""
    # insert in reverse so they land in the given order
    ref = anchor_para
    inserted = []
    for line in lines:
        p = ref.insert_paragraph_before(line)
        r = p.runs[0] if p.runs else p.add_run(line)
        r.font.highlight_color = WD_COLOR_INDEX.YELLOW
        inserted.append(p)
    if inserted:
        inserted[0].runs[0].bold = True


def find(text_starts_with: str, occurrence: int = 0):
    hits = [p for p in doc.paragraphs if p.text.strip().startswith(text_starts_with)]
    return hits[occurrence]


# ------------------------------------------------------------- top banner
guide_before(paras[0], [
    "[UPDATED 2026-07-29 - READ THIS BLOCK FIRST, delete when done]",
    "This pass adds correction blocks on top of your July 22/26 draft. It "
    "does not touch your prose. Numbers below are sourced from "
    "docs_FrozenBN/RESULTS_SECTION.md (updated 2026-07-29) and the "
    "2026-07-28 Jesse meeting deck (docs_FrozenBN/2026-07-28_jesse_update.md) "
    "-- both are newer than the 92.9%/81.4% and structural-mapping numbers "
    "already in this file. Three things changed since you wrote this draft:",
    "(1) The severity numbers you have (93%/81% in the Discussion, 92.9%/"
    "81.4% in the old draft below) were computed BEFORE a leakage fix. "
    "Corrected, leak-safe numbers are 90.9% injury / 77.4% damage. The old "
    "numbers must not appear in the paper as results.",
    "(2) A new result exists that is not in this draft at all: diagnosis "
    "(cause-category prediction) on the same 296 held-out accidents, "
    "84.2% top-1. This needs its own results subsection.",
    "(3) Structural mapping was tested far more exhaustively after you "
    "wrote the old draft's Section 7-9 (7 variants total, including trying "
    "it as a hard replacement for retrieval, not just a rerank). It never "
    "beat the baseline and one variant was actively harmful. It is a "
    "confirmed negative result, not the 'small consistent positive' the "
    "old draft's Section 9 describes -- the framing needs to flip before "
    "it goes in Appendix A.",
])

# ------------------------------------------------------- breadth paragraph
anchor = find("-analyses of risks of transportation safety")
guide_before(anchor, [
    "[GAP - breadth paragraph still stub questions, not written]",
    "These four lines and 'Bayesian networks have been widely utilized...' "
    "below are still questions/placeholders, not prose. Checklist refs "
    "13-15 (Ale et al. CATS, Luxhoj 2003, Ancel et al. 2015) and ref 12 "
    "(Zhang & Mahadevan DSS 2019, pairs with your LR baseline) are the "
    "citations to build this paragraph from. See paper_revision_checklist"
    ".docx for links and what each abstract says.",
])

# ------------------------------------------------------- 4.1 reproduction
anchor = find("Zhang\u2019s methodology")
guide_before(anchor, [
    "[BIGGEST REMAINING WRITING JOB - Section 4.1 is bullets only]",
    "Everything from here through 'One pointer sentence: fidelity "
    "validated...' (including the 'I DON'T GET THIS PART' note) is still "
    "outline, not prose. This is the section that makes your reproduction "
    "claim in 5.1 credible, so it can't stay as bullets. Write: (a) "
    "Zhang's recipe in your own words (priors = event count / BTS "
    "departures, his Eq. 6; edge strength via co-occurrence; Beta-CDF "
    "conditional probabilities); (b) your pyAgrum implementation and the "
    "deterministic tie-break difference; (c) person-finding nodes and "
    "multi-state severity nodes, and WHY each was needed (cite "
    "tests/bn_upgraded.py, docs/ALL_TABLES_EXACT_COMPARISON.md). On the "
    "'I DON'T GET THIS PART' line: ask in the group chat or re-read RESS "
    "Section 4.3-4.4 (the CPT pruning to 12 parents, ranked by P(w|e)) "
    "before writing this paragraph -- don't guess.",
])

# ------------------------------------------------ held-out evaluation stub
anchor = find("On 296 held out accidents (2007-2019), we predict")
guide_before(anchor, [
    "[CRITICAL - use these leak-safe numbers, not any earlier figure]",
    "Source: docs_FrozenBN/RESULTS_SECTION.md \u00a75.2-5.3 (2026-07-29), "
    "outputs/heldout_significance.md. All numbers below are AFTER outcome "
    "phrases (\"was destroyed\", \"received fatal injuries\") are redacted "
    "from every narrative before any embedding or parsing -- this is why "
    "they differ from the 92.9%/81.4% figures elsewhere in this file.",
    "Severity table to build (4-class top-1 accuracy / Macro-F1, n=296, "
    "bootstrap 95% CI, exact McNemar):",
    "  Majority class (prior): 58.4% / 0.184 injury; 42.6% / 0.149 damage",
    "  Parsed events only (hard+soft): 82.4% / 0.422 injury; 50.7% / 0.309 "
    "damage",
    "  Supervised LR, parsed features: 85.5% / 0.440 injury; 60.1% / 0.408 "
    "damage",
    "  Supervised LR, TF-IDF text: 92.2% / 0.603 injury; 73.3% / 0.621 "
    "damage",
    "  Supervised LR, narrative embedding: 91.6% / 0.474 injury; 74.0% / "
    "0.543 damage",
    "  Narrative -> BN, k-NN severity readout (bn-sev, OURS, primary "
    "result): 90.9% / 0.470 injury; 77.4% / 0.697 damage",
    "State plainly (Maha will check for this): the chain beats the "
    "network prior by +32.5 pts injury / +34.8 pts damage (Holm-adjusted "
    "p < 1e-4 both) and beats the parsed-feature LR (Holm p = 0.006 "
    "injury, p < 1e-4 damage); it TIES the two strongest supervised text "
    "baselines (embedding-LR and TF-IDF-LR, all Holm p >= 0.29) -- it does "
    "not beat them, and TF-IDF-LR is numerically the best injury "
    "predictor. The chain IS numerically best on damage and clearly best "
    "on damage Macro-F1 (0.697).",
    "Also state, in the same paragraph, the honesty point: bn-sev and a "
    "pure k-NN readout (retrieval-sev, no network at all) are identical "
    "on all 296 accidents (0 discordant) -- the network contributes ZERO "
    "predictive accuracy here. The accuracy comes entirely from the "
    "narrative-retrieval signal; the network's contribution is the "
    "reasoning layer (joint queries, what-ifs, explanations) at zero "
    "accuracy cost. Do not imply the BN improves accuracy anywhere in "
    "this section.",
    "Severe-outcome screening (fatal-or-serious vs rest, useful for "
    "triage framing): injury 93.5% sensitivity / 96.8% specificity; "
    "damage 75.3% / 89.8%.",
])

# ---------------------------------------------- NEW diagnosis-on-296 section
anchor = find("Scenario analyses")
guide_before(anchor, [
    "[NEW SUBSECTION - MISSING ENTIRELY - insert before Scenario analyses, "
    "after held-out severity]",
    "Title suggestion: 'Diagnosis: held-out cause-category prediction'. "
    "Source: RESULTS_SECTION.md \u00a75.4, outputs/diagnosis_heldout_eval.md, "
    "outputs/diagnosis_emb_lr.md. This result is not in your draft at all "
    "yet and it is a real, current headline number -- do not skip it.",
    "Context to explain first: NTSB changed its coding taxonomy in 2008 "
    "(legacy subject codes -> CICTT), so exact cause-code matching across "
    "the 1982-2006 / 2007-2019 split is impossible by design. Evaluation "
    "is therefore at the four CICTT top-level categories (Personnel, "
    "Aircraft, Environment, Organizational); a prediction counts as "
    "correct if its top category is among an accident's coded causes "
    "(n = 253).",
    "Table to build (top-1 accuracy, 95% CI, mean reciprocal rank):",
    "  Category frequency baseline: 45.8% [39.5, 52.2], MRR 0.685",
    "  Frozen BN, event evidence alone: 57.7% [51.4, 63.6], MRR 0.759",
    "  Narrative retrieval, zero-parameter (OURS, primary): 84.2% "
    "[79.4, 88.5], MRR 0.915",
    "  Supervised LR, embedding: 88.1% [84.2, 91.7], MRR 0.936",
    "Be honest about both halves of the BN event-path result (57.7%): it "
    "is significantly above the frequency baseline (Holm p = 0.0007) -- a "
    "real structured-inference result -- AND simultaneously a partial "
    "negative result, losing to plain retrieval by 26.5 points. Report "
    "both, don't lead with only the flattering half.",
    "Also disclose: recall is balanced across the three common categories "
    "(68/62/72% Personnel/Aircraft/Environment) but no method recovers "
    "the rare Organizational class (25 cases); the supervised embedding "
    "model beats retrieval by 3.9 points (p = 0.041, exploratory) -- the "
    "price of retrieval needing zero labeled training; the category "
    "rollup itself is a documented keyword rule set, not independently "
    "validated coding -- its influence is bounded at <=2 accuracy points "
    "(6.2% of mapped findings are contested), same direction for every "
    "predictor, so no claimed ordering can flip.",
])

# --------------------------------------------------- ablations / struct map
anchor = find("Structural reranking")
guide_before(anchor, [
    "[UPDATE - structural mapping is now a stronger, more exhaustively "
    "tested negative result than earlier notes assumed]",
    "Source: docs_FrozenBN/2026-07-28_jesse_update.md, slides 8-11 (also "
    "outputs/structmap_final_verdict/REPORT.md). Since the old Section 7-9 "
    "material (in the archived block below) was written, structural "
    "mapping was tested in SEVEN forms, not just the top-50 rerank: "
    "rerank top-50 (the original A2), bypass to single best match, wider "
    "pool then rerank, struct-first over ALL accidents (not just top-50), "
    "hybrid embedding (structure folded into the embedding itself), and "
    "struct-weighted severity voting on the main 296-accident task.",
    "None of the seven beat the embedding-only baseline. One (best-match "
    "bypass) was actively HARMFUL: -1.6 to -2.6 points on diagnosis "
    "(p < 0.001). Struct-first over all accidents was not just neutral on "
    "diagnosis, it was catastrophic on severity (-11.8 pts injury / -10.5 "
    "pts damage vs the no-structure baseline). The single best-looking "
    "struct-fused cell (+1.4 pp damage, p = 0.125) was indistinguishable "
    "from pool-size noise (changing k alone with NO structure gave +1.0 "
    "pp) -- 1 flattering result out of 28 tested cells is not signal.",
    "Write this as: 'we tested structural mapping in seven architectural "
    "forms (Appendix A); none improved on the embedding baseline, and one "
    "was actively harmful; we conclude the causal-chain representation is "
    "too coarse to add signal beyond what the narrative embedding already "
    "captures (structural-score correlation with correctness rho ~ "
    "0.01).' Do NOT reuse the old draft's language ('a small consistent "
    "positive', 'doesn't negatively impact') -- that framing is now "
    "contradicted by the fuller test suite.",
])

# --------------------------------------------------- discussion 93% flag
anchor = find("Held out injury 93%")
guide_before(anchor, [
    "[STOP - do not submit this number]",
    "'93%' here (and '~87.5%' for the LR comparison) is the pre-leakage-"
    "fix figure. docs_FrozenBN/RESULTS_SECTION.md explicitly lists 93%/81% "
    "under 'Numbers you must NOT put in the paper'. Replace this whole "
    "bullet with: the leak-safe chain reaches 90.9% injury / 77.4% "
    "damage, statistically indistinguishable from the strongest "
    "supervised baselines (Holm p >= 0.29) while needing no training "
    "data. See the corrected table inserted above Section 5.3.",
])

# --------------------------------------------------- declarations reminder
anchor = find("Declarations")
guide_before(anchor, [
    "[DECLARATIONS - agreed wording, from the earlier revision checklist]",
    "Four blocks needed: (1) CRediT author statement; (2) Declaration of "
    "competing interest (standard 'none' sentence); (3) Data availability "
    "(NTSB data are public; decide with Maha whether code is shared on "
    "request or via repository); (4) Generative-AI-use declaration, "
    "wording agreed with Maha: 'AI tools were used to assist with "
    "software development and editing suggestions; all methodology, "
    "analysis, and final text are the authors'; the authors reviewed and "
    "take full responsibility for all content.' Put it in your own words "
    "and match RESS/Elsevier's exact required format/heading.",
])

# --------------------------------------------------- archive boundary
anchor = find("Ap(Old version)")
guide_before(anchor, [
    "[ARCHIVE - DO NOT BUILD ON OR SUBMIT ANYTHING BELOW THIS LINE AS-IS]",
    "Everything from here to the end of the document is your July 21 "
    "full-prose draft, kept as raw material per the original restructuring "
    "guide. Three concrete problems with it, as of 2026-07-29:",
    "(1) Severity numbers (92.9%/81.4% in its Section 5.3-era text, if "
    "present) are pre-leakage-fix -- see the correction above Section 5.3.",
    "(2) Its Section 7-9 present structural mapping (A2) as a working, "
    "mildly positive method ('A2 is marginally better than A0 on every "
    "metric'). This is now contradicted by the fuller 7-variant test -- "
    "see the correction above 'Structural reranking'.",
    "(3) Its worked-example numbers (Section 8, engine fire / landing "
    "gear) were computed with the FUSED structural+cosine score, which "
    "the current pipeline does not use -- a reviewer who recomputes Step "
    "2 in that walkthrough would get different numbers.",
    "What is still usable: the two worked-example SCENARIOS themselves "
    "(the engine-fire and landing-gear query framing) as illustrations, "
    "IF recomputed with the current cosine-only pipeline before they go "
    "back in the paper -- do not copy the numbers as they stand. The "
    "reference list at the very end (5 entries) is still valid input to "
    "merge into your final References section.",
])

doc.save(OUT)
print(f"saved: {OUT}")
