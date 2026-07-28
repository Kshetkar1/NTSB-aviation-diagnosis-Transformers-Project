"""Generate docs/tuesday_jul14_update.pptx — Tuesday advisor deck (30 min).

Built ON TOP of the user's own Google-Slides deck (same theme/master/layouts),
so it looks exactly like the decks they normally make:
  template: /Users/kanushetkar/Downloads/Copy of NTSB project update 6_30_26.pptx
  style   : white background, Arial, plain title top-left, short bullets,
            plain tables, no decorations.

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 docs/build_tuesday_jul14_pptx.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

TEMPLATE = Path("/Users/kanushetkar/Downloads/Copy of NTSB project update 6_30_26.pptx")
OUT = Path(__file__).resolve().parent / "tuesday_jul14_update.pptx"

prs = Presentation(str(TEMPLATE))

# ---- wipe the template's slides, keep its theme/layouts ---------------------
xml_slides = prs.slides._sldIdLst
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
for sld in list(xml_slides):
    prs.part.drop_rel(sld.get(R))
    xml_slides.remove(sld)

LAYOUTS = {l.name: l for l in prs.slide_layouts}
TITLE_LAYOUT = LAYOUTS.get("TITLE", prs.slide_layouts[0])
OBJECT_LAYOUT = LAYOUTS.get("OBJECT", prs.slide_layouts[1])


def _placeholder(slide, want_title):
    for ph in slide.placeholders:
        is_title = ph.placeholder_format.idx == 0 or "title" in ph.name.lower()
        if is_title == want_title:
            return ph
    return None


def add_content_slide(title, bullets=None, note=None):
    """bullets: list of (text, level) or (text, level, dict-of-run-opts)."""
    s = prs.slides.add_slide(OBJECT_LAYOUT)
    t = _placeholder(s, True)
    if t is not None:
        t.text_frame.text = title
    body = _placeholder(s, False)
    if bullets and body is not None:
        tf = body.text_frame
        tf.clear()
        first = True
        for item in bullets:
            text, level = item[0], item[1]
            opts = item[2] if len(item) > 2 else {}
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.level = level
            r = p.add_run()
            r.text = text
            if opts.get("bold"):
                r.font.bold = True
            if opts.get("size"):
                r.font.size = Pt(opts["size"])
    elif body is not None:
        body.text_frame.clear()
    if note:
        s.notes_slide.notes_text_frame.text = note
    return s


def add_table(slide, rows, left, top, width, height, col_widths=None,
              font_size=13, bold_header=True):
    nrows, ncols = len(rows), len(rows[0])
    gfx = slide.shapes.add_table(nrows, ncols, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    tbl = gfx.table
    if col_widths:
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Inches(w)
    for r in range(nrows):
        for c in range(ncols):
            cell = tbl.cell(r, c)
            para = cell.text_frame.paragraphs[0]
            run = para.add_run()
            run.text = str(rows[r][c])
            run.font.size = Pt(font_size)
            run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)
            if r == 0 and bold_header:
                run.font.bold = True
    return tbl


# ============================================================ 1: TITLE
s = prs.slides.add_slide(TITLE_LAYOUT)
tp = _placeholder(s, True)
if tp is not None:
    tp.text_frame.text = "NTSB Project Update July 14th, 2026"
sp = _placeholder(s, False)
if sp is not None:
    sp.text_frame.text = "Kanu Shetkar"

# ============================================================ 2: PREV + TODAY
add_content_slide(
    "Previous Meeting",
    [
        ("Table 4 → “how far apart are they from his numbers?”", 0),
        ("“if they don't match, at least we should know WHY they don't match”", 0),
        ("“what is he counting versus what are you counting?”", 0),
        ("Explain the comparison panel (neutral vs similarity) properly", 0),
        ("", 0),
        ("Today’s Meeting", 0, {"bold": True, "size": 28}),
        ("Table 4 → counted every cell + why they don’t match + the real version", 0),
        ("What we changed (Part A: clusters, Part B: law of total probability)", 0),
        ("Diagnosis & Prognosis trees (Streamlit)", 0),
        ("Why the remaining tables need a Bayesian network → Friday", 0),
    ],
    note=("Close out the Table 4 questions from last week first, then the two "
          "changes in the pipeline, then the live demo, then the road to Friday."))

# ============================================================ 3: TABLE 4 COUNTED
s = add_content_slide(
    "Table 4  P(fire | brake wear, wiring overheating) — counted",
    note=("SAY: 'Here is the table with what we counted and Zhang's numbers, using "
          "his exact two parents. You can see the first one is close and works — "
          "it's 1 out of 1, which his own sparse cap turns into 0.95, right next to "
          "his 0.99. But the other three are not close at all.' Then bridge: 'So the "
          "next question is WHY — and the paper itself answers that.'\n\n"
          "BACKUP if he asks 'did you try other windows or estimators?': same "
          "result on the full corpus (89/2,212 = 0.040), with his own Eq. 8/9 "
          "estimator (0.39), and with his Beta-CDF recipe (0.39) — no method "
          "reaches 0.93 / 0.95 / 2e-9. These counts use the same conventions that "
          "reproduce Table 7 exactly."))
add_table(s, [
    ["brake", "wiring", "n/N", "Direct counting", "Zhang Table 4"],
    ["YES", "YES", "1/1", "0.950 (sparse → capped)", "0.99"],
    ["YES", "NO", "2/9", "0.222", "0.93"],
    ["NO", "YES", "10/21", "0.476", "0.95"],
    ["NO", "NO", "89/1,711", "0.052", "2 x 10^-9"],
], left=1.4, top=1.9, width=10.5, height=3.4,
   col_widths=[1.5, 1.5, 1.8, 3.4, 2.3], font_size=16)
tb = s.shapes.add_textbox(Inches(1.4), Inches(5.8), Inches(10.5), Inches(0.8))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run()
r.text = "First cell: close (sparse cap) — the other three: not close at all"
r.font.size = Pt(18)
r.font.bold = True

# ============================================================ 4: WHY NO MATCH
add_content_slide(
    "So where do the Table 4 numbers come from?",
    [
        ("My counts didn’t match — so I went looking for where his numbers could come from", 0),
        ("The paper’s own math (Eq. 9, p.8) computes this same conditional:", 0),
        ("P(fire | wiring overheating) = 1/102 = 0.0098   …but Table 4 prints 0.95", 1),
        ("The paper’s own words (p.7):", 0),
        ("“No record about the joint occurrence of overheating of electric wiring and "
         "landing gear brake system wear-out…” → nothing to count for the key cell", 1),
        ("The paper’s own demonstration (Fig. 3):", 0),
        ("Tables 3+4+5 + Bayes’ rule reproduce every Fig. 3 number exactly → "
         "Table 4 is the INPUT to the Section 3 walkthrough", 1),
        ("Wherever the paper counts, my counts match → Table 4 was never counted", 0,
         {"bold": True}),
    ],
    note=("Tell it as a story, in this order: 'My counts didn't match, so instead of "
          "assuming I was wrong, I went looking for where his numbers could come "
          "from. First I checked the paper's own math: Equation 9 on page 8 computes "
          "this exact conditional — fire given wiring overheating — and gets 1 over "
          "102, about 0.01. Table 4 prints 0.95 for the same thing. So the paper "
          "disagrees with itself by a factor of a hundred. Then I checked the "
          "paper's own words: page 7 says there is no record of wiring overheating "
          "and brake wear happening together — so the key cell literally has nothing "
          "to count; I verified that, it's empty. Then the last clue: if you take "
          "Tables 3, 4 and 5 and just apply Bayes' rule, you reproduce every number "
          "in the Figure 3 demonstration exactly. That's what Table 4 is FOR — it's "
          "the input to the Section 3 teaching walkthrough, not an output from the "
          "data.' Then the closing line: 'So wherever the paper counts — Equation 9, "
          "Table 7, the priors — my counts match it to the digit. Table 4 was never "
          "counted.' Then PAUSE and ask: 'Does that match your reading of it?'\n\n"
          "BACKUP if he pushes back: (1) 2e-9 arithmetic — a counted ratio of 2e-9 "
          "would need ~1 fire per 500 million records; the dataset has 2,243. "
          "(2) His released code & network: 0.99/0.93/2e-9 appear nowhere, and his "
          "generated Fire node doesn't even have brake+wiring as parents. "
          "NEVER say 'made up' or 'fake' — say 'illustrative' / 'the Section 3 "
          "teaching example' / 'not produced by counting.'"))

# ============================================================ 5: REAL TABLE 4
s = add_content_slide(
    "The real Table 4  P(fire | wiring, fuel) — goes in the paper",
    note=("SAY: 'This is the constructive ending — Table 4's exact format, filled "
          "entirely from real data, two independent ways. Left lane = direct "
          "counting: count(fire & combination)/count(combination), same conventions "
          "as Table 7. Right lane = Zhang's own Section 4.3 machinery, which I "
          "implemented myself. Both lanes agree on the picture: the both-present "
          "cell is sparse (one incident), the singles sit around 0.4-0.5, and the "
          "neither cell is the ~4% base rate. This is the version that goes in our "
          "paper.'\n\n"
          "IF HE ASKS 'how did you do Zhang's recreation numbers?' — exact recipe:\n"
          "1) SINGLE cells = his Eq. 9 edge ratio: accidents where the cause LEADS "
          "TO fire / accidents containing the cause. Wiring: 11/28 = 0.393. Fuel "
          "family: 18/51 = 0.353. (Differs from the left lane's 10/21 and 23/45 "
          "because Eq. 9 counts event-sequence edges, direct counting conditions on "
          "the finding being present — two definitions, both countable.)\n"
          "2) BOTH-present cell = his multi-parent Beta-CDF rule (constructCPT): "
          "contribution = these two parents' ratios as a share of ALL fire-parent "
          "ratios = 0.0193 → Beta-CDF(0.0193) = 0.033 → floored by the largest "
          "active single ratio → max(0.033, 0.393) = 0.393. That's why both-present "
          "equals the wiring cell.\n"
          "3) NEITHER cell = 0 — his constructCPT returns 0 when no parent is "
          "active.\n"
          "4) The 0.95 cap: any raw ratio of exactly 1.0 gets multiplied by 0.95 — "
          "that rule is from HIS released code, not mine.\n"
          "Short version: 'Singles are his Eq. 9 edge ratios — 11/28 and 18/51. The "
          "both-cell is his Beta-CDF rule floored by the biggest single: 0.393. The "
          "neither-cell is zero by his constructCPT. Every number is a count you "
          "can check.'"))
add_table(s, [
    ["wiring", "fuel", "n/N", "Direct counting", "Zhang’s method (Sec. 4.3)"],
    ["YES", "YES", "1/1", "0.950 (sparse → capped)", "0.393"],
    ["YES", "NO", "10/21", "0.476", "0.393"],
    ["NO", "YES", "23/45", "0.511", "0.353"],
    ["NO", "NO", "68/1,675", "0.041", "0.000"],
], left=0.8, top=1.45, width=10.4, height=2.5,
   col_widths=[1.2, 1.2, 1.6, 3.0, 3.4], font_size=14)
tb = s.shapes.add_textbox(Inches(0.8), Inches(4.25), Inches(11.8), Inches(2.7))
tf = tb.text_frame; tf.word_wrap = True
_bullets5 = [
    ("Same format as Zhang’s Table 4 — every cell counted from real NTSB data "
     "(1,742 accidents, 1982–2006)", False),
    ("Two independent methods agree — direct counting and Zhang’s own Sec. 4.3 "
     "recipe (Eq. 9 + Beta-CDF + 0.95 cap)", False),
    ("The 1/1 cell shows the sparse-data problem live — one observation shouldn’t "
     "produce certainty → the 0.95 cap engages", False),
    ("This is the version that goes in our paper", True),
]
for _i, (_t, _bold) in enumerate(_bullets5):
    p = tf.paragraphs[0] if _i == 0 else tf.add_paragraph()
    p.space_after = Pt(6)
    r = p.add_run()
    r.text = "\u2022  " + _t
    r.font.size = Pt(15)
    r.font.bold = _bold

# ============================================================ 6: WHOLE PROCESS
s = add_content_slide(
    "The Whole Process",
    note=("SAY (tell it as one continuous story): 'Let me walk you through the "
          "whole process on one real example, start to finish. Imagine an "
          "investigator types in: engine caught fire during takeoff.\n\n"
          "First, the system reads that query and detects the outcome — fire. And "
          "here's the anchor: that maps to the exact same 102 fire accidents as "
          "your Table 7. Not a similar set — the same 102. Everything downstream "
          "is built on that population.\n\n"
          "Second, we embed the query — turn it into a vector — and score EVERY "
          "incident in the dataset by similarity to it. Not a top-400 cutoff — "
          "every incident gets a score. That's the ranked list you see in the app: "
          "each row is one past accident, its score, and a snippet of its "
          "narrative. So for our query, the top of the list is full of engine-fire "
          "accidents, as you'd expect.\n\n"
          "Third — the partition, and this is the first thing we changed. The 102 "
          "fires are already grouped into fifteen clusters of similar accidents: "
          "for example, cabin fires is the biggest with 32 accidents, electrical "
          "fires has 21, engine fires has 16. Every fire belongs to exactly one "
          "cluster — no overlaps, no gaps. The fix was the gaps: nineteen fires "
          "used to have no cluster at all. Now it's 102 out of 102 — I'll show "
          "that on the next slide.\n\n"
          "Fourth — the second change, and the important one. Inside each cluster "
          "we ask: given an accident in this cluster, what were the causes? So "
          "inside the electrical cluster, wiring shows up at a high rate; inside "
          "the engine cluster, loss of engine power does. Before, I computed these "
          "with my own counting rules. Now it is exactly your Table 7 counting — "
          "same cause labels, same contributory-factor rule, same denominator "
          "convention — just restricted to the cluster.\n\n"
          "Fifth — the cluster weights. This is the ONLY place the query enters "
          "the math. Two modes. Neutral: the query adds nothing, each cluster is "
          "weighted by its size — cabin fires gets 32 over 102, about 31 percent. "
          "Similarity: each cluster is weighted by how much similarity score its "
          "members got — so for our engine query, the engine-fire clusters get "
          "extra weight.\n\n"
          "Sixth — the law of total probability puts it together: the probability "
          "of a cause given the query is the sum, over the fifteen clusters, of "
          "the probability of the cause within the cluster, times the weight of "
          "the cluster. And this is the guarantee: with neutral weights the "
          "algebra collapses and you get your published Table 7 back exactly — "
          "airframe 0.3137, wiring 0.0882, all 85 causes. With similarity weights, "
          "the answer tilts toward the query — loss of engine power rises from "
          "0.088 to 0.110 for this query — and that difference IS what the "
          "narrative added.\n\n"
          "And seventh, from that distribution we build the diagnosis tree — "
          "causes of causes — and the prognosis tree, which I'll show you live.\n\n"
          "One thing to be completely clear about: similarity never invents a "
          "probability. It only decides which clusters get more weight. Every "
          "probability in this pipeline is a count over real accidents.'\n\n"
          "PAUSE: 'Does this picture make sense before I show the two changes in "
          "detail?'\n\n"
          "LIKELY QUESTIONS — pocket answers:\n"
          "Q1 'Where do the 15 clusters come from? Who decided them?' -> 'They're "
          "precomputed, before any query — we embed every accident's narrative and "
          "cluster the embeddings, so accidents describing similar situations group "
          "together. The query never changes the clusters, only their weights.'\n"
          "Q2 'What exactly are Table 7 rules?' -> 'Only findings coded by "
          "investigators as Cause or Factor count; each cause counted once per "
          "accident; denominator = number of accidents in the cluster. Same three "
          "conventions as your published table — just restricted to the cluster.'\n"
          "Q3 'Why does neutral give Table 7 back EXACTLY?' (the prove-it moment — "
          "memorize this) -> 'Because the sizes cancel. Each cluster contributes "
          "count-of-cause-in-cluster over N_K, times N_K over 102. The N_K cancels, "
          "so the sum is just total count over 102 — literally the Table 7 cell. "
          "It's algebra, not approximation.'\n"
          "Q4 'How is the similarity weight computed, exactly?' -> 'Add up the "
          "similarity scores of the cluster's members, then normalize across "
          "clusters so the weights sum to one. A cluster full of accidents that "
          "look like the query gets more mass.'\n\n"
          "WATCH OUT (from practice): (1) Step 5 neutral weight is N_K over 102 — "
          "NOT 'N divided by 2'. (2) Step 4 counts WITHIN the 102 fires — the 184M "
          "flights denominator belongs only to the prior P(fire). (3) Say 'law of "
          "total probability', not 'chain rule'. (4) Retrieval scores EVERY "
          "incident — the app's breadth slider only limits the retrieval-counting "
          "comparison lane, and the LTP always uses all 102 fires."))
add_table(s, [
    ["", "What happens", "Ex: “engine caught fire during takeoff”"],
    ["1. Detect outcome", "Parse the query for an NTSB occurrence",
     "fire → 102 fire accidents (Zhang’s Table 7 denominator)"],
    ["2. Embed + retrieve", "Embed query; score EVERY incident by similarity",
     "every incident gets a similarity score — the ranked list in the app"],
    ["3. Partition  (CHANGED)", "Split the 102 fires into precomputed clusters "
     "(the query does NOT change them)",
     "15 clusters: cabin fire (32), electrical (21), engine fire (16)…"],
    ["4. P(cause | K)  (CHANGED)", "Zhang’s Table-7 counting INSIDE each cluster",
     "his labels • contributory-only • denominator = cluster size"],
    ["5. P(K | query)", "Cluster weights — the ONLY place the query enters",
     "neutral: N_K/102  •  similarity: cluster’s share of similarity scores"],
    ["6. Law of total probability", "P(cause|Q) = Σ_K P(cause|K)·P(K|Q)",
     "neutral → Table 7 EXACTLY; similarity → shifts toward the query (auditable)"],
    ["7. Build tree", "Recurse: causes of causes, path probabilities",
     "diagnosis / prognosis trees (Streamlit)"],
], left=0.55, top=1.2, width=12.2, height=4.8,
   col_widths=[2.5, 4.4, 5.3], font_size=12)

# ============================================================ 7: CHANGED PART A
s = add_content_slide(
    "What we Changed Part A — cluster coverage",
    note=("SAY: 'So the first change. This one we found ourselves, while "
          "validating. When I audited the clusters, nineteen of the 102 fire "
          "accidents had no cluster at all — they were invisible to the "
          "law-of-total-probability step. The math was running on 83 fires and "
          "silently ignoring nineteen. Why? Those nineteen are older records: they "
          "have all the investigator-coded data — occurrence codes, findings — but "
          "no written narrative text. The original pipeline created clusters by "
          "embedding the narrative; no narrative, nothing to embed, no cluster. Why "
          "does it matter? Because the guarantee on the next slide — getting your "
          "Table 7 back exactly — requires the clusters to cover all 102 fires. "
          "It's a partition argument: every fire in exactly one bucket. Nineteen "
          "fires in no bucket = a hole in the algebra. The fix: for those records I "
          "build a short pseudo-narrative from the data they DO have — the coded "
          "occurrence sequence and findings — embed it with the same model as "
          "everyone else, and assign each one to its nearest existing cluster. No "
          "new information is invented — same investigator-coded facts, just made "
          "visible to the same embedding step. So now: 102 out of 102. And I "
          "re-ran every validation after this change — everything still passes.'\n\n"
          "LIKELY QUESTIONS:\n"
          "Q 'Doesn't the pseudo-narrative fabricate data?' -> 'No — it only "
          "restates fields already in the record, written by the investigators. "
          "Nothing is added; the record just becomes readable to the embedding "
          "model.'\n"
          "Q 'Does this change Table 7?' -> 'No. Table 7 counting never used "
          "clusters — it's unchanged and still exact. This only completes the "
          "partition the LTP layer sits on.'\n"
          "Q 'How do you pick which cluster they go to?' -> 'Nearest cluster "
          "centroid in embedding space — the same distance measure the original "
          "clustering used.'"))
add_table(s, [
    ["What", "June Draft", "Why it was a problem", "Now"],
    ["Fires with a cluster", "83 / 102",
     "19 legacy records have NO narrative text → never embedded → not clustered → "
     "invisible to the LTP step",
     "102 / 102"],
    ["Fix", "—",
     "",
     "pseudo-narrative from coded findings → embedded with same model → nearest cluster"],
    ["Why it matters", "partition had holes",
     "exact recovery (Part B) requires ALL 102 fires covered",
     "partition complete → algebra works"],
], left=0.55, top=1.4, width=12.2, height=3.2,
   col_widths=[2.2, 2.0, 4.6, 3.4], font_size=13)

# ============================================================ 8: CHANGED PART B
s = add_content_slide(
    "What we Changed Part B — law of total probability",
    note=("SAY: 'Now the second change — the important one. This goes straight "
          "back to the question you asked me: are your probabilities actually "
          "close to Zhang's? Honest answer: before, they weren't. The structure "
          "was right — I was already using the law of total probability over "
          "clusters, as we discussed. But INSIDE each cluster I was counting "
          "causes with my own rules — different label handling, different "
          "denominator. And the result [point at table]: for a fire query, my "
          "method gave airframe 0.059; your published table says 0.314. Across the "
          "whole distribution the distance was 1.96 out of a maximum possible 2. "
          "Not close — and worse, no reason it SHOULD be close. Here's the "
          "realization: the law-of-total-probability decomposition was never the "
          "problem — the problem was what was inside it. So we kept the structure "
          "and swapped the inside: the probability of a cause given a cluster is "
          "now computed with exactly your Table 7 estimator — same cause labels, "
          "same contributory-factor rule, same denominator — just restricted to "
          "that cluster's accidents. And that buys something I can prove, not just "
          "claim. When the query carries no information — neutral weights — the "
          "cluster sizes cancel algebraically and the output is your published "
          "Table 7, exactly: L1 distance zero, machine precision, all 85 causes. "
          "There's an offline test that checks every cause, and it passes. And "
          "when the query DOES carry information — similarity weights — the "
          "output moves away from Table 7, and that movement is meaningful: for "
          "the engine-fire query it's 0.33, and I can decompose it cluster by "
          "cluster — the engine clusters got more weight, so loss of engine power "
          "rose from 0.088 to 0.110. So the summary: your Table 7 is now the "
          "zero-information special case of our method. When the narrative says "
          "nothing, we give your table back exactly. When it says something, we "
          "show precisely what it added and where.'\n\n"
          "LIKELY QUESTIONS:\n"
          "Q 'What exactly were your old counting rules?' -> 'We counted every "
          "MENTION of every finding label — one accident could contribute five "
          "labels, non-contributory included — and divided by total mentions. "
          "Zhang counts each cause once per ACCIDENT and divides by accidents. "
          "Different units, so the numbers couldn't match — airframe 0.059 vs "
          "0.314. Wrong conventions, not a bug.'\n"
          "Q 'What was the old weight formula?' -> 'A heuristic: average "
          "similarity times cluster size over total. The output wasn't a "
          "probability of anything, so there was nothing it SHOULD equal and no "
          "way to check it. The new weights are honest probabilities: neutral is "
          "N_K over 102; similarity is each cluster's share of the summed "
          "similarity scores, normalized to 1.'\n"
          "Q 'Isn't this just copying Zhang?' -> 'Inside a cluster, deliberately "
          "yes — that's what makes the two methods comparable at all. The layer on "
          "top — clusters, query weighting, the decomposition — is ours. His table "
          "is the special case; our method generalizes it.'\n"
          "Q 'How do I know exact is really exact?' -> 'A standalone test runs the "
          "LTP with neutral weights and compares every one of the 85 causes to "
          "Table 7. The L1 comes out at ten to the minus seventeen — "
          "floating-point zero.'"))
add_table(s, [
    ["What", "June Draft", "Why it was a problem", "Now"],
    ["P(C | K) cause given cluster",
     "counted every finding MENTION, divided by total mentions",
     "probabilities spread thin: airframe 0.059 vs Zhang’s 0.314",
     "Zhang’s Table-7 counting: each cause once per ACCIDENT, ÷ cluster size"],
    ["P(K | Q) cluster weight", "heuristic: avg similarity × cluster size / total",
     "not a probability → nothing it SHOULD equal, no way to check",
     "neutral = N_K/102  •  similarity = cluster’s share of similarity scores"],
    ["Guarantee", "none",
     "our distribution sat at L1 = 1.96 from Table 7 (max 2.0)",
     "neutral weights → Table 7 EXACTLY (L1 = 0, all 85 causes, tested)"],
    ["Query signal", "two things differed at once: counting AND query",
     "couldn’t tell WHICH caused a difference from Table 7",
     "counting now identical → any deviation = the query (0.33 for fire query)"],
], left=0.55, top=1.3, width=12.2, height=4.0,
   col_widths=[2.6, 2.9, 3.3, 3.4], font_size=12)

# ============================================================ 9: PANEL DECODED
add_content_slide(
    "The comparison panel — what did the narrative add?",
    [
        ("Run the query → the panel shows the answers side by side: Zhang’s Table 7, "
         "retrieval counting, and our law-of-total-probability answer", 0),
        ("First, flip to NEUTRAL — the query is silenced:", 0),
        ("our answer becomes IDENTICAL to Table 7 → L1 reads 0.000 (the calibration check)", 1),
        ("Then flip to SIMILARITY — the query speaks:", 0),
        ("the answer tilts toward accidents like ours → L1 = 0.33 = what the narrative added", 1),
        ("engine query → engine clusters gain weight → loss of engine power rises 0.088 → 0.110", 1),
        ("Sanity check: all lanes agree on the top-5 causes (5/5)", 0),
        ("L1 = distance between two distributions (0 = identical, 2 = opposite)", 1),
        ("Zhang’s Table 7 = the zero-information special case of our method", 0, {"bold": True}),
    ],
    note=("SAY (as a story — this was his other request, the panel decoded): "
          "'Last time I showed you this panel and it wasn't clear what the numbers "
          "meant, so let me tell it properly this time. The whole panel exists to "
          "answer one question: WHAT DID THE NARRATIVE ADD?\n\n"
          "When I run the query, the panel puts the answers side by side: your "
          "published Table 7, the plain retrieval counting, and our "
          "law-of-total-probability answer.\n\n"
          "Now watch what happens when I flip the toggle to NEUTRAL. That silences "
          "the query — every cluster is weighted only by its size. And our answer "
          "becomes IDENTICAL to your Table 7 — the L1 distance reads 0.000. That's "
          "the calibration check: when the narrative says nothing, we say exactly "
          "what your table says. If that number were anything but zero, something "
          "would be broken.\n\n"
          "Then I flip it to SIMILARITY, and the query gets to speak. The clusters "
          "that look like our engine-fire story gain weight, and the answer tilts: "
          "the L1 to Table 7 becomes 0.33. That 0.33 IS the narrative signal — "
          "it's not error, it's the measured amount the story changed the answer. "
          "And I can show WHERE it went: the engine clusters gained weight, so "
          "loss of engine power rose from 0.088 to 0.110.\n\n"
          "And a sanity check on the side: all the lanes still agree on the top "
          "five causes — five out of five. The tilt re-weights; it doesn't invent "
          "new causes.\n\n"
          "So one sentence to summarize the whole panel: your Table 7 is the "
          "zero-information special case of our method — and the distance from it "
          "is exactly what the narrative contributed.'\n\n"
          "[THEN GO LIVE: run the query in the app, flip neutral -> similarity, "
          "point at the L1 changing from 0.000 to 0.33, then show the diagnosis "
          "tree, then prognosis.]\n\n"
          "LIKELY QUESTIONS:\n"
          "Q 'What is L1 exactly?' -> 'Take the two probability distributions, "
          "line up all 85 causes, add up the absolute differences cell by cell. "
          "Zero means identical; two is the maximum possible — completely "
          "disjoint.'\n"
          "Q 'What is the retrieval-counting lane?' -> 'Zhang-style counting over "
          "just the retrieved incidents instead of all 102 — it sits at L1 0.248 "
          "from Table 7. It's the middle ground between his table and our LTP.'\n"
          "Q 'What is Spearman?' (a column in the app) -> 'Rank correlation — do "
          "the two lanes put the causes in the same ORDER, ignoring the exact "
          "values. 1.0 means the same ordering.'\n"
          "Q 'Is 0.33 good or bad?' -> 'Neither — it's the size of the query's "
          "influence. Zero would mean the narrative was ignored; a huge number "
          "would mean we abandoned the data. 0.33 means a real but controlled "
          "tilt, and every piece of it is auditable per cluster.'"))

# ============================================================ 10: DEMO
add_content_slide(
    "Diagnosis & Prognosis Tree",
    [("Show Streamlit app", 0)],
    note=("Run 'engine caught fire during takeoff'. Root = fire, 102 accidents. "
          "Level 1 = the Table 7 cells (wiring 9/102, LOEP 9/102, fluid/fuel 6/102, "
          "APU 5/102). Every edge: p = one-step probability, n/N = raw count, "
          "path p = product down the chain. No evacuation / emergency procedure — "
          "responses excluded per Maha's correction. Then toggle to prognosis."))

# ============================================================ 11: TREES VS BN
add_content_slide(
    "Why the remaining tables need a Bayesian network",
    [
        ("My trees: follow ONE path at a time, every hop a real count (n/N visible)", 0),
        ("Table 8, lower Table 9, Figs 11–12 need:", 0),
        ("combine ALL paths between two events at once", 1),
        ("propagate BACKWARD (observe effect → update causes → re-predict)", 1),
        ("fuse MULTIPLE evidences at the same time", 1),
        ("what-ifs: “set this prior to 0.3” → hypothetical, nothing to count", 1),
        ("Those 4 capabilities ARE a Bayesian network (nodes + CPTs + propagation)", 0),
        ("Building it from OUR OWN pipeline probabilities, open-source inference — "
         "nothing taken from Zhang’s implementation", 0),
        ("", 0),
        ("Results Friday → then 100% of the paper is matched or explained", 0, {"bold": True}),
    ],
    note=("A real mathematical boundary, not a gap in my implementation. Already "
          "exact by counting alone: Table 7 (85/85), P(fire) prior, Table 9 forward "
          "edges (0.95/0.50/0.1429), Beta-CDF fit. If asked how the BN is looking: "
          "'Promising — some published posteriors already reproduce — but I want to "
          "show it properly with slides on Friday.' Then stop."))

# ============================================================ 12: NEXT STEP
add_content_slide(
    "Next Step",
    [
        ("Finish the Bayesian network built from our own probabilities", 0),
        ("Friday → full BN construction + Table 8 / Table 9 / Figs 11–12 comparisons", 0),
        ("Then start writing the paper", 0),
        ("", 0),
        ("Anything you want prioritized for Friday?", 0, {"bold": True}),
    ],
    note=("Wrap up: Table 4 closed (counted, explained, real version built), two "
          "pipeline changes proven, panel explained, demo shown. BN is the last "
          "piece — ask Maha what to prioritize for Friday."))

# ============================================================ 13: APPENDIX
add_content_slide("Appendix", [("", 0)])

# ============================================================ 14: APPENDIX - ZHANG LANE RECIPE
s = add_content_slide(
    "Appendix — the “Zhang’s method (Sec. 4.3)” lane",
    [
        ("Single cells → his Eq. 9 edge ratio: accidents where the cause LEADS TO "
         "fire / accidents containing the cause", 0),
        ("wiring: 11/28 = 0.393      fuel family: 18/51 = 0.353", 1),
        ("(direct counting lane uses 10/21 and 23/45 — Eq. 9 counts event-sequence "
         "edges; direct counting conditions on the finding being present)", 1),
        ("Both-present cell → his multi-parent Beta-CDF rule (constructCPT):", 0),
        ("contribution = these 2 parents’ share of ALL fire-parent ratios = 0.0193", 1),
        ("Beta-CDF(0.0193) = 0.033 → floored by largest single → max(0.033, 0.393) = 0.393", 1),
        ("Neither cell → 0 (his constructCPT returns 0 when no parent is active)", 0),
        ("0.95 cap → any raw ratio of exactly 1.0 is multiplied by 0.95 (rule from "
         "HIS released code)", 0),
    ],
    note=("Jump here only if Maha asks 'how did you do Zhang's recreation numbers?'. "
          "Walk top to bottom: singles are his Eq. 9 edge ratios (11/28, 18/51); the "
          "both-cell is his Beta-CDF rule floored by the biggest single (0.393); the "
          "neither-cell is zero by his constructCPT; the 0.95 cap is from his own "
          "code. Every number is a count you can check — "
          "tests/build_table4_analogue.py prints all of them."))

prs.save(str(OUT))
print(f"Saved {OUT} with {len(prs.slides._sldIdLst)} slides")
for i, slide in enumerate(prs.slides, 1):
    title = ""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            title = shape.text_frame.text.splitlines()[0][:70]
            break
    print(f"  {i}. {title}")
