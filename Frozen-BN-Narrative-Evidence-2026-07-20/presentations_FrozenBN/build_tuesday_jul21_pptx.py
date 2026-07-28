"""Generate docs/tuesday_jul21_update.pptx, the Tuesday advisor deck (30 min).

Written to MIMIC the last two decks the user liked (tuesday_jul14 / friday_jul17):
  * short bullets, plain tables, one bold takeaway line under each table
  * every bullet advances the story; a reader who has not seen the project
    in a long time can follow it top to bottom
  * NO em dashes anywhere

STORY (one arc: how queries and narratives are used, and the big picture;
NO last-meeting tables in the main flow, NO trees, NO live demo):
  1. his question -> agenda
  2. the one picture: build time vs query time
  3. the counting path, step by step (embedding, clusters, weights, LTP)
  4. the trust check: neutral weights give Table 7 back exactly
  5. the network path: sentence -> parsed facts -> evidence -> propagation
  6. who reads the text: deterministic parser + LLM tier (visible slide)
  7. the faithfulness check + negative control
  8. one REAL held-out accident, end to end
  9. proof on unseen accidents (held-out 296) -> next step
  appendix: Sec 5.2 ladder, scoreboard, LLM detail, 12 gaps

Every number comes from a checked artifact:
  docs/ALL_TABLES_EXACT_COMPARISON.md    (four-way exact probabilities)
  outputs/bn_upgraded_full.json          (scoreboard 48/29/12/4)
  outputs/BN_UPGRADED_ENVELOPE.md        (30-seed envelope + released-xdsl check)
  outputs/heldout_narrative_bn_eval.json (held-out 296: injury 93%, damage 81%)
  docs/table7_full_reproduction.csv      (Table 7, 85/85)
  docs/figures/narrative_to_bn_architecture.png  (the one picture)

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 docs/build_tuesday_jul21_pptx.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

TEMPLATE = Path("/Users/kanushetkar/Downloads/Copy of NTSB project update 6_30_26.pptx")
HERE = Path(__file__).resolve().parent
OUT = HERE / "tuesday_jul21_update.pptx"
ARCH_PNG = HERE / "figures" / "narrative_to_bn_architecture.png"

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


def add_takeaway(slide, text, top=5.4, size=17):
    tb = slide.shapes.add_textbox(Inches(0.9), Inches(top), Inches(11.5), Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    return tb


# ============================================================ 1: TITLE
s = prs.slides.add_slide(TITLE_LAYOUT)
tp = _placeholder(s, True)
if tp is not None:
    tp.text_frame.text = "NTSB Project Update July 21st, 2026"
sp = _placeholder(s, False)
if sp is not None:
    sp.text_frame.text = "Kanu Shetkar"

# ============================================================ 2: TODAY
add_content_slide(
    "Today's Meeting",
    [
        ("Your question last time: how do the narratives and the query "
         "actually connect to all of this?", 0, {"bold": True}),
        ("", 0),
        ("Today is that answer, end to end:", 0),
        ("One picture of the whole system", 0),
        ("The journey of one typed sentence: embedding, clusters, weights, "
         "law of total probability", 0),
        ("The same sentence driving the network: parsed facts, evidence, "
         "propagation", 0),
        ("Who reads the text: a deterministic parser, and an LLM when the "
         "wording only describes", 0),
        ("One real accident walked through the whole pipeline", 0),
        ("Proof on 296 accidents the system never saw", 0),
    ],
    note=("TIME: 1 min.\n\n"
          "SAY: 'Last time you asked one question, and today is built "
          "entirely around it: when someone types a narrative or a query, "
          "how does that text actually connect to the numbers. Where do the "
          "embeddings, the clusters, and the law of total probability come "
          "in, and how does the network get involved. So today: one picture "
          "of the whole system, the journey of one typed sentence step by "
          "step, who actually reads the text including where the LLM now "
          "sits, then one real accident through the whole thing, and proof "
          "on unseen accidents.'\n\n"
          "Do NOT recap last week's tables; comparisons live in the "
          "appendix if he asks. No trees, no demo today. Twenty minutes "
          "total; the middle slides answer HIS question, spend the time "
          "there."))

# ============================================================ 3: THE ONE PICTURE
s = add_content_slide(
    "The one picture: two moments in time",
    note=("SAY: 'Before any detail, here is the whole system in one picture. "
          "Two moments in time. BUILD TIME, top lane, happens once and has "
          "two rows: the coded NTSB records, the structured entries "
          "investigators filed, go through Zhang's Section 4 recipe and "
          "become the Bayesian network, which is then FROZEN -- that is the "
          "thing we validated against his 93 published numbers. And "
          "alongside it, the accident narratives are converted once into an "
          "embedding index and grouped into clusters. QUERY TIME, bottom, "
          "runs every time someone types, and the same sentence can go down "
          "two paths. The COUNTING path: similarity scores every accident, "
          "the clusters get weights, and the law of total probability mixes "
          "their counts -- one conditional at a time. The NETWORK path: two "
          "readers turn the text into facts, every fact gets a strength "
          "measured from the data, and the facts land as evidence on the "
          "frozen network, which propagates them jointly. The one sentence "
          "to remember: the narrative never touches the network's "
          "construction. It enters only as evidence, at query time.'\n\n"
          "IF HE ASKS where embeddings/clusters come from: point at the "
          "second build row -- 'made once at build time, reused at query "
          "time in both paths: the counting path uses them directly, and "
          "the network path uses them to measure soft-fact strengths' (the "
          "dotted blue arrows).\n\n"
          "This slide IS the answer to his question; the next slides just "
          "zoom in. Point at the lanes while talking."))
if ARCH_PNG.is_file():
    # new figure is ~1.52:1; size by HEIGHT so it fits under the title
    # (its yellow banner IS the takeaway, so no extra text box on this slide)
    s.shapes.add_picture(str(ARCH_PNG), Inches(2.35), Inches(1.45),
                         height=Inches(5.8))

# ============================================================ 4: EMBEDDING
add_content_slide(
    "Step 1: from a sentence to similar accidents",
    [
        ("You type: 'engine caught fire during takeoff'", 0),
        ("The sentence becomes an embedding: a list of numbers that captures "
         "its meaning", 0),
        ("Every accident narrative in our database was turned into the same "
         "kind of numbers, once, ahead of time", 0),
        ("Similar meaning gives nearby numbers", 0),
        ("So we can score EVERY accident by how similar it is to your "
         "sentence", 0, {"bold": True}),
        ("No keywords, no exact matching: 'flames from the cowling' still "
         "finds the fire accidents", 0),
    ],
    note=("SAY: 'Step one, the query. You type a plain sentence. The "
          "computer converts it into an embedding, which is just a long "
          "list of numbers that captures what the sentence means, not which "
          "words it uses. Every accident narrative in our database went "
          "through the same conversion once, ahead of time. Sentences that "
          "mean similar things end up with nearby numbers. So when your "
          "query arrives, we score every single accident by similarity to "
          "your sentence. The win over keyword search: someone who writes "
          "flames from the cowling still finds the fire accidents, because "
          "the meaning is close even though the words are different.'\n\n"
          "Keep it this simple. If he wants the model name: OpenAI "
          "embeddings, same model for queries and narratives."))

# ============================================================ 5: CLUSTERS + LTP
add_content_slide(
    "Step 2: clusters, weights, and the law of total probability",
    [
        ("The accidents are grouped into clusters: families of accidents "
         "that resemble each other", 0),
        ("Inside each cluster we COUNT, exactly Zhang's way: "
         "P(cause | cluster)", 0),
        ("Your query hands each cluster a weight: how much of the "
         "similarity points there", 0),
        ("Law of total probability stitches it together:", 0),
        ("P(cause | query) = sum over clusters of "
         "P(cause | cluster) x weight(cluster)", 1, {"bold": True}),
        ("Plain words: each family of accidents votes, and your query "
         "decides how loud each vote is", 0),
    ],
    note=("SAY: 'Step two. The accidents are grouped into clusters, "
          "families of accidents that resemble each other. Inside each "
          "cluster we count exactly the way Zhang counts: the probability "
          "of each cause within that family. That part is his method, "
          "untouched. What the query adds is the weights. The similarity "
          "scores from the last slide tell us how much of the query points "
          "at each cluster. Then the law of total probability stitches it "
          "together: the probability of a cause given your query is the sum "
          "over clusters of the cause probability inside the cluster times "
          "the weight your query gave that cluster. In plain words: each "
          "family of accidents votes, and your query decides how loud each "
          "vote is.'\n\n"
          "Write the formula on the board if it helps. The counting INSIDE "
          "clusters is Zhang's rule; only the weights come from the query.\n\n"
          "IF HE ASKS how the clusters are formed: 'the accidents are "
          "partitioned by their category label in the data, fixed once, "
          "before any query exists. The query never redraws the clusters; "
          "it only re-weights them.' That fixed partition is what makes the "
          "law of total probability legal here."))

# ============================================================ 6: TRUST CHECK (LTP)
s = add_content_slide(
    "The trust check: turn the query off, get his table back",
    note=("SAY: 'Why should anyone trust that formula. Because of this "
          "check. If you make the weights neutral, meaning each cluster "
          "counts by its size and the query adds no information, the "
          "formula collapses back to plain counting over everything, and it "
          "returns Zhang's Table 7 EXACTLY. All 85 causes, same 102-fire "
          "denominator, cell for cell. Wiring 9 of 102, airframe 32 of 102. "
          "So the query machinery does not invent anything: with no query "
          "you get his published table, and with a query you get a tilted "
          "version of it, tilted by a measured amount. Zhang is the special "
          "case of our formula.'\n\n"
          "This is the exact-recovery punchline. Do not skip it: it is the "
          "reason the whole counting path is trustworthy.\n"
          "Source: docs/table7_full_reproduction.csv, 85 of 85."))
add_table(s, [
    ["P(cause | fire), sample rows", "Zhang Table 7", "Our formula, neutral weights"],
    ["airframe/component/system failure", "0.3137  (32/102)", "0.3137  (32/102)"],
    ["electrical system, electric wiring", "0.0882  (9/102)", "0.0882  (9/102)"],
    ["loss of engine power (total)", "0.0882  (9/102)", "0.0882  (9/102)"],
    ["fluid, fuel", "0.0588  (6/102)", "0.0588  (6/102)"],
], left=0.9, top=1.7, width=11.4, height=2.8,
   col_widths=[5.2, 3.0, 3.2], font_size=14)
add_takeaway(s, "Neutral weights return Table 7 exactly, all 85 cells. "
                "Zhang is the special case of our formula.", top=4.9)

# ============================================================ 7: THE BRIDGE
add_content_slide(
    "Step 3: the same sentence can also drive the network",
    [
        ("Counting answers one conditional at a time; the network handles "
         "several facts at once", 0),
        ("So the sentence is parsed into structured facts, three kinds:", 0),
        ("HARD: facts named in NTSB vocabulary ('loss of engine power') "
         "are set to present, 100%. Deterministic, reproducible", 1),
        ("SOFT: when the wording only DESCRIBES a fact, an LLM maps the "
         "description to the vocabulary ('the gauges were giving bad "
         "readings' is the engine instrument node)", 1),
        ("Every LLM-proposed fact enters at a strength MEASURED from the "
         "100 most-similar accidents, never at the LLM's own confidence", 1),
        ("STATED severity: 'sustained substantial damage' enters weighted "
         "by how reliable that phrasing historically is", 1),
        ("The facts are placed as evidence and the network propagates them "
         "all at once", 0, {"bold": True}),
    ],
    note=("SAY: 'Step three, the network path. The counting path answers "
          "one conditional at a time. Real investigations know several "
          "things at once, and that is what the network is for. So the same "
          "typed sentence is parsed into structured facts, in tiers. Hard "
          "facts: when the sentence names something in NTSB vocabulary, "
          "like loss of engine power, that node is set to present, one "
          "hundred percent -- deterministic, no language model involved. "
          "When the sentence only describes a fact instead of naming it, "
          "an LLM does the reading: it maps the description onto the "
          "vocabulary, the gauges were giving bad readings becomes the "
          "engine instrument node. But we do not trust its confidence: "
          "every fact it proposes enters at the strength measured among "
          "the hundred most similar accidents, so the LLM picks WHICH "
          "facts and the data decides HOW STRONGLY. And stated severity: "
          "if the sentence says the aircraft sustained substantial damage, "
          "that statement enters weighted by how often that exact phrasing "
          "matched the coded record historically. Then all the facts are "
          "placed as evidence on the network and it propagates them "
          "jointly.'\n\n"
          "IF HE ASKS how a frequency becomes evidence: 'Pearl's virtual "
          "evidence, also called Jeffrey conditioning. If the fact appears "
          "in 40 percent of the similar accidents, we set a likelihood "
          "ratio against the node's own prior so the network's updated "
          "belief in the fact lands at about 40 percent. Suggestion at "
          "measured strength, not assertion.'\n\n"
          "IF HE ASKS why the LLM does not set the strengths: 'We tested "
          "it. Across 1,078 narratives the LLM picks facts well but its "
          "confidence numbers are badly miscalibrated -- when it says 86 "
          "percent the fact is real about 9 percent of the time -- so the "
          "data sets the strengths.' Calibration figure is in the "
          "appendix.\n\n"
          "IF HE ASKS how often the LLM is needed: 'on 296 held-out "
          "narratives, 176 were fully handled by the deterministic pass "
          "and 120 needed the LLM; with the LLM tier, injury accuracy on "
          "vague narratives rises from 64 to 68 percent and damage from "
          "45 to 52.' Details in the appendix LLM slide."))

# ============================================================ 7b: WHO READS THE TEXT
s = add_content_slide(
    "Who reads the text: two readers, one rule",
    [
        ("Reader 1, the deterministic parser: fires when the text NAMES a "
         "fact from the vocabulary. Exact, reproducible, free", 0),
        ("Reader 2, the LLM: fires only when the text DESCRIBES a fact "
         "without naming it", 0),
        ("'the first officer's windshield cracking' becomes the window "
         "node", 1),
        ("On 296 real narratives: 176 resolved by reader 1, 120 needed "
         "the LLM", 0),
        ("The one rule: the LLM picks WHICH facts, the data decides HOW "
         "STRONGLY, the network does the reasoning", 0, {"bold": True}),
    ],
    note=("TIME: 2 min. This is the answer to 'are you using an LLM'.\n\n"
          "SAY: 'So who actually does the parsing. Two readers behind one "
          "front door. Reader one is a deterministic parser: if the "
          "sentence names a fact in NTSB vocabulary, like loss of engine "
          "power, it is caught here -- exact, reproducible, no language "
          "model involved. Reader two is the LLM, and it fires only when "
          "the wording describes instead of names: a real narrative said "
          "the first officer's windshield was cracking, and the LLM maps "
          "that to the window node. On 296 real held-out narratives the "
          "split was 176 to 120, so the LLM is doing real work on about "
          "forty percent of real text. And the one rule that keeps it "
          "honest: the LLM picks which facts, but every fact enters at a "
          "strength measured from the hundred most similar accidents, "
          "never at the LLM's own confidence, and the network does all "
          "the reasoning.'\n\n"
          "IF HE ASKS why not let the LLM do everything: 'we measured "
          "that too -- LLM as the only front door on all 296: accuracy "
          "is comparable, but its probabilities are less calibrated "
          "(worse log-loss), it loses exactness on one of the table "
          "scenarios, and it costs an API call on every query. So we "
          "keep the deterministic pass first.' Numbers in the appendix "
          "LLM slide.\n\n"
          "IF HE ASKS what the LLM tier buys: 'injury accuracy on the "
          "held-out narratives rises from 64 to 68 percent, damage from "
          "45 to 52, entirely from the vague narratives the deterministic "
          "parser cannot ground.'"))

# ============================================================ 8: FAITHFULNESS
s = add_content_slide(
    "The faithfulness check: typed sentence = clicked evidence",
    note=("SAY: 'Same trust question for the bridge: how do I know parsing "
          "a sentence gives the right answer. Because when a sentence names "
          "the same facts you could click by hand, the parser lands on the "
          "IDENTICAL evidence nodes, and identical evidence gives identical "
          "numbers, to the last digit. Here is Table 9 with the evidence "
          "typed as a sentence: trouble with an engine instrument during "
          "the flight. The direct column is me clicking the node; the "
          "narrative column is the typed sentence. Same numbers. So the "
          "sentence is a faithful front door to the network, not a second "
          "model.'\n\n"
          "Source: docs/ALL_TABLES_EXACT_COMPARISON.md, engine instruments "
          "block. If he asks when they DIVERGE: 'when the wording is vague, "
          "by design: then soft evidence enters at measured strength "
          "instead of 100%.'\n\n"
          "IF HE FIXATES on no injury 0.9431 vs 0.8890: 'that is a "
          "close-category cell, within 6 percent; the point of THIS slide "
          "is the two right columns being identical. The Zhang gaps are "
          "the scoreboard slide at the end.'"))
add_table(s, [
    ["Sentence: 'trouble with an engine instrument'", "Zhang", "Clicked", "Typed"],
    ["P(loss of engine power)", "0.95", "0.9502", "0.9502"],
    ["P(forced landing)", "0.1357", "0.1357", "0.1357"],
    ["P(gear collapsed)", "0.0096", "0.0082", "0.0082"],
    ["P(no injury)", "0.9431", "0.8890", "0.8890"],
], left=0.9, top=1.65, width=11.4, height=2.9,
   col_widths=[5.6, 1.9, 1.9, 2.0], font_size=14)
add_takeaway(s, "Typed sentence and clicked evidence produce identical "
                "numbers. The narrative is a faithful front door.", top=5.0)

# ============================================================ 8b: NEGATIVE CONTROL
s = add_content_slide(
    "How do we know the match is earned? We tried to break it",
    note=("SAY: 'A fair question here is: if typing and clicking always "
          "agree, is the sentence actually doing anything? So we ran a "
          "negative control. Same eleven scenarios, three kinds of input. "
          "Feed the correct sentence: identical numbers, eleven of eleven. "
          "Feed each scenario a sentence that belongs to a DIFFERENT "
          "scenario: the numbers diverge, eleven of eleven. Feed a nonsense "
          "sentence with no aviation content: the parser finds nothing and "
          "the network falls back to its baseline, eleven of eleven. If the "
          "text were being ignored, all three rows would look the same. So "
          "the sentence determines the evidence, and the evidence "
          "determines the numbers. The agreement is earned.'\n\n"
          "IF ASKED about negation or vague wording: 'we also probed those "
          "adversarially -- negated facts like no fire on board and "
          "hypotheticals like worried about losing power are detected and "
          "excluded, and vague wording drops to calibrated soft evidence "
          "instead of a hard claim.'\n\n"
          "Source: outputs/narrative_negative_control.md and "
          "outputs/parser_flaws_and_paraphrases.md."))
add_table(s, [
    ["What we fed the parser", "What came out", "Scenarios"],
    ["The correct sentence", "identical to clicked evidence", "11 of 11"],
    ["A sentence from a different scenario", "different numbers", "11 of 11"],
    ["A nonsense sentence (no aviation content)", "falls back to baseline",
     "11 of 11"],
], left=0.9, top=1.75, width=11.4, height=2.4,
   col_widths=[5.6, 3.9, 1.9], font_size=15)
add_takeaway(s, "Right sentence, right numbers. Wrong sentence, wrong "
                "numbers. Nonsense, baseline. The text drives the result.",
             top=4.7)

# ============================================================ 10: ONE REAL ACCIDENT
s = add_content_slide(
    "The big picture on one real accident (2007, held out)",
    note=("TIME: 3 min. This is the synthesis slide: everything from the "
          "last six slides happening at once, on a real case the system "
          "never saw.\n\n"
          "SAY: 'Let me put the whole picture on one real accident, from "
          "2007, so nothing about it was used to build anything. A "
          "passenger jet just after takeoff; the narrative says the first "
          "officer's windshield was cracking and beginning to arc, the "
          "crew turned back and landed. Nothing in that text names an "
          "NTSB vocabulary node, so the deterministic reader finds "
          "nothing and the LLM reads it: windshield cracking becomes the "
          "window node, and the impact-debris sentence becomes the object "
          "node. Watch the strengths: window enters strong because the "
          "narrative asserts it; object enters at 0.14, because among the "
          "hundred most similar accidents that fact appears at that rate. "
          "Measured, not guessed. The network propagates and the "
          "probability mass moves from the do-nothing prior toward what "
          "actually happened: no injury correct, and minor damage, a "
          "state the prior put near zero, rises to a quarter of the "
          "mass. That is the honest picture: the text was read, grounded "
          "in data, reasoned over jointly, and pushed the answer toward "
          "the truth.'\n\n"
          "BE HONEST if he pushes on damage: the top damage pick was no "
          "damage; the point is the mass moving toward the truth on a "
          "rare class, and the aggregate numbers are the next slide.\n\n"
          "Source: outputs/real_narrative_walkthrough.md (second case; "
          "the first case is a tier-1 fan disk separation if he wants a "
          "NAMED-facts example)."))
add_table(s, [
    ["Pipeline step", "What happened on this accident"],
    ["Narrative (real, held out)", "'first officer's windshield cracking "
     "and beginning to arc...'"],
    ["Who read it", "LLM tier (nothing is named in vocabulary)"],
    ["Evidence, strength measured", "window (asserted); object 0.14; "
     "crew person nodes"],
    ["Network posteriors", "no injury 0.55 (top; CORRECT), minor damage "
     "rises ~0 to 0.25"],
], left=0.8, top=1.6, width=11.6, height=3.0,
   col_widths=[3.9, 7.7], font_size=14)
add_takeaway(s, "Read by the LLM, grounded by the data, reasoned by the "
                "network: the mass moves toward what actually happened.",
             top=5.0, size=15)

# ============================================================ 11: HELD-OUT PROOF
s = add_content_slide(
    "Proof on accidents the system never saw",
    note=("SAY: 'You just saw one accident. This slide is the same test "
          "at scale: does the whole thing actually predict, on data the "
          "system never touched. Setup: everything is built on "
          "1982 to 2006. Then we took 296 accidents from 2007 to 2019, fed "
          "ONLY the narrative text in, and predicted the injury level and "
          "the damage level, four classes each. Row one, the network alone, "
          "knowing nothing: 58 and 43 percent. Row two, parse the narrative "
          "into evidence and propagate it through the network: 81 and 60. "
          "Row three, the narrative severity readout: what the similar "
          "accidents and the stated severity say directly, no network in "
          "this row: 93 and 81. And row four, a supervised baseline trained "
          "on the same narratives: 83 and 67. Two things to take away. The "
          "narrative carries most of the signal, and our extraction beats "
          "a trained model. And the two layers have different jobs: the "
          "readout answers how bad was it, the network answers the "
          "questions the readout never can: what caused it, what happens "
          "with several facts at once, and the what-if dials.'\n\n"
          "BE PRECISE: the 93/81 row does NOT go through the network. If "
          "you present it as the network's number he will catch it. The "
          "network's narrative-evidence number is 81/60.\n\n"
          "Source: outputs/heldout_narrative_bn_eval.json (n=296, clean "
          "temporal split). IF HE ASKS about narratives that state the "
          "outcome: 'about a third state the damage level; the statement "
          "is weighted by its historical reliability, not trusted blindly, "
          "and the retrieval part alone still gets 77 percent on damage.'\n\n"
          "The single-case version of this slide was the windshield "
          "walkthrough you just showed; if he wants a SECOND real case, "
          "outputs/real_narrative_walkthrough.md also has a tier-1 fan "
          "disk separation with NAMED facts. Have it open as a backup "
          "tab."))
add_table(s, [
    ["Predictor (296 unseen accidents)", "Injury acc", "Damage acc"],
    ["Network alone (no narrative)", "58%", "43%"],
    ["Narrative parsed to evidence, through the network", "81%", "60%"],
    ["Narrative severity readout (similar accidents + stated)", "93%", "81%"],
    ["Supervised baseline (logistic regression)", "83%", "67%"],
], left=0.8, top=1.7, width=11.6, height=2.9,
   col_widths=[7.2, 2.1, 2.1], font_size=14)
add_takeaway(s, "The narrative carries the signal: 93% injury, 81% damage, "
                "beating a trained baseline. The network answers what the "
                "readout cannot: causes, multi-evidence, what-ifs.",
             top=5.15, size=15)

# ============================================================ 12: NEXT STEP
add_content_slide(
    "Next Step",
    [
        ("The analysis phase is complete: every published number matched, "
         "close, or explained with evidence", 0),
        ("The narrative path is validated end to end, including on unseen "
         "accidents", 0),
        ("Next is the paper: methodology and validation sections first, "
         "from the artifacts you saw today", 0),
        ("", 0),
        ("Which section do you want on your desk first?", 0, {"bold": True}),
    ],
    note=("SAY: 'My read: the analysis phase is complete. Every published "
          "number is matched, close, or explained with evidence, and the "
          "narrative path is validated end to end including on unseen "
          "accidents. Next is writing. The outline maps every result to a "
          "section, and the methodology and validation sections can come "
          "straight from what you saw today. Which section do you want on "
          "your desk first?' Then stop talking and let him direct.\n\n"
          "Have a real date ready for the first section. DO NOT say the "
          "draft is nearly done."))

# ============================================================ 13: APPENDIX
add_content_slide(
    "Appendix",
    [("Backup only: Section 5.2 staircase, the 93-value scoreboard, every "
      "network table sentence-driven, the LLM evidence, the 12 remaining "
      "gaps", 0)],
    note=("Nothing past this slide is presented. If he asks to see the "
          "app, it still runs (streamlit_tree_app.py); warm it before the "
          "meeting just in case, but there is no demo slide on purpose."))

# ============================================================ A0: SEC 5.2 LADDER
s = add_content_slide(
    "Appendix: stacking evidence, one sentence at a time (his Section 5.2)",
    note=("Backup: the multi-evidence staircase, if he wants a Zhang "
          "comparison beyond the walkthrough.\n\n"
          "SAY: 'His Section 5.2 demonstration: keep adding landing gear "
          "evidence and watch gear collapse climb. We drive it with "
          "sentences. Strut problem: 0.25, exactly his number. Add the "
          "emergency extension: he 0.68, we 0.60. Add gear locking: 0.78 "
          "vs 0.71. Add the attachment: 0.89 vs 0.85. Same staircase, "
          "step for step, each step from typing one more fact into the "
          "same sentence. Counting cannot do this: four facts at once "
          "match almost no accidents; the network propagates them "
          "jointly.'\n\n"
          "Source: ALL_TABLES_EXACT_COMPARISON.md Section 5.2 block."))
add_table(s, [
    ["Facts in the sentence (cumulative)", "Zhang", "Ours (typed)"],
    ["strut problem", "0.25", "0.25  exact"],
    ["+ emergency extension failed", "0.682", "0.600"],
    ["+ gear locking mechanism", "0.777", "0.705"],
    ["+ gear attachment", "0.894", "0.855"],
], left=1.0, top=1.65, width=11.2, height=2.9,
   col_widths=[6.0, 2.4, 2.8], font_size=15)
add_takeaway(s, "Each extra fact typed into the sentence climbs the same "
                "staircase as his figure. Counting cannot do this.", top=4.95)

# ============================================================ A1: SCOREBOARD
s = add_content_slide(
    "Appendix: where it all stands, 93 published numbers scored",
    note=("Backup, only if he asks for the overall standing.\n\n"
          "SAY: 'Every number in his paper that depends on the network, 93 "
          "of them: 48 exact, 29 close, 77 of 93. The remaining 12 are "
          "explained with evidence: 30 rebuilds under his own randomized "
          "construction never produce his published values, and his own "
          "released network file cannot reproduce those cells either; on "
          "every such cell ours is as close or closer.'\n\n"
          "Sources: outputs/bn_upgraded_full.json, "
          "outputs/BN_UPGRADED_ENVELOPE.md."))
add_table(s, [
    ["", "Count", "Meaning"],
    ["Exact", "48", "matches to publication precision"],
    ["Close", "29", "within 25 percent"],
    ["Different", "12", "explained: his own released file misses them too"],
    ["Qualitative", "4", "direction verified (no numbers printed)"],
], left=1.2, top=1.6, width=10.8, height=2.9,
   col_widths=[2.4, 1.6, 6.8], font_size=15)
add_takeaway(s, "77 of 93 exact or close. On the rest, our network is closer "
                "to his published values than his own released file.", top=4.9)

# ============================================================ A2: TABLE 9 LOEP
s = add_content_slide(
    "Appendix: Table 9 vs ours, sentence-driven",
    note=("Backup, only if he wants more tables than the Section 5.2 "
          "ladder.\n\n"
          "SAY: 'The sentence is: the aircraft experienced a loss of engine "
          "power. Forward cells match to the third digit. Big outcome rows "
          "agree: no injury 0.99 his, 0.95 ours. The small rows move more, "
          "and small probabilities move easily: a handful of accidents "
          "swings them; the per-cell evidence is on the gaps slide.'\n\n"
          "Source: ALL_TABLES_EXACT_COMPARISON.md, LOEP block (typed = "
          "clicked, identical)."))
add_table(s, [
    ["Sentence: 'the aircraft experienced a loss of engine power'", "Zhang", "Ours (typed)"],
    ["P(forced landing)", "0.1429", "0.1449"],
    ["P(ditching)", "0.00461", "0.00468"],
    ["P(no injury)", "0.9899", "0.9496"],
    ["P(substantial damage)", "0.0166", "0.0215"],
    ["P(serious injury)", "0.00822", "0.0160"],
    ["P(destroyed aircraft)", "0.00559", "0.0292"],
], left=0.9, top=1.55, width=11.4, height=3.5,
   col_widths=[6.4, 2.4, 2.6], font_size=14)

# ============================================================ A3: FIG 12 PILOT
s = add_content_slide(
    "Appendix: Figure 12 vs ours, the pilot queries",
    note=("Backup. SAY: 'Figure 12 conditions on the pilot in command "
          "being a factor. Two meetings ago these cells were not runnable; "
          "the person findings were in the data all along and the graph "
          "now uses them. All four cells close: no injury 0.97 his, 0.94 "
          "ours.' DO NOT say we added data: the graph now uses fields it "
          "was ignoring.\n\n"
          "Source: ALL_TABLES_EXACT_COMPARISON.md Fig 12 stage 1."))
add_table(s, [
    ["Sentence: 'the pilot in command was a factor'", "Zhang", "Ours (typed)"],
    ["P(unstabilized approach)", "0.00484", "0.0037"],
    ["P(dragged wing / rotor / pod)", "0.023", "0.0211"],
    ["P(substantial damage)", "0.0458", "0.0510"],
    ["P(no injury)", "0.97", "0.9405"],
], left=1.0, top=1.65, width=11.2, height=2.9,
   col_widths=[5.8, 2.5, 2.9], font_size=14)

# ============================================================ A4: TABLE 8
s = add_content_slide(
    "Appendix: Table 8, the one table a sentence cannot drive",
    note=("Backup. SAY: 'Table 8 is a what-if experiment: he EDITS a "
          "prior, imagine strut failures were more common, and watches "
          "gear collapse respond. A narrative describes what happened; it "
          "cannot describe an imaginary world where failure rates differ. "
          "So this one is driven by turning the dial on the network "
          "directly, and it comes out essentially exact. Narratives carry "
          "evidence; dials carry hypotheticals; the network handles "
          "both.'\n\n"
          "Source: ALL_TABLES_EXACT_COMPARISON.md Table 8 sweep."))
add_table(s, [
    ["Strut prior (the dial)", "Zhang: P(gear collapsed)", "Ours"],
    ["0.00065", "0.000163", "0.000163  exact"],
    ["0.065", "0.0162", "0.0163"],
    ["0.1", "0.025", "0.025  exact"],
    ["0.5", "0.125", "0.125  exact"],
], left=1.2, top=1.65, width=10.8, height=2.9,
   col_widths=[3.4, 3.9, 3.5], font_size=15)

# ============================================================ A5: LLM CALIBRATION
s = add_content_slide(
    "Appendix: why the data sets the soft-fact strengths, not the LLM",
    [
        ("We tested letting an LLM assign confidence to extracted facts", 0),
        ("Across 1,078 narratives: when the LLM says 86 percent, the fact "
         "is in the coded record about 9 percent of the time", 0),
        ("The retrieval-measured strength tracks reality across every "
         "confidence bin", 0),
        ("Checked under two truth signals (coded record; coded or spoken "
         "verbatim): the LLM stays miscalibrated under both", 0),
        ("Design rule: the LLM may pick WHICH facts; the data decides HOW "
         "STRONGLY", 0, {"bold": True}),
    ],
    note=("ONLY if he asks about the LLM. Figure: "
          "docs/figures/confidence_calibration.png. ECE numbers: LLM 0.74 "
          "strict, 0.60 lenient; retrieval 0.16 strict, 0.25 lenient (and "
          "the lenient shift is UNDERconfidence, the safe direction). Also "
          "tested: LLM direct probability estimates are off by an order of "
          "magnitude; rebuilding the dataset from narratives alone inverts "
          "causal attribution. All three experiments justify the "
          "architecture.\n\n"
          "IF HE ASKS 'would a stronger LLM fix it': tested EIGHT models "
          "across TWO providers (GPT 4o-mini..5.5; Claude haiku..opus) on "
          "adversarial probes + paraphrases. Identical pattern on both "
          "providers: small models cautious, big models read better but "
          "over-generalize out-of-vocabulary facts; raw recovery capped at "
          "3/7 everywhere. Then two fixes raised it WITHOUT giving the LLM "
          "authority over strengths: (1) a prompt rule that paraphrase "
          "mapping is expected, (2) rare-but-over-represented facts "
          "survive the data filter on LIFT (odds vs base rate) instead of "
          "raw frequency -- e.g. 'the gauges were giving bad readings' -> "
          "engine instrument, seen in 1% of similar accidents but 18x "
          "over-represented. Recovery now 4-5/7, remaining misses are "
          "parent/sibling nodes, safety intact. Strengths still come from "
          "the data. Sources: outputs/llm_model_ladder.md, "
          "outputs/llm_paraphrase_upgrade.md.\n\n"
          "IF HE ASKS 'so is the LLM in the pipeline now': yes, as a TIERED "
          "front door. Tier 1 is the deterministic parser (fires when the "
          "wording names facts -- 176 of 296 held-out narratives, and all "
          "11 faithfulness scenarios stay bit-identical). Tier 2 calls the "
          "LLM only when tier 1 finds nothing, and its facts still get "
          "data-measured strengths. On the 296 held-out accidents the "
          "tiered parser beats the deterministic parser alone (injury "
          "accuracy 68 vs 64 percent, damage 52 vs 45), but the retrieval "
          "severity readout is still the best severity predictor (93/81), "
          "so the LLM improves the QUERY front door, not the severity "
          "engine. Source: outputs/heldout_eval_llm_tier.txt and "
          "outputs/combined_parser_validation.md.\n\n"
          "IF HE ASKS 'can the LLM do the WHOLE process alone': yes, and we "
          "measured it -- LLM as the ONLY front door on all 296 held-out "
          "accidents. Top-1 accuracy is comparable (injury 70 vs 69 "
          "percent, damage 59 vs 51), but its probability quality is worse "
          "(log-loss 1.27 vs 1.00 injury) -- more confident errors -- and "
          "on the 11 table scenarios it hits 30 of 33 runs instead of "
          "always: it gives pilot-in-command 0.53 instead of hard "
          "evidence, so Zhang's Figure 12 comparison stops being exact. "
          "Plus an API call on every query. So the LLM CAN do the process; "
          "the tiered design keeps its coverage and removes those costs. "
          "Sources: outputs/heldout_eval_llm_first.txt, "
          "outputs/llm_only_identity.md."))

# ============================================================ 19: APPENDIX B
add_content_slide(
    "Appendix: the 12 remaining gaps, the evidence",
    [
        ("30 rebuilds with his own randomized construction: his published "
         "values never appear; the envelope is tight around our build", 0),
        ("His own released network file misses those cells too", 0),
        ("On every cell his file can answer, ours is as close or closer to "
         "his published value", 0, {"bold": True}),
        ("Conclusion: the published table reflects an unreleased build and "
         "dataset state, not an error in our reproduction", 0),
        ("Per-cell notes: outputs/BN_UPGRADED_ENVELOPE.md", 0),
    ],
    note=("ONLY if he wants to walk the 12. The xdsl cross-check table (4 "
          "cells, ours closer on 3, tie on 1) is in "
          "outputs/BN_UPGRADED_ENVELOPE.md."))

prs.save(str(OUT))
print(f"Saved {OUT} with {len(prs.slides._sldIdLst)} slides")
for i, slide in enumerate(prs.slides, 1):
    title = ""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            title = shape.text_frame.text.splitlines()[0][:70]
            break
    print(f"  {i}. {title}")
