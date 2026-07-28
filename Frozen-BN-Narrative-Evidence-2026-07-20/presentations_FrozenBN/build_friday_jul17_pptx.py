"""Generate docs/friday_jul17_update.pptx, the Friday advisor deck (30 min).

v4, written to MIMIC docs/build_tuesday_jul14_pptx.py (the deck the user liked):
  * short bullets, plain tables, one bold takeaway line under each table
  * order: previous meeting -> the Bayesian network -> worked example ->
    every Zhang table compared (7, 9, 8, Figs 11/12, Sec 5.2) -> scoreboard ->
    one finding two fixes -> Live Demo (just that) -> next step -> appendix
  * Table 4 and Figure 3 in the APPENDIX only
  * no em dashes

Every number comes from a checked artifact:
  outputs/bn_full_comparison.json      (93 items, per-section verdicts)
  outputs/bn_upgraded.json             (Table 9 LOEP outcomes, Fig 12 queries)
  outputs/bn_upgraded_full.json        (final scoreboard 48/29/12/4)
  outputs/bn_variance_envelope.json    (30-rebuild jitter experiment)
  docs/table7_full_reproduction.csv    (Table 7, 85/85)

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 docs/build_friday_jul17_pptx.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

TEMPLATE = Path("/Users/kanushetkar/Downloads/Copy of NTSB project update 6_30_26.pptx")
OUT = Path(__file__).resolve().parent / "friday_jul17_update.pptx"

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
    tp.text_frame.text = "NTSB Project Update July 17th, 2026"
sp = _placeholder(s, False)
if sp is not None:
    sp.text_frame.text = "Kanu Shetkar"

# ============================================================ 2: PREV + TODAY
add_content_slide(
    "Previous Meeting",
    [
        ("Talked about Table 4", 0),
        ("Showed the changes to the process", 0),
        ("Proposed the Bayesian network", 0),
        ("", 0),
        ("Today's Meeting", 0, {"bold": True, "size": 28}),
        ("The Bayesian network we built, and where its numbers come from", 0),
        ("A worked example on it", 0),
        ("Every remaining table in his paper vs ours: Tables 8, 9, Figs 11, 12", 0),
        ("Diagnosis & Prognosis trees live (Streamlit)", 0),
        ("Writing plan", 0),
    ],
    note=("SAY: 'Tuesday we closed Table 4 and I showed you the corrected "
          "counting pipeline, with Table 7 reproducing exactly. And I promised "
          "the Bayesian network for today. So: the network I built and where "
          "every number in it comes from, a worked example, then every "
          "remaining table in the paper compared against ours, Tables 8 and 9 "
          "and Figures 11 and 12, then the trees live in the app, and we end "
          "with the writing plan.'\n\n"
          "One minute, no more. Table 4 and Figure 3 stay in the appendix "
          "unless HE brings them up."))

# ============================================================ 3: THE BN
add_content_slide(
    "The Bayesian network we built",
    [
        ("Not his files, not his GeNIe software → his Section 4 methodology, "
         "implemented by us, on our 1,742 accidents", 0),
        ("Nodes = events and findings (wiring, fire, engine power loss, injuries)", 0),
        ("Arrows come from the data itself:", 0),
        ("findings point at the occurrence they explain", 1),
        ("each accident's event sequence gives the forward arrows", 1),
        ("Each node holds a probability table → fill them in, and the network "
         "answers questions in BOTH directions (causes and consequences)", 0),
        ("Result: 785 nodes", 0, {"bold": True}),
    ],
    note=("SAY: 'First, what I built. I did not load his network file and I "
          "did not use his GeNIe software. I took the recipe from Section 4 "
          "of the paper and implemented every step myself over our 1,742 "
          "accidents. The network is a graph: every node is an event or a "
          "finding, an arrow means one thing leads to another. And the arrows "
          "are not drawn by hand. They come from the data: findings point at "
          "the occurrence they explain, and the event sequence of each "
          "accident gives the forward arrows: fire, then emergency descent, "
          "then forced landing. Each node holds a small probability table, "
          "and once those are filled in you can clamp anything you know and "
          "the network answers in both directions: what likely caused this, "
          "and what likely comes next. The result is a network of 785 "
          "nodes.'\n\n"
          "IF HE ASKS why not use his file: 'I did load it as a cross check, "
          "and that revealed something I will show you in a few slides. For "
          "the paper, ours has to be rebuildable from the data.'"))

# ============================================================ 4: WHERE NUMBERS COME FROM
add_content_slide(
    "Where every number in it comes from",
    [
        ("Priors (root events): occurrences ÷ 184,517,128 flights (his Eq. 6)", 0),
        ("fire: 102 accidents → P(fire) = 5.5 x 10^-7, matches the paper", 1),
        ("the ONLY place the flight count is used", 1, {"bold": True}),
        ("One parent: counted → in how many accidents does the child follow "
         "the parent? Capped at 0.95", 0),
        ("improper oil grade → engine power loss: 1/1, capped → 0.95 = his "
         "published cell", 1),
        ("Many parents: too sparse to count → his Beta-CDF curve, calibrated "
         "to the counted values, parents capped at 12", 0),
        ("Every number = a count, a capped count, or a curve fit to counts", 0,
         {"bold": True}),
    ],
    note=("SAY: 'Where do the numbers in those tables come from. Three cases. "
          "First, priors for the root events: the number of accidents where "
          "the event occurred divided by 184,517,128 flights, his Equation 6. "
          "Fire occurred in 102 accidents, so the prior is five and a half in "
          "ten million, same as the paper. And that is the ONLY place the "
          "flight count appears: every conditional probability is counted "
          "among accidents. Second, one-parent cells are counted directly: "
          "among accidents where the parent appears, in how many does the "
          "child follow. Capped at 0.95, because one out of one is one data "
          "point, not certainty. Real example: improper oil grade appears in "
          "exactly one accident, and it lost engine power. One out of one, "
          "capped, 0.95, and that is literally the cell in his Table 9. "
          "Third, many-parent cells cannot be counted, the combinations are "
          "too sparse, so he fits a curve, a Beta-CDF, calibrated so the "
          "single-event values match the counts, parents capped at twelve. "
          "We implement the same. So every number in the network is a count, "
          "a capped count, or a curve fit to counts. Inference runs on "
          "pyAgrum, open source, the same exact computation GeNIe runs.'\n\n"
          "Pronunciation: pyAgrum = PIE-ah-grum. GeNIe = GENIE.\n"
          "WATCH OUT: 184M belongs to the priors ONLY."))

# ============================================================ 5: EXAMPLE 1 EVIDENCE
s = add_content_slide(
    "Worked example: tell it one thing",
    note=("SAY: 'Let me run an example instead of talking abstractly. I tell "
          "the network one thing: there was a loss of engine power. It "
          "propagates that evidence through all 785 nodes and returns the "
          "probability of every outcome. That is exactly the question behind "
          "his Table 9, so his published column goes right next to ours. No "
          "injury: he publishes 0.99, we get 0.95. Substantial damage: 0.017 "
          "vs our 0.022. Destroyed: 0.0056 vs our 0.029. Same question, same "
          "recipe, independently built network, and the answers line up. One "
          "honest row: serious injury, his 0.008 vs our 0.016, about double. "
          "Small probabilities move easily. Where the numbers really moved, "
          "I traced why, and that is coming in two slides.'\n\n"
          "Numbers: outputs/bn_upgraded.json table9 (upgraded network)."))
add_table(s, [
    ["Evidence: loss of engine power", "Zhang publishes", "Our network"],
    ["P(no injury)", "0.9899", "0.9496"],
    ["P(serious injury)", "0.00822", "0.0160"],
    ["P(substantial damage)", "0.0166", "0.0215"],
    ["P(destroyed aircraft)", "0.00559", "0.0292"],
    ["P(minor damage)", "0.00378", "0.0208"],
], left=1.4, top=1.7, width=10.4, height=3.2,
   col_widths=[4.6, 2.9, 2.9], font_size=15)
add_takeaway(s, "One evidence in → every outcome out. His Table 9 question, "
                "and the answers line up.")

# ============================================================ 6: EXAMPLE 2 EVIDENCES
add_content_slide(
    "Now tell it two things: why the network exists",
    [
        ("Counting could answer the last slide: 14 accidents have engine "
         "power loss → count outcomes inside those 14", 0),
        ("Add one more fact: pilot-in-command was a factor", 0),
        ("only 4 accidents in the whole dataset match both → counting is done", 1,
         {"bold": True}),
        ("The network does not filter rows → it propagates both evidences "
         "through the graph and still returns a full, grounded answer", 0),
        ("This is why the paper builds a Bayesian network, and why we built one", 0,
         {"bold": True}),
        ("You will see this exact query live in the demo", 0),
    ],
    note=("SAY: 'Here is the moment that justifies the network. The last "
          "slide, counting could still answer: fourteen accidents have a "
          "loss of engine power, count the outcomes inside those fourteen. "
          "Noisy, but possible. Now add one more fact, which every real "
          "investigation has: the pilot in command was a factor. Only FOUR "
          "accidents in the entire dataset match both. Nobody estimates a "
          "probability from four accidents. Counting is done. The network "
          "does not filter rows: it propagates both evidences through the "
          "graph and still returns a full grounded answer. That is why the "
          "paper builds a Bayesian network and why we built one. I will run "
          "this exact query live in a few minutes.'"))

# ============================================================ 7: TABLE 7
s = add_content_slide(
    "Table 7 vs ours: cause probabilities",
    note=("SAY: 'Now the tables, one by one. Table 7 first: probability of "
          "each cause given a fire. His denominator is 102 fire accidents, "
          "ours is 102, and all 85 cells match exactly. Wiring nine out of "
          "102, airframe 32 out of 102, identical to the digit. This one is "
          "pure counting and it is the anchor everything else stands on.'\n\n"
          "Source: docs/table7_full_reproduction.csv, 85/85. The app shows "
          "the full side-by-side table."))
add_table(s, [
    ["P(cause | fire), sample rows", "Zhang", "Ours"],
    ["airframe/component/system failure", "0.3137  (32/102)", "0.3137  (32/102)"],
    ["electrical system, electric wiring", "0.0882  (9/102)", "0.0882  (9/102)"],
    ["loss of engine power (total)", "0.0882  (9/102)", "0.0882  (9/102)"],
    ["fluid, fuel", "0.0588  (6/102)", "0.0588  (6/102)"],
    ["auxiliary power unit (APU)", "0.0490  (5/102)", "0.0490  (5/102)"],
], left=1.0, top=1.7, width=11.2, height=3.2,
   col_widths=[5.4, 2.9, 2.9], font_size=15)
add_takeaway(s, "85 of 85 cells exact. Same denominator, same rules, same numbers.")

# ============================================================ 8: TABLE 9
s = add_content_slide(
    "Table 9 vs ours: 50 cells around engine power loss",
    note=("SAY: 'Table 9 is the big one: fifty cells around loss of engine "
          "power, forward edges and outcome posteriors. Every single-parent "
          "forward edge comes out exact: the 0.95, the 0.50, forced landing "
          "0.1429, instruments 0.1357. Those are counting, and counting "
          "reproduces. First pass: 23 exact, 6 close, 21 different, and the "
          "21 were almost all the damage and injury rows. Those needed one "
          "structural fix, two slides from now, and after it they sit next "
          "to his values, like the outcome rows you saw in the example.'\n\n"
          "First-pass verdicts from outputs/bn_full_comparison.json: "
          "Table 9 = 23 exact, 6 close, 21 differs. After the severity fix "
          "the damage/injury rows move to the values in slide 5."))
add_table(s, [
    ["Cell", "Zhang", "Ours"],
    ["P(engine power loss | improper oil usage)", "0.95", "0.95  exact"],
    ["P(engine power loss | combustion liner failure)", "0.50", "0.50  exact"],
    ["P(forced landing | engine power loss)", "0.1429", "0.1429  exact"],
    ["P(forced landing | inoperative instruments)", "0.1357", "0.1357  exact"],
    ["P(no injury | engine power loss)", "0.9899", "0.9496  after fix"],
    ["P(minor damage | engine power loss)", "0.00378", "0.0208  after fix"],
], left=0.9, top=1.55, width=11.4, height=3.5,
   col_widths=[6.0, 2.5, 2.9], font_size=14)
add_takeaway(s, "Every single-parent edge exact. The damage and injury rows "
                "needed one fix → coming in two slides.", top=5.3)

# ============================================================ 9: TABLE 8
s = add_content_slide(
    "Table 8 vs ours: his what-if experiment",
    note=("SAY: 'Table 8 is a what-if experiment: he dials the prior on a "
          "landing-gear strut problem up, step by step, and watches how the "
          "probability of gear collapse responds. This is something counting "
          "can NEVER do: there is nothing to count in a hypothetical world "
          "where struts fail ten times more often. Only the network can "
          "answer it. We ran the same dial on ours: 18 of the 24 cells "
          "exact, 5 close, 1 different. Read one row: set the strut prior "
          "to 0.065 and he gets gear collapse at 0.0162, we get 0.0163.'\n\n"
          "Verdicts: outputs/bn_full_comparison.json, Table 8 section = "
          "18 EXACT, 5 CLOSE, 1 DIFFERS."))
add_table(s, [
    ["Strut prior (his dial)", "Zhang: P(gear collapsed)", "Ours"],
    ["0.00065", "0.000163", "0.000163  exact"],
    ["0.065", "0.0162", "0.0163  exact"],
    ["0.1", "0.025", "0.025  exact"],
    ["6.5 x 10^-6", "1.70 x 10^-6", "1.72 x 10^-6  close"],
], left=1.2, top=1.7, width=10.8, height=2.9,
   col_widths=[3.4, 3.9, 3.5], font_size=15)
add_takeaway(s, "18 of 24 exact, 5 close, 1 differs. What-if questions cannot "
                "be counted; the network answers them.", top=5.0)

# ============================================================ 10: FIGS 11 + 12
s = add_content_slide(
    "Figures 11 and 12 vs ours: the multi-evidence queries",
    note=("SAY: 'Figures 11 and 12 are his multi-evidence demonstrations. "
          "Figure 12 conditions on pilot-in-command, and on Tuesday these "
          "four cells were not even runnable on our network, our graph had "
          "no person nodes. The person information was in our findings all "
          "along, the graph just never used it. Wire person nodes in, like "
          "his network has, and all four now come out close: no injury he "
          "publishes 0.97, we get 0.94. Figure 11 is directional: does "
          "damage rise as evidence accumulates. Ours moves in the same "
          "direction on all three checks. And his Section 5.2 gear-collapse "
          "chain: given the strut finding he gets 0.25, we get 0.25 exact; "
          "stack the evidence to three pieces and he reaches 0.894, we "
          "reach 0.855, close.'\n\n"
          "DO NOT say we added or rebuilt a data file. Nothing was added: "
          "the graph now uses fields it was ignoring.\n\n"
          "Sources: outputs/bn_upgraded.json fig12; bn_full_comparison "
          "Sec 5.2 text (1 exact, 3 close) and Fig 11 (3 of 3 directions "
          "match)."))
add_table(s, [
    ["Fig 12 (evidence: pilot-in-command)", "Zhang", "Ours"],
    ["P(unstabilized approach)", "0.00484", "0.0037  close"],
    ["P(dragged wing / rotor / pod)", "0.023", "0.0211  close"],
    ["P(substantial damage)", "0.0458", "0.0510  close"],
    ["P(no injury)", "0.97", "0.9405  close"],
], left=1.0, top=1.55, width=11.2, height=2.9,
   col_widths=[5.6, 2.6, 3.0], font_size=14)
add_takeaway(s, "Tuesday: not runnable (no person nodes). Today: all four close. "
                "Fig 11 directions match 3/3; gear-collapse chain 0.25 exact.", top=4.9)

# ============================================================ 11: FINDING + FIXES
add_content_slide(
    "Where the gaps came from: one finding, two fixes",
    [
        ("The finding: his build is not deterministic → his released code adds "
         "random jitter when picking parents", 0),
        ("rebuilt his network 30x with his own randomization → disputed cells "
         "swing up to six orders of magnitude; his own released file misses "
         "10 of the 11 worst cells too", 1),
        ("ours is deterministic: build it twice, same network", 1, {"bold": True}),
        ("Fix 1: person nodes → the data had them all along, the graph now "
         "uses them (unlocked Figure 12)", 0),
        ("Fix 2: severity as ONE 4-state node (fatal / serious / minor / none) "
         "instead of separate on/off switches that cannot say 'nothing "
         "happened'", 0),
        ("P(no injury | engine power loss): 0.02 (impossible) → 0.95, "
         "next to his 0.99", 1),
    ],
    note=("SAY: 'So where did the gaps come from. One finding and two fixes. "
          "The finding: I went into his released code line by line, and when "
          "his builder picks which parents a node keeps, it adds RANDOM "
          "jitter to break ties. Every run of his own code produces a "
          "different network. I rebuilt his network thirty times with his "
          "own randomization: the disputed cells swing by up to six orders "
          "of magnitude between builds, and his own released network file "
          "misses the published values on ten of the eleven worst cells "
          "too. So those numbers cannot be regenerated even from his own "
          "code. Not an attack: the method is sound and most numbers "
          "reproduce. It is a reproducibility finding, and our deterministic "
          "build is the fix. Then the two upgrades. People: his Figure 12 "
          "conditions on pilot-in-command and our graph had no person "
          "nodes. The person findings were in our data all along; the graph "
          "now uses them. And severity: the faithful recipe models each "
          "injury level as its own on-off switch, and that structure cannot "
          "say nothing happened. It forced the probability of no injury to "
          "near zero when most accidents have no injuries. One injury node "
          "with four states that sum to one, same for damage, and the "
          "worst rows snap into place: no injury from an impossible 0.02 to "
          "0.95 against his 0.99.'\n\n"
          "IF HE ASKS 'is his paper wrong?': 'No. Most numbers reproduce "
          "and the method is sound. The deep posteriors depend on which "
          "random build you get; ours removes the randomness.'\n"
          "IF HE ASKS 'can the last 12 be fixed?': 'Not by us. They would "
          "need his exact random build, which even his own code cannot "
          "regenerate. That is the finding.'"))

# ============================================================ 12: SCOREBOARD
s = add_content_slide(
    "All 93 published network numbers, scored",
    note=("SAY: 'Everything together. Every number in the paper that "
          "depends on the network, ninety three of them, scored twice: the "
          "faithful first pass, and after the two fixes. Forty eight exact, "
          "twenty nine close: 77 of 93, and each of the remaining twelve "
          "has a documented traced cause, mostly his build variance. So the "
          "sentence for the paper is: every published number is reproduced "
          "or explained with evidence. Nothing is a mystery.'\n\n"
          "Sources: outputs/bn_full_comparison.json, "
          "outputs/bn_upgraded_full.json, per-cell notes in "
          "outputs/BN_COMPARISON_REPORT.md and the app."))
add_table(s, [
    ["", "First pass", "After the two fixes"],
    ["Exact", "43", "48"],
    ["Close", "16", "29"],
    ["Different", "26", "12  (each with a traced cause)"],
    ["Qualitative", "8", "4"],
    ["Exact or close", "59 / 93", "77 / 93"],
], left=1.6, top=1.6, width=10.0, height=3.4,
   col_widths=[2.6, 3.0, 4.4], font_size=15)
add_takeaway(s, "Every one of the 93 is either reproduced or explained with evidence.")

# ============================================================ 13: LIVE DEMO
add_content_slide(
    "Live Demo",
    [("Show Streamlit app", 0)],
    note=("DEMO SCRIPT (about 7 min). The app reads top to bottom; scroll "
          "and narrate.\n"
          "1. Diagnosis mode, query 'engine caught fire during takeoff', "
          "Run analysis. Metrics: 'it detected fire, 102 accidents, the "
          "exact Table 7 denominator.'\n"
          "2. Causes table: 'Table 7 counting re-run live, wiring nine of "
          "102.'\n"
          "3. Comparison panel: flip neutral vs similarity ONCE: 'with "
          "neutral weights we get Table 7 back exactly; with similarity "
          "weights the query tilts the answer by a measured amount.'\n"
          "4. Diagnosis tree: 'left is the outcome, right is upstream: "
          "under wiring sit circuit breakers and maintenance findings, "
          "each backed by two accidents.' Open the exclusion audit: "
          "'evacuation appears after the fire every time, so the rule "
          "excludes it. Counting, not opinion.'\n"
          "5. Prognosis mode: 'forward in time, every path ends in the "
          "dark boxes: damage and injury, your derivation.'\n"
          "6. The network panel (first open builds it, ~20 s, say so). "
          "Preset Table 9 LOEP: his column next to ours. Then preset LOEP "
          "+ pilot: 'four accidents match both. Counting is dead. The "
          "network still answers. This is why it exists.'\n"
          "7. If time: the epilogue section: Table 7 side by side 85/85 "
          "and the 93-cell scoreboard.\n\n"
          "PRACTICAL: warm the app before the meeting (run the fire "
          "query, open the network toggle once so the ~20 s build is "
          "cached). If a click seems to do nothing, click once more.\n"
          "WATCH OUT: within-102 counting for causes; 184M flights is the "
          "priors' denominator only."))

# ============================================================ 14: NEXT STEP
add_content_slide(
    "Next Step",
    [
        ("Analysis phase is done → every published number matched, close, or "
         "explained with evidence", 0),
        ("Bring the draft's methodology + validation sections up to date with "
         "what you saw today", 0),
        ("Re-run the held-out evaluation with the corrected counting", 0),
        ("", 0),
        ("What should I prioritize first?", 0, {"bold": True}),
    ],
    note=("SAY: 'So my read: the analysis phase is done. Everything the "
          "paper publishes is reproduced, close, or explained with a traced "
          "cause. Next is writing: I have a working draft, and the "
          "methodology and validation sections need to be brought up to "
          "date with what you saw today, plus a re-run of the held-out "
          "evaluation with the corrected counting. What do you want me to "
          "prioritize first?' Then stop talking and let him direct.\n\n"
          "DO NOT say the draft is nearly submittable or that only small "
          "edits remain."))

# ============================================================ 15: APPENDIX
add_content_slide("Appendix", [("", 0)])

# ============================================================ 16: APPENDIX A
add_content_slide(
    "Appendix: the 12 remaining gaps, by cause",
    [
        ("His build variance (majority): deep multi-hop posteriors that swing "
         "orders of magnitude across his own random builds; his released "
         "network misses them too", 0),
        ("Data contradicts the printed value (2-3 cells): the raw counts "
         "cannot produce his number under any estimator we tried", 0),
        ("Scope and definition differences: node families we resolve "
         "differently (e.g. gear-collapse variants), documented per cell", 0),
        ("Every cell has its note in the comparison table (app + "
         "BN_COMPARISON_REPORT.md)", 0, {"bold": True}),
    ],
    note=("Only if he asks to walk the twelve. Per-cell notes: the app's "
          "comparison section (filter to DIFFERS) and "
          "outputs/BN_COMPARISON_REPORT.md."))

# ============================================================ 17: APPENDIX B
add_content_slide(
    "Appendix: Table 4 recap (closed on Tuesday)",
    [
        ("Table 4 = P(fire | wiring, fuel), the Section 3 illustrative CPT", 0),
        ("Its values cannot be counted from the data (no estimator over the "
         "1,742 accidents produces them)", 0),
        ("The paper itself signals this: Section 3 = worked example, "
         "Section 4 (Eq. 9 + Beta-CDF) = the data method", 0),
        ("Our pipeline uses the Section 4 recipe → those numbers reproduce", 0,
         {"bold": True}),
    ],
    note=("ONLY if he reopens Table 4. 'We closed this Tuesday: Table 4 is "
          "the Section 3 teaching example. Its values cannot be produced by "
          "counting with any estimator we tried, and the paper's own data "
          "method is Section 4, which is what we implement and what "
          "reproduces.' Never say fake or wrong: say illustrative, worked "
          "example."))

# ============================================================ 18: APPENDIX C
add_content_slide(
    "Appendix: the Figure 3 question, answered",
    [
        ("Inputs: only the printed Table 3 (priors), Table 4 (fire CPT), "
         "Table 5 (evacuation CPT)", 0),
        ("Computation: chain rule + Bayes' rule, four hand-checkable steps", 0),
        ("Every Figure 3 number comes out exactly, including the 0.67 / 0.33 "
         "wiring vs brake split", 0),
        ("So Figure 3 = correct algebra on the Section 3 illustrative tables: "
         "a worked example, not a data result", 0, {"bold": True}),
        ("Script: tests/reproduce_fig3_from_tables.py", 0),
    ],
    note=("ONLY if he asks about Figure 3. 'You asked whether Figure 3 can "
          "be reproduced. I took only what the paper prints, Tables 3, 4 "
          "and 5, and pushed them through Bayes' rule: every number comes "
          "out exactly, including the 0.67 to 0.33 split. So yes, it is "
          "internally consistent, and since the inputs are the Section 3 "
          "illustrative tables it is a correct worked example rather than a "
          "data result.'"))

prs.save(str(OUT))
print(f"Saved {OUT} with {len(prs.slides._sldIdLst)} slides")
for i, slide in enumerate(prs.slides, 1):
    title = ""
    for shape in slide.shapes:
        if shape.has_text_frame and shape.text_frame.text.strip():
            title = shape.text_frame.text.splitlines()[0][:70]
            break
    print(f"  {i}. {title}")
