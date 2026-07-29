"""Generate maha_jul31_update.pptx - Friday in-person deck for Maha.

Source: docs_FrozenBN/2026-07-31_maha_friday.md. Plain-language method
names (Maha's Tuesday feedback), leak-safe numbers only, every slide
footer cites the generating file.

Run:
  python3.11 presentations_FrozenBN/build_maha_jul31_pptx.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

TEMPLATE = Path("/Users/kanushetkar/Downloads/Copy of NTSB project update 6_30_26.pptx")
HERE = Path(__file__).resolve().parent
OUT = HERE / "maha_jul31_update.pptx"

prs = Presentation(str(TEMPLATE)) if TEMPLATE.is_file() else Presentation()
if TEMPLATE.is_file():
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
    elif body is not None:
        body.text_frame.clear()
    if note:
        s.notes_slide.notes_text_frame.text = note
    return s


def add_table(slide, rows, left, top, width, height, col_widths=None,
              font_size=13, bold_row=None):
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
            run.font.color.rgb = RGBColor(0, 0, 0)
            if r == 0 or r == bold_row:
                run.font.bold = True
    return tbl


def add_footer(slide, text, top=6.35):
    tb = slide.shapes.add_textbox(Inches(0.9), Inches(top), Inches(11.5),
                                  Inches(0.4))
    p = tb.text_frame.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(90, 90, 90)


def add_takeaway(slide, text, top=5.55, size=16):
    tb = slide.shapes.add_textbox(Inches(0.9), Inches(top), Inches(11.5),
                                  Inches(0.85))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True


# ---- 1 TITLE
s = prs.slides.add_slide(TITLE_LAYOUT)
tp = _placeholder(s, True)
if tp is not None:
    tp.text_frame.text = "NTSB Narrative Project - Results & Paper Status"
sp = _placeholder(s, False)
if sp is not None:
    sp.text_frame.text = "Kanu Shetkar  -  Dr. Mahadevan  -  July 31, 2026"

# ---- 2 WHAT CHANGED SINCE TUESDAY
add_content_slide(
    "What you asked for Tuesday - and what changed",
    [
        ("\u201cClean up the confusing table\u201d \u2192 every method now has a plain-English name", 0),
        ("One table, one metric, metric defined on the same slide", 1),
        ("\u201cExplain everything yourself\u201d \u2192 every slide footer names the file behind its numbers", 0),
        ("All results re-run under the no-leakage protocol \u2014 these are the honest numbers", 0),
    ],
)

# ---- 3 SYSTEM IN ONE SENTENCE
s = add_content_slide(
    "The system in one sentence",
    [
        ("A narrative comes in; a reader turns it into facts the network knows;", 0),
        ("each fact gets a strength counted from the 100 most similar past accidents;", 0),
        ("the frozen network (Zhang's, rebuilt & verified) combines them into probabilities.", 0),
        ("Reader = vocabulary match first; LLM only when the text describes without naming", 1),
        ("The LLM never supplies a number \u2014 every strength is a counted fraction from data", 1),
        ("Nothing in this chain is trained; the network is frozen after the 1982\u20132006 build", 1),
    ],
)
add_takeaway(s, "Narrative \u2192 named facts \u2192 counted strengths \u2192 frozen network \u2192 probabilities.")

# ---- 4 ZHANG MATCH
s = add_content_slide("The bar you set: match Zhang exactly", [])
add_table(s, [
    ["Quantity Zhang published", "Zhang", "Ours", "Status"],
    ["Fire occurrences 1982\u20132006", "102", "102", "exact"],
    ["Prior P(fire)", "5.53e-7", "5.527942e-7", "exact"],
    ["Table 7 cause distribution (published)", "85 rows", "85/85", "exact"],
], 0.9, 1.7, 11.4, 2.2, col_widths=[5.2, 2.2, 2.4, 1.6], font_size=14)
add_takeaway(s, "Everything Zhang published, we reproduce to the digit \u2014 the network we query is his.")
add_footer(s, "docs_FrozenBN/ZHANG_REPRODUCTION_REPORT.md")

# ---- 5 SEVERITY TABLE
s = add_content_slide(
    "Severity on 296 unseen accidents (2007\u20132019)",
    [("Metric: 4-way exact match \u2014 top choice among fatal/serious/minor/none "
      "(injury), destroyed/substantial/minor/none (damage) equals the NTSB code", 0)],
)
add_table(s, [
    ["Method", "Injury", "Damage"],
    ["Always guess the most common outcome", "58.4%", "42.6%"],
    ["Facts parsed from the text \u2192 network", "82.4%", "50.7%"],
    ["Similar-accidents readout \u2192 network (OURS)", "90.9%", "77.4%"],
    ["Trained embedding model (comparison)", "91.6%", "74.0%"],
    ["Trained word-frequency model (comparison)", "92.2%", "73.3%"],
], 0.9, 2.15, 11.4, 2.9, col_widths=[7.4, 2.0, 2.0], font_size=14, bold_row=3)
add_takeaway(s, "Outcome sentences are stripped before ANY use of the text \u2014 the honest numbers "
                "(93/81 was pre-fix). Statistical ties with both trained models; stronger on severe "
                "damage \u2014 and zero training.",
             top=5.35, size=14)
add_footer(s, "outputs/heldout_significance.md")

# ---- 6 WHAT ACCURACY HIDES
add_content_slide(
    "Because you'll ask what accuracy hides",
    [
        ("Class balance: accuracy flatters \u201calways none\u201d \u2192 we also report Macro-F1", 0),
        ("Ours 0.470 injury / 0.697 damage  vs  baseline 0.184 / 0.149", 1),
        ("Triage view \u2014 \u201cis this severe?\u201d", 0),
        ("Injury: 93.5% of severe cases caught, 96.8% of non-severe correctly cleared", 1),
        ("Damage: 75.3% caught, 89.8% cleared", 1),
        ("Honest misses: fatal (3 cases) and minor injury (16) never ranked top-1 \u2014 stated in the paper", 0),
        ("Every comparison carries an exact McNemar test and a bootstrap confidence interval", 0),
    ],
)

# ---- 7 DIAGNOSIS
s = add_content_slide(
    "NEW: diagnosis on unseen accidents (cause categories)",
    [("NTSB changed coding systems in 2008, so both eras roll up to the four "
      "official top-level cause categories; correct = top category is among "
      "the coded causes (n=253)", 0)],
)
add_table(s, [
    ["Method", "Top-1", "Rank quality (MRR)"],
    ["Always guess the most common category", "45.8%", "0.685"],
    ["Facts parsed from the text \u2192 network", "57.7%", "0.759"],
    ["Similar-accidents vote (OURS)", "83.8%", "0.912"],
    ["Trained model on the same text (comparison)", "88.1%", "0.936"],
], 0.9, 2.15, 11.4, 2.6, col_widths=[7.0, 1.8, 2.6], font_size=14, bold_row=3)
add_takeaway(s, "Balanced across Personnel/Aircraft/Environment (68/62/72% recall). Disclosed: "
                "trained model +4 pts (p=0.03); nobody catches the rare Organizational class.",
             top=5.15, size=14)
add_footer(s, "outputs/diagnosis_heldout_eval.md  -  outputs/diagnosis_emb_lr.md")

# ---- 8 WHAT THE NETWORK CONTRIBUTES
s = add_content_slide(
    "What the network itself contributes",
    [
        ("The predictive signal comes from the narratives (similar-accident evidence)", 0),
        ("The network preserves it exactly \u2014 verified by a built-in self-test \u2014 and adds:", 0),
        ("A verified causal structure (slide 4: Zhang match to the digit)", 1),
        ("Joint reasoning: several pieces of evidence combined in one coherent calculation", 1),
        ("What-if questions (\u201csame accident but IMC at night?\u201d) a counting table can't answer", 1),
    ],
)
add_takeaway(s, "Narratives give the signal; the network makes it explainable and interrogable.")

# ---- 9 PAPER STATUS
add_content_slide(
    "Paper status",
    [
        ("Results section: written; every number regenerates from one command", 0),
        ("docs_FrozenBN/RESULTS_SECTION.md + REPRODUCE.md", 1),
        ("\u201cWhat is trained?\u201d inventory: frozen / measured / selected \u2014 nothing fitted in the chain", 0),
        ("Leakage protocol + two audits: written and committed", 0),
        ("Remaining: intro & related-work polish, figures, your pass on the slide-8 framing", 0),
    ],
)

# ---- 10 PLAN
s = add_content_slide(
    "Plan to submission",
    [
        ("This week: full draft assembled from the verified sections", 0),
        ("You receive: draft + reproduction commands + one-page crib sheet", 0),
        ("Decision I need from you:", 0),
        ("Lead the paper with \u201cprediction from narratives, reasoning from the network\u201d?", 1),
    ],
)
add_takeaway(s, "Ask: confirm the framing Friday so the draft locks this weekend.")

prs.save(str(OUT))
print(f"wrote {OUT} ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
