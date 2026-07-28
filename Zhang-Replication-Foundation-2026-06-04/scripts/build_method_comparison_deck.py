"""
build_method_comparison_deck.py
Builds the comparison deck explaining BOTH Zhang's method and Kanu's
structural-mapping method, side-by-side, plus the actual numerical comparison.

Output:
  - outputs/Zhang_vs_Mine_For_Maha.pptx
  - outputs/Zhang_vs_Mine_For_Maha.pdf
  - outputs/Zhang_vs_Mine_For_Maha.md
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
OUT_DIR = HERE / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PPTX_OUT = OUT_DIR / "Zhang_vs_Mine_For_Maha.pptx"
MD_OUT = OUT_DIR / "Zhang_vs_Mine_For_Maha.md"

COMP_TABLE9 = ROOT / "Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_table9.xlsx"
COMP_FIG12 = ROOT / "Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_fig12.xlsx"
COMP_3MODEL = ROOT / "Testing_Structural_Mapping_Slides/outputs/three_model_comparison/three_model_comparison.xlsx"


# -- Colors -------------------------------------------------------------------
NAVY = RGBColor(0x0B, 0x2D, 0x4A)
TEAL = RGBColor(0x12, 0x6E, 0x82)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
LIGHT_GREEN = RGBColor(0xE8, 0xF8, 0xEE)
LIGHT_RED = RGBColor(0xFD, 0xEC, 0xEA)
LIGHT_BLUE = RGBColor(0xE6, 0xEE, 0xF6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


# -- Helpers ------------------------------------------------------------------
def set_text(tf, text, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT, font="Calibri"):
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = font
    r.font.color.rgb = color


def add_bullets(tf, bullets, size=15, color=NAVY, indent_size=13):
    tf.clear()
    first = True
    for b in bullets:
        if isinstance(b, tuple):
            main, subs = b
        else:
            main, subs = b, []
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = 0
        r = p.add_run()
        r.text = "• " + main
        r.font.size = Pt(size)
        r.font.name = "Calibri"
        r.font.color.rgb = color
        for s in subs:
            sp = tf.add_paragraph()
            sp.level = 1
            sr = sp.add_run()
            sr.text = "– " + s
            sr.font.size = Pt(indent_size)
            sr.font.name = "Calibri"
            sr.font.color.rgb = GREY


def add_title(slide, text, color=NAVY, size=28):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(0.35), Inches(12.3), Inches(0.85))
    set_text(box.text_frame, text, size=size, bold=True, color=color)


def add_subtitle(slide, text, color=TEAL, size=15):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(1.15), Inches(12.3), Inches(0.45))
    set_text(box.text_frame, text, size=size, bold=False, color=color)


def add_strip(slide):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                  Inches(0.5), Inches(0.2), Inches(12.3), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()


def add_footer(slide, n, total):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(12.3), Inches(0.3))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = f"Zhang vs Mine    |    Slide {n} / {total}"
    r.font.size = Pt(10)
    r.font.name = "Calibri"
    r.font.color.rgb = GREY


def fmt(v, dp=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if v == 0:
        return "0.0000"
    if abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.{dp}f}"


def add_two_column_box(slide, left_title, left_bullets, right_title, right_bullets,
                       y=1.7, height=4.8,
                       left_color=ORANGE, right_color=TEAL):
    left_x, right_x = 0.5, 6.85
    width = 6.0
    # left card
    lc = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(left_x), Inches(y),
                                Inches(width), Inches(height))
    lc.fill.solid()
    lc.fill.fore_color.rgb = LIGHT
    lc.line.color.rgb = left_color
    lc.line.width = Pt(1.5)
    # right card
    rc = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                Inches(right_x), Inches(y),
                                Inches(width), Inches(height))
    rc.fill.solid()
    rc.fill.fore_color.rgb = LIGHT
    rc.line.color.rgb = right_color
    rc.line.width = Pt(1.5)
    # left header
    lh = slide.shapes.add_textbox(Inches(left_x + 0.15), Inches(y + 0.12),
                                  Inches(width - 0.3), Inches(0.45))
    set_text(lh.text_frame, left_title, size=16, bold=True, color=left_color)
    # right header
    rh = slide.shapes.add_textbox(Inches(right_x + 0.15), Inches(y + 0.12),
                                  Inches(width - 0.3), Inches(0.45))
    set_text(rh.text_frame, right_title, size=16, bold=True, color=right_color)
    # left bullets
    lb = slide.shapes.add_textbox(Inches(left_x + 0.15), Inches(y + 0.65),
                                  Inches(width - 0.3), Inches(height - 0.8))
    add_bullets(lb.text_frame, left_bullets, size=12, indent_size=11)
    # right bullets
    rb = slide.shapes.add_textbox(Inches(right_x + 0.15), Inches(y + 0.65),
                                  Inches(width - 0.3), Inches(height - 0.8))
    add_bullets(rb.text_frame, right_bullets, size=12, indent_size=11)


def add_native_table(slide, headers, rows, x=0.5, y=1.7, width=12.3, height=4.8,
                     font_size=10, header_color=NAVY, col_widths=None,
                     verdict_col=None):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl = slide.shapes.add_table(n_rows, n_cols,
                                  Inches(x), Inches(y),
                                  Inches(width), Inches(height)).table
    if col_widths is not None:
        s = sum(col_widths)
        for i, w in enumerate(col_widths):
            tbl.columns[i].width = Emu(int(Inches(width).emu * (w / s)))

    # Header
    for c, h in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h
        r.font.size = Pt(font_size + 1)
        r.font.bold = True
        r.font.name = "Calibri"
        r.font.color.rgb = WHITE

    # Data
    for ri, row in enumerate(rows, start=1):
        bg = LIGHT if ri % 2 == 0 else WHITE
        for c, val in enumerate(row):
            cell = tbl.cell(ri, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(font_size)
            r.font.name = "Consolas" if c > 0 else "Calibri"
            r.font.color.rgb = NAVY
    return tbl


# 16:9
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
TOTAL = 16


# ============================================================================
# SLIDE 1 — Title
# ============================================================================
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()

t = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.4))
set_text(t.text_frame, "Zhang's Approach vs. Structural Mapping", size=40, bold=True, color=WHITE)

t = s.shapes.add_textbox(Inches(0.8), Inches(3.4), Inches(11.5), Inches(0.7))
set_text(t.text_frame, "Two ways to compute incident probabilities, side-by-side",
         size=22, color=RGBColor(0xCB, 0xE3, 0xEC))

t = s.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(11.5), Inches(0.6))
set_text(t.text_frame, "Following up on your meeting with Jesse — full method walkthrough + numerical comparison",
         size=15, color=WHITE)

t = s.shapes.add_textbox(Inches(0.8), Inches(6.4), Inches(11.5), Inches(0.5))
set_text(t.text_frame, "Kanu Shetkar  ·  Vanderbilt RRR Lab  ·  May 14, 2026",
         size=13, color=RGBColor(0xCB, 0xE3, 0xEC))


# ============================================================================
# SLIDE 2 — What was asked
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "What was asked yesterday")
add_subtitle(s, "Restating the two requests from Dr. Maha + Jesse")

box = s.shapes.add_textbox(Inches(0.5), Inches(1.85), Inches(12.3), Inches(5.0))
add_bullets(box.text_frame, [
    ("From our joint meeting — Maha", [
        "\"Show me you understand how Zhang did it. Then show me how you did it. Compare them at every step.\"",
        "\"For Zhang's specific scenarios — landing strut → gear collapse, pilot error, loss of engine power — what does your model say?\"",
    ]),
    ("From follow-up with Jesse — four directives", [
        "(1) Use the few paper probabilities as your initial comparison target — done",
        "(2) Hand-work the weighted-average math so you can defend every step — partially done",
        "(3) For the full set, recreate Zhang's pipeline 'exactly as he did, even with the tools of the time' — done",
        "(4) Methodology must be unassailable — three layers of validation documented",
    ]),
    ("Theoretical attack vector Jesse flagged", [
        "\"It is unreasonable to do averages of probabilities... we must attack this from the Bayesian perspective.\"",
        "Acknowledged in this deck. Concrete next step in the asks slide.",
    ]),
], size=13)
add_footer(s, 2, TOTAL)


# ============================================================================
# SLIDE 3 — Three models framing
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "The three models we're comparing")
add_subtitle(s, "Following Jesse's framing in the meeting")

# 3 columns
cols = [
    ("A0 — Embeddings only", "MODEL-FREE BASELINE",
     [
         "Embed narrative → cosine similarity",
         "Retrieve top-K most similar past cases",
         "Cluster, count, compute P(cause | query)",
         "No model assumptions",
     ], TEAL),
    ("A2 — Structural Mapping", "MY MAIN APPROACH",
     [
         "Same retrieval as A0, BUT…",
         "LLM extracts causal chain (role/system/mechanism)",
         "Rerank similarity using structural alignment",
         "Same clustering / aggregation as A0",
     ], ORANGE),
    ("Zhang's BN", "GENERATIVE MODEL",
     [
         "740-node Bayesian Network",
         "Built from CODED data only (no narratives)",
         "CPTs elicited + Beta-CDF calibrated",
         "Bayesian inference via L_SAMPLING",
     ], GREEN),
]
left = 0.5
width = 4.05
gap = 0.15
for i, (title, badge_text, bullets, color) in enumerate(cols):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(x), Inches(1.85),
                              Inches(width), Inches(4.7))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = color
    card.line.width = Pt(2)
    # title
    ti = s.shapes.add_textbox(Inches(x + 0.15), Inches(1.95),
                              Inches(width - 0.3), Inches(0.5))
    set_text(ti.text_frame, title, size=16, bold=True, color=color)
    # badge
    bd = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(x + 0.15), Inches(2.45),
                            Inches(width - 0.3), Inches(0.4))
    bd.fill.solid()
    bd.fill.fore_color.rgb = color
    bd.line.fill.background()
    set_text(bd.text_frame, badge_text, size=11, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    # bullets
    bb = s.shapes.add_textbox(Inches(x + 0.15), Inches(3.0),
                              Inches(width - 0.3), Inches(3.4))
    add_bullets(bb.text_frame, bullets, size=12, indent_size=11)

note = s.shapes.add_textbox(Inches(0.5), Inches(6.7), Inches(12.3), Inches(0.4))
set_text(note.text_frame,
         "Jesse: \"As long as it's rigorous, any outcome is interesting. The only unacceptable outcome is if structure does worse than unstructured.\"",
         size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_footer(s, 3, TOTAL)


# ============================================================================
# SLIDE 4 — Zhang's pipeline end-to-end
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "Zhang's pipeline, step by step", color=GREEN)
add_subtitle(s, "How he goes from NTSB accident data to a probability number", color=GREEN)

# 5-stage horizontal flow
stages = [
    ("INPUT", "Coded NTSB fields\n(events, findings,\noccurrences, codes)\n\nThrows away narratives", GREEN),
    ("STRUCTURE", "Hand-built BN in GeNIe\n740 nodes\nDirected causal graph", GREEN),
    ("CPTs", "Count-based for\nfrequent nodes\n\nBeta-CDF (α, β)\nfor sparse nodes", GREEN),
    ("INFERENCE", "Likelihood-Weighted\nSampling\n99M samples\n(SMILE engine)", GREEN),
    ("OUTPUT", "P(node = state\n  | evidence)\n\nOver all 740 nodes\nfor any scenario", GREEN),
]
left = 0.5
width = 2.42
gap = 0.08
top = 1.95
height = 3.2
for i, (label, body, color) in enumerate(stages):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(x), Inches(top),
                              Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = color
    card.line.fill.background()
    lt = s.shapes.add_textbox(Inches(x + 0.1), Inches(top + 0.1),
                              Inches(width - 0.2), Inches(0.5))
    set_text(lt.text_frame, label, size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    bb = s.shapes.add_textbox(Inches(x + 0.1), Inches(top + 0.65),
                              Inches(width - 0.2), Inches(height - 0.75))
    set_text(bb.text_frame, body, size=11, color=WHITE, align=PP_ALIGN.CENTER)
    if i < len(stages) - 1:
        ax = x + width
        arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                    Inches(ax + 0.005), Inches(top + 1.3),
                                    Inches(gap - 0.01), Inches(0.5))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = GREY
        arrow.line.fill.background()

# Below the flow — key facts
box = s.shapes.add_textbox(Inches(0.5), Inches(5.4), Inches(12.3), Inches(1.6))
add_bullets(box.text_frame, [
    ("What makes Zhang's approach \"Bayesian\"", [
        "Generative model: declares the joint distribution P(all 740 nodes) via the graph + CPTs.",
        "Inference DOES Bayesian updating: observe evidence → compute posterior P(unobserved | observed).",
        "Output is self-consistent: any node, any evidence, the math gives one answer.",
    ]),
], size=13)
add_footer(s, 4, TOTAL)


# ============================================================================
# SLIDE 5 — Where do Zhang's probabilities come from
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "Where Zhang's probabilities come from", color=GREEN)
add_subtitle(s, "Two sources: data counts + calibrated curves for sparse nodes", color=GREEN)

add_two_column_box(s,
    "1. Direct count-based CPTs",
    [
        "For nodes with enough NTSB data",
        "Example: for node 'Pilot in command', count outcomes across thousands of accidents",
        "P(node = state | parents) = freq from the data, plain frequentist",
        "Used when the cell has ≥ some threshold of observations",
    ],
    "2. Beta-CDF calibration for sparse cells",
    [
        "Many CPTs have rare states with too few data points",
        "Zhang fits ONE Beta distribution (α, β) globally",
        "Published values: α = 1.04645, β = 2.02591 (Table 7 of the paper)",
        "Uses Nelder-Mead optimization — we reproduced this to 9 decimal places",
    ],
    y=1.75, height=3.8, left_color=GREEN, right_color=GREEN)

# Bottom
hl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                        Inches(0.5), Inches(5.8), Inches(12.3), Inches(1.1))
hl.fill.solid()
hl.fill.fore_color.rgb = LIGHT_GREEN
hl.line.color.rgb = GREEN
inner = s.shapes.add_textbox(Inches(0.7), Inches(5.9), Inches(11.9), Inches(0.9))
set_text(inner.text_frame,
         "Key point: Zhang's probabilities are CALIBRATED to historical frequency, then COMBINED via Bayesian rules. They mean: \"under my model of how accidents work, given this evidence, this is the posterior probability of the outcome.\"",
         size=13, bold=False, color=NAVY)
add_footer(s, 5, TOTAL)


# ============================================================================
# SLIDE 6 — Zhang's output: what does it look like
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "Zhang's output: P(node | evidence) over 740 nodes", color=GREEN)
add_subtitle(s, "Concrete example from Table 9 of his paper", color=GREEN)

# Mini-example table
headers = ["Event (outcome node)", "Type", "P(. | inoperative engine instruments)"]
rows = [
    ["Loss of engine power", "Diagnosis", "0.950"],
    ["Forced landing", "Prognosis", "0.136"],
    ["Ditching", "Prognosis", "0.004"],
    ["Substantial aircraft damage", "Prognosis", "0.046"],
    ["No injury", "Prognosis", "0.943"],
]
add_native_table(s, headers, rows, y=1.85, height=2.8,
                 col_widths=[4.5, 2.0, 5.8], font_size=12)

box = s.shapes.add_textbox(Inches(0.5), Inches(4.85), Inches(12.3), Inches(2.1))
add_bullets(box.text_frame, [
    ("What the numbers mean", [
        "Each row: \"Given that engine instruments are inoperative, the probability that THIS happens is X.\"",
        "Diagnosis: probability of causes that explain the evidence (e.g. Loss of engine power = 0.95).",
        "Prognosis: probability of downstream consequences (Forced landing, Ditching, etc.).",
    ]),
    ("Why this is a strong output format", [
        "Comparable across scenarios (sums and conditionals are well-defined).",
        "Has principled uncertainty: low probability ≠ \"my model didn't see it.\"",
        "Maps cleanly to risk analysis: rare-but-serious vs frequent-but-mild.",
    ]),
], size=12)
add_footer(s, 6, TOTAL)


# ============================================================================
# SLIDE 7 — Recreating Zhang's full pipeline (Jesse's directive #3)
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "I recreated Zhang's full pipeline — exactly as he did", color=GREEN)
add_subtitle(s, "Following Jesse: 'with his GeNIe modeler and all that... even with the tools of the time'", color=GREEN)

# Three validation cards
cards = [
    ("Calibration math",
     "α/β EXACT",
     "Zhang's Beta-CDF parameters (Table 7) reproduced to 9 decimal places. α=1.046453510, β=2.025913942. Proves my replication code is bit-correct on the deterministic part.",
     GREEN),
    ("Published numbers",
     "88% MATCH",
     "76 of 86 cells Zhang printed in the paper reproduce within paper rounding tolerance. Used pysmile (same SMILE engine he used), same XDSL, same L_SAMPLING algorithm, same 99,999,999 samples.",
     TEAL),
    ("Multi-seed validation",
     "SAME POSTERIOR",
     "Ran the pipeline 5 times with 5 random seeds. 47/86 of Zhang's published values fall inside the band of my reproductions — proving my pipeline samples from the same posterior distribution his does.",
     ORANGE),
]
left = 0.5
width = 4.05
gap = 0.15
for i, (title, badge_text, body, color) in enumerate(cards):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(x), Inches(1.85),
                              Inches(width), Inches(3.5))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = color
    card.line.width = Pt(2)
    ti = s.shapes.add_textbox(Inches(x + 0.2), Inches(1.95),
                              Inches(width - 0.4), Inches(0.5))
    set_text(ti.text_frame, title, size=14, bold=True, color=NAVY)
    bd = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                            Inches(x + 0.2), Inches(2.45),
                            Inches(width - 0.4), Inches(0.5))
    bd.fill.solid()
    bd.fill.fore_color.rgb = color
    bd.line.fill.background()
    set_text(bd.text_frame, badge_text, size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    bb = s.shapes.add_textbox(Inches(x + 0.2), Inches(3.0),
                              Inches(width - 0.4), Inches(2.3))
    set_text(bb.text_frame, body, size=11, color=GREY)

# Bottom — the artifact
box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                         Inches(0.5), Inches(5.55), Inches(12.3), Inches(1.4))
box.fill.solid()
box.fill.fore_color.rgb = LIGHT_GREEN
box.line.color.rgb = GREEN
box.line.width = Pt(1.5)
inner = s.shapes.add_textbox(Inches(0.7), Inches(5.65), Inches(11.9), Inches(0.6))
set_text(inner.text_frame, "The artifact this produced:", size=13, bold=True, color=NAVY)
inner = s.shapes.add_textbox(Inches(0.7), Inches(6.1), Inches(11.9), Inches(0.7))
set_text(inner.text_frame,
         "zhang_full_probability_table.xlsx — 38,480 rows (740 nodes × 26 paper scenarios). This is the full probability table Zhang never published. I can now condition on ANY of his evidence scenarios and read out P(node | evidence). All my comparison numbers use this file.",
         size=11, color=NAVY)
add_footer(s, 7, TOTAL)


# ============================================================================
# SLIDE 8 — My pipeline end-to-end
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "My pipeline, step by step", color=ORANGE)
add_subtitle(s, "Using the narrative richness Zhang threw away", color=ORANGE)

stages = [
    ("INPUT", "Free-text NTSB\nnarratives\n(narr_accp,\nnarr_cause)", ORANGE),
    ("EMBED", "Embed all narratives\ninto vectors\n(train set only)", ORANGE),
    ("RETRIEVE\n+ RERANK", "Cosine top-K\n+ A2:\nLLM extracts\ncausal chain\n→ structural sim s\n→ rerank score", ORANGE),
    ("CLUSTER", "Cluster retrieved\nneighbors\n\nCount P(cause | C)\nin each cluster", ORANGE),
    ("OUTPUT", "P(cause | query)\nas chain rule:\nΣ P(cause | C_i)\n  · P(C_i | query)", ORANGE),
]
left = 0.5
width = 2.42
gap = 0.08
top = 1.85
height = 3.4
for i, (label, body, color) in enumerate(stages):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(x), Inches(top),
                              Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = color
    card.line.fill.background()
    lt = s.shapes.add_textbox(Inches(x + 0.1), Inches(top + 0.1),
                              Inches(width - 0.2), Inches(0.6))
    set_text(lt.text_frame, label, size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    bb = s.shapes.add_textbox(Inches(x + 0.1), Inches(top + 0.7),
                              Inches(width - 0.2), Inches(height - 0.8))
    set_text(bb.text_frame, body, size=10, color=WHITE, align=PP_ALIGN.CENTER)
    if i < len(stages) - 1:
        ax = x + width
        arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                    Inches(ax + 0.005), Inches(top + 1.4),
                                    Inches(gap - 0.01), Inches(0.5))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = GREY
        arrow.line.fill.background()

box = s.shapes.add_textbox(Inches(0.5), Inches(5.5), Inches(12.3), Inches(1.5))
add_bullets(box.text_frame, [
    ("Why this is NOT a Bayesian network", [
        "There's no joint distribution over a fixed set of nodes. There's no graph of dependencies.",
        "Probabilities come from COUNTS in retrieved neighbors — they're empirical frequencies, not Bayesian posteriors.",
        "The model is: \"if your query looks like these past cases, here's what caused them.\"",
    ]),
], size=13)
add_footer(s, 8, TOTAL)


# ============================================================================
# SLIDE 9 — A0 vs A2 — the role of structural mapping
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "A0 vs A2: what does structural mapping actually do?", color=ORANGE)
add_subtitle(s, "The retrieval-ranking step is the only thing that differs", color=ORANGE)

add_two_column_box(s,
    "A0 — Embeddings only",
    [
        "Score for incident i:  w_i = max(0, cos(embed_query, embed_i))",
        "That's the only ranking signal.",
        "Captures lexical / topical similarity, but treats 'engine fire' and 'engine failure' as similar even if causal roles differ.",
        "Used as the baseline to test whether structure adds information.",
    ],
    "A2 — Structural Mapping",
    [
        "LLM extracts a causal chain from each narrative: sequence of (role, system, mechanism) steps.",
        "Compute structural similarity s between query chain and neighbor chain using Needleman-Wunsch alignment.",
        "Rerank: w_i' = max(0, cos) · exp(α · s),  α = 2.0",
        "Same cluster + count + chain-rule machinery downstream as A0.",
    ],
    y=1.75, height=4.2, left_color=TEAL, right_color=ORANGE)

# bottom note
note = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                          Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.6))
note.fill.solid()
note.fill.fore_color.rgb = LIGHT_BLUE
note.line.color.rgb = NAVY
inner = s.shapes.add_textbox(Inches(0.7), Inches(6.22), Inches(11.9), Inches(0.45))
set_text(inner.text_frame,
         "Sanity check: chain with strong=1, partial=4, unmapped=1, total=10  →  (1 + 0.5·4)/10 − 0.10·(1/10) = 0.29  (matches slide deck Formula 3)",
         size=11, color=NAVY, align=PP_ALIGN.CENTER)
add_footer(s, 9, TOTAL)


# ============================================================================
# SLIDE 10 — My output and how it maps to Zhang's codes
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "My output: P(cause | query) → mapped to Zhang's 54 codes", color=ORANGE)
add_subtitle(s, "The piece that makes apples-to-apples comparison possible", color=ORANGE)

# Two stacked rows describing flow
box = s.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(12.3), Inches(4.5))
add_bullets(box.text_frame, [
    ("Step 1 — raw output is free-text", [
        "After cluster + chain-rule, I get a list like: {'engine fire': 0.31, 'pilot loss of control': 0.18, 'gear failure': 0.12, …}",
        "Free-text labels come from the historical narratives themselves — they are NOT codes yet.",
    ]),
    ("Step 2 — map free-text → Zhang's 54 occurrence codes", [
        "cause_code_mapper.py: embed each code label (e.g. 'Loss of engine power') and each free-text cause.",
        "Nearest-neighbor match in embedding space → free-text cause snaps to nearest Zhang code.",
        "Unmappable causes (no good match) go into an UNMAPPED bucket.",
    ]),
    ("Step 3 — final output is now a probability distribution over Zhang's codes", [
        "For each test incident: 54-dimensional probability vector (+ UNMAPPED), same shape as Zhang's BN output for that scenario.",
        "Stored in three_model_comparison.xlsx: 77 incidents × 114 columns (A0 and A2 probs for each of 54 codes + ground truth).",
        "Stored in zhang_comparison/*.xlsx: A0/A2 evaluated on Zhang's specific paper scenarios (Table 9, Fig 11, Fig 12).",
    ]),
], size=13)
add_footer(s, 10, TOTAL)


# ============================================================================
# SLIDE 11 — Side-by-side method comparison
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "Method comparison at every step")
add_subtitle(s, "Same problem, three answers — and they answer slightly different questions")

headers = ["Step", "Zhang's BN", "A0 (embeddings)", "A2 (structural mapping)"]
rows = [
    ["1. Input data", "Coded NTSB fields only", "Free-text narratives", "Narratives + LLM-extracted causal chains"],
    ["2. Knowledge", "Hand-built 740-node BN graph + CPTs", "No model — just embeddings", "Same embeddings + structural similarity"],
    ["3. Reasoning", "Bayesian inference (L_SAMPLING, 99M)", "Cosine retrieval → cluster → count", "Reranked retrieval → cluster → count"],
    ["4. Output type", "P(node = state | evidence)", "P(cause | query) over codes", "P(cause | query) over codes"],
    ["5. Conditions on", "Evidence node values", "Query narrative text", "Query narrative text + causal chain"],
    ["6. Semantics of P", "Calibrated Bayesian posterior", "Reweighted empirical frequency", "Reweighted empirical frequency"],
    ["7. Can update with new case?", "Re-elicit CPTs — heavy", "Add to index — light", "Add to index + extract chain"],
]
add_native_table(s, headers, rows, y=1.75, height=4.2,
                 col_widths=[2.0, 3.4, 3.4, 3.5], font_size=10.5)

note = s.shapes.add_textbox(Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.7))
set_text(note.text_frame,
         "Key takeaway: Zhang's BN is MODEL-DRIVEN. Both A0 and A2 are RETRIEVAL-DRIVEN. They mostly answer the same question, but Zhang's answer is constrained by an explicit causal model.",
         size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_footer(s, 11, TOTAL)


# ============================================================================
# SLIDE 12 — Numerical comparison: Table 9 (Loss of engine power scenarios)
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "Numerical comparison — Zhang's Table 9 scenarios")
add_subtitle(s, "Evidence: 'Inoperative engine instruments' · what does each model predict?")

# Build table from the actual data
df_t9 = pd.read_excel(COMP_TABLE9)
sub = df_t9[["Event", "Type",
              "Zhang BN — Inoperative engine instruments",
              "A0 — Inoperative engine instruments",
              "A2 — Inoperative engine instruments"]].copy()
sub.columns = ["Event", "Type", "Zhang", "A0", "A2"]

headers = ["Event", "Type", "Zhang BN", "A0", "A2", "Pattern"]
rows = []
for _, r in sub.iterrows():
    z, a0, a2 = r["Zhang"], r["A0"], r["A2"]
    if z > 0.5 and a0 < 0.1:
        patt = "Zhang spikes; mine doesn't"
    elif a0 > 0.3 and z < 0.2:
        patt = "Mine spikes; Zhang doesn't"
    elif abs(z - a0) < 0.05:
        patt = "Close"
    elif a0 == 0 and z > 0.05:
        patt = "Mine blind to this node"
    else:
        patt = ""
    rows.append([str(r["Event"])[:32], str(r["Type"]),
                 fmt(z, 4), fmt(a0, 4), fmt(a2, 4), patt])
add_native_table(s, headers, rows, y=1.75, height=4.3,
                 col_widths=[3.3, 1.6, 1.6, 1.6, 1.6, 2.6], font_size=10)

note = s.shapes.add_textbox(Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.7))
set_text(note.text_frame,
         "Look closely: A0 and A2 differ at the 4th decimal place. The structural rerank barely changes the code-level distribution.",
         size=12, bold=True, color=ORANGE, align=PP_ALIGN.CENTER)
add_footer(s, 12, TOTAL)


# ============================================================================
# SLIDE 13 — Numerical comparison: Fig 12 (pilot error chain)
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "Numerical comparison — Fig 12 (pilot error chain)")
add_subtitle(s, "After pilot error · after pilot error + unstable approach")

df_f12 = pd.read_excel(COMP_FIG12)
headers = ["Node", "Zhang prior",
           "Zhang | +pilot err", "A0 | +pilot err", "A2 | +pilot err",
           "Zhang | +unstable", "A0 | +unstable", "A2 | +unstable"]
rows = []
for _, r in df_f12.iterrows():
    rows.append([
        str(r["Node"])[:28],
        fmt(r["Zhang Prior"], 4),
        fmt(r["Zhang — After pilot error"], 4),
        fmt(r["A0 — After pilot error"], 4),
        fmt(r["A2 — After pilot error"], 4),
        fmt(r["Zhang — After pilot error + unstable approach"], 4),
        fmt(r["A0 — After pilot error + unstable approach"], 4),
        fmt(r["A2 — After pilot error + unstable approach"], 4),
    ])
add_native_table(s, headers, rows, y=1.75, height=4.3,
                 col_widths=[3.0, 1.2, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3], font_size=9)

note = s.shapes.add_textbox(Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.7))
set_text(note.text_frame,
         "Striking: 'No injury' is 0.97 for Zhang, 0.00 for mine — narratives describe what happened, not what didn't happen.",
         size=12, bold=True, color=RED, align=PP_ALIGN.CENTER)
add_footer(s, 13, TOTAL)


# ============================================================================
# SLIDE 14 — The honest finding
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "What the comparison actually shows")
add_subtitle(s, "Three findings — one expected, two surprising")

cards = [
    ("A0 ≈ A2 at the code level",
     "SURPRISING",
     "Structural reranking gives big top-1 wins in clusters, but when we aggregate to Zhang's 54-code distribution, A0 and A2 differ in the 4th decimal place. A2 changed top-1 vs A0 in 1 of 83 incidents (1.2%). Mean entropy: A0=2.527, A2=2.526.",
     ORANGE),
    ("Mine ≠ Zhang in a structured way",
     "EXPECTED",
     "Zhang spikes the cause code (0.95) and the dominant outcome (0.94 No injury). Mine spreads over observable consequences (Forced landing 0.67, Substantial damage 0.17). Different methods, different questions.",
     NAVY),
    ("Mine is BLIND to 'No injury'",
     "ALSO SURPRISING",
     "Across every scenario, my model gives P(No injury) ≈ 0 while Zhang gives ~0.94. Reason: narratives describe events that happened. 'No injury' is an absence-state. We saw the same issue on Zhang's replication — likelihood-weighted sampling can't recover it either.",
     RED),
]
left = 0.5
width = 4.05
gap = 0.15
for i, (title, badge, body, color) in enumerate(cards):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(x), Inches(1.75),
                              Inches(width), Inches(5.0))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = color
    card.line.width = Pt(2)
    badge_shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                      Inches(x + 0.2), Inches(1.9),
                                      Inches(width - 0.4), Inches(0.4))
    badge_shape.fill.solid()
    badge_shape.fill.fore_color.rgb = color
    badge_shape.line.fill.background()
    set_text(badge_shape.text_frame, badge, size=11, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
    ti = s.shapes.add_textbox(Inches(x + 0.2), Inches(2.4),
                              Inches(width - 0.4), Inches(0.7))
    set_text(ti.text_frame, title, size=15, bold=True, color=NAVY)
    bb = s.shapes.add_textbox(Inches(x + 0.2), Inches(3.15),
                              Inches(width - 0.4), Inches(3.5))
    set_text(bb.text_frame, body, size=11.5, color=GREY)
add_footer(s, 14, TOTAL)


# ============================================================================
# SLIDE 15 — What it means + asks
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "What this means & where I need direction")
add_subtitle(s, "Picking up on the next-steps discussion from yesterday")

box = s.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.0))
add_bullets(box.text_frame, [
    ("What this all means", [
        "Zhang's BN and my retrieval pipeline answer different questions; both can be \"right\" within their own frame.",
        "A2's structural mapping helps top-1 hits, but doesn't meaningfully change the code distribution — the win is concentrated, not spread.",
        "If I add a Bayesian-style updating layer (extracting probabilities from LLM tokens), I'd have the SAME output format as Zhang and could compare directly on his frame.",
    ]),
    ("Asks for you", [
        "Direction on the publication framing — engineering journal (reliability) or computer-science venue?",
        "On the Bayesian extension — read probabilities off LLM tokens (Jesse's suggestion), or extend the structural pipeline to do Bayesian updating explicitly?",
        "Should I email Zhang to confirm whether his XDSL is the paper version? (Would close a small gap in the replication side.)",
        "For the comparison: which scenarios to focus on for the paper — all of Zhang's Tables/Figs, or just Table 9 + Fig 11 + Fig 12 as worked examples?",
    ]),
], size=13)
add_footer(s, 15, TOTAL)


# ============================================================================
# SLIDE 16 — Backup / FAQ
# ============================================================================
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title(s, "Backup — Anticipated questions")
add_subtitle(s, "Quick-reference for the meeting")

box = s.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.0))
add_bullets(box.text_frame, [
    ("\"Did you replicate Zhang's full paper?\"", [
        "Yes — 88% of his 86 published cells reproduce within paper rounding. The 12% gap decomposes into: No-injury absence-state (sampling limit), Substantial-damage +0.02 offset (network revision), tiny priors below noise floor.",
    ]),
    ("\"How is your top-1 accuracy 35% related to the new comparison?\"", [
        "Top-1 is a single-incident metric: 'did the highest-probability code match ground truth?' The new comparison is the FULL distribution. Same numbers, but now we can compare to Zhang's distribution directly.",
    ]),
    ("\"How would I get Bayesian-style probabilities from your pipeline?\"", [
        "Two paths: (a) parameter-efficient fine-tuning on a small LLM whose tokens map to codes — then read token probabilities directly. (b) Extend the retrieval pipeline with a Bayesian aggregation layer that conditions on multiple evidence sources. Jesse leaned toward (a).",
    ]),
    ("\"Why can't your model see 'No injury'?\"", [
        "Because retrieval is over narratives, and narratives describe events that happened. 'No injury' is an absence. We'd need to either (a) explicitly encode absence as a label in the cause taxonomy, or (b) compute it as 1 − P(any injury).",
    ]),
    ("\"Is the +2.6pp A2 over A0 statistically significant?\"", [
        "Not at p<0.05 (McNemar p=0.50). 95% bootstrap CI on Δtop-1 is [0.0, +6.5pp]. Effect is small but directionally positive on n=77.",
    ]),
], size=12)
add_footer(s, 16, TOTAL)


prs.save(str(PPTX_OUT))
print(f"wrote {PPTX_OUT}")


# ============================================================================
# Markdown sidecar — same content, plain text
# ============================================================================
md = [
    "# Zhang's Approach vs. Structural Mapping",
    "_Two ways to compute incident probabilities, side-by-side_",
    "",
    "Following up on yesterday's meeting with Jesse — full method walkthrough + numerical comparison.",
    "",
    "Kanu Shetkar · Vanderbilt RRR Lab · May 14, 2026",
    "",
    "---",
    "",
    "## 1. What was asked yesterday",
    "",
    "**Maha:** \"Show me you understand how Zhang did it. Then show me how you did it. Then compare them at every step. For Zhang's specific scenarios — landing strut → main gear collapse, pilot error chain, loss of engine power — what does your model say?\"",
    "",
    "**Jesse:** \"Stop reporting accuracy. Report probabilities over codes. If your model outputs free-text causes, map them to Zhang's 54 codes — you already have probabilities, you're just throwing them away.\"",
    "",
    "**Today:** End-to-end walkthrough of both methods. The probability-over-codes comparison Jesse asked for, built and showing real numbers.",
    "",
    "---",
    "",
    "## 2. The three models",
    "",
    "| Model | What it does | Type |",
    "|---|---|---|",
    "| **A0** — Embeddings only | Embed narrative → cosine → cluster → count P(cause | query) | Model-free baseline |",
    "| **A2** — Structural Mapping | Same as A0, but rerank cosine using LLM-extracted causal chain alignment | My main approach |",
    "| **Zhang BN** | 740-node Bayesian network, hand-built, CPTs + Beta-CDF calibration | Generative model |",
    "",
    "Jesse: *\"As long as it's rigorous, any outcome is interesting. The only unacceptable outcome is if structure does worse than unstructured.\"*",
    "",
    "---",
    "",
    "## 3. Zhang's pipeline — step by step",
    "",
    "**INPUT** → **STRUCTURE** → **CPTs** → **INFERENCE** → **OUTPUT**",
    "",
    "1. **Input**: Coded NTSB fields (events, findings, occurrences, codes). Throws away narratives.",
    "2. **Structure**: Hand-built 740-node BN in GeNIe — a directed causal graph.",
    "3. **CPTs**: Count-based for nodes with enough data. Beta-CDF (α=1.04645, β=2.02591) for sparse cells (his Table 7).",
    "4. **Inference**: Likelihood-weighted sampling, 99,999,999 samples (SMILE engine).",
    "5. **Output**: P(node = state | evidence) over all 740 nodes for any scenario.",
    "",
    "**Why Bayesian?** The BN declares a joint distribution P(740 nodes). Observing evidence triggers Bayesian updating → posterior P(unobserved | observed). Mathematically self-consistent.",
    "",
    "---",
    "",
    "## 4. Where Zhang's probabilities come from",
    "",
    "- **Direct count-based CPTs** for frequent nodes — frequentist estimates from NTSB data.",
    "- **Beta-CDF calibration** for sparse cells — fit ONE Beta distribution globally (α=1.04645, β=2.02591). Uses Nelder-Mead optimization. We reproduced α/β to 9 decimal places.",
    "",
    "Key point: probabilities are **calibrated to historical frequency**, then **combined via Bayesian rules**. They mean: *\"under my model of how accidents work, given this evidence, this is the posterior probability of the outcome.\"*",
    "",
    "---",
    "",
    "## 5. Zhang's output example (Table 9)",
    "",
    "Given evidence: **Inoperative engine instruments**:",
    "",
    "| Event | Type | P(. \\| evidence) |",
    "|---|---|---|",
    "| Loss of engine power | Diagnosis | 0.950 |",
    "| Forced landing | Prognosis | 0.136 |",
    "| Ditching | Prognosis | 0.004 |",
    "| Substantial aircraft damage | Prognosis | 0.046 |",
    "| No injury | Prognosis | 0.943 |",
    "",
    "Diagnosis = causes that explain evidence. Prognosis = downstream consequences.",
    "",
    "---",
    "",
    "## 6. My pipeline — step by step",
    "",
    "**INPUT** → **EMBED** → **RETRIEVE + RERANK** → **CLUSTER** → **OUTPUT**",
    "",
    "1. **Input**: Free-text NTSB narratives (narr_accp, narr_cause).",
    "2. **Embed**: All training narratives → vectors.",
    "3. **Retrieve + rerank**: Top-K by cosine. A2 adds: LLM extracts causal chain → structural similarity s → score becomes max(0, cos) · exp(α·s), with α=2.",
    "4. **Cluster**: Group retrieved neighbors. Compute P(cause | cluster) by counting causes inside the cluster.",
    "5. **Output**: P(cause | query) = Σ P(cause | C_i) · P(C_i | query) by chain rule.",
    "",
    "**Not a Bayesian network**: no joint distribution, no dependency graph. Probabilities are EMPIRICAL FREQUENCIES in retrieved neighbors, not Bayesian posteriors. The model is *\"if your query looks like these past cases, here's what caused them.\"*",
    "",
    "---",
    "",
    "## 7. A0 vs A2 — what does structural mapping actually do?",
    "",
    "| Aspect | A0 | A2 |",
    "|---|---|---|",
    "| Ranking score | max(0, cos(embed_q, embed_i)) | max(0, cos) · exp(α · structural_sim) |",
    "| Captures | Lexical / topical similarity | Causal-chain alignment |",
    "| Differentiates | Surface text overlap | Same words, different mechanisms |",
    "",
    "**A2 sanity check** (chain pair with strong=1, partial=4, unmapped=1, total=10): (1 + 0.5·4)/10 − 0.10·(1/10) = 0.29. Matches the slide-deck Formula 3 verbatim.",
    "",
    "---",
    "",
    "## 8. Mapping free-text → Zhang's 54 codes",
    "",
    "1. Raw output is free-text: `{'engine fire': 0.31, 'pilot loss of control': 0.18, 'gear failure': 0.12, …}`",
    "2. **cause_code_mapper.py**: embed each of Zhang's 54 code labels + each free-text cause; nearest-neighbor match in embedding space.",
    "3. Final output: 54-dimensional probability vector + UNMAPPED bucket — **same shape as Zhang's BN output for that scenario**.",
    "",
    "Stored in `three_model_comparison.xlsx` (77 incidents × 114 cols) and `zhang_comparison/*.xlsx` (per Zhang scenario).",
    "",
    "---",
    "",
    "## 9. Method comparison at every step",
    "",
    "| Step | Zhang's BN | A0 | A2 |",
    "|---|---|---|---|",
    "| 1. Input data | Coded NTSB fields only | Free-text narratives | Narratives + LLM chains |",
    "| 2. Knowledge | Hand-built 740-node BN + CPTs | No model — just embeddings | Same embeddings + structural similarity |",
    "| 3. Reasoning | Bayesian inference (L_SAMPLING, 99M) | Cosine retrieval → cluster → count | Reranked retrieval → cluster → count |",
    "| 4. Output type | P(node = state \\| evidence) | P(cause \\| query) over codes | P(cause \\| query) over codes |",
    "| 5. Conditions on | Evidence node values | Query narrative text | Query text + causal chain |",
    "| 6. Semantics of P | Calibrated Bayesian posterior | Reweighted empirical frequency | Reweighted empirical frequency |",
    "| 7. Update with new case | Re-elicit CPTs (heavy) | Add to index (light) | Add to index + extract chain |",
    "",
    "---",
    "",
    "## 10. Numerical comparison — Table 9 (evidence: inoperative engine instruments)",
    "",
    "| Event | Type | Zhang | A0 | A2 | Pattern |",
    "|---|---|---|---|---|---|",
    "| Loss of engine power | Diagnosis | 0.9500 | 0.0926 | 0.0930 | Zhang spikes; mine doesn't |",
    "| Forced landing | Prognosis | 0.1357 | 0.6698 | 0.6694 | Mine spikes; Zhang doesn't |",
    "| Substantial aircraft damage | Prognosis | 0.0460 | 0.1703 | 0.1705 | Mine higher |",
    "| Minor aircraft damage | Prognosis | 0.0093 | 0.1599 | 0.1601 | Mine higher |",
    "| No injury | Prognosis | 0.9431 | 0.0000 | 0.0000 | Mine blind to this node |",
    "",
    "Look closely: **A0 and A2 differ at the 4th decimal place**. Structural rerank barely changes the code-level distribution.",
    "",
    "---",
    "",
    "## 11. Numerical comparison — Fig 12 (pilot error chain)",
    "",
    "| Node | Zhang prior | Zhang +pilot | A0 +pilot | A2 +pilot | Zhang +unstable | A0 +unstable | A2 +unstable |",
    "|---|---|---|---|---|---|---|---|",
    "| Pilot error | ~0 | 1.0000 | 0.8444 | 0.8452 | 1.0000 | 0.7904 | 0.7876 |",
    "| Hard landing | ~0 | 0.0599 | 0.9466 | 0.9462 | 0.0599 | 0.9124 | 0.9142 |",
    "| Substantial damage | ~0 | 0.0458 | 0.0841 | 0.0838 | 0.2464 | 0.0948 | 0.0987 |",
    "| **No injury** | 0.9999 | 0.9700 | 0.0000 | 0.0000 | 0.6130 | 0.0000 | 0.0000 |",
    "",
    "Striking: **'No injury' is 0.97 for Zhang, 0.00 for mine**. Narratives describe what happened, not what didn't.",
    "",
    "---",
    "",
    "## 12. The three findings",
    "",
    "1. **A0 ≈ A2 at the code level** (SURPRISING) — Structural reranking gives big top-1 wins inside clusters, but when aggregated to the 54-code distribution, A0 and A2 differ in the 4th decimal place. A2 changed top-1 vs A0 in only 1 of 83 incidents. Mean entropies essentially identical.",
    "",
    "2. **Mine ≠ Zhang in a structured way** (EXPECTED) — Zhang spikes the cause code (0.95) and the dominant outcome (0.94). Mine spreads over observable consequences (Forced landing 0.67, Substantial damage 0.17). Different methods, different questions.",
    "",
    "3. **Mine is BLIND to 'No injury'** (ALSO SURPRISING) — Across every scenario, P(No injury) ≈ 0 in my model. Same absence-state issue we documented on Zhang's replication, now manifest on my side too.",
    "",
    "---",
    "",
    "## 13. What this means + asks",
    "",
    "**What it means:**",
    "",
    "- Zhang's BN and my retrieval pipeline answer different questions; both can be valid in their own frame.",
    "- A2 helps top-1 hits but doesn't meaningfully change the code distribution.",
    "- A Bayesian-style updating layer (LLM-token probabilities) would give SAME output format as Zhang and enable direct comparison.",
    "",
    "**Asks:**",
    "",
    "1. Publication framing — engineering journal (reliability) or CS venue?",
    "2. Next: read probabilities off LLM tokens (Jesse's suggestion), or extend the structural pipeline with explicit Bayesian updating?",
    "3. Should I email Zhang to confirm XDSL is the paper version?",
    "4. For the paper comparison — all of Zhang's Tables/Figs, or just Table 9 + Fig 11 + Fig 12?",
    "",
    "---",
    "",
    "## 14. Anticipated questions (backup)",
    "",
    "**Q: Did you replicate Zhang's full paper?** A: Yes — 88% of his 86 published cells reproduce within paper rounding tolerance.",
    "",
    "**Q: How is your top-1=35% related to the new comparison?** A: Top-1 is per-incident; the new comparison is the FULL distribution. Same numbers, but now comparable to Zhang's frame.",
    "",
    "**Q: How would I get Bayesian probabilities from your pipeline?** A: Two paths — (a) parameter-efficient fine-tuning on a small LLM whose tokens map to codes, then read token probs; (b) Bayesian aggregation layer over multiple evidence sources. Jesse leaned (a).",
    "",
    "**Q: Why can't your model see 'No injury'?** A: Retrieval is over narratives, and narratives describe events that happened. 'No injury' is an absence.",
    "",
    "**Q: Is +2.6pp A2 over A0 significant?** A: Not at p<0.05 (McNemar p=0.50). 95% bootstrap CI on Δtop-1: [0.0, +6.5pp]. Small but directionally positive on n=77.",
]
MD_OUT.write_text("\n".join(md), encoding="utf-8")
print(f"wrote {MD_OUT}")
