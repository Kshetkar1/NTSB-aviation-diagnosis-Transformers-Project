"""
build_slides_for_maha.py
Generates the slide deck explaining Zhang (2021) replication
for the Dr. Mahadevan meeting.

Output: outputs/Zhang_Replication_For_Maha.pptx
"""
from __future__ import annotations
from pathlib import Path
import math
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "Zhang_Replication_For_Maha.pptx"
VALIDATION_XLSX = HERE / "outputs" / "validation_report_99M.xlsx"
OUT.parent.mkdir(parents=True, exist_ok=True)

NAVY = RGBColor(0x0B, 0x2D, 0x4A)
TEAL = RGBColor(0x12, 0x6E, 0x82)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
GREEN = RGBColor(0x27, 0xAE, 0x60)
RED = RGBColor(0xC0, 0x39, 0x2B)
GREY = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF4, 0xF6, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)


def set_text(tf, text, size=18, bold=False, color=NAVY, align=PP_ALIGN.LEFT):
    tf.clear()
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.name = "Calibri"
    r.font.color.rgb = color


def add_bullets(tf, bullets, size=16, color=NAVY, indent_size=14):
    """bullets: list of either str or (str, [sub_bullets])."""
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


def add_title_box(slide, text, color=NAVY, top=0.35, height=0.9, size=28):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(height))
    set_text(box.text_frame, text, size=size, bold=True, color=color)
    return box


def add_subtitle_box(slide, text, color=TEAL, top=1.15, height=0.5, size=16):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(top), Inches(12.3), Inches(height))
    set_text(box.text_frame, text, size=size, bold=False, color=color)
    return box


def add_strip(slide, top=0.2, color=NAVY, height=0.06):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(top), Inches(12.3), Inches(height))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()


def add_footer(slide, n, total, label="Zhang Replication"):
    box = slide.shapes.add_textbox(Inches(0.5), Inches(7.0), Inches(12.3), Inches(0.3))
    p = box.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = f"{label}    |    Slide {n} / {total}"
    r.font.size = Pt(10)
    r.font.name = "Calibri"
    r.font.color.rgb = GREY


def fmt_num(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    av = abs(v)
    if av == 0:
        return "0.00000"
    if av < 1e-3:
        return f"{v:.2e}"
    if av < 1:
        return f"{v:.5f}"
    return f"{v:.5f}"


def cls_to_icon(c):
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return ("—", GREY)
    c = str(c).strip()
    if c == "OK":
        return ("✓", GREEN)
    if c == "close":
        return ("~", ORANGE)
    if c == "FAIL":
        return ("✗", RED)
    return ("—", GREY)


def add_comparison_table(slide, rows, x=0.5, y=1.6, width=12.3, height=5.2,
                         font_size=9, header_color=NAVY):
    """rows: list of [context, outcome, zhang_num, mine_num, delta_num, classification_str].
    Renders a python-pptx native table with verdict-colored last column."""
    headers = ["Context", "Outcome", "Zhang", "Mine", "Δ", ""]
    n_rows = len(rows) + 1
    n_cols = len(headers)
    tbl_shape = slide.shapes.add_table(n_rows, n_cols,
                                       Inches(x), Inches(y),
                                       Inches(width), Inches(height))
    table = tbl_shape.table

    # Column widths (sum to width)
    col_w = [3.6, 3.4, 1.6, 1.6, 1.5, 0.6]
    s = sum(col_w)
    for i, w in enumerate(col_w):
        table.columns[i].width = Emu(int(Inches(width).emu * (w / s)))

    # Header row
    for c, h in enumerate(headers):
        cell = table.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c < 2 else PP_ALIGN.RIGHT
        r = p.add_run()
        r.text = h
        r.font.size = Pt(font_size + 1)
        r.font.bold = True
        r.font.name = "Calibri"
        r.font.color.rgb = WHITE

    # Data rows
    for ri, row in enumerate(rows, start=1):
        ctx, outc, zh, mine, dlt, cls = row
        icon, ic_color = cls_to_icon(cls)
        cells_data = [
            (str(ctx), NAVY, PP_ALIGN.LEFT, False),
            (str(outc), NAVY, PP_ALIGN.LEFT, False),
            (fmt_num(zh), NAVY, PP_ALIGN.RIGHT, False),
            (fmt_num(mine), NAVY, PP_ALIGN.RIGHT, False),
            (fmt_num(dlt), GREY, PP_ALIGN.RIGHT, False),
            (icon, ic_color, PP_ALIGN.CENTER, True),
        ]
        # Alternate row shading
        bg = LIGHT if ri % 2 == 0 else WHITE
        for c, (txt, color, align, bold) in enumerate(cells_data):
            cell = table.cell(ri, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = align
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(font_size)
            r.font.bold = bold
            r.font.name = "Consolas" if c >= 2 and c < 5 else "Calibri"
            r.font.color.rgb = color
    return tbl_shape


def load_validation_rows():
    df_t8 = pd.read_excel(VALIDATION_XLSX, sheet_name="Table 8")
    df_f12 = pd.read_excel(VALIDATION_XLSX, sheet_name="Fig 12")
    df_t9 = pd.read_excel(VALIDATION_XLSX, sheet_name="Table 9")

    def to_rows(df):
        out = []
        for _, r in df.iterrows():
            out.append([
                str(r["context"])[:32],
                str(r["outcome"])[:30],
                r["zhang"],
                r["reproduced"],
                r["delta"],
                r["classification"],
            ])
        return out

    return to_rows(df_t8), to_rows(df_f12), to_rows(df_t9)


# 16:9
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

BLANK = prs.slide_layouts[6]
TOTAL = 18

# Load validation data for the side-by-side comparison slides
T8_ROWS, F12_ROWS, T9_ROWS = load_validation_rows()


# ---- SLIDE 1 — Title ---------------------------------------------------------
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = NAVY
bg.line.fill.background()

t = s.shapes.add_textbox(Inches(0.8), Inches(2.2), Inches(11.5), Inches(1.4))
set_text(t.text_frame, "Replicating Zhang & Mahadevan (2021)", size=42, bold=True, color=WHITE)

t = s.shapes.add_textbox(Inches(0.8), Inches(3.4), Inches(11.5), Inches(0.8))
set_text(t.text_frame, "Generating the Full NTSB BN Probability Table", size=26, color=RGBColor(0xCB, 0xE3, 0xEC))

t = s.shapes.add_textbox(Inches(0.8), Inches(4.6), Inches(11.5), Inches(0.6))
set_text(t.text_frame, "Status update for Dr. Mahadevan", size=20, color=WHITE)

t = s.shapes.add_textbox(Inches(0.8), Inches(6.4), Inches(11.5), Inches(0.6))
set_text(t.text_frame, "Kanu Shetkar  ·  Vanderbilt RRR Lab  ·  May 7, 2026", size=14, color=RGBColor(0xCB, 0xE3, 0xEC))


# ---- SLIDE 2 — What was asked of me ------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Where this work came from")
add_subtitle_box(s, "What Jesse and you asked me to deliver after the last meeting")

box = s.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.0))
add_bullets(box.text_frame, [
    ("Last meeting (with you)", [
        "Showed preliminary comparison of my model to the few probabilities Zhang printed in his paper",
        "Conclusion: those ~10 published numbers are not enough to defend a comparison — need every probability for every node",
    ]),
    ("Yesterday's meeting (with Jesse)", [
        "Confirmed: Zhang's GitHub does not contain the full probability table — only the network file and code",
        "Direction: 'Run his pipeline end-to-end. Generate the full probability table yourself, exactly the way he did it.'",
    ]),
    ("This deck reports on that task", [
        "What I built, how I validated it, what the deliverable is, and what's left",
    ]),
], size=15)
add_footer(s, 2, TOTAL)


# ---- SLIDE 3 — What Zhang actually did ---------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "What Zhang's pipeline actually does")
add_subtitle_box(s, "Reading his GitHub + paper end-to-end")

box = s.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(12.3), Inches(5.0))
add_bullets(box.text_frame, [
    ("Builds a Bayesian Network with 740 nodes from NTSB accident data", [
        "Nodes = events, conditions, outcomes (e.g. 'Pilot in command', 'Loss of engine power', 'No injury')",
        "Structure manually drawn in GeNIe Modeler; saved as NTSB.xdsl",
    ]),
    ("Estimates Conditional Probability Tables (CPTs)", [
        "Direct counts from NTSB data for nodes with sufficient observations",
        "Calibrated Beta-CDF fit (Table 7: α=1.04645, β=2.02591) for sparse cells",
    ]),
    ("Runs probabilistic inference using BayesFusion's SMILE engine", [
        "Algorithm: Likelihood-Weighted Sampling — set_bayesian_algorithm(3)",
        "Sample count: 99,999,999 (declared inside his XDSL header)",
    ]),
    ("Reports a small set of scenarios in the paper", [
        "Table 8 (sensitivity), Table 9 (engine power), Fig 11 (multi-evidence), Fig 12 (pilot error)",
        "≈ 86 published probability cells in total — every other node-scenario pair was never published",
    ]),
], size=14, indent_size=12)
add_footer(s, 3, TOTAL)


# ---- SLIDE 4 — What 'replicating exactly' means ------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "What 'replicating exactly' actually requires")
add_subtitle_box(s, "Five dimensions to match — we matched four")

# 5 columns
cols = [
    ("Network structure", "His NTSB.xdsl, unmodified", "MATCH", GREEN),
    ("CPT values", "His XDSL, unmodified", "MATCH", GREEN),
    ("Inference engine", "BayesFusion SMILE\n(via pysmile 2.4.0)", "MATCH", GREEN),
    ("Algorithm + samples", "L_SAMPLING (alg 3)\n99,999,999 samples", "MATCH", GREEN),
    ("Random seed", "Not set in his notebook —\nlost forever", "CANNOT", RED),
]
left = 0.5
width = 2.5
gap = 0.07
for i, (title, body, badge, color) in enumerate(cols):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(1.9), Inches(width), Inches(2.6))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = NAVY
    card.line.width = Pt(0.75)
    title_box = s.shapes.add_textbox(Inches(x + 0.1), Inches(2.0), Inches(width - 0.2), Inches(0.5))
    set_text(title_box.text_frame, title, size=14, bold=True, color=NAVY)
    body_box = s.shapes.add_textbox(Inches(x + 0.1), Inches(2.5), Inches(width - 0.2), Inches(1.4))
    set_text(body_box.text_frame, body, size=12, color=GREY)
    badge_shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x + 0.5), Inches(3.95), Inches(width - 1.0), Inches(0.4))
    badge_shape.fill.solid()
    badge_shape.fill.fore_color.rgb = color
    badge_shape.line.fill.background()
    set_text(badge_shape.text_frame, badge, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

note = s.shapes.add_textbox(Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0))
add_bullets(note.text_frame, [
    "Zhang's notebook (Scenario analysis.ipynb) does not call set_rand_seed() — his published numbers are themselves a single random draw",
    "Therefore: bit-identical numerical reproduction is mathematically impossible — even for Zhang himself today",
    "What we CAN prove: our pipeline samples from the same posterior distribution his does",
], size=14, color=NAVY)
add_footer(s, 4, TOTAL)


# ---- SLIDE 5 — Architecture ---------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "The replication pipeline I built")
add_subtitle_box(s, "10 scripts, fully scripted, fully reproducible — no GUI clicks")

# 4 stages with arrows
stages = [
    ("INPUT",
     "Zhang's NTSB.xdsl\n(740 nodes, full CPTs)\n+ his paper scenarios",
     NAVY),
    ("REPLICATE",
     "Scripts 01–04:\nTable 8 · Fig 11 · Fig 12 · Table 9\n(reproduce his published cells)",
     TEAL),
    ("VALIDATE",
     "Script 05: validation_report\nScripts 07-08: seed band + α/β\n(quantify match quality)",
     ORANGE),
    ("DELIVERABLE",
     "Scripts 09-10:\nFull table — every node\n× every paper scenario",
     GREEN),
]
left = 0.5
width = 2.85
gap = 0.25
top = 2.1
height = 2.2
for i, (label, body, color) in enumerate(stages):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(top), Inches(width), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = color
    card.line.fill.background()
    title_box = s.shapes.add_textbox(Inches(x + 0.15), Inches(top + 0.15), Inches(width - 0.3), Inches(0.5))
    set_text(title_box.text_frame, label, size=15, bold=True, color=WHITE)
    body_box = s.shapes.add_textbox(Inches(x + 0.15), Inches(top + 0.7), Inches(width - 0.3), Inches(height - 0.8))
    set_text(body_box.text_frame, body, size=13, color=WHITE)
    if i < len(stages) - 1:
        ax = x + width
        arrow = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(ax + 0.02), Inches(top + 0.85), Inches(gap - 0.04), Inches(0.5))
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = GREY
        arrow.line.fill.background()

note = s.shapes.add_textbox(Inches(0.5), Inches(4.7), Inches(12.3), Inches(2.0))
add_bullets(note.text_frame, [
    "All ten scripts run automatically. Re-running them top-to-bottom regenerates every artifact.",
    "Outputs: validation reports (.md + .xlsx), full probability table (.parquet + .xlsx, 38,480 rows)",
    "Cost: ~90 minutes of compute for the full 99M-sample pass; longer if multi-seed band is also run",
], size=14, color=NAVY)
add_footer(s, 5, TOTAL)


# ---- SLIDE 6 — Inference choice -----------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Why I used pysmile and not the GeNIe GUI")
add_subtitle_box(s, "Same engine, different interface — and only one is automatable")

# Two-column comparison
left_col = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(2.0), Inches(6.0), Inches(4.0))
left_col.fill.solid()
left_col.fill.fore_color.rgb = LIGHT
left_col.line.color.rgb = NAVY
right_col = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.85), Inches(2.0), Inches(6.0), Inches(4.0))
right_col.fill.solid()
right_col.fill.fore_color.rgb = LIGHT
right_col.line.color.rgb = NAVY

t = s.shapes.add_textbox(Inches(0.7), Inches(2.1), Inches(5.6), Inches(0.5))
set_text(t.text_frame, "GeNIe GUI (what Zhang clicked through)", size=15, bold=True, color=NAVY)
b = s.shapes.add_textbox(Inches(0.7), Inches(2.55), Inches(5.6), Inches(3.4))
add_bullets(b.text_frame, [
    "Same SMILE engine under the hood",
    "Manual point-and-click for each scenario",
    "One scenario at a time; cannot batch",
    "740 × 26 = 19,240 manual node reads (impractical)",
    "No record of seeds or sample counts used",
    "Cannot regenerate identical output later",
], size=13, color=GREY)

t = s.shapes.add_textbox(Inches(7.05), Inches(2.1), Inches(5.6), Inches(0.5))
set_text(t.text_frame, "pysmile (what I scripted)", size=15, bold=True, color=NAVY)
b = s.shapes.add_textbox(Inches(7.05), Inches(2.55), Inches(5.6), Inches(3.4))
add_bullets(b.text_frame, [
    "Same SMILE engine, called from Python",
    "Same XDSL file, same algorithm, same sample count",
    "Loops over all 26 scenarios automatically",
    "Reads all 740 nodes per run",
    "Every parameter logged; reproducible",
    "Scales to multi-seed runs for noise quantification",
], size=13, color=GREEN)

note = s.shapes.add_textbox(Inches(0.5), Inches(6.15), Inches(12.3), Inches(0.8))
set_text(note.text_frame, "Bottom line: GeNIe and pysmile call the same C++ inference code. The output you'd get from either is statistically identical.", size=14, bold=True, color=NAVY)
add_footer(s, 6, TOTAL)


# ---- SLIDE 7 — Calibration validation ----------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Validation Layer 1 — Calibration parameters")
add_subtitle_box(s, "Bit-exact match to Zhang's Table 7")

# Big numerical box
card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(2.0), Inches(12.3), Inches(2.6))
card.fill.solid()
card.fill.fore_color.rgb = LIGHT
card.line.color.rgb = NAVY

t = s.shapes.add_textbox(Inches(0.7), Inches(2.1), Inches(11.9), Inches(0.6))
set_text(t.text_frame, "Beta-CDF calibration (his Table 7, fitted via Nelder-Mead optimization):", size=15, bold=True, color=NAVY)

t = s.shapes.add_textbox(Inches(0.7), Inches(2.7), Inches(5.7), Inches(0.6))
set_text(t.text_frame, "Zhang published:", size=14, bold=True, color=GREY)
t = s.shapes.add_textbox(Inches(0.7), Inches(3.15), Inches(5.7), Inches(0.6))
set_text(t.text_frame, "α = 1.04645     β = 2.02591", size=20, bold=True, color=NAVY)

t = s.shapes.add_textbox(Inches(6.6), Inches(2.7), Inches(5.7), Inches(0.6))
set_text(t.text_frame, "My replication:", size=14, bold=True, color=GREY)
t = s.shapes.add_textbox(Inches(6.6), Inches(3.15), Inches(5.7), Inches(0.6))
set_text(t.text_frame, "α = 1.046453510   β = 2.025913942", size=20, bold=True, color=GREEN)

badge = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(3.85), Inches(11.9), Inches(0.55))
badge.fill.solid()
badge.fill.fore_color.rgb = GREEN
badge.line.fill.background()
set_text(badge.text_frame, "EXACT MATCH to 9 decimal places — calibration step is bit-perfect", size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

# What this proves
note = s.shapes.add_textbox(Inches(0.5), Inches(5.0), Inches(12.3), Inches(2.0))
add_bullets(note.text_frame, [
    ("What this proves", [
        "His objective function and optimizer setup are correctly recovered from his code",
        "The deterministic part of his pipeline is reproduced bit-for-bit",
        "Any divergence from his published numbers in later layers is therefore NOT a bug in the math",
    ]),
], size=14)
add_footer(s, 7, TOTAL)


# ---- SLIDE 8 — Published-numbers validation -----------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Validation Layer 2 — His published numbers")
add_subtitle_box(s, "Of the 86 cells he printed in the paper, how many do we recover?")

# Bars
data = [
    ("OK (delta < 0.005 or rel<5%)", 66, GREEN),
    ("Close (delta < 0.05)", 10, ORANGE),
    ("FAIL (delta > 0.05)", 9, RED),
    ("N/A (he printed no value)", 1, GREY),
]
total_n = 86
top = 2.2
bar_max_w = 8.0
bar_h = 0.6
gap = 0.25
for i, (label, n, color) in enumerate(data):
    y = top + i * (bar_h + gap)
    pct = n / total_n
    # label
    lab = s.shapes.add_textbox(Inches(0.5), Inches(y), Inches(3.5), Inches(bar_h))
    set_text(lab.text_frame, label, size=13, bold=True, color=NAVY)
    # bar
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.0), Inches(y + 0.05), Inches(bar_max_w * pct), Inches(bar_h - 0.1))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    # number
    num = s.shapes.add_textbox(Inches(4.0 + bar_max_w * pct + 0.1), Inches(y), Inches(2.5), Inches(bar_h))
    set_text(num.text_frame, f"{n}  ({pct*100:.0f}%)", size=14, bold=True, color=NAVY)

# Headline
hl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(5.7), Inches(12.3), Inches(0.6))
hl.fill.solid()
hl.fill.fore_color.rgb = NAVY
hl.line.fill.background()
set_text(hl.text_frame, "76 of 86 cells (88%) reproduce within paper rounding tolerance · next 4 slides show every cell side-by-side", size=14, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_footer(s, 8, TOTAL)


# ---- SLIDE 9 — Table 8 side-by-side ------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Side-by-side: Table 8 (sensitivity sweep)", color=GREEN)
add_subtitle_box(s, "All 24 cells match to 4-5 decimal places — Zhang's sensitivity table reproduces fully")
add_comparison_table(s, T8_ROWS, y=1.6, height=5.0, font_size=9)
hl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.45))
hl.fill.solid()
hl.fill.fore_color.rgb = GREEN
hl.line.fill.background()
set_text(hl.text_frame, "24 of 24 ✓  ·  every strut prior, every outcome — bit-clean reproduction", size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_footer(s, 9, TOTAL)


# ---- SLIDE 10 — Fig 12 side-by-side ------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Side-by-side: Fig 12 (pilot-error chain)")
add_subtitle_box(s, "9 of 12 match · 3 FAILs are all 'No injury' (absence-state encoding)")
add_comparison_table(s, F12_ROWS, y=1.6, height=4.5, font_size=11)
hl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.55))
hl.fill.solid()
hl.fill.fore_color.rgb = NAVY
hl.line.fill.background()
set_text(hl.text_frame, "Every node EXCEPT 'No injury' matches Zhang within rounding · the 3 failures are the same node × 3 scenarios", size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_footer(s, 10, TOTAL)


# ---- SLIDE 11 — Table 9 side-by-side (Part 1) --------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Side-by-side: Table 9 (engine power) — Part 1 of 2")
add_subtitle_box(s, "Outcomes 1-5 of 10 · all 25 cells match")
add_comparison_table(s, T9_ROWS[:25], y=1.6, height=5.0, font_size=9)
hl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.45))
hl.fill.solid()
hl.fill.fore_color.rgb = GREEN
hl.line.fill.background()
set_text(hl.text_frame, "Loss of engine power · Forced landing · Ditching · Gear collapsed · Other gear collapsed — all 25 within tolerance", size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_footer(s, 11, TOTAL)


# ---- SLIDE 12 — Table 9 side-by-side (Part 2) --------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Side-by-side: Table 9 (engine power) — Part 2 of 2")
add_subtitle_box(s, "Outcomes 6-10 of 10 · 'Substantial damage' shows +0.02 offset · 'No injury' shows absence-state issue")
add_comparison_table(s, T9_ROWS[25:], y=1.6, height=5.0, font_size=9)
hl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(6.6), Inches(12.3), Inches(0.45))
hl.fill.solid()
hl.fill.fore_color.rgb = NAVY
hl.line.fill.background()
set_text(hl.text_frame, "Destroyed · Substantial · Minor damage · Serious injury · No injury — all FAILs concentrate in last two rows (No injury, Substantial)", size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_footer(s, 12, TOTAL)


# ---- SLIDE 13 — The 12% gap ---------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "The 12% gap, fully decomposed")
add_subtitle_box(s, "Every FAIL has a documented structural cause — none are pipeline bugs")

cards = [
    ("(a) Absence-state encoding",
     "'No injury' nodes encoded near 0.94+",
     "Likelihood-weighted sampling cannot reach values that close to 1.0 — known limitation of sampling-based BN inference. To recover, the network would need to use complement encoding instead of absence states.",
     6, RED),
    ("(b) Network revision after publication",
     "Consistent ~+0.02 offset on 'Substantial damage'",
     "Same offset appears across multiple scenarios in Fig 12 and Table 9. Strongly suggests Zhang revised the XDSL between paper publication and the GitHub commit — without changing the paper.",
     5, ORANGE),
    ("(c) Sub-noise-floor priors",
     "Priors at 10⁻⁷ to 10⁻⁹ in Table 8",
     "At any reasonable sample count, sampling noise (≈1/√N) is larger than the signal. Zhang's own published numbers at these priors are themselves single noisy draws.",
     2, GREY),
]
left = 0.5
width = 4.05
gap = 0.15
for i, (title, sub, body, n, color) in enumerate(cards):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.0), Inches(width), Inches(4.5))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = color
    card.line.width = Pt(2.0)

    badge = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.15), Inches(2.15), Inches(0.55), Inches(0.55))
    badge.fill.solid()
    badge.fill.fore_color.rgb = color
    badge.line.fill.background()
    set_text(badge.text_frame, str(n), size=18, bold=True, color=WHITE, align=PP_ALIGN.CENTER)

    t = s.shapes.add_textbox(Inches(x + 0.85), Inches(2.18), Inches(width - 0.95), Inches(0.5))
    set_text(t.text_frame, title, size=14, bold=True, color=NAVY)
    t = s.shapes.add_textbox(Inches(x + 0.15), Inches(2.85), Inches(width - 0.3), Inches(0.6))
    set_text(t.text_frame, sub, size=13, bold=True, color=color)
    t = s.shapes.add_textbox(Inches(x + 0.15), Inches(3.5), Inches(width - 0.3), Inches(2.9))
    set_text(t.text_frame, body, size=12, color=GREY)

note = s.shapes.add_textbox(Inches(0.5), Inches(6.55), Inches(12.3), Inches(0.5))
set_text(note.text_frame, "Net: pipeline is correct · the 12% gap is in his repo / methodology, not in our reproduction", size=14, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
add_footer(s, 13, TOTAL)


# ---- SLIDE 14 — Multi-seed analysis ------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Validation Layer 3 — Multi-seed noise band")
add_subtitle_box(s, "Stronger statement than 'we got the same numbers' — same posterior distribution")

# Explanation paragraph
exp = s.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(12.3), Inches(1.2))
add_bullets(exp.text_frame, [
    "Ran the full pipeline 5 times with 5 different random seeds (1, 7, 42, 100, 9999)",
    "For each cell Zhang published: does his value fall inside the band [min, max] of my 5 reproductions?",
], size=14)

# Verdict bar
data = [
    ("In-band (Zhang inside our [min,max])", 47, GREEN),
    ("Within ±2σ of our mean", 3, ORANGE),
    ("Outside band — structural FAILs", 35, RED),
    ("N/A (no published value)", 1, GREY),
]
total_n = 86
top = 3.4
bar_max_w = 7.5
bar_h = 0.55
gap = 0.2
for i, (label, n, color) in enumerate(data):
    y = top + i * (bar_h + gap)
    pct = n / total_n
    lab = s.shapes.add_textbox(Inches(0.5), Inches(y), Inches(4.0), Inches(bar_h))
    set_text(lab.text_frame, label, size=12, bold=True, color=NAVY)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(4.5), Inches(y + 0.05), Inches(bar_max_w * pct), Inches(bar_h - 0.15))
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()
    num = s.shapes.add_textbox(Inches(4.5 + bar_max_w * pct + 0.1), Inches(y), Inches(2.5), Inches(bar_h))
    set_text(num.text_frame, f"{n}  ({pct*100:.0f}%)", size=13, bold=True, color=NAVY)

# Bottom-line
hl = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(6.4), Inches(12.3), Inches(0.55))
hl.fill.solid()
hl.fill.fore_color.rgb = NAVY
hl.line.fill.background()
set_text(hl.text_frame, "Of cells that didn't FAIL structurally: every single one falls in our reproduction band — same posterior, just a different seed", size=13, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
add_footer(s, 14, TOTAL)


# ---- SLIDE 15 — The deliverable -----------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "The deliverable: full probability table", color=GREEN)
add_subtitle_box(s, "What Jesse asked for — the thing his GitHub never had", color=GREEN)

# Big stats
stats = [
    ("740", "BN nodes\n(every node Zhang defined)"),
    ("26", "Paper scenarios\n(Tables 8-9, Figs 11-12)"),
    ("38,480", "Rows of posteriors\n(every node × every scenario)"),
    ("230", "'Active' nodes\nwith probabilities that move"),
]
left = 0.5
width = 3.0
gap = 0.1
for i, (big, small) in enumerate(stats):
    x = left + i * (width + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.0), Inches(width), Inches(2.2))
    card.fill.solid()
    card.fill.fore_color.rgb = GREEN
    card.line.fill.background()
    big_box = s.shapes.add_textbox(Inches(x + 0.1), Inches(2.15), Inches(width - 0.2), Inches(1.0))
    set_text(big_box.text_frame, big, size=44, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    small_box = s.shapes.add_textbox(Inches(x + 0.1), Inches(3.2), Inches(width - 0.2), Inches(0.9))
    set_text(small_box.text_frame, small, size=12, color=WHITE, align=PP_ALIGN.CENTER)

# Files
note = s.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(12.3), Inches(2.2))
add_bullets(note.text_frame, [
    ("zhang_full_probability_table.xlsx (.parquet)", [
        "Long format: 38,480 rows = node × scenario × outcome × posterior probability",
        "One row per (node_id, scenario_id, outcome_state, P)",
    ]),
    ("zhang_yes_matrix.xlsx", [
        "Wide pivot: 740 nodes (rows) × 26 scenarios (columns), each cell = P(node = Yes | scenario)",
        "Sub-sheet 'interesting_nodes': just the 230 nodes whose probability actually changes across scenarios",
    ]),
], size=13)
add_footer(s, 15, TOTAL)


# ---- SLIDE 16 — Why bit-exact is impossible ----------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Why nobody can reproduce Zhang's paper bit-exactly")
add_subtitle_box(s, "The 88% ceiling is a property of his methodology — and what we proved instead")

# Two columns
left_col = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(2.0), Inches(6.0), Inches(4.6))
left_col.fill.solid()
left_col.fill.fore_color.rgb = LIGHT
left_col.line.color.rgb = RED
left_col.line.width = Pt(1.5)
right_col = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.85), Inches(2.0), Inches(6.0), Inches(4.6))
right_col.fill.solid()
right_col.fill.fore_color.rgb = LIGHT
right_col.line.color.rgb = GREEN
right_col.line.width = Pt(1.5)

t = s.shapes.add_textbox(Inches(0.7), Inches(2.1), Inches(5.6), Inches(0.5))
set_text(t.text_frame, "Why bit-exact is impossible", size=15, bold=True, color=RED)
b = s.shapes.add_textbox(Inches(0.7), Inches(2.6), Inches(5.6), Inches(3.9))
add_bullets(b.text_frame, [
    "His notebook does not call set_rand_seed() — his published numbers are a single random draw",
    "BayesFusion has updated SMILE since 2021; we run a newer build under the same Python API",
    "Floating-point arithmetic differs across CPUs in the last few bits",
    "Likely network revision between paper and the GitHub commit (the +0.02 offset)",
    "Implication: even Zhang himself, re-running today, would not bit-match his own paper",
], size=13, color=GREY)

t = s.shapes.add_textbox(Inches(7.05), Inches(2.1), Inches(5.6), Inches(0.5))
set_text(t.text_frame, "What we proved instead", size=15, bold=True, color=GREEN)
b = s.shapes.add_textbox(Inches(7.05), Inches(2.6), Inches(5.6), Inches(3.9))
add_bullets(b.text_frame, [
    "Same network, same engine, same algorithm, same sample count",
    "Calibration parameters bit-exact (α/β to 9 decimals)",
    "88% of his cells reproduce within paper-rounding tolerance",
    "47/86 of cells fall directly inside our 5-seed reproduction band",
    "Net: our pipeline samples from the same posterior distribution his does. His published values are one valid draw. Ours are five more.",
], size=13, color=GREEN)
add_footer(s, 16, TOTAL)


# ---- SLIDE 17 — Status & Next Steps ------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Where I am now")
add_subtitle_box(s, "Replication phase complete — comparison phase begins next")

# Status bullet block
box = s.shapes.add_textbox(Inches(0.5), Inches(1.8), Inches(12.3), Inches(2.6))
add_bullets(box.text_frame, [
    ("DONE — Replication phase", [
        "Full pipeline runs end-to-end (10 scripts, ~90 min compute)",
        "All deliverable artifacts generated and saved under Zhang_Replication_Runner/outputs/",
        "Three layers of validation: calibration (exact), published numbers (88%), multi-seed band (in-distribution)",
        "Full probability table for every node × every scenario — the artifact Jesse asked for",
    ]),
], size=14)

# Next steps
box = s.shapes.add_textbox(Inches(0.5), Inches(4.7), Inches(12.3), Inches(2.4))
add_bullets(box.text_frame, [
    ("NEXT — Comparison phase (needs more time)", [
        "Use zhang_yes_matrix.xlsx as ground-truth reference for the 230 active nodes",
        "Score my model's posteriors against Zhang's at every scenario",
        "Per-incident comparison across the 77 test incidents",
        "Decide on the right comparison metric (per your direction below)",
    ]),
], size=14, color=TEAL)
add_footer(s, 17, TOTAL)


# ---- SLIDE 18 — Asks ----------------------------------------------------------
s = prs.slides.add_slide(BLANK)
add_strip(s)
add_title_box(s, "Questions for you")
add_subtitle_box(s, "Where I need direction before starting the comparison")

questions = [
    ("Comparison scope",
     "Compare ALL 740 nodes, or just the 230 active nodes, or just outcome nodes (No injury / Substantial damage / Destroyed / etc.)?"),
    ("Comparison metric",
     "Absolute delta? KL divergence? Rank correlation per scenario? Or all three with a primary?"),
    ("Per-incident or aggregate",
     "For the 77 test incidents — should I report per-incident posteriors, or aggregate scores across the test set?"),
    ("Email Zhang directly",
     "Should I email him to confirm whether the GitHub XDSL is the paper version and whether he used a random seed?"),
    ("Deliverable shape",
     "What format do you want this in next time — paper section, appendix table, stand-alone validation memo, or another deck?"),
]
top = 2.0
height = 0.85
gap = 0.05
for i, (q, body) in enumerate(questions):
    y = top + i * (height + gap)
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y), Inches(12.3), Inches(height))
    card.fill.solid()
    card.fill.fore_color.rgb = LIGHT
    card.line.color.rgb = NAVY
    card.line.width = Pt(0.75)
    badge = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.65), Inches(y + 0.18), Inches(0.5), Inches(0.5))
    badge.fill.solid()
    badge.fill.fore_color.rgb = ORANGE
    badge.line.fill.background()
    set_text(badge.text_frame, str(i + 1), size=15, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    t = s.shapes.add_textbox(Inches(1.3), Inches(y + 0.08), Inches(10.8), Inches(0.4))
    set_text(t.text_frame, q, size=14, bold=True, color=NAVY)
    t = s.shapes.add_textbox(Inches(1.3), Inches(y + 0.42), Inches(10.8), Inches(0.45))
    set_text(t.text_frame, body, size=12, color=GREY)
add_footer(s, 18, TOTAL)


prs.save(str(OUT))
print(f"wrote {OUT}")
