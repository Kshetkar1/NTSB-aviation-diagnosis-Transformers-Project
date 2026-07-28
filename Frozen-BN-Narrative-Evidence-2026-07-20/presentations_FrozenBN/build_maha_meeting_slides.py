"""Build today's advisor meeting deck with Table 4, full Table 7, and progress slides.

Output: docs/maha_meeting_update.pptx

Run:
  python3 docs/build_maha_meeting_slides.py
"""
from __future__ import annotations

import csv
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "maha_meeting_update.pptx"
TABLE4_CSV = HERE / "presentation_table4.csv"
TABLE7_CSV = HERE / "presentation_table7_full.csv"
SPARSE_FIG = HERE / "figures" / "sparse_robustness.png"
TREE_FIG = HERE / "figures" / "tree_diagnosis_fire.png"

NAVY = RGBColor(0x0B, 0x2E, 0x59)
ACCENT = RGBColor(0x1F, 0x6F, 0xB2)
GREEN = RGBColor(0x1B, 0x7A, 0x33)
GREY = RGBColor(0x44, 0x44, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xEE, 0xF3, 0xF8)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def add_title_bar(slide, title, sub=None):
    bar = slide.shapes.add_shape(1, 0, 0, SW, Inches(1.15))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.margin_left = Inches(0.45)
    tf.margin_top = Inches(0.12)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = title
    r.font.size = Pt(26)
    r.font.bold = True
    r.font.color.rgb = WHITE
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(14)
        r2.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEE)


def add_footer(slide):
    box = slide.shapes.add_textbox(Inches(0.4), SH - Inches(0.42),
                                   SW - Inches(0.8), Inches(0.32))
    tf = box.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Vanderbilt University  •  NTSB Project  •  Kanu Shetkar"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x8A, 0x97, 0xA6)


def add_bullets(slide, bullets, top=1.45, left=0.55, width=None, height=None, size=17):
    width = width or (SW - Inches(1.1))
    height = height or Inches(4.7)
    box = slide.shapes.add_textbox(Inches(left), Inches(top), width, height)
    tf = box.text_frame
    tf.word_wrap = True
    # Gray card background via shape behind text
    card = slide.shapes.add_shape(1, Inches(left) - Inches(0.08), Inches(top) - Inches(0.08),
                                  width + Inches(0.16), height + Inches(0.12))
    card.fill.solid()
    card.fill.fore_color.rgb = CARD
    card.line.color.rgb = RGBColor(0xBB, 0xC7, 0xD4)
    card.line.width = Pt(0.5)
    card.shadow.inherit = False
    # Send card to back
    slide.shapes._spTree.remove(card._element)
    slide.shapes._spTree.insert(2, card._element)

    first = True
    for text, level in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(5)
        marker = "•  " if level == 0 else "–  "
        r = p.add_run()
        r.text = marker + text
        r.font.size = Pt(size - level * 2)
        r.font.color.rgb = GREY if level else RGBColor(0x1A, 0x1A, 0x1A)
    return box


def add_speaker_note(slide, note):
    slide.notes_slide.notes_text_frame.text = note


def add_table(slide, rows, left, top, width, height, header_fill=ACCENT,
              col_widths=None, font_size=12, highlight_rows=None):
    highlight_rows = highlight_rows or set()
    nrows, ncols = len(rows), len(rows[0])
    gfx = slide.shapes.add_table(nrows, ncols, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    table = gfx.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)
    for r in range(nrows):
        for c in range(ncols):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.04)
            cell.margin_top = Inches(0.01)
            para = cell.text_frame.paragraphs[0]
            run = para.add_run()
            run.text = str(rows[r][c])
            run.font.size = Pt(font_size)
            if r == 0:
                run.font.bold = True
                run.font.color.rgb = WHITE
                cell.fill.solid()
                cell.fill.fore_color.rgb = header_fill
            else:
                cell.fill.solid()
                if r in highlight_rows:
                    cell.fill.fore_color.rgb = RGBColor(0xE6, 0xF4, 0xEA)
                else:
                    cell.fill.fore_color.rgb = RGBColor(0xF4, 0xF7, 0xFA) if r % 2 else WHITE
                run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            if c > 0:
                para.alignment = PP_ALIGN.CENTER
    return table


def short_cause(name: str, n: int = 42) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def load_table7():
    rows = list(csv.DictReader(TABLE7_CSV.open(encoding="utf-8")))
    return rows


# ── 1 Title ──────────────────────────────────────────────────────────────────
s = add_slide()
band = s.shapes.add_shape(1, 0, 0, SW, SH)
band.fill.solid()
band.fill.fore_color.rgb = NAVY
band.line.fill.background()
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.8), SW - Inches(1.6), Inches(3.5))
tf = tb.text_frame
p = tf.paragraphs[0]
r = p.add_run()
r.text = "NTSB Project Update"
r.font.size = Pt(40)
r.font.bold = True
r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
r = p2.add_run()
r.text = "Diagnosis progress — Table 4, Table 7, conditional diagnosis, sparse data, trees"
r.font.size = Pt(20)
r.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEE)
p3 = tf.add_paragraph()
p3.space_before = Pt(20)
r = p3.add_run()
r.text = "Kanu Shetkar  •  Meeting with Maha"
r.font.size = Pt(16)
r.font.color.rgb = WHITE
add_speaker_note(s, "Quick update on what you asked me to check since last time.")

# ── 2 Agenda: last meeting → today ───────────────────────────────────────────
s = add_slide()
add_title_bar(s, "Last meeting → Today's update")
add_table(s, [
    ["Last meeting (you asked)", "Today's update"],
    ["Recheck code & math", "Fixed data merge + prior denominator"],
    ["P(fire) = 5.52×10⁻⁷ should match", "✓ Matches after 102 fires + 184.5M flights"],
    ["Test 5 causes from Table 7", "✓ Full Table 7 — all 85 causes match"],
    ["—", "Table 4: illustrative vs our real analogue"],
    ["—", "Conditional diagnosis (LTP + exact filter)"],
    ["—", "Sparse data plan → Beta-CDF decision"],
    ["—", "Diagnosis tree preview"],
], left=0.55, top=1.55, width=12.2, height=4.8,
   col_widths=[5.5, 6.7], font_size=14, highlight_rows={2, 3})
add_footer(s)
add_speaker_note(s, "Connect last meeting to today. You asked me to verify the fire prior and Table 7 — both done.")

# ── 3 Table 4 ────────────────────────────────────────────────────────────────
s = add_slide()
add_title_bar(s, "Table 4 — Zhang's example vs our similar real example",
              sub="P(fire | electrical wiring, fuel system) — 2-parent CPT from NTSB data")
add_bullets(s, [
    ("Zhang's Table 4 uses hand-picked teaching values (0.99, 0.93, 0.95) — not from counting.", 0),
    ("I rebuilt the same 2×2 layout with real causes: wiring + fuel (1982–2006 data).", 0),
    ("Our raw counts and Zhang's recreated estimator do NOT produce 0.99.", 0),
], top=1.35, height=Inches(1.55), size=15)
add_table(s, [
    ["Wiring", "Fuel", "Count", "Zhang T4\n(paper)", "Our\ncount", "Zhang\nestimator"],
    ["Yes", "Yes", "1/1", "0.99", "1.00", "0.39"],
    ["Yes", "No", "10/21", "0.93", "0.48", "0.39"],
    ["No", "Yes", "23/45", "0.95", "0.51", "0.35"],
    ["No", "No", "68/1675", "≈0", "0.04", "0.00"],
], left=0.5, top=3.05, width=12.3, height=2.55,
   col_widths=[1.0, 0.9, 1.1, 1.5, 1.3, 1.5], font_size=13, highlight_rows={1})
add_bullets(s, [
    ("Takeaway: Table 4 is illustrative. Real data gives ~0.35–0.51, not 0.99.", 0),
], top=5.85, height=Inches(0.8), size=15)
add_footer(s)
add_speaker_note(s, "Walk through the table. Both-present cell is literally 1 fire out of 1 incident.")

# ── 4 Table 7 headline ───────────────────────────────────────────────────────
t7 = load_table7()
s = add_slide()
add_title_bar(s, "Table 7 — all 85 causes match Zhang",
              sub="P(cause | fire) = count / 102 fire accidents")
add_bullets(s, [
    ("Question: given a fire, what contributory cause?  Denominator = 102.", 0),
    ("After merging occurrences + Cause/Factor labeling: 85 / 85 match (±0.00001 rounding).", 0),
    ("Formula is simple counting: (# fires with that cause) ÷ 102.", 0),
    ("Top causes below — full 85-row table on next slides.", 0),
], top=1.35, height=Inches(1.65), size=15)
top8 = t7[:8]
t7_top_rows = [["Cause", "n/102", "Zhang P", "Our P"]]
for r in top8:
    t7_top_rows.append([
        short_cause(r["cause"], 38),
        f"{r['our_n']}/102",
        f"{float(r['zhang_prob']):.5f}",
        f"{float(r['our_prob']):.5f}",
    ])
add_table(s, t7_top_rows, left=0.5, top=3.15, width=12.3, height=2.85,
          col_widths=[6.5, 1.2, 2.3, 2.3], font_size=12)
add_footer(s)
add_speaker_note(s, "Airframe 32/102 = 31.4%. Every cause matches.")

# ── 5–7 Table 7 full (3 slides, two columns) ─────────────────────────────────
def t7_chunk_slide(part_label, chunk_left, chunk_right, start_idx):
    s = add_slide()
    add_title_bar(s, f"Table 7 — full comparison ({part_label})",
                  sub="Zhang P vs Our P — all rows match YES")
    def col_rows(chunk, col_start):
        rows = [["#", "Cause", "n", "Zhang", "Ours"]]
        for i, r in enumerate(chunk, col_start):
            rows.append([
                str(i),
                short_cause(r["cause"], 28),
                r["our_n"],
                f"{float(r['zhang_prob']):.4f}",
                f"{float(r['our_prob']):.4f}",
            ])
        return rows

    add_table(s, col_rows(chunk_left, start_idx),
              left=0.35, top=1.45, width=6.35, height=5.55,
              col_widths=[0.45, 3.5, 0.55, 0.95, 0.95], font_size=9)
    add_table(s, col_rows(chunk_right, start_idx + len(chunk_left)),
              left=6.85, top=1.45, width=6.35, height=5.55,
              col_widths=[0.45, 3.5, 0.55, 0.95, 0.95], font_size=9)
    add_footer(s)


chunks = [t7[i:i + 29] for i in range(0, 85, 29)]
labels = ["rows 1–29", "rows 30–58", "rows 59–85"]
for label, chunk in zip(labels, chunks):
    mid = (len(chunk) + 1) // 2
    start = t7.index(chunk[0]) + 1
    t7_chunk_slide(label, chunk[:mid], chunk[mid:], start)

# ── 8 Conditional diagnosis ──────────────────────────────────────────────────
s = add_slide()
add_title_bar(s, "Conditional diagnosis — two ways to narrow the population")
add_bullets(s, [
    ("Standard (Table 7): given fire → all 102 fires → e.g. wiring = 9/102 = 8.8%.", 0),
    ("", 0),
    ("My core method — similarity + LTP (what Maha asked for):", 0),
    ("embed query → retrieve similar incidents → cluster by type", 1),
    ("Law of Total Probability: P(cause|query) = Σ P(cause|cluster) × P(cluster|query)", 1),
    ("", 0),
    ("Exact filter mode (when we know structured facts):", 0),
    ("fire + electric wiring → keep only 14 incidents with both in the record", 1),
    ("count causes in that cohort → wiring = 11/14 = 78.6%", 1),
    ("", 0),
    ("Exact is better when facts are known (auditable cohort). LTP when user only has narrative.", 0),
], top=1.35, height=Inches(5.5), size=15)
add_footer(s)
add_speaker_note(s, "Do not say we dropped LTP. Exact filter is additive.")

# ── 9 Sparse data ────────────────────────────────────────────────────────────
s = add_slide()
add_title_bar(s, "Sparse data — plan, test, decision (Beta-CDF)",
              sub="When a probability rests on only 1–2 incidents")
add_bullets(s, [
    ("Problem: some cells have almost no data → raw count jumps to 0% or 100%.", 0),
    ("Plan: test narrative smoothing (borrow from similar incident stories).", 0),
    ("Result: looked good on simulated sparse cells, but strict held-out test did not clearly beat Beta-CDF.", 0),
    ("Decision: use Zhang's Beta-CDF (you said yes) as default for sparse cells.", 0),
], top=1.35, left=0.55, width=Inches(5.8), height=Inches(2.4), size=14)
if SPARSE_FIG.is_file():
    s.shapes.add_picture(str(SPARSE_FIG), Inches(6.6), Inches(1.45), Inches(6.2), Inches(4.8))
add_bullets(s, [
    ("Graph: green = our narrative method, blue = Beta-CDF. Lower error = better.", 0),
    ("Narrative wins at tiny n in simulation; Beta-CDF is our approved fallback.", 0),
], top=6.35, height=Inches(0.9), size=13)
add_footer(s)
add_speaker_note(s, "Appendix has Beta-CDF explainer if he asks.")

# ── 10 Diagnosis tree ────────────────────────────────────────────────────────
s = add_slide()
add_title_bar(s, "Diagnosis tree — progress preview",
              sub="Branching causes, not a single ranked list")
add_bullets(s, [
    ("Steps: query → embed → similar incidents → detect fire (root)", 0),
    ("Level 1: P(cause | fire) → top branches (airframe, wiring, fuel…)", 0),
    ("Level 2: filter to incidents with that cause → sub-causes", 0),
    ("Each edge shows probability + n/denom (support visible)", 0),
    ("Status: built + JSON export; Streamlit demo; validating vs Zhang examples next", 0),
], top=1.35, left=0.55, width=Inches(5.5), height=Inches(3.2), size=14)
if TREE_FIG.is_file():
    s.shapes.add_picture(str(TREE_FIG), Inches(6.2), Inches(1.4), Inches(6.5), Inches(5.2))
add_footer(s)
add_speaker_note(s, "Tease only — full validation next meeting.")

# ── 11 Closing ───────────────────────────────────────────────────────────────
s = add_slide()
add_title_bar(s, "Summary")
add_bullets(s, [
    ("Data + prior fixed; Table 7: 85/85 match.", 0),
    ("Table 4: illustrative — our real analogue shown.", 0),
    ("Conditional diagnosis: LTP (narrative) + exact filter (known facts).", 0),
    ("Sparse cells: tested narrative → using Beta-CDF.", 0),
    ("Diagnosis tree: preview built; more validation next.", 0),
], top=1.55, size=18)
add_footer(s)

prs.save(str(OUT))
print(f"Saved {OUT} ({len(prs.slides._sldIdLst)} slides)")
