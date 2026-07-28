"""Generate docs/tuesday_update.pptx — the Tuesday advisor-update deck (~11 content slides).

Standalone build script for the NTSB Bayesian-Network reproduction project. It
matches the visual style of docs/build_diagnosis_core_pptx.py (navy title bar,
accent rule, bullet helpers, speaker notes, side-by-side tables) and additionally
renders two tree figures (diagnosis + prognosis, fire query) into docs/figures/
before assembling the deck.

Every slide carries plain-language speaker notes the student can read aloud.

Run with the framework interpreter (NOT anaconda):
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 docs/build_tuesday_pptx.py
"""
import json
import os
import tempfile
from pathlib import Path

# Matplotlib needs a writable config dir in this environment.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl-"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "tuesday_update.pptx"
FIG_ROOT = HERE / "figures"
OUTPUTS = ROOT / "outputs"

NAVY = RGBColor(0x0B, 0x2E, 0x59)
ACCENT = RGBColor(0x1F, 0x6F, 0xB2)
GREEN = RGBColor(0x1B, 0x7A, 0x33)
GREY = RGBColor(0x44, 0x44, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
CARD = RGBColor(0xEE, 0xF3, 0xF8)

# Matplotlib hex equivalents of the brand palette.
MPL_NAVY = "#0B2E59"
MPL_ACCENT = "#1F6FB2"
MPL_GREEN = "#1B7A33"
MPL_CARD = "#EEF3F8"
MPL_EDGE = "#BBC7D4"


# ============================================================ FIGURE GENERATION
def _short(label, n=34):
    return label if len(label) <= n else label[: n - 1] + "\u2026"


def _count_leaves(node):
    kids = node.get("children") or []
    if not kids:
        return 1
    return sum(_count_leaves(k) for k in kids)


def render_tree_figure(root, out_path, title, kind, root_caption):
    """Render a left-to-right hierarchical tree to a PNG.

    kind = "diagnosis" (<- inferred causes) or "prognosis" (-> forward events).
    Each non-root edge box shows label, edge probability and support n/denom.
    """
    arrow = "\u2190" if kind == "diagnosis" else "\u2192"
    leaf_count = _count_leaves(root)
    fig_h = max(4.2, 0.62 * leaf_count)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.set_axis_off()

    x_gap = 3.55
    leaf_cursor = [0]
    positions = {}

    def assign(node, depth):
        kids = node.get("children") or []
        if not kids:
            y = leaf_cursor[0]
            leaf_cursor[0] += 1
        else:
            ys = [assign(k, depth + 1) for k in kids]
            y = sum(ys) / len(ys)
        positions[id(node)] = (depth * x_gap, y)
        return y

    assign(root, 0)

    def color_for(depth):
        if depth == 0:
            return MPL_NAVY, "white"
        if depth == 1:
            return MPL_ACCENT, "white"
        return MPL_CARD, "#1A1A1A"

    def draw(node, depth, parent_xy=None):
        x, y = positions[id(node)]
        # edge line from parent
        if parent_xy is not None:
            px, py = parent_xy
            ax.plot([px + 1.45, x - 0.02], [py, y], color=MPL_EDGE, lw=1.3,
                    zorder=1, solid_capstyle="round")
        face, txt = color_for(depth)
        box_w, box_h = 2.95, 0.62
        box = FancyBboxPatch((x, y - box_h / 2), box_w, box_h,
                             boxstyle="round,pad=0.02,rounding_size=0.08",
                             linewidth=1.0, edgecolor=MPL_EDGE, facecolor=face,
                             zorder=2)
        ax.add_patch(box)
        if depth == 0:
            label = f"{node['label'].upper()}\n{root_caption}"
            ax.text(x + box_w / 2, y, label, ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=txt, zorder=3)
        else:
            p = node.get("edge_prob", 0.0)
            n = node.get("n")
            denom = node.get("denom")
            supp = f"{n}/{denom}" if n is not None and denom is not None else f"{denom}"
            head = f"{arrow} {_short(node['label'])}"
            sub = f"p={p:.3f}   N={supp}   path={node.get('path_prob', 0.0):.3f}"
            ax.text(x + box_w / 2, y + 0.10, head, ha="center", va="center",
                    fontsize=8.0, fontweight="bold", color=txt, zorder=3)
            ax.text(x + box_w / 2, y - 0.15, sub, ha="center", va="center",
                    fontsize=6.8, color=txt, zorder=3)
        for k in (node.get("children") or []):
            draw(k, depth + 1, (x, y))

    draw(root, 0)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 0.3, max(xs) + 3.4)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.9)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=12, fontweight="bold", color=MPL_NAVY, loc="left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Diagnosis fire tree — generated from tests/build_diagnosis_figure.py
# (faithful Table-7 counting, upstream causes only, N=102).
DIAGNOSIS_FIRE_TREE = {
    "label": "fire", "depth": 0, "denom": 102, "children": [
        {"label": "Electrical system, electric wiring", "edge_prob": 0.088, "n": 9, "denom": 102,
         "path_prob": 0.088, "children": [
            {"label": "Electrical system, circuit breaker", "edge_prob": 0.143, "n": 2, "denom": 14, "path_prob": 0.013, "children": []},
            {"label": "Maintenance, installation", "edge_prob": 0.143, "n": 2, "denom": 14, "path_prob": 0.013, "children": []},
            {"label": "Maintenance, modification", "edge_prob": 0.143, "n": 2, "denom": 14, "path_prob": 0.013, "children": []},
            {"label": "Maintenance, service of aircraft/equipment", "edge_prob": 0.071, "n": 1, "denom": 14, "path_prob": 0.006, "children": []},
         ]},
        {"label": "Loss of engine power (total) - mechanical failure/malfunction", "edge_prob": 0.088, "n": 9, "denom": 102,
         "path_prob": 0.088, "children": [
            {"label": "Engine compartment", "edge_prob": 0.111, "n": 1, "denom": 9, "path_prob": 0.010, "children": []},
            {"label": "Wing", "edge_prob": 0.111, "n": 1, "denom": 9, "path_prob": 0.010, "children": []},
            {"label": "Maintenance, service of aircraft/equipment", "edge_prob": 0.111, "n": 1, "denom": 9, "path_prob": 0.010, "children": []},
         ]},
        {"label": "Fluid, fuel", "edge_prob": 0.059, "n": 6, "denom": 102,
         "path_prob": 0.059, "children": [
            {"label": "Ignition system, ignition harness", "edge_prob": 0.125, "n": 1, "denom": 8, "path_prob": 0.007, "children": []},
            {"label": "Fuel system, line fitting", "edge_prob": 0.125, "n": 1, "denom": 8, "path_prob": 0.007, "children": []},
            {"label": "Ignition system, exciter", "edge_prob": 0.125, "n": 1, "denom": 8, "path_prob": 0.007, "children": []},
         ]},
        {"label": "Auxiliary power unit (APU)", "edge_prob": 0.049, "n": 5, "denom": 102,
         "path_prob": 0.049, "children": [
            {"label": "Engine accessories, engine starter", "edge_prob": 0.125, "n": 1, "denom": 8, "path_prob": 0.006, "children": []},
            {"label": "Procedure inadequate", "edge_prob": 0.125, "n": 1, "denom": 8, "path_prob": 0.006, "children": []},
         ]},
    ],
}


def generate_figures():
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    diag_png = FIG_ROOT / "tree_diagnosis_fire.png"
    prog_png = FIG_ROOT / "tree_prognosis_fire.png"

    render_tree_figure(
        DIAGNOSIS_FIRE_TREE, str(diag_png),
        title="Diagnosis tree  \u2014  query: \u201cengine caught fire during takeoff\u201d",
        kind="diagnosis",
        root_caption="N=102 fire accidents  (upstream causes only)",
    )

    # Prognosis fire tree straight from the on-disk artifact.
    with open(OUTPUTS / "prognosis_tree.json") as f:
        prog = json.load(f)
    render_tree_figure(
        prog["tree"], str(prog_png),
        title="Prognosis tree  \u2014  same fire query, forward escalation",
        kind="prognosis",
        root_caption="seed event (denom=15)",
    )
    return diag_png.name, prog_png.name


# ============================================================ PPTX HELPERS
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
    tf.word_wrap = False
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Vanderbilt University  \u2022  NTSB Bayesian-Network Reproduction (Zhang & Mahadevan, RESS 2021)"
    r.font.size = Pt(9)
    r.font.color.rgb = RGBColor(0x8A, 0x97, 0xA6)


def add_bullets(slide, bullets, top=1.45, left=0.55, width=None, height=None, size=18):
    width = width or (SW - Inches(1.1))
    height = height or Inches(4.7)
    box = slide.shapes.add_textbox(Inches(left), Inches(top), width, height)
    tf = box.text_frame
    tf.word_wrap = True
    first = True
    for text, level in bullets:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(6)
        marker = "\u2022  " if level == 0 else "\u2013  "
        r = p.add_run()
        r.text = marker + text
        r.font.size = Pt(size - level * 2)
        r.font.color.rgb = GREY if level else RGBColor(0x1A, 0x1A, 0x1A)
    return box


def add_speaker_note(slide, note):
    slide.notes_slide.notes_text_frame.text = note


def add_picture_fit(slide, path, left, top, max_w, max_h, center=True, border=True):
    with PILImage.open(str(path)) as im:
        iw, ih = im.size
    ar = iw / ih
    box_ar = max_w / max_h
    if ar > box_ar:
        w = max_w
        h = max_w / ar
    else:
        h = max_h
        w = max_h * ar
    if center:
        left = left + (max_w - w) / 2.0
        top = top + (max_h - h) / 2.0
    pic = slide.shapes.add_picture(str(path), Inches(left), Inches(top), Inches(w), Inches(h))
    if border:
        pic.line.color.rgb = RGBColor(0xBB, 0xC7, 0xD4)
        pic.line.width = Pt(0.75)
    return pic


def add_table(slide, rows, left, top, width, height, header_fill=ACCENT,
              col_widths=None, font_size=13, highlight_rows=None):
    highlight_rows = highlight_rows or set()
    nrows = len(rows)
    ncols = len(rows[0])
    gfx = slide.shapes.add_table(nrows, ncols, Inches(left), Inches(top),
                                 Inches(width), Inches(height))
    table = gfx.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)
    for r in range(nrows):
        for c in range(ncols):
            cell = table.cell(r, c)
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.02)
            cell.margin_bottom = Inches(0.02)
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
                    run.font.bold = True
                else:
                    cell.fill.fore_color.rgb = RGBColor(0xF4, 0xF7, 0xFA) if r % 2 else WHITE
                run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
            if c > 0:
                para.alignment = PP_ALIGN.CENTER
    return table


# ============================================================ BUILD
DIAG_FIG, PROG_FIG = generate_figures()

# ---------------------------------------------------------- 1: title
s = add_slide()
band = s.shapes.add_shape(1, 0, 0, SW, SH)
band.fill.solid(); band.fill.fore_color.rgb = NAVY
band.line.fill.background(); band.shadow.inherit = False
accent = s.shapes.add_shape(1, 0, Inches(4.35), SW, Inches(0.08))
accent.fill.solid(); accent.fill.fore_color.rgb = ACCENT
accent.line.fill.background(); accent.shadow.inherit = False
tb = s.shapes.add_textbox(Inches(0.8), Inches(1.7), SW - Inches(1.6), Inches(3.6))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "NTSB Bayesian Network \u2014 Tuesday Update"
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
r = p2.add_run(); r.text = "Reproducing & extending Zhang et al. (RESS 2021)"
r.font.size = Pt(22); r.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEE)
p3 = tf.add_paragraph(); p3.space_before = Pt(22)
r = p3.add_run(); r.text = "Presenter: Kanu Shetkar    \u2022    Advisor: Maha    \u2022    Tuesday"
r.font.size = Pt(16); r.font.color.rgb = WHITE
p4 = tf.add_paragraph()
r = p4.add_run(); r.text = "What you asked for  \u2192  what I fixed  \u2192  trees (the new direction)"
r.font.size = Pt(16); r.font.italic = True; r.font.color.rgb = RGBColor(0x9F, 0xC2, 0xE0)
add_speaker_note(s,
    "Open by reminding Maha what she asked me to do: reproduce Zhang's fire numbers "
    "exactly, settle the Table 4 question, validate the evidence-filtered diagnosis "
    "mode and the sparse-cell method, and move toward the tree picture of diagnosis "
    "and prognosis. This talk walks through each: two data bugs I found and fixed, a "
    "true cell-by-cell match of Table 7, the Table 4 clarification, the two validated "
    "add-ons, and then the headline new direction \u2014 trees. Keep it to fifteen minutes; "
    "the appendix has the full tables if she wants detail.")

# ---------------------------------------------------------- 2: two data bugs
s = add_slide()
add_title_bar(s, "Two separate data bugs \u2014 found & fixed",
              sub="Two different fixes for two different quantities (do not conflate them)")
add_table(s, [
    ["", "Fix #1 \u2014 the FIRE COUNT", "Fix #2 \u2014 the PRIOR P(fire)"],
    ["Bug",
     "occurrences / seq-of-events\nnever merged into findings",
     "prior divided by 2,243 accidents\n(wrong denominator)"],
    ["Effect of bug",
     "only 38 fires counted",
     "P(fire) far too large"],
    ["Fix",
     "merge legacy occurrences +\nseq_of_events with findings",
     "divide by TOTAL flights\n184.5M departures (1982\u20132006)"],
    ["Result",
     "38 \u2192 102 fires (matches Zhang)",
     "P(fire) \u2248 5.5\u00d710\u207b\u2077 (matches Zhang exactly)"],
], left=0.5, top=1.55, width=12.3, height=4.3,
   col_widths=[2.1, 5.1, 5.1], font_size=14, highlight_rows={4})
add_bullets(s, [
    ("Two quantities, two fixes: the MERGE fixed the fire count (and therefore Table 7); "
     "the DENOMINATOR fixed the prior P(fire). They are independent.", 0),
], top=6.0, size=14)
add_footer(s)
add_speaker_note(s,
    "This is the slide to be precise on. There were two different problems with two "
    "different fixes \u2014 don't let them blur together. Fix one: the dataset never merged "
    "the legacy occurrence and sequence-of-events records into the findings, so only 38 "
    "fires were counted; after merging, 102 fires, which is Zhang's number and which "
    "drives Table 7. Fix two is about a completely different quantity \u2014 the prior "
    "probability of fire. I had been dividing by the 2,243 accidents in the set, but "
    "Zhang's prior is per flight: divide the fire count by the total number of flights "
    "flown, about 184.5 million departures from 1982 to 2006. That gives P(fire) around "
    "5.5 times ten-to-the-minus-seven, matching Zhang exactly. So: merge fixed the "
    "count, denominator fixed the prior \u2014 two separate things.")

# ---------------------------------------------------------- 3: Table 7
s = add_slide()
add_title_bar(s, "Table 7 \u2014 first TRUE cell-by-cell match: 85/85 exact",
              sub="Denominator = 102 fires; anchor Airframe = 0.31372 (n=32); tol \u00b10.0005")
add_table(s, [
    ["Cause (excerpt)", "Zhang P", "n", "Our P", "Match"],
    ["Airframe/component/system failure/malfunction", "0.31372", "32", "0.31373", "\u2713"],
    ["Loss of engine power (total) - mechanical", "0.08823", "9", "0.08824", "\u2713"],
    ["Electrical system, electric wiring", "0.08823", "9", "0.08824", "\u2713"],
    ["Fluid, fuel", "0.05882", "6", "0.05882", "\u2713"],
    ["Auxiliary power unit (APU)", "0.04901", "5", "0.04902", "\u2713"],
    ["Unknown quantity (code 92000 fallback)", "0.01960", "2", "0.01961", "\u2713"],
], left=0.45, top=1.55, width=7.9, height=3.5,
   col_widths=[4.5, 1.1, 0.6, 1.1, 0.6], font_size=12, highlight_rows={1})
add_bullets(s, [
    ("Be honest about what changed:", 0),
    ("The earlier \u201call match\u201d was a self-consistency check (our engine vs itself).", 1),
    ("This is the FIRST true cell-by-cell match to the printed paper.", 1),
    ("Closed the last 18 gaps by counting only contributory findings \u2014 "
     "Cause_Factor \u2208 {C, F}.", 0),
    ("Plus the \u201cUnknown quantity\u201d fallback label for unresolved code 92000 (n=2).", 1),
    ("No dataset rows were edited \u2014 fix lives in the label/edge layer only.", 0),
], top=1.6, left=8.5, width=Inches(4.5), size=13)
add_footer(s)
add_speaker_note(s,
    "Table 7 is now a genuine, cell-by-cell reproduction of the paper: all 85 causes "
    "match within Zhang's own rounding. The honesty point matters \u2014 last time the 'all "
    "match' was really our engine agreeing with itself, a self-consistency check. This "
    "time I'm comparing each cell against the numbers printed in Zhang's Table 7, and "
    "they line up, anchored by the dominant Airframe cell at 0.31372 with 32 of 102 "
    "fires. The last 18 gaps were not logic errors; they were label-mapping. Zhang "
    "counts only contributory findings \u2014 the ones flagged Cause or Factor \u2014 so I "
    "filtered to those, and I matched his fallback label 'Unknown quantity' for one "
    "unresolved code. Importantly I changed no data rows; the fix is purely in how "
    "labels are counted.")

# ---------------------------------------------------------- 4: Table 4
s = add_slide()
add_title_bar(s, "Table 4 \u2014 REAL data-derived example (per Maha)",
              sub="Zhang's Table 4 is illustrative; here is the same format populated from real NTSB data")
add_bullets(s, [
    ("Zhang's Table 4 (0.99 / 0.93 / 0.95 / 2e-9) is a TEACHING table \u2014 \u201cfor the sake "
     "of demonstration, assume\u2026\u201d. Agreed with Maha: we build a REAL one for the paper.", 0),
    ("Real CPT: P(fire | electrical wiring, fuel system) \u2014 two real contributory-factor "
     "parents, all four cells counted from the 1982\u20132006 window (1,742 accidents):", 0),
], top=1.45, height=Inches(1.4), size=15)
add_table(s, [
    ["wiring", "fuel", "n / denom", "raw", "Beta-CDF", "final (0.95 cap)", "Zhang T4 (toy)"],
    ["Yes", "Yes", "1 / 1", "1.000", "1.000", "0.950  (sparse)", "0.99"],
    ["Yes", "No", "10 / 21", "0.476", "0.717", "0.476", "0.93"],
    ["No", "Yes", "23 / 45", "0.511", "0.753", "0.511", "0.95"],
    ["No", "No", "68 / 1675", "0.041", "0.071", "0.041", "2e-9"],
], left=0.5, top=2.9, width=10.0, height=2.2,
   col_widths=[1.0, 1.0, 1.5, 1.2, 1.4, 2.2, 1.7], font_size=13, highlight_rows={1})
add_bullets(s, [
    ("Sparse cell (both present, n=1) is exactly where Zhang's 0.95 cap / Beta-CDF "
     "smoothing engages \u2014 the real table demonstrates the sparse-data machinery.", 0),
    ("The toy 0.99 / 0.93 are unreachable from real counts (real single-cause rates "
     "\u2248 0.35\u20130.51) \u2014 confirming Table 4 was illustrative, as Maha said.", 0),
    ("Reproduce: tests/build_table4_analogue.py (offline, reads the corrected window).", 0),
], top=5.25, size=13)
add_footer(s)
add_speaker_note(s,
    "This is the real Table 4 example Maha asked for. His Table 4 is a teaching table "
    "for a toy network, so instead of chasing its hand-picked numbers we built the same "
    "FORMAT from real data: probability of fire given two real parent causes, electrical "
    "wiring and the fuel system, all four present-absent combinations, counted over the "
    "corrected 1982 to 2006 window. Three things to point out. First, the both-present "
    "cell has exactly one incident \u2014 a live example of the sparse-cell problem, and you "
    "can see Zhang's 0.95 cap engage there. Second, the neither-present cell is about "
    "four percent, the base rate of fire without those causes \u2014 not two times ten to the "
    "minus nine. Third, the toy values 0.99 and 0.93 are unreachable from real counts, "
    "which confirms Table 4 was illustrative. This real table is what goes in the paper.")

# ---------------------------------------------------------- 5: evidence-filtered cohort diagnosis
s = add_slide()
add_title_bar(s, "Evidence-Filtered Cohort Diagnosis  (renamed per Maha)",
              sub="Was \u201cconditional diagnosis\u201d \u2014 an ADD-ON to query-first similarity, not a replacement")
add_bullets(s, [
    ("What it is: filter the corpus to incidents matching the outcome AND the known "
     "facts, then count causes over that cohort (no Bayesian-conditioning claim).", 0),
    ("Validated on multiple outcome+fact combinations, e.g.:", 0),
], top=1.45, height=Inches(1.5), size=16)
add_table(s, [
    ["Outcome", "Known fact(s)", "Cohort n", "Top cause (P)"],
    ["fire", "electric wiring", "14", "Electrical wiring (78.6%)"],
    ["fire", "fuel", "8", "Fluid, fuel (87.5%)"],
    ["loss of engine power", "fuel", "9", "Fluid, fuel (100%)"],
    ["fire", "wiring + maintenance", "0", "empty cohort (safe)"],
], left=0.7, top=3.0, width=9.4, height=2.3,
   col_widths=[3.0, 2.6, 1.2, 2.6], font_size=13, highlight_rows={1})
add_bullets(s, [
    ("Stays QUERY-FIRST: it's a targeted add-on to the narrative similarity mode.", 0),
    ("Degrades safely \u2014 narrowing the filter shrinks n; an impossible combo returns an "
     "empty cohort, no crash. Flag tiny-n (n<5) as low-confidence.", 0),
], top=5.45, size=14)
add_footer(s)
add_speaker_note(s,
    "Per Maha's note I've renamed this 'evidence-filtered cohort diagnosis' instead of "
    "'conditional diagnosis', because it doesn't claim Bayesian conditioning \u2014 it just "
    "filters to a cohort by the evidence and reports frequencies. Mechanically: given an "
    "outcome and some known facts, keep only incidents that contain all of them, then "
    "count causes. It reproduces the canonical example \u2014 fire plus electric wiring gives "
    "14 incidents with wiring dominating at 78.6 percent \u2014 and it generalizes to other "
    "combinations. Two things to stress: it stays query-first, it's an add-on to the "
    "similarity search, not a replacement; and it degrades safely \u2014 narrow filters give "
    "small cohorts, impossible combinations give an empty cohort rather than an error, "
    "and I flag tiny cohorts as low-confidence.")

# ---------------------------------------------------------- 6: sparse data
s = add_slide()
add_title_bar(s, "Sparse cells: tested narrative smoothing, decided on Beta-CDF",
              sub="Revised per Maha \u2014 every term on this slide defined")
add_bullets(s, [
    ("Problem: some forward cells P(outcome | cause) have almost no data \u2192 raw "
     "counting gives 0% or 100%. Zhang's fix: Beta-CDF smoothing (we match it).", 0),
    ("Our test: can narrative smoothing (similar incident stories) beat Beta-CDF?", 0),
    ("How we tested \u2014 the terms:", 0),
    ("a PAIR = one probability question, one cause + one outcome (e.g. P(fire | fuel)). "
     "158 pairs where the full-data answer is trusted.", 1),
    ("n = how many incidents we PRETEND to have for that question (1, 2, 3, 5, 10) "
     "after randomly hiding the rest.", 1),
    ("400 random subsamples per n, so no single lucky draw decides it. "
     "Methods compared: raw count, Zhang cap, Beta-CDF, narrative.", 1),
    ("HELD-OUT: each scored incident hidden from the index so narrative cannot peek.", 1),
    ("Result: narrative did NOT clearly beat Beta-CDF on the fair held-out test.", 0),
    ("Decision (Maha approved): use Beta-CDF for sparse forward cells. "
     "Table 7 unchanged \u2014 still plain counting over 102 fires.", 0),
], size=15)
add_footer(s)
add_speaker_note(s,
    "For sparse FORWARD probabilities \u2014 not Table 7 \u2014 some cells only have one or two "
    "incidents, so plain counting gives zero or one hundred percent. Zhang smooths those "
    "with a Beta-CDF curve. We wanted to see if narrative smoothing could do better. We "
    "couldn't score real one-incident cells without knowing the truth, so we used about "
    "158 cause-and-outcome pairs where we DO trust the full-data rate. Then we pretended "
    "we only had one, two, three, five, or ten incidents \u2014 four hundred random "
    "subsamples each \u2014 and asked four methods to guess against that known answer. On a "
    "strict held-out test, hiding each incident so narrative can't peek, narrative did "
    "not clearly beat Beta-CDF. So we're using Beta-CDF to match Zhang, and Table 7 "
    "stays plain counting over 102 fires. If he asks what n is: n is how many incidents "
    "we pretend to have for that one question \u2014 not 158, not 400. 158 is how many "
    "questions; 400 is how many repeats per question.")

# ---------------------------------------------------------- 7: trees concept + diagnosis tree
s = add_slide()
add_title_bar(s, "The new direction: diagnosis & prognosis are TREES",
              sub="From one event, MULTIPLE outcomes branch \u2014 not a single linear chain")
add_picture_fit(s, FIG_ROOT / DIAG_FIG, left=0.35, top=1.35, max_w=8.5, max_h=5.5)
add_bullets(s, [
    ("Maha's point: from one event, many outcomes branch, each with its own "
     "probability \u2014 a TREE, not a chain.", 0),
    ("Query-first generation:", 0),
    ("free-text query \u2192 embed \u2192 retrieve similar incidents \u2192 build tree over "
     "that pool.", 1),
    ("Diagnosis edge = P(cause | outcome AND ancestors) \u2014 Zhang's Table-7 counting, "
     "made recursive.", 0),
    ("Each edge shows p, support N=n/denom, and cumulative path probability.", 0),
    ("Figure: real run, query \u201cengine caught fire during takeoff\u201d.", 1),
], top=1.5, left=8.95, width=Inches(4.1), size=13)
add_footer(s)
add_speaker_note(s,
    "Here's the headline new direction. Maha clarified that diagnosis and prognosis "
    "aren't single linear chains \u2014 they're trees. From one event, several different "
    "things can branch out, each with a probability, and each of those can branch "
    "again. Two things to emphasize. First, it's query-first: I start from the user's "
    "plain-text query, embed it, retrieve similar incidents, and build the tree only "
    "over that relevant pool. Second, the math at each branch is exactly Zhang's Table 7 "
    "quantity \u2014 probability of a cause given the outcome \u2014 just made recursive: causes "
    "of the outcome, then sub-causes that co-occur with each cause. The figure is a real "
    "run for a fire query: the root is fire across all 102 fire accidents \u2014 the Table 7 "
    "denominator \u2014 and each branch shows the edge probability, the support like 32 out "
    "of 102, and the running path probability. Downstream responses like evacuation are "
    "excluded; only upstream causes appear, per Maha's diagnosis/prognosis separation.")

# ---------------------------------------------------------- 8: prognosis tree + publication gate
s = add_slide()
add_title_bar(s, "Prognosis tree + the publication gate",
              sub="Same fire query, read forward: P(next event | current event) escalation")
add_picture_fit(s, FIG_ROOT / PROG_FIG, left=0.35, top=1.35, max_w=8.5, max_h=5.5)
add_bullets(s, [
    ("Prognosis = the forward direction: from the seed event, where does it escalate?", 0),
    ("Edge = P(next event | current event), a Markov hop with visible support N.", 0),
    ("Same query-first generation; branches top-B next-events, prunes by prob & support.", 0),
    ("Both trees export to JSON \u2014 ready for the Streamlit demo.", 0),
    ("Publication gate:", 0),
    ("a full diagnosis tree AND a full prognosis tree, validated against Zhang's "
     "paper examples.", 1),
], top=1.5, left=8.95, width=Inches(4.1), size=13)
add_footer(s)
add_speaker_note(s,
    "The same idea run forward is prognosis: starting from an event, what does it "
    "escalate into? Each edge here is the probability of the next event given the "
    "current one \u2014 an honest Markov hop \u2014 and I keep the support N visible at every "
    "edge so you can see how thin the data gets deeper down. It's built the same "
    "query-first way and both trees already export to JSON, so they drop straight into "
    "the Streamlit demo. The bottom line for Maha is the publication gate: what we need "
    "to publish is a complete diagnosis tree and a complete prognosis tree, validated "
    "against Zhang's own paper examples. That's the target the rest of the work points "
    "at.")

# ---------------------------------------------------------- 9: next steps / timeline
s = add_slide()
add_title_bar(s, "Next steps & timeline")
add_table(s, [
    ["Track", "Status", "Next step"],
    ["Table 7 reproduction", "Done \u2014 85/85 exact", "Lock as the reproduction anchor"],
    ["Table 4", "Resolved (illustrative)", "Optional: build data-derived CPT for a real node"],
    ["Evidence-filtered cohort diagnosis", "Validated, usable today", "Surface cohort n + low-confidence flags"],
    ["Sparse cells", "Tested \u2192 default Beta-CDF", "Keep semantic as experimental n\u22643"],
    ["Trees (diagnosis + prognosis)", "Implemented + demo PASS", "Finish, validate vs all Zhang examples"],
    ["Streamlit demo", "App exists", "Wire trees JSON for live demo"],
], left=0.5, top=1.5, width=12.3, height=3.9,
   col_widths=[3.8, 3.6, 4.9], font_size=13, highlight_rows={5})
add_bullets(s, [
    ("Headline next step: finish & validate the trees, build them against ALL of Zhang's "
     "paper examples, then demo live in Streamlit.", 0),
], top=5.7, size=15)
add_footer(s)
add_speaker_note(s,
    "Where things go from here. The reproduction work is done and solid \u2014 Table 7 is the "
    "anchor, Table 4 is resolved, and both add-ons are validated with clear decisions. "
    "The real forward energy is on the trees: they're implemented and the demo passes, "
    "and the job now is to finish them and validate them against all of Zhang's paper "
    "examples, not just the fire case. In parallel, the Streamlit app already exists, so "
    "I can wire the tree JSON into it and show this live. If you want, the one optional "
    "extra is building a populated, data-derived CPT for a real node as the honest "
    "Table-4 analogue.")

# ---------------------------------------------------------- 10: appendix - full Table 7
s = add_slide()
add_title_bar(s, "Appendix \u2014 Table 7 reproduction (top 14 of 85)",
              sub="Full 85/85 in docs/TABLE7_FULL_REPRODUCTION.md and table7_full_reproduction.csv")
left_rows = [["#", "Cause", "Z P", "n"]]
right_rows = [["#", "Cause", "Z P", "n"]]
t7 = [
    (1, "Airframe/component/system failure", "0.31372", 32),
    (2, "Loss of engine power (total) - mech", "0.08823", 9),
    (3, "Electrical system, electric wiring", "0.08823", 9),
    (4, "Fluid, fuel", "0.05882", 6),
    (5, "Auxiliary power unit (APU)", "0.04901", 5),
    (6, "Maintenance, installation", "0.03921", 4),
    (7, "Procedure inadequate", "0.03921", 4),
    (8, "Loss of engine power (partial) - mech", "0.03921", 4),
    (9, "Maintenance, service bulletin/letter", "0.02941", 3),
    (10, "Engine compartment", "0.02941", 3),
    (11, "Maintenance", "0.02941", 3),
    (12, "Cargo/baggage", "0.02941", 3),
    (13, "Fuel system, nozzle", "0.01960", 2),
    (14, "Fuel system, drain", "0.01960", 2),
]
for row in t7[:7]:
    left_rows.append([row[0], row[1], row[2], row[3]])
for row in t7[7:]:
    right_rows.append([row[0], row[1], row[2], row[3]])
add_table(s, left_rows, left=0.4, top=1.55, width=6.2, height=3.6,
          col_widths=[0.5, 4.0, 1.0, 0.7], font_size=12)
add_table(s, right_rows, left=6.85, top=1.55, width=6.2, height=3.6,
          col_widths=[0.5, 4.0, 1.0, 0.7], font_size=12)
add_bullets(s, [
    ("Our P matches Zhang P for all 85 causes within \u00b10.0005 (Zhang's 5-dp rounding); "
     "denominator = 102.", 0),
], top=5.4, size=14)
add_footer(s)
add_speaker_note(s,
    "Backup detail if Maha wants to see more of the table. These are the top rows of "
    "Zhang's Table 7 with the cause, his probability and the count out of 102; ours "
    "matches every one within his own rounding, all 85 of them. The complete table and "
    "a CSV are in the docs folder if she wants to inspect any specific cell.")

# ---------------------------------------------------------- 11: appendix - methods
s = add_slide()
add_title_bar(s, "Appendix \u2014 methods & sources")
add_bullets(s, [
    ("Data: data/processed/refined_dataset_1982_2006.json (1,742 incidents), legacy "
     "occurrences + seq_of_events merged with findings.", 0),
    ("Table 7: empirical_cause_distribution(\u201cfire\u201d, cause_factor_only=True), "
     "denominator = 102, contributory findings (Cause_Factor \u2208 {C,F}); no rows mutated.", 0),
    ("Prior: P(fire) = 102 / 184,517,128 total departures (BTS, 1982\u20132006) = 5.53e-7.", 0),
    ("Evidence-filtered cohort diagnosis: hard set-intersection over outcome + facts, "
     "then Zhang count(cause & outcome)/count(outcome); validated in "
     "tests/exact_filter_validation.py.", 0),
    ("Sparse cells: leave-one-incident-out comparison (raw / cap / Beta-CDF / semantic); "
     "tests/sparse_robustness_validation.py.", 0),
    ("Trees: trees.py (imports zhang_diagnosis, prognosis, main_app unchanged); "
     "outputs/diagnosis_tree.json, outputs/prognosis_tree.json; tests/tree_demo.py.", 0),
    ("Source: Zhang & Mahadevan, Reliability Engineering and System Safety 209 (2021) "
     "107371.", 0),
], size=14)
add_footer(s)
add_speaker_note(s,
    "This is a reference slide so every claim is traceable. It lists the dataset, the "
    "exact function call and denominator behind Table 7, the prior formula and its "
    "184.5-million-flight denominator, how the evidence-filtered diagnosis and the "
    "sparse-cell comparison were run and which test files reproduce them, and where the "
    "tree code and JSON exports live. It also cites the Zhang and Mahadevan paper. I can "
    "reproduce any number on request with framework Python 3.11.")

prs.save(str(OUT))
n = len(prs.slides._sldIdLst)
print(f"Saved {OUT} with {n} slides")
print(f"Figures generated: {DIAG_FIG}, {PROG_FIG}")
for i, sl in enumerate(prs.slides, 1):
    title = ""
    for sh in sl.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            title = sh.text_frame.text.strip().splitlines()[0]
            break
    print(f"  {i:2d}. {title}")
