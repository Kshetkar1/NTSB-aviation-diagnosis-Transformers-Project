"""Generate docs/diagnosis_slides_core.pptx — a TRIMMED CORE deck (~15 slides).

This is a standalone companion to docs/build_diagnosis_pptx.py. It reuses the SAME
slide-building helpers and copies the SAME slide content (bullets + speaker notes +
figures) verbatim for ONLY the core slides needed for a 15-minute talk. The full
deck script (docs/build_diagnosis_pptx.py) is left completely untouched as the backup.

Run with the framework interpreter:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 docs/build_diagnosis_core_pptx.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from PIL import Image as PILImage

HERE = Path(__file__).resolve().parent
OUT = HERE / "diagnosis_slides_core.pptx"
FIG = HERE / "figures" / "zhang"
FIG_ROOT = HERE / "figures"

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
    r.font.size = Pt(28)
    r.font.bold = True
    r.font.color.rgb = WHITE
    if sub:
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = sub
        r2.font.size = Pt(15)
        r2.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEE)


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
        marker = "•  " if level == 0 else "–  "
        r = p.add_run()
        r.text = marker + text
        r.font.size = Pt(size - level * 2)
        r.font.color.rgb = GREY if level else RGBColor(0x1A, 0x1A, 0x1A)
    return box


def add_speaker_note(slide, note):
    slide.notes_slide.notes_text_frame.text = note


def add_caption(slide, text, left, top, width, size=13, color=None, italic=True, bold=False):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(0.5))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.italic = italic
    r.font.bold = bold
    r.font.color.rgb = color or GREY
    return box


def add_picture_fit(slide, img_name, left, top, max_w, max_h, center=True, border=True,
                    base=None):
    """Add a picture scaled to fit within (max_w x max_h) inches, preserving aspect."""
    path = (base or FIG) / img_name
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


# ============================================================ CORE 1: title
s = add_slide()
band = s.shapes.add_shape(1, 0, 0, SW, SH)
band.fill.solid()
band.fill.fore_color.rgb = NAVY
band.line.fill.background()
band.shadow.inherit = False
accent = s.shapes.add_shape(1, 0, Inches(4.35), SW, Inches(0.08))
accent.fill.solid()
accent.fill.fore_color.rgb = ACCENT
accent.line.fill.background()
accent.shadow.inherit = False

tb = s.shapes.add_textbox(Inches(0.8), Inches(1.9), SW - Inches(1.6), Inches(3.2))
tf = tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
r = p.add_run(); r.text = "NTSB Bayesian Network — Diagnosis"
r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
p2 = tf.add_paragraph()
r = p2.add_run(); r.text = "Reproduction & generalization of Zhang et al. (RESS 2021)"
r.font.size = Pt(22); r.font.color.rgb = RGBColor(0xCF, 0xDD, 0xEE)
p3 = tf.add_paragraph(); p3.space_before = Pt(24)
r = p3.add_run(); r.text = "Presenter: [student name]    •    Advisor: Maha    •    June 25, 2026"
r.font.size = Pt(16); r.font.color.rgb = WHITE
p4 = tf.add_paragraph()
r = p4.add_run(); r.text = "The issue I found  →  how I fixed it  →  the diagnosis results"
r.font.size = Pt(16); r.font.italic = True; r.font.color.rgb = RGBColor(0x9F, 0xC2, 0xE0)
p5 = tf.add_paragraph(); p5.space_before = Pt(10)
r = p5.add_run(); r.text = "Zhang's original figures/tables shown side-by-side with ours throughout"
r.font.size = Pt(14); r.font.italic = True; r.font.color.rgb = RGBColor(0x9F, 0xC2, 0xE0)
add_speaker_note(s, "Three beats: the bug I found, the fix, the results. New this version: every "
                    "key result is paired with the actual figure/table from Zhang's paper so you "
                    "can verify the numbers. Headline: same data -> same numbers as Zhang, and we "
                    "go a bit beyond. Scope is the fire node, 1982-2006 window.")

# ============================================================ CORE 2: problem
s = add_slide()
add_title_bar(s, "The problem: my numbers didn't match Zhang")
add_bullets(s, [
    ("My diagnosis probabilities diverged from Zhang's across the board.", 0),
    ("The prior was off, and Table 7 cause shares didn't line up.", 0),
    ("My engine overlapped Zhang on only 3 of 10 top causes.", 0),
    ("This pointed to something systematic — not a tuning issue.", 0),
])
add_speaker_note(s, "This wasn't close-but-off. Everything was wrong at once, which signals a "
                    "data problem upstream, not a parameter to nudge.")

# ============================================================ CORE 3: root cause 1
s = add_slide()
add_title_bar(s, "Root cause #1: missing legacy data  (38 vs 102 fires)")
add_bullets(s, [
    ("NTSB changed coding schemes (eADMS, the electronic Aviation Data Management "
     "System) ~2007.", 0),
    ("Pre-2007 accidents store event chains + findings in legacy files "
     "(Occurrences.txt, seq_of_events.txt).", 0),
    ("My processed dataset never merged that legacy layer — 1,742 pre-2007 "
     "incidents had empty sequence_of_events and findings.", 0),
    ("Result: only 38 fires counted instead of 102 (~78% of the window missing causal data).", 0),
], height=Inches(2.6))
add_table(s, [
    ["Symptom", "Before", "After"],
    ["Fire occurrences", "38", "102"],
    ["Incidents w/ event sequences", "~500", "2,009"],
    ["Incidents w/ findings", "~900", "1,938"],
], left=1.4, top=4.7, width=10.5, height=1.9, col_widths=[5.5, 2.5, 2.5],
   font_size=15, highlight_rows={1})
add_speaker_note(s, "This single gap explains why nothing lined up: the fire denominator was "
                    "wrong, so every P(cause|fire) was computed on the wrong population.")

# ============================================================ CORE 4: what changed (knob table)
s = add_slide()
add_title_bar(s, "What changed in our process (and why)")
add_table(s, [
    ["Knob", "Before (wrong)", "After (Zhang-aligned)", "Why it matters"],
    ["Data layer", "findings only;\npre-2007 empty", "legacy occurrences +\nfindings merged",
     "restores 64 fires\n(38 -> 102)"],
    ["Denominator", "retrieval\n'similarity-mass' (~100+)", "count / 102\nfire accidents",
     "un-shrinks every prob.\nto Zhang's scale"],
    ["Vocabulary", "findings + modifiers\n(wiring, burned)", "findings + occurrences,\nclean labels",
     "recovers #1 cause\n(airframe)"],
    ["Index coverage", "340 records (19 fires)\nun-embedded", "all narratives\nembedded",
     "retrieval sees all\n102 fire accidents"],
], left=0.45, top=1.5, width=12.45, height=4.6,
   col_widths=[2.0, 3.3, 3.5, 3.65], font_size=13)
add_bullets(s, [
    ("Net effect: same data, same definitions, same denominator as Zhang -> the numbers line up.", 0),
], top=6.3, size=16)
add_speaker_note(s, "The one slide that makes the 'process change' explicit. Each row is a knob we "
                    "turned from a wrong setting to the setting Zhang actually used; the right "
                    "column says exactly what that buys us. No tuning, just alignment.")

# ============================================================ CORE 5: our prior result
s = add_slide()
add_title_bar(s, "Our result: the prior reproduced EXACTLY",
              sub="Compare directly with Zhang's Eq. (6) on the previous slide")
add_table(s, [
    ["Quantity", "Zhang (paper)", "Ours", "Match"],
    ["Prior P(fire)", "5.53 x 10^-7", "5.527942 x 10^-7", "EXACT"],
    ["Fire occurrences (1982-2006)", "102", "102", "EXACT"],
], left=1.0, top=1.7, width=11.3, height=1.6, col_widths=[4.3, 2.6, 2.7, 1.7],
   font_size=15, highlight_rows={1, 2})
add_bullets(s, [
    ("P(fire) = 102 / 184,517,128 = 5.527942 x 10^-7  — same formula, same denominator as Zhang.", 0),
    ("Denominator = total U.S. air-carrier departures 1982-2006 (BTS), interpolated & summed.", 0),
    ("The '184,572,128' said verbally is a digit transposition of 184,517,128.", 1),
], top=3.9)
add_speaker_note(s, "The sanity check passing to the digit, against Zhang's own equation one slide "
                    "back. That's the gate cleared.")

# ============================================================ CORE 6: our Table 7
s = add_slide()
add_title_bar(s, "Our result: Table 7 reproduced  (P(cause | fire))",
              sub="Put next to Zhang's Table 7 on the previous slide")
add_table(s, [
    ["Cause", "n", "Zhang / Ours"],
    ["Airframe/component/system failure", "32", "0.3137"],
    ["Electrical system, electric wiring", "11", "0.1078"],
    ["Loss of engine power (total) - mech", "9", "0.0882"],
    ["Fluid, fuel", "7", "0.0686"],
    ["Auxiliary power unit (APU)", "6", "0.0588"],
    ["Maintenance, installation", "4", "0.0392"],
    ["Landing gear, tire", "4", "0.0392"],
], left=0.7, top=1.45, width=8.0, height=4.6, col_widths=[5.2, 1.0, 1.8],
   font_size=13, highlight_rows={1})
add_bullets(s, [
    ("Dominant cell exact vs Zhang: Airframe 0.3137 = 0.31372.", 0),
    ("Secondary cells within ~1 accident.", 0),
    ("At full retrieval breadth, 113/113 causes match Zhang exactly.", 0),
    ("top-50 -> 36 fires (0.222); top-200 -> 84 (0.286); all -> 102 (0.3137).", 0),
], top=1.7, left=8.9, width=Inches(4.0), size=15)
add_speaker_note(s, "Put this next to the previous slide — same causes, same probabilities. "
                    "Convergence is clean: tighter retrieval is query-focused; full breadth "
                    "reproduces the whole population and equals Zhang.")

# ============================================================ CORE 7: Zhang Table 4 + Beta-CDF
s = add_slide()
add_title_bar(s, "Zhang's Table 4 vs his actual method  (pedagogical 0.99 != output)",
              sub="Zhang Table 4 (top) and Fig. 8 Beta-CDF estimator (bottom) — actual figures")
add_picture_fit(s, "zhang_table4.png", left=0.4, top=1.35, max_w=12.5, max_h=1.85, center=True)
add_picture_fit(s, "zhang_betacdf.png", left=0.4, top=3.45, max_w=5.6, max_h=3.55)
add_bullets(s, [
    ("Top = Zhang's Table 4: the CPT that motivated this work —", 0),
    ("'if brake worn AND wiring overheated -> P(fire) = 0.99'.", 1),
    ("It is a hand-set TEACHING example for the Fig. 2 toy net, NOT an estimator output.", 1),
    ("Bottom = Zhang's real method (Fig. 8 Beta-CDF, a*b ~ 1.05/2.03).", 0),
    ("On the same inputs it yields ~0.09-0.11, nowhere near 0.99.", 1),
    ("So 'get Table 4 to 0.9' is unreachable by ANY method, including Zhang's own.", 0),
    ("We relate our conditional diagnosis to Table 4's idea — honestly — next slide.", 0),
], top=3.4, left=6.4, width=Inches(6.5), size=14)
add_speaker_note(s, "This directly answers the long-standing Table 4 question. The 0.99 is "
                    "pedagogical — Zhang hand-fills that CPT to illustrate the toy network. His "
                    "genuine estimator (Fig. 8 Beta-CDF) gives ~0.1 on the same inputs. So we don't "
                    "chase 0.99; we reproduce the legitimate quantities (prior, Table 7) and "
                    "reinterpret Table 4 as 'condition on more evidence', shown next.")

# ============================================================ CORE 8: conditional
s = add_slide()
add_title_bar(s, "Beyond Zhang: conditional diagnosis  (the honest Table-4 analogue)")
add_bullets(s, [
    ("P(cause | outcome AND condition) — restrict the population, then diagnose.", 0),
    ("Example: P(cause | fire AND electrical wiring)", 0),
    ("Population restricted to the 14 wiring-related fire incidents.", 1),
    ("Wiring jumps to ~78.6% as the dominant cause.", 1),
    ("This is the honest version of Table 4's idea: instead of hand-setting 0.99, we "
     "sub-set to the matching incidents and report what the data says.", 0),
    ("Lets the advisor ask targeted 'what if we already know X' questions.", 0),
])
add_speaker_note(s, "Table 4 asked 'given brake + wiring, how likely is fire?'. Our conditional "
                    "asks the diagnosis-direction analogue — 'given fire AND wiring, which cause "
                    "dominates?' — by restricting the population (14 incidents), not inventing a "
                    "0.99. Answer: wiring ~79%.")

# ============================================================ CORE 9: Part 2 intro / claim
s = add_slide()
add_title_bar(s, "Part 2: A more robust estimator for sparse cells",
              sub="The new, validated contribution since the last meeting")
add_bullets(s, [
    ("Part 1 was about DENSE numbers — the fire prior and Table 7 — where we have "
     "102 fires to count, so plain counting works.", 0),
    ("This part is the opposite situation: cells with almost no data (1-10 "
     "observations) — which is exactly where Zhang's famous 0.95 lives.", 0),
    ("Claim, stated honestly up front:", 0),
    ("Our semantic-neighbour smoothing is more accurate AND far more stable than "
     "Zhang's smoother exactly where his estimates are most fragile.", 1),
    ("It is also a strict, clean replacement for his Beta-CDF smoother.", 1),
    ("NOT a blanket 'we beat everything' — the honest caveat travels with us all "
     "the way through.", 1),
])
add_speaker_note(s, "Slow down here. Everything before was reproduction — proving I can get "
                    "Zhang's numbers. This is the actual new science: a better way to estimate "
                    "probabilities when there's almost no data, and I validated it. I'll teach the "
                    "concepts first, show the result, then be very honest about where it does and "
                    "doesn't win.")

# ============================================================ CORE 10: results figure (sparse_robustness.png)
s = add_slide()
add_title_bar(s, "Results: accuracy and stability vs sparsity",
              sub="Left = accuracy (MAE) vs n; right = stability (across-draw STD) vs n")
add_picture_fit(s, "sparse_robustness.png", left=0.4, top=1.35, max_w=8.4, max_h=4.9,
                base=FIG_ROOT)
add_bullets(s, [
    ("Accuracy (left): semantic sits flat at MAE ~ 0.11 for every n; counting starts "
     "near 0.27-0.28 at n=1 and only catches it around n ~ 7.", 0),
    ("Stability (right): semantic spread ~ 0.000 (same answer every draw); "
     "counting / Beta-CDF wobble at STD 0.10-0.37.", 0),
    ("Beta-CDF (blue) is the WORST line at every n — the global curve distorts these "
     "cells.", 0),
    ("Significance: vs Beta-CDF, semantic wins at EVERY n (Wilcoxon p < 1e-4, "
     "bootstrap CIs exclude 0); vs raw count, it wins big for n <= 5.", 0),
], top=1.5, left=8.9, width=Inches(4.1), size=13)
add_speaker_note(s, "Walk through the picture. Left is accuracy, lower is better — the green "
                    "semantic line is flat and low, around 0.11, no matter how little data there "
                    "is. The grey and red counting lines start near 0.28 at one observation and "
                    "come down, crossing green around seven. Right is stability — how much the "
                    "answer jumps between random draws; semantic is pinned to zero, the others "
                    "bounce. The blue Beta-CDF line is worst on accuracy everywhere. And it's all "
                    "statistically significant — p below one in ten thousand against Beta-CDF.")

# ============================================================ CORE 11: honest answer
s = add_slide()
add_title_bar(s, "So... is this better than Zhang?  (the honest answer)")
add_bullets(s, [
    ("Yes — exactly where it matters most. In the ultra-sparse regime (n <= 5, where "
     "Zhang's fragile 0.95-type cells live), semantic is significantly more accurate "
     "and far more stable than raw count, the 0.95-cap, and Beta-CDF.", 0),
    ("It is a strict, clean replacement for Zhang's Beta-CDF smoother — better on "
     "every metric, at every n, AND under the strict leave-one-out check.", 0),
    ("Honest caveat (must say this): semantic does NOT beat a plain raw count at "
     "~10+ observations, and in strict held-out (LOO) prediction it LOSES to raw count:", 0),
    ("LOO Brier (lower better): raw/cap 0.148 < semantic 0.167 < Beta-CDF 0.171.", 1),
    ("Scientifically honest framing -> a HYBRID: semantic smoothing for sparse cells, "
     "plain counting for dense cells. The natural next step / future work.", 0),
])
add_speaker_note(s, "This slide arms me for the inevitable question — is it actually better than "
                    "Zhang? Honest answer: yes, where it matters most. In the sparse regime, n of "
                    "five or fewer, which is precisely where Zhang's 0.95 lives, it's significantly "
                    "more accurate and dramatically more stable. And it's a clean win over his "
                    "actual Beta-CDF smoother across the board. But I won't oversell it: once you "
                    "have ten or more observations a plain raw count is better, and in the strictest "
                    "leave-one-out test semantic actually loses to raw counting because its "
                    "neighbour pool carries a small bias. So the honest takeaway isn't 'throw out "
                    "counting,' it's 'use the right tool for the regime' — semantic when starved "
                    "for data, counting when rich. That hybrid is the clean story for the paper.")

# ============================================================ CORE 12: Part 3 intro
s = add_slide()
add_title_bar(s, "Part 3: Does the narrative method actually work?",
              sub="The open item ('validate query-conditioning') is now closed")
add_bullets(s, [
    ("Part 1 = reproduction (counting -> the population numbers). Part 2 = a better "
     "estimator for sparse cells.", 0),
    ("Part 3 = the narrative engine's OWN unique value: per-incident query-conditioning.", 0),
    ("The whole question: does using THIS incident's story sharpen the cause estimate "
     "beyond just using overall frequencies?", 0),
    ("Four things we now validated, in order:", 0),
    ("(1) conditioning works (the headline)  (2) the probabilities are calibrated  "
     "(3) a safety gate  (4) robustness to rough real queries.", 1),
    ("Honest framing up front: the win is in the LIFT over the baseline, NOT a high "
     "absolute accuracy. Absolute top-1 stays a modest ~47%.", 0),
])
add_speaker_note(s, "This part closes the open item from last time. Beat one was reproduction, beat "
                    "two was the sparse-cell estimator. This third part is the one thing only the "
                    "narrative method can do: take a single incident's write-up and use it to guess "
                    "that incident's cause better than just going with the most common cause "
                    "overall. I validated four things — conditioning helps, the probabilities are "
                    "honest, I added a safety switch, and it survives messy queries. The one honest "
                    "thing I repeat throughout: I'm claiming a reliable improvement over the "
                    "baseline, not high accuracy. Lead with lift, not the 47 percent.")

# ============================================================ CORE 13: results table (cond vs prior vs random)
s = add_slide()
add_title_bar(s, "Results: conditioning vs the prior vs a random pool",
              sub="cond / unc / rand — specific-cause space; lifts are paired means")
add_table(s, [
    ["Stratum (specific)", "n", "top-1 (c/u/r)", "MRR (c/u/r)", "MRR lift\nvs prior", "MRR lift\nvs random"],
    ["A-clean (leakage-free)", "1085", "47.4/42.2/38.9", "0.577/0.525/0.473", "+0.052\n(p~1e-11)", "+0.103\n(p~1e-29)"],
    ["A-all (factual)", "1254", "46.9/40.9/38.0", "0.570/0.511/0.463", "+0.059", "+0.108"],
    ["A-leak (echoes cause)", "169", "43.8/32.5/32.5", "0.529/0.423/0.394", "+0.106", "+0.135"],
    ["B-cause-prose (leaky)", "746", "45.6/40.6/35.1", "0.568/0.518/0.442", "+0.050", "+0.126"],
], left=0.45, top=1.5, width=12.45, height=2.6,
   col_widths=[2.7, 0.8, 2.65, 2.9, 1.7, 1.7], font_size=12, highlight_rows={1})
add_bullets(s, [
    ("Headline (honest A-clean): conditioning beats the prior +5.2 pp top-1 / +0.052 "
     "MRR (p~1e-11), and beats a concentration-matched random pool +8.5 pp / +0.103 "
     "(p~1e-29).", 0),
    ("Every comparison is paired per incident with overwhelming significance — not a "
     "small-sample fluke.", 0),
], top=4.4, size=15)
add_speaker_note(s, "Read the top row, A-clean — the honest, leakage-free number, the one to quote. "
                    "Conditioning gets top-1 of 47.4 percent versus 42.2 for the prior, about five "
                    "points better, and the MRR goes up by about 0.052, p around 1e-11. Against the "
                    "random pool the gap is even bigger, 8.5 points and 0.103, 1e-29. Beating the "
                    "prior says the narrative helps; beating the random pool says it's not a "
                    "small-pool artefact. Both hold decisively.")

# ============================================================ CORE 14: leakage check
s = add_slide()
add_title_bar(s, "Is it real, or is it leakage?  (the decisive check)")
add_bullets(s, [
    ("The proof it's NOT leakage: the leaky ceiling B (+0.050 MRR) is NO bigger than "
     "the honest A-clean (+0.052 MRR).", 0),
    ("If A were secretly leakage, B (which DOES contain the cause text) would dominate "
     "it. It doesn't -> A is clean.", 1),
    ("A-leak confirms the detector works: factual narratives that DO echo the cause "
     "show the expected bigger lift (+0.106).", 0),
    ("Conditioning helps where it should, and only there: the win is on specific-"
     "mechanism causes.", 0),
    ("On incidents whose only true cause is a generic catch-all (n=29), the frequency "
     "prior is already optimal, so conditioning LOSES (top-1 -20.7 pp) — the predicted "
     "pattern, not a bug.", 1),
])
add_speaker_note(s, "This slide kills the 'isn't this just leakage?' objection. Stratum B is the "
                    "cause-prose — text that literally describes the cause. If my method were "
                    "winning by cheating off that, B would show a much bigger lift than clean A. It "
                    "doesn't — B is +0.050, clean A is +0.052, basically the same. So A isn't "
                    "riding on leakage. As a sanity check, the A-leak slice — factual reports that "
                    "echo the cause — does show a bigger lift, +0.106, proving the detector "
                    "detects leakage. And conditioning only helps when there's a specific mechanism "
                    "to find; on the handful of generic-only cases the overall frequency is already "
                    "best, so conditioning slightly hurts there — exactly as expected, and it "
                    "motivates the safety gate.")

# ============================================================ CORE 15: status & next steps
s = add_slide()
add_title_bar(s, "Status & next steps")
add_bullets(s, [
    ("Diagnosis reproduction: done / solid.", 0),
    ("Prior + Table 7 reproduced exactly; method generalized; conditional + "
     "confidence-aware modes added. The credibility result.", 1),
    ("TWO validated contributions:", 0),
    ("Query-conditioning (Part 3): a narrative-specific win — beats the no-narrative "
     "baseline +5.2 pp top-1 / +0.052 MRR (p~1e-11), leakage-controlled, not a "
     "concentration artefact.", 1),
    ("Sparse-cell robustness (Part 2): semantic smoothing fixes Zhang's fragility — "
     "strictly beats his Beta-CDF, most accurate/stable where he is weakest (n <= 5).", 1),
    ("The conditioning method is now hardened: calibrated (ECE ~0.059 -> 0.038 via a "
     "one-dial temperature fix), gated (a safety switch — never worse than the prior), "
     "and robust to rough real-world queries.", 0),
    ("Honest open caveats (each travels with its claim): absolute accuracy modest "
     "(~47%); gating is safety not accuracy (harm regime ~2.3%); ultra-terse one-liners "
     "lose most of the ranking lift; sparse win is not blanket (raw count wins at "
     "n >~ 10) -> a hybrid is next.", 0),
])
add_speaker_note(s, "Where things stand now. The reproduction is rock-solid and buys credibility. I "
                    "have two genuinely validated contributions: query-conditioning, the narrative "
                    "method's own win — it beats the no-narrative baseline with overwhelming "
                    "significance and I've shown it's not leakage and not a small-pool artefact — "
                    "and the sparse-cell robustness work that fixes Zhang's fragile smoother. And "
                    "the conditioning method isn't just validated, it's hardened: the probabilities "
                    "are calibrated and a one-parameter fix makes them honest to a few points, "
                    "there's a safety gate so it's never worse than the prior, and it survives "
                    "messy real queries. Every caveat stays attached: modest absolute accuracy, the "
                    "gate is safety not score, very short queries lose most of the ranking benefit, "
                    "and the sparse-cell method isn't a blanket win — all pointing to the obvious "
                    "next step, a hybrid that uses each tool where it's best.")

prs.save(str(OUT))
print(f"Saved {OUT} with {len(prs.slides._sldIdLst)} slides")
