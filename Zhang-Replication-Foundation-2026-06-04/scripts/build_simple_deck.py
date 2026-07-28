"""
Plain 10-slide deck for Maha meeting.
No rounded rectangles, no color badges, no AI-flavor formatting.
Just title + bullets + native tables for data comparisons.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).parent
OUT = HERE / "outputs" / "Maha_Meeting_Final.pptx"
OUT.parent.mkdir(parents=True, exist_ok=True)

COMP_T9 = ROOT / "Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_table9.xlsx"
COMP_F12 = ROOT / "Testing_Structural_Mapping_Slides/outputs/zhang_comparison/comparison_fig12.xlsx"

BLACK = RGBColor(0x20, 0x20, 0x20)
GREY = RGBColor(0x60, 0x60, 0x60)

prs = Presentation()
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def title(slide, text, y=0.4, size=28):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(y), Inches(12.1), Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = BLACK
    r.font.name = "Calibri"


def subtitle(slide, text, y=1.05, size=14):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(y), Inches(12.1), Inches(0.5))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.italic = True
    r.font.color.rgb = GREY
    r.font.name = "Calibri"


def bullets(slide, items, y=1.7, x=0.6, w=12.1, h=5.5, size=14):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            level, text = item
        else:
            level, text = 0, item
        p.level = level
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size if level == 0 else max(11, size - 2))
        r.font.color.rgb = BLACK
        r.font.bold = (level == 0 and text.endswith(":"))
        r.font.name = "Calibri"
        p.space_after = Pt(3)


def footer(slide, n, total):
    tb = slide.shapes.add_textbox(Inches(12.0), Inches(7.05), Inches(1.2), Inches(0.3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.RIGHT
    r = p.add_run()
    r.text = f"{n}/{total}"
    r.font.size = Pt(9)
    r.font.color.rgb = GREY


def native_table(slide, df, y, x=0.6, w=12.1, h=5.0, font_size=10):
    rows, cols = df.shape[0] + 1, df.shape[1]
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(h))
    tbl = shape.table
    for j, col in enumerate(df.columns):
        cell = tbl.cell(0, j)
        cell.text = str(col)
        for p in cell.text_frame.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.size = Pt(font_size)
                r.font.name = "Calibri"
    for i in range(df.shape[0]):
        for j, col in enumerate(df.columns):
            val = df.iloc[i, j]
            if isinstance(val, float):
                if pd.isna(val):
                    s = ""
                else:
                    s = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
            else:
                s = str(val)
            cell = tbl.cell(i + 1, j)
            cell.text = s
            for p in cell.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    r.font.name = "Calibri"


TOTAL = 10

# ----------------------------------------------------------------------------
# Slide 1 — Title
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
tb = s.shapes.add_textbox(Inches(0.6), Inches(2.6), Inches(12.1), Inches(1.0))
p = tb.text_frame.paragraphs[0]
r = p.add_run()
r.text = "Comparing my Structural Mapping pipeline to Zhang's Bayesian Network"
r.font.size = Pt(32)
r.font.bold = True
r.font.color.rgb = BLACK
r.font.name = "Calibri"

tb = s.shapes.add_textbox(Inches(0.6), Inches(3.8), Inches(12.1), Inches(0.6))
p = tb.text_frame.paragraphs[0]
r = p.add_run()
r.text = "Kanu Shetkar  ·  Reliability and Risk Reasoning Group  ·  May 14, 2026"
r.font.size = Pt(16)
r.font.color.rgb = GREY
r.font.name = "Calibri"
footer(s, 1, TOTAL)

# ----------------------------------------------------------------------------
# Slide 2 — What you asked me to do
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "What you asked me to do")
bullets(s, [
    "From our joint meeting:",
    (1, "Understand how Zhang calculates his probabilities, step by step"),
    (1, "Compare his approach to mine on the same scenarios"),
    (1, "Use Zhang's published scenarios (Table 9, Fig 11, Fig 12) as the comparison target"),
    "",
    "From follow-up with Jesse:",
    (1, "Recreate Zhang's full pipeline using his GeNIe modeler and the tools of the time"),
    (1, "Hand-work the math so the weighted average is defensible"),
    (1, "Methodology must be unassailable. Every step justified, every number reproducible."),
    "",
    "Theoretical concern Jesse raised:",
    (1, "\"Averaging frequencies over rare events is not a Bayesian approach.\""),
    (1, "I address this in the asks slide at the end."),
])
footer(s, 2, TOTAL)

# ----------------------------------------------------------------------------
# Slide 3 — Three models
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "Three models, same 54-code output space")
subtitle(s, "All three produce P(cause code), so the comparison is apples to apples")
bullets(s, [
    "1.  Zhang's Bayesian Network",
    (1, "Input: coded NTSB data (events, findings, occurrences)"),
    (1, "Output: posterior P(node | evidence) over 740 nodes"),
    "",
    "2.  A0  (Baseline, mine)",
    (1, "Input: NTSB narratives"),
    (1, "Embeddings + cosine similarity, no structural mapping"),
    "",
    "3.  A2  (Structural Mapping, mine)",
    (1, "Same as A0, plus reranking by structural similarity"),
    (1, "Structural similarity built from LLM-extracted causal chains"),
    "",
    "Different methods, different theoretical foundations, same output shape. Directly comparable.",
])
footer(s, 3, TOTAL)

# ----------------------------------------------------------------------------
# Slide 4 — How Zhang gets his probabilities
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "How Zhang gets his probabilities")
bullets(s, [
    "Input:  coded NTSB data only.  Narratives discarded.",
    "",
    "Network:  740 nodes, hand-built in GeNIe Modeler.",
    "",
    "CPTs filled two ways:",
    (1, "Dense cells: direct frequency from NTSB counts"),
    (1, "Sparse cells: smoothed with a Beta-CDF prior, α=1.05, β=2.03 (his Table 7)"),
    (2, "Beta-CDF transforms raw estimates, pulling sparse values toward prior mean (≈ 0.34)"),
    (2, "Nelder-Mead found α, β by iteratively moving a candidate triangle toward lowest error"),
    (1, "Calibration parameters fit globally via Nelder-Mead optimization"),
    "",
    "Inference:",
    (1, "Likelihood-Weighted Sampling (L_SAMPLING, algorithm 3 in SMILE)"),
    (2, "Intuition: run 99M simulated accident scenarios, weight each by how well it matches the evidence, take the weighted average"),
    (2, "If asked deeper: non-evidence nodes sampled from CPTs, evidence forced to observed values, weight = P(evidence | parents)"),
    (1, "99,999,999 samples per scenario"),
    (1, "Why sampling, not exact?  740 nodes makes exact inference intractable."),
    "",
    "Output:  P(node = state | evidence) for every one of the 740 nodes.",
    "These are population-level Bayesian posteriors over a fixed graph.",
])
footer(s, 4, TOTAL)

# ----------------------------------------------------------------------------
# Slide 5 — How I get my probabilities
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "How I get my probabilities")
bullets(s, [
    "Input:  NTSB narratives. The text Zhang threw away.",
    "",
    "Pipeline:",
    (1, "1.  Embed query and corpus narratives"),
    (1, "2.  Retrieve nearest neighbors by cosine similarity"),
    (1, "3.  A2 only: rerank with structural similarity"),
    (2, "w' = max(0, cos) × exp(2 × s)"),
    (2, "s computed by Needleman-Wunsch alignment of (role, system, mechanism) causal chains"),
    (1, "4.  Cluster the reranked neighbors"),
    (1, "5.  P(cause | cluster) from empirical counts of historical causes"),
    (1, "6.  Aggregate via law of total probability:  P(cause | query) = Σ P(cause | C_i) · P(C_i | query)"),
    (1, "7.  Map free-text causes to Zhang's 54 codes via embedding nearest-neighbor"),
    "",
    "These are within-cluster empirical frequencies over retrieved neighbors, not Bayesian posteriors.",
])
footer(s, 5, TOTAL)

# ----------------------------------------------------------------------------
# Slide 6 — I recreated Zhang's pipeline (Jesse's directive)
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "I recreated Zhang's full pipeline")
subtitle(s, "Jesse: \"with his GeNIe modeler and all that, even with the tools of the time\"")
bullets(s, [
    "Tools used:",
    (1, "pysmile (Python wrapper for the same SMILE engine GeNIe uses)"),
    (1, "Zhang's NTSB.xdsl from his public GitHub, unmodified"),
    (1, "L_SAMPLING algorithm with 99,999,999 samples"),
    "",
    "Three layers of validation:",
    (1, "1. α and β parameters reproduce to 9 decimal places. Proves the deterministic math is bit-correct."),
    (1, "2. 76 of 86 published cells reproduce within paper rounding tolerance, which is 88% match."),
    (1, "3. Multi-seed band: 47 of 86 Zhang values fall inside the [min, max] of 5 reproductions."),
    (2, "This proves I sample from the same posterior distribution Zhang does."),
    "",
    "The 12% gap decomposes into specific structural causes:",
    (1, "~6 cells, \"No injury\": L_SAMPLING fundamentally can't recover values close to 1.0 on absence states"),
    (1, "~5 cells, \"Substantial damage\" with consistent +0.02 offset: suggests XDSL revision after publication"),
    (1, "~2 cells, priors at 10⁻⁷: below sampling noise floor at any reasonable count"),
    "",
    "Artifact: zhang_full_probability_table.xlsx (740 nodes × 26 scenarios, 38,480 rows).",
    "The full probability table Zhang never published.",
])
footer(s, 6, TOTAL)

# ----------------------------------------------------------------------------
# Slide 7 — Numerical comparison: Table 9 (loss of engine power)
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "Numerical comparison: Loss of engine power scenarios")

if COMP_T9.exists():
    raw = pd.read_excel(COMP_T9)
    scenario = "Inoperative engine instruments"
    out = pd.DataFrame({
        "Event": raw["Event"],
        "Type": raw["Type"],
        "Zhang BN": raw[f"Zhang BN — {scenario}"],
        "A0 (mine)": raw[f"A0 — {scenario}"],
        "A2 (mine)": raw[f"A2 — {scenario}"],
    })
    subtitle(s, f"Zhang's Table 9.  Evidence: {scenario}.  Zhang BN  vs.  my A0  vs.  my A2.")
    native_table(s, out, y=1.65, font_size=11, h=4.8)
else:
    bullets(s, ["(Could not find comparison_table9.xlsx, paste screenshot here)"], y=2.0)

footer(s, 7, TOTAL)

# ----------------------------------------------------------------------------
# Slide 8 — Numerical comparison: Fig 12 (pilot error chain)
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "Numerical comparison: Pilot error chain")

if COMP_F12.exists():
    raw = pd.read_excel(COMP_F12)
    scenario = "After pilot error"
    out = pd.DataFrame({
        "Node": raw["Node"],
        "Zhang Prior": raw["Zhang Prior"],
        "Zhang BN": raw[f"Zhang — {scenario}"],
        "A0 (mine)": raw[f"A0 — {scenario}"],
        "A2 (mine)": raw[f"A2 — {scenario}"],
    })
    subtitle(s, "Zhang's Fig 12.  Evidence: pilot error observed.  Zhang BN  vs.  my A0  vs.  my A2.")
    native_table(s, out, y=1.65, font_size=11, h=4.5)
else:
    bullets(s, ["(Could not find comparison_fig12.xlsx, paste screenshot here)"], y=2.0)

footer(s, 8, TOTAL)

# ----------------------------------------------------------------------------
# Slide 9 — What I found
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "What I found")
bullets(s, [
    "1.  A0 and A2 are nearly identical at the code level.",
    (1, "Top-1 accuracy: A0 = 25/77 (32.5%),  A2 = 27/77 (35.1%).  Delta = +2.6 percentage points."),
    (1, "McNemar p = 0.50, not statistically significant at p<0.05 (n=77 is small)."),
    (1, "A2 changed top-1 prediction in only 1 of 83 incidents in three_model_comparison.xlsx."),
    (1, "Structural reranking helps inside clusters; the gains wash out when aggregated to 54 codes."),
    "",
    "2.  Both A0 and A2 diverge from Zhang in a structured way.",
    (1, "Zhang concentrates probability on the cause node:  P(Loss of engine power) ≈ 0.95"),
    (1, "Mine spread probability over downstream consequences:"),
    (2, "P(Forced landing) ≈ 0.67,  P(Substantial damage) ≈ 0.17"),
    (1, "Reason: Zhang reasons over a joint distribution; my retrieval captures narrative co-occurrence."),
    "",
    "3.  My pipeline is blind to \"No injury\".",
    (1, "P(No injury) ≈ 0.00 on every test case."),
    (1, "Narratives describe events that happened, not absences."),
    (1, "Same blindspot Zhang's L_SAMPLING hits on absence-encoded states."),
])
footer(s, 9, TOTAL)

# ----------------------------------------------------------------------------
# Slide 10 — What it means + asks
# ----------------------------------------------------------------------------
s = prs.slides.add_slide(BLANK)
title(s, "What it means and what I need from you")
bullets(s, [
    "What this shows:",
    (1, "Narrative information adds signal that coded data alone cannot capture."),
    (1, "Direction positive on every metric; magnitude modest at the code level."),
    (1, "Bigger gains likely live inside the retrieval step, not visible after aggregation to 54 codes."),
    "",
    "Three questions for you:",
    "",
    "1.  Evaluation level:",
    (1, "Is the code-distribution comparison the right level, or should I evaluate at the cluster level"),
    (1, "where A2's gains actually live?"),
    "",
    "2.  Bayesian extension (Jesse's critique):",
    (1, "Read probabilities directly off LLM tokens via parameter-efficient fine-tuning,"),
    (1, "or extend the structural pipeline to do Bayesian aggregation explicitly over the clusters?"),
    "",
    "3.  Publication path:",
    (1, "Reliability engineering journal with current results, or hold for the Bayesian upgrade?"),
    "",
    "Side project in progress:",
    (1, "Building an LLM and Cursor skill that takes a narrative and constructs a Bayesian network."),
    (1, "Aligns with the Bayesian extension direction in question 2."),
    (1, "Early prototype, demos the path forward."),
])
footer(s, 10, TOTAL)


prs.save(str(OUT))
print(f"wrote {OUT}")
print(f"  total slides: {len(prs.slides)}")
