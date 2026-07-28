"""Rebuild Section 9 tables + figure captions from the SAME summary.json that
powers the graphs, so the text and figures are guaranteed consistent."""
import json
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

S = json.load(open("Worked_Examples/section9_graphs/summary.json"))
OUT = "/Users/kanushetkar/Desktop/NTSB Paper Process/Section 9 Tables and Captions.docx"

n = S["n"]
a0, a2 = S["A0"], S["A2"]
mc = S["mcnemar"]["table"]
p_val = S["mcnemar"]["p_value_two_sided"]


def pct(x):
    return f"{x*100:.1f}%"


def ci_pct(lo, hi):
    return f"[{lo*100:.1f}, {hi*100:.1f}]"


doc = Document()
doc.styles["Normal"].font.name = "Times New Roman"
doc.styles["Normal"].font.size = Pt(11)


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, head in enumerate(headers):
        t.rows[0].cells[i].text = head
        for par in t.rows[0].cells[i].paragraphs:
            for run in par.runs:
                run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    doc.add_paragraph()


def cap(label, text):
    par = doc.add_paragraph()
    r = par.add_run(label + " ")
    r.bold = True
    r.italic = True
    rr = par.add_run(text)
    rr.italic = True


doc.add_heading("Section 9 - Aggregate Validation (tables + figure captions)", 1)
doc.add_paragraph(
    f"All numbers below are computed on the same N = {n} held-out test incidents and from the "
    "same evaluation run that produced the figures, so the tables and figures are consistent.")

# Metric definition
doc.add_heading("9.2 Metric definition", 2)
doc.add_paragraph(
    "For each test incident the system predicts the most likely primary cause (the NTSB "
    "\"C\" / probable-cause finding). We compare the predicted cause text to the official NTSB "
    "probable-cause text using cosine similarity of their embeddings. We report:")
doc.add_paragraph(
    "  - Top-1 accuracy: the prediction is counted a hit when its similarity clears the match "
    "threshold (match% >= 75, i.e. cosine >= 0.75).", style="List Bullet")
doc.add_paragraph(
    "  - Recall@5: a hit appears anywhere in the top 5 predicted causes.", style="List Bullet")
doc.add_paragraph(
    "  - MRR: mean reciprocal rank of the first hit (rewards ranking the right cause higher).",
    style="List Bullet")
doc.add_paragraph(
    "  - Avg match%: mean similarity (cosine x 100) of the top prediction across all cases, a "
    "threshold-free view.", style="List Bullet")

# 9.3 results table
doc.add_heading("9.3 Aggregate results table", 2)
table(
    ["Metric", "A0 (embedding)", "A2 (structural)", "Delta (A2 - A0)", "A2 95% CI"],
    [
        ["Top-1 accuracy", f"{pct(a0['top1'])} ({a0['n_hits']}/{n})",
         f"{pct(a2['top1'])} ({a2['n_hits']}/{n})",
         f"+{(a2['top1']-a0['top1'])*100:.1f} pp", ci_pct(*a2["top1_ci95"])],
        ["Recall@5", f"{pct(a0['recall5'])} ({a0['n_r5_hits']}/{n})",
         f"{pct(a2['recall5'])} ({a2['n_r5_hits']}/{n})",
         f"+{(a2['recall5']-a0['recall5'])*100:.1f} pp", ci_pct(*a2["recall5_ci95"])],
        ["MRR", f"{a0['mrr']:.3f}", f"{a2['mrr']:.3f}",
         f"+{a2['mrr']-a0['mrr']:.3f}", f"[{a2['mrr_ci95'][0]:.3f}, {a2['mrr_ci95'][1]:.3f}]"],
        ["Avg match %", f"{a0['avg_match_pct']:.1f}%", f"{a2['avg_match_pct']:.1f}%",
         f"+{a2['avg_match_pct']-a0['avg_match_pct']:.2f} pp",
         f"[{a2['avg_match_pct_ci95'][0]:.1f}, {a2['avg_match_pct_ci95'][1]:.1f}]"],
    ],
)
doc.add_paragraph(
    "Reading the table: A2 (structural reranking) is marginally better than A0 on every metric, "
    "but the gains are small and the confidence intervals overlap. The honest summary is that "
    "structural reranking does not hurt and gives a small, consistent nudge in the right "
    "direction.")

# 9.4 McNemar
doc.add_heading("9.4 A2 vs A0 - paired (McNemar) test", 2)
table(
    ["", "A2 = pass", "A2 = fail"],
    [
        ["A0 = pass", mc["both_pass"], mc["A0_only_pass"]],
        ["A0 = fail", mc["A2_only_pass"], mc["both_fail"]],
    ],
)
doc.add_paragraph(
    f"On {mc['both_pass']} incidents both pipelines hit; on {mc['both_fail']} both miss. They "
    f"disagree on only {mc['A0_only_pass'] + mc['A2_only_pass']} case "
    f"(A2 wins {mc['A2_only_pass']}, A0 wins {mc['A0_only_pass']}). McNemar exact two-sided "
    f"p = {p_val:.3f}: the difference is not statistically significant at this sample size, "
    "consistent with the small, non-harmful effect seen in the table above.")

# Figure captions
doc.add_heading("Figure captions", 2)
cap("Figure 9.1 (bar_with_ci.png).",
    f"Top-1 accuracy, Recall@5, and average match% for A0 (embedding only) versus A2 "
    f"(structural reranking) on the N={n} held-out test set, with 95% confidence intervals. "
    "A2 is marginally higher on each metric; intervals overlap.")
cap("Figure 9.2 (mcnemar_table.png).",
    f"Paired flip table of per-incident hit/miss outcomes. Both pipelines agree on "
    f"{mc['both_pass'] + mc['both_fail']} of {n} cases; they disagree on "
    f"{mc['A0_only_pass'] + mc['A2_only_pass']} (McNemar exact p = {p_val:.3f}).")
cap("Figure 9.3 (match_pct_distribution.png).",
    "Distribution of top-1 match% (cosine x 100) for A0 and A2, shown as a histogram and box "
    "plot with the 0.75 hit threshold marked. The two distributions nearly overlap, confirming "
    "structural reranking shifts the whole curve only slightly.")
cap("Figure 9.4 (rank_distribution.png).",
    "Rank position of the first 'good-enough' cause (rank 1 through 5, or no hit in top 5) for "
    "A0 and A2. Most hits occur at rank 1; A2 recovers a few additional cases into the top 5.")
cap("Figure 9.5 (paired_delta.png).",
    "Per-incident difference in match% (A2 minus A0), sorted. Most incidents are unchanged; a "
    "small majority of the non-zero cases favor A2, with a near-zero mean delta.")

doc.save(OUT)
print("Saved:", OUT)
