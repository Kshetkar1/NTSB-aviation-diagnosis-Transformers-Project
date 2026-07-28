"""Build ONE complete, self-consistent Section 9 with figures embedded inline.
Metric = cause-level (Top-1@0.75 / Recall@5 / MRR / match%), matching the graphs."""
import json
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

S = json.load(open("Worked_Examples/section9_graphs/summary.json"))
GDIR = "Worked_Examples/section9_graphs"
OUT = "/Users/kanushetkar/Desktop/NTSB Paper Process/Section 9 FULL (paste-ready).docx"

n = S["n"]
a0, a2 = S["A0"], S["A2"]
mc = S["mcnemar"]["table"]
p_val = S["mcnemar"]["p_value_two_sided"]

doc = Document()
doc.styles["Normal"].font.name = "Times New Roman"
doc.styles["Normal"].font.size = Pt(11)


def p(text=""):
    return doc.add_paragraph(text)


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


def figure(path, caption_label, caption_text, width=5.5):
    par = doc.add_paragraph()
    par.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = par.add_run()
    run.add_picture(f"{GDIR}/{path}", width=Inches(width))
    cpar = doc.add_paragraph()
    cpar.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cpar.add_run(caption_label + " ")
    r.bold = True
    r.italic = True
    r.font.size = Pt(10)
    rr = cpar.add_run(caption_text)
    rr.italic = True
    rr.font.size = Pt(10)
    doc.add_paragraph()


def pc(x):
    return f"{x*100:.1f}%"


# ===================== SECTION 9 =====================
doc.add_heading("9. Aggregate Validation", 1)
p(f"The worked examples in section 8 cover only two incidents. This section tests the pipeline "
  f"on all {n} held-out incidents at once, so the results are not cherry-picked. We run both the "
  "embedding baseline (A0) and the structural-mapping pipeline (A2) on every test incident, "
  "compare the top predicted cause against the official NTSB probable cause, and report how "
  "close the predictions land and how A2 compares to A0.")
p("Section 8 reported cluster-level localization - the headline output that tells the user "
  "where the problem lies. Here we validate the harder task: how close the single top predicted "
  "cause is to the official NTSB probable cause. Cause-level accuracy is naturally lower than "
  "cluster localization, because it targets one specific code rather than a broad failure area.")

doc.add_heading("9.1 Setup", 2)
p("We use a train/test split of the FAR-121 data. From the subset of incidents that have both a "
  "complete narrative and a probable-cause finding, we use 254 incidents: 177 to build the "
  "retrieval index and 77 held out for testing. For each test incident we use its narrative as "
  "the query and run the full pipeline under both A0 (embedding only) and A2 (structural "
  "mapping), so the two pipelines are compared on exactly the same cases.")

doc.add_heading("9.2 Metric definition", 2)
p("The ground truth for each incident is its NTSB probable cause. For each prediction we take "
  "the top predicted cause, embed it, and measure cosine similarity to the NTSB probable-cause "
  "text; match% is that similarity times 100. We report four metrics so the reader is not "
  "locked into a single cutoff:")
doc.add_paragraph("Top-1 accuracy - the top prediction clears the match threshold (match% >= 75, "
                  "i.e. cosine >= 0.75).", style="List Bullet")
doc.add_paragraph("Recall@5 - a matching cause appears anywhere in the top 5 predictions.",
                  style="List Bullet")
doc.add_paragraph("Avg match% - mean similarity of the top prediction across all cases "
                  "(a threshold-free view).", style="List Bullet")

doc.add_heading("9.3 Aggregate results", 2)
table(
    ["Metric", "A0 (embedding)", "A2 (structural)", "Delta (A2 - A0)", "A2 95% CI"],
    [
        ["Top-1 accuracy", f"{pc(a0['top1'])} ({a0['n_hits']}/{n})",
         f"{pc(a2['top1'])} ({a2['n_hits']}/{n})",
         f"+{(a2['top1']-a0['top1'])*100:.1f} pp",
         f"[{a2['top1_ci95'][0]*100:.1f}, {a2['top1_ci95'][1]*100:.1f}]"],
        ["Recall@5", f"{pc(a0['recall5'])} ({a0['n_r5_hits']}/{n})",
         f"{pc(a2['recall5'])} ({a2['n_r5_hits']}/{n})",
         f"+{(a2['recall5']-a0['recall5'])*100:.1f} pp",
         f"[{a2['recall5_ci95'][0]*100:.1f}, {a2['recall5_ci95'][1]*100:.1f}]"],
        ["Avg match %", f"{a0['avg_match_pct']:.1f}%", f"{a2['avg_match_pct']:.1f}%",
         f"+{a2['avg_match_pct']-a0['avg_match_pct']:.2f} pp",
         f"[{a2['avg_match_pct_ci95'][0]:.1f}, {a2['avg_match_pct_ci95'][1]:.1f}]"],
    ],
)
figure("bar_with_ci.png", "Figure 9.1.",
       f"Top-1 accuracy, Recall@5, and average match% for A0 versus A2 on the N={n} held-out "
       "test set, with 95% confidence intervals. A2 is marginally higher on each metric; the "
       "intervals overlap.")
p("A2 (structural reranking) is marginally better than A0 on every metric, but the gains are "
  "small and the confidence intervals overlap. The honest reading is that structural reranking "
  "does not hurt and gives a small, consistent nudge in the right direction.")

doc.add_heading("9.4 A2 vs A0 (paired comparison)", 2)
p("Because both pipelines run on the same incidents, we compare them case by case with a "
  "McNemar paired test on the top-1 hit outcome.")
table(
    ["", "A2 = pass", "A2 = fail"],
    [["A0 = pass", mc["both_pass"], mc["A0_only_pass"]],
     ["A0 = fail", mc["A2_only_pass"], mc["both_fail"]]],
)
figure("mcnemar_table.png", "Figure 9.2.",
       f"Paired flip table of per-incident hit/miss outcomes. The pipelines agree on "
       f"{mc['both_pass']+mc['both_fail']} of {n} cases and disagree on "
       f"{mc['A0_only_pass']+mc['A2_only_pass']} (McNemar exact p = {p_val:.3f}).")
p(f"Both pipelines hit on {mc['both_pass']} incidents and miss on {mc['both_fail']}; they "
  f"disagree on only {mc['A0_only_pass']+mc['A2_only_pass']} case "
  f"(A2 wins {mc['A2_only_pass']}, A0 wins {mc['A0_only_pass']}). The McNemar exact two-sided "
  f"p-value is {p_val:.3f}, so the difference is not statistically significant at this sample "
  "size - consistent with the small, non-harmful effect in the table above.")

doc.add_heading("9.5 Limitations of the validation", 2)
p("This validation has several limitations. The hit threshold (cosine 0.75) is a modeling "
  "choice rather than a standard, so we report Recall@5 and mean match% alongside it to keep the "
  "results from hinging on a single cutoff. The same embedding model is used both to retrieve "
  "incidents and to score the prediction against the NTSB text, which makes the metric somewhat "
  "self-referential. The test set is small (77 incidents), which widens the confidence intervals "
  "and is the main reason a real A2 effect, if one exists, may be too small to reach statistical "
  "significance here. Because NTSB probable-cause text is short, even a correct prediction tends "
  "to cap out at a moderate cosine rather than near 1.0. Finally, this validation covers "
  "diagnosis only: prognosis is not aggregate-validated, because too few retrieved incidents "
  "carry a recorded next event to support a stable held-out metric, so prognosis is demonstrated "
  "qualitatively in the worked examples instead.")

# ---- Appendix figures (optional) ----
doc.add_page_break()
doc.add_heading("Appendix: additional validation figures (optional)", 1)
p("These give a more detailed view of the same 77-case comparison and can be moved to an "
  "appendix or omitted from the main text.")
figure("match_pct_distribution.png", "Figure A.1.",
       "Distribution of top-1 match% for A0 and A2 (histogram + box plot) with the 0.75 hit "
       "threshold marked. The two distributions nearly overlap.")
figure("rank_distribution.png", "Figure A.2.",
       "Rank position of the first matching cause (rank 1-5 or no hit in top 5). Most hits "
       "occur at rank 1; A2 recovers a few additional cases into the top 5.")
figure("paired_delta.png", "Figure A.3.",
       "Per-incident difference in match% (A2 minus A0), sorted. Most incidents are unchanged "
       "and the mean delta is near zero.")

doc.save(OUT)
print("Saved:", OUT)
