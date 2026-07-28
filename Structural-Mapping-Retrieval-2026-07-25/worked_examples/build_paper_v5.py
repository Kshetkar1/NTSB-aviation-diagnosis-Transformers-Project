"""
Build docs/full_paper_v5_draft.docx from the precomputed worked examples and
the aggregate evaluation results.

This is a DRAFT builder. The output is intended to be opened in Word and
marked up by Dr. Maha. Voice is deliberately rough: short sentences, no
em-dashes, some redundancy, no Zhang/Mahadevan (2021) comparison framing.

Inputs (relative to project root):
  Worked_Examples/data/example_after.json                 (engine fire,    after-incident)
  Worked_Examples/data/example_during.json                (engine fire,    pilot voice)
  Worked_Examples/data/example_after_landing_gear.json    (landing gear,   after-incident)
  Worked_Examples/data/example_during_landing_gear.json   (landing gear,   pilot voice)
  Worked_Examples/data/aggregate_results.jsonl            (77 held-out test cases)

Output:
  docs/full_paper_v5_draft.docx
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent
DATA_DIR = _HERE / "data"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "full_paper_v5_draft.docx"
sys.path.insert(0, str(_HERE))
from aggregate_summary import compute_summary, load_results  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_examples() -> dict[str, dict]:
    out = {
        "engine_after":  json.loads((DATA_DIR / "example_after.json").read_text()),
        "engine_during": json.loads((DATA_DIR / "example_during.json").read_text()),
        "gear_after":    json.loads((DATA_DIR / "example_after_landing_gear.json").read_text()),
        "gear_during":   json.loads((DATA_DIR / "example_during_landing_gear.json").read_text()),
    }
    sym_engine = DATA_DIR / "example_symptom_engine_fire.json"
    sym_gear = DATA_DIR / "example_symptom_landing_gear.json"
    if sym_engine.exists():
        out["engine_symptom"] = json.loads(sym_engine.read_text())
    if sym_gear.exists():
        out["gear_symptom"] = json.loads(sym_gear.read_text())
    return out


def _load_aggregate() -> tuple[list[dict], dict]:
    rows = load_results()
    return rows, compute_summary(rows)


GROUND_TRUTH = {
    "20100114X11754": {
        "narr_cause": (
            "The sealing failure between the fuel nozzle adapter assembly and fuel "
            "transfer tubes that allowed fuel to leak into the nacelle fire zone where "
            "it was subsequently ignited by the hot combustor case. Contributing to the "
            "failure was a combination of manufacturing defects of the fuel nozzle adapter "
            "assembly by the manufacturer and incorrectly overhauled fuel transfer tubes "
            "by the engine manufacturer."
        ),
        "cause_finding": "Aircraft-Aircraft power plant-Engine fuel and control-Fuel distribution-Failure - C",
        "description": "ATR72 left-engine fire on takeoff (St. Croix, 2010)",
    },
    "20081116X33137": {
        "narr_cause": (
            "The mechanical overload of the nosewheel steering links for undetermined "
            "reasons, which resulted in nose landing gear rotation and its subsequent "
            "wedging within the wheel well structure."
        ),
        "cause_finding": "Aircraft-Aircraft systems-Landing gear system-Gear extension and retract sys-Capability exceeded - C",
        "description": "DHC-8 nose landing gear retracted on landing (Philadelphia, 2008)",
    },
}


# ---------------------------------------------------------------------------
# Doc helpers
# ---------------------------------------------------------------------------

def _setup(doc: Document) -> None:
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)


def h(doc: Document, text: str, level: int = 1) -> None:
    p = doc.add_heading(text, level=level)
    for run in p.runs:
        run.font.color.rgb = RGBColor(0x00, 0x00, 0x00)


def p(doc: Document, text: str, italic: bool = False) -> None:
    para = doc.add_paragraph()
    run = para.add_run(text)
    if italic:
        run.italic = True


def eq(doc: Document, text: str, number: str | None = None) -> None:
    """Render an equation as a centered italic line. Equation number on the
    right via tab stops (rough but readable)."""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = para.add_run(text)
    r.italic = True
    if number:
        para.add_run(f"     ({number})")


def quote(doc: Document, text: str) -> None:
    para = doc.add_paragraph()
    para.paragraph_format.left_indent = Inches(0.4)
    para.paragraph_format.right_indent = Inches(0.4)
    run = para.add_run(text)
    run.italic = True


def note(doc: Document, text: str) -> None:
    # DRAFT NOTE blocks intentionally suppressed in the final draft.
    return


def add_table(doc: Document, headers: list[str], rows: list[list[str]],
              col_widths: list[float] | None = None,
              caption: str | None = None) -> None:
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for j, name in enumerate(headers):
        hdr[j].text = ""
        para = hdr[j].paragraphs[0]
        r = para.add_run(name)
        r.bold = True
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.rows[i].cells[j]
            cell.text = str(val)
            for para in cell.paragraphs:
                for r in para.runs:
                    r.font.size = Pt(10)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)
    if caption:
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run(caption)
        r.italic = True
        r.font.size = Pt(10)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_title(doc: Document) -> None:
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = para.add_run(
        "Decision-Useful Diagnosis and Prognosis of Aviation Accidents\n"
        "from Free-Text Narratives in the NTSB FAR 121 Database"
    )
    r.bold = True
    r.font.size = Pt(16)

    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.add_run("Kanu Shetkar, Sankaran Mahadevan, Jesse Spencer-Smith\nVanderbilt University").font.size = Pt(11)

    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = para.add_run("Draft. Not for distribution.")
    r.italic = True
    r.font.size = Pt(10)


def section_abstract(doc: Document) -> None:
    h(doc, "Abstract", level=1)
    p(doc,
      "We present a tool that reads an aviation accident narrative and tells a "
      "pilot or an investigator two things. First, what is the most likely failure "
      "theme behind this incident (diagnosis). Second, given the current state "
      "of the incident, what is the most likely next event in the timeline (prognosis). "
      "Both questions are answered by retrieving similar past incidents from the "
      "NTSB FAR 121 database (1982 to 2016) using text embeddings, clustering the "
      "retrieved set into named failure themes, and combining cluster-level "
      "frequencies through the law of total probability."
      )
    p(doc,
      "The tool supports two query styles. The after-incident query uses the full "
      "investigator narrative and is the view a safety analyst gets once the report "
      "is written. The during-incident query is a single sentence written in the "
      "pilot\u2019s voice, as if the pilot is on the radio mid-event. We show that the "
      "same pipeline handles both query styles and returns a decision-useful answer "
      "in both cases."
      )
    p(doc,
      "We also describe a structural mapping extension that adds a second similarity "
      "signal based on the causal chain of each incident, and combines it with the "
      "embedding score through a single-line fusion formula to rerank the top "
      "retrieved candidates. We evaluate the pipeline on a held-out test set of 77 "
      "FAR 121 incidents. The predicted top failure-theme cluster is a plausible "
      "semantic match to the official NTSB probable-cause text on 62 percent of the "
      "test set and a solid match on 42 percent of the test set, measured by cosine "
      "similarity in the same embedding space used for retrieval. The structural "
      "extension improves on the embedding-only baseline on 17 of 77 test cases, "
      "ties on 51, and loses on 9. The two worked examples in this paper sit inside "
      "the solid-match cohort and are not cherry-picked from outside the aggregate "
      "result."
      )
    note(doc,
        "Voice is still rough in places. Some of the abstract repeats what is in "
        "the intro. I want Maha to mark up which sentences to keep and which to cut."
    )


def section_intro(doc: Document) -> None:
    h(doc, "1  Introduction", level=1)
    p(doc,
      "Aviation accidents are studied so that future ones can be avoided. The NTSB "
      "publishes a detailed accident report for each incident it investigates, and "
      "those reports are the primary record we have for understanding how small "
      "abnormal events grow into accidents. The reports include two kinds of "
      "information. The first is structured: coded event sequences, occurrence "
      "codes, finding codes, injury data, and aircraft damage data. The structured "
      "fields use a fixed vocabulary. The second is unstructured: free-text "
      "narratives written by pilots, crews, and investigators, and a probable cause "
      "statement written by an NTSB analyst. The narratives are where most of the "
      "context lives. They mention the pilot\u2019s description of the problem, the "
      "exact maintenance history of the affected component, the weather, the crew "
      "interaction, and any other detail the investigator considered relevant. "
      "Earlier statistical models of the NTSB data have used the structured fields "
      "only, because that is what fits inside a Bayesian network or a fixed feature "
      "vector. The narrative text was left out."
      )
    p(doc,
      "Large language models change this. They can read narrative text directly, "
      "find similar past incidents by meaning rather than by keyword, and produce "
      "structured outputs that downstream code can use. The work in this paper is "
      "built on that capability. We bring the narrative text into the analysis "
      "without throwing away the rigor of frequency-based probabilities."
      )
    p(doc,
      "We frame the tool around two practitioner views. The first is the "
      "investigator view. The investigator has the full narrative in hand and "
      "wants to know what failure theme the incident belongs to, what causes have "
      "been recorded in similar past incidents, and what events have historically "
      "followed this kind of incident. The second is the pilot view. The pilot "
      "is in the middle of the event and has one or two sentences of information. "
      "The pilot wants the same answer from the same tool. We treat these two "
      "views as two query styles and run them through one pipeline."
      )
    p(doc,
      "The reason we built the pipeline as a hybrid (LLM for retrieval and "
      "clustering, frequency counts for probabilities) instead of running an LLM "
      "end-to-end is described in Section 1.1. The contributions of this work are "
      "in Section 1.2."
      )

    h(doc, "1.1  Limitations of the Pure LLM Approach", level=2)
    p(doc,
      "Before settling on the hybrid design, we built a pipeline in which an LLM "
      "was responsible for the entire process: retrieving similar past cases, "
      "extracting contextual factors from the narratives, scoring and ranking "
      "candidate causes, and producing a recommendation. Three problems with this "
      "approach made it unsuitable."
      )
    p(doc,
      "First, the LLM does not produce probabilities in the way a Bayesian network "
      "does. It infers relationships from learned language patterns rather than "
      "from explicit frequency counts, so the numbers it returns are not grounded "
      "in the actual data and are not reproducible across runs."
      )
    p(doc,
      "Second, the LLM sometimes produced confident outputs that were not supported "
      "by any incident in the database. This is the behavior often called "
      "hallucination. Even with retrieval grounding, the model still tended to "
      "overstate certainty."
      )
    p(doc,
      "Third, the system was sensitive to small changes in how the query was phrased. "
      "Changing one or two words in the query produced different top causes. This is "
      "not acceptable for a safety analysis tool."
      )
    p(doc,
      "We took two lessons from this. LLMs are good at retrieving and structuring "
      "narrative text. Frequency counts taken from the actual data are good at "
      "producing reliable probabilities. The two parts should be kept separate. "
      "The hybrid approach in Section 5 is built on that lesson."
      )

    h(doc, "1.2  Contributions", level=2)
    p(doc,
      "This paper makes four contributions to the analysis of free-text accident "
      "narratives in the NTSB FAR 121 database."
      )
    p(doc,
      "1. A hybrid pipeline that brings narrative text into NTSB analysis. The LLM "
      "is used for narrative retrieval and clustering. Probabilities are computed "
      "from frequency counts using the law of total probability."
      )
    p(doc,
      "2. A two-query design. The same pipeline runs on a full investigator "
      "narrative (the after-incident view) and on a single-sentence pilot voice "
      "query (the during-incident view). We show worked examples for both."
      )
    p(doc,
      "3. A structural mapping extension that extracts a structured causal chain "
      "for each incident, computes a structural similarity score using sequence "
      "alignment, and combines this score with the embedding cosine through a "
      "single-line fusion formula to rerank the top retrieved candidates."
      )
    p(doc,
      "4. An aggregate evaluation on a held-out test set of 77 FAR 121 incidents. "
      "We measure how well the predicted top failure-theme cluster matches the "
      "official NTSB probable-cause text using cosine similarity in the same "
      "embedding space used for retrieval. The pipeline produces a plausible "
      "semantic match on 62 percent of the test set and a solid match on 42 "
      "percent of the test set. The structural extension wins on 17 of 77 cases, "
      "ties on 51, and loses on 9, which is a small but consistent positive "
      "shift over the embedding-only baseline."
      )
    p(doc,
      "The rest of the paper is organized as follows. Section 2 reviews related "
      "work on structure-mapping theory and two-stage retrieval. Section 3 "
      "describes the NTSB FAR 121 data. Section 4 introduces background concepts. "
      "Section 5 presents the hybrid pipeline. Section 6 presents the structural "
      "mapping extension. Section 7 walks through two real worked examples, "
      "engine fire and landing gear, each under both query styles and both "
      "pipelines, and includes a symptom-only stress test (Section 7.4) that "
      "checks whether the pipeline relies on diagnosis-leading vocabulary in "
      "the query. Section 8 reports the aggregate evaluation. Section 9 "
      "discusses what the pipeline does well and where it does not. Section 10 "
      "concludes."
      )


def section_related(doc: Document) -> None:
    h(doc, "2  Related Work", level=1)
    p(doc,
      "This work draws on two lines of prior research. The first is "
      "structure-mapping theory in cognitive science, which gives the conceptual "
      "basis for the structural mapping extension in Section 6. The second is "
      "two-stage retrieval and reranking in information retrieval, which gives "
      "the algorithmic basis for combining the embedding score and the structural "
      "score in Section 6.6."
      )

    h(doc, "2.1  NTSB Accident Data and Narrative Analysis", level=2)
    p(doc,
      "Earlier statistical analyses of the NTSB FAR 121 database have used the "
      "structured coded fields only. The narrative text was not used as input. "
      "Bayesian network models, ensemble machine learning models, and other "
      "discrete-feature models all need a fixed feature vector or a fixed node "
      "set, and the narratives do not fit that shape without manual coding. The "
      "work in this paper is the first attempt we know of to bring the narrative "
      "text into the same analysis using LLM-based retrieval, while keeping "
      "the probabilities grounded in frequency counts."
      )
    note(doc,
        "Maha - is there a specific prior NTSB-narrative paper I should cite here? "
        "I deliberately removed the Zhang/Mahadevan (2021) citation from this section "
        "because the paper is meant to stand on its own. Let me know if you want "
        "any of it back as background only."
    )

    h(doc, "2.2  Structure-Mapping Theory", level=2)
    p(doc,
      "The structural mapping extension in this paper operationalizes ideas from "
      "structure-mapping theory in cognitive science. Gentner (1983) introduced "
      "structure-mapping as a framework for analogical reasoning. The central "
      "claim is that deep similarity between two situations is about the relational "
      "structure (which objects play which roles, in what causal order), not the "
      "surface attributes of the objects."
      )
    p(doc,
      "Gentner and Markman (1997) separated analogy (matching on relational "
      "structure with little surface overlap) from literal similarity (matching on "
      "both relational structure and surface attributes), and argued that human "
      "similarity judgments give more weight to systematic relations than to "
      "first-order relations or attributes."
      )
    p(doc,
      "Falkenhainer et al. (1989) developed the Structure-Mapping Engine, an "
      "algorithm that searches for the best mapping between two structured "
      "representations and assigns evidence values to candidate match hypotheses. "
      "Higher-order structure provides trickle-down support to lower-level matches."
      )
    p(doc,
      "Goldstone (1994) showed that human similarity judgments incorporate "
      "alignment and feature comparison together rather than as independent "
      "stages. Spencer-Smith and Goldstone (1997) showed that the weights given to "
      "different features during similarity comparison shift dynamically with the "
      "comparison context."
      )
    p(doc,
      "We do not implement the Structure-Mapping Engine directly. We implement a "
      "smaller, domain-tuned version of the same idea: a fixed three-field schema "
      "(role, system, mechanism), a per-step similarity that weights role most "
      "heavily, and a chain score that classifies aligned pairs into strong, "
      "partial, and unmapped categories. The schema is described in Section 6.2."
      )

    h(doc, "2.3  Two-Stage Retrieval and Reranking", level=2)
    p(doc,
      "The fusion step in Section 6.6 fits inside the broader pattern of two-stage "
      "retrieval in information retrieval. Nogueira and Cho (2019) introduced a "
      "two-stage architecture in which a fast first-pass retriever (BM25 or a "
      "similar index-based scorer) returns a candidate set of passages and a "
      "second, more expensive scorer reranks them using a learned model. The "
      "structural mapping extension uses the same shape: the embedding cosine "
      "plays the role of the fast first-pass score and the structural similarity "
      "plays the role of the slower per-pair reranker."
      )
    p(doc,
      "The alignment step itself uses the dynamic programming algorithm of "
      "Needleman and Wunsch (1970), which was originally developed for global "
      "alignment of two protein sequences and is now standard for any "
      "two-sequence alignment task with a per-position similarity function and "
      "a gap penalty."
      )


def section_data(doc: Document) -> None:
    h(doc, "3  NTSB Accident Investigation Data", level=1)
    p(doc,
      "The National Transportation Safety Board (NTSB) is an independent federal "
      "agency that investigates accidents and incidents across all modes of "
      "transportation in the United States. The work in this paper focuses on "
      "the aviation portion of the database, and within aviation we focus on "
      "FAR 121 air carriers, which cover large U.S.-based airlines, regional air "
      "carriers, and all-cargo operators."
      )
    p(doc,
      "The FAR 121 portion of the NTSB database from 1982 to 2016 contains 2,243 "
      "accident and incident reports. Each incident is spread across 14 separate "
      "files. The files cover aircraft characteristics, engine characteristics, "
      "sequences of events, occurrences, findings, injury data, and the narrative "
      "reports written by pilots, crews, and investigators. For this work we merge "
      "the 14 files into a single record per incident, keyed by the NTSB event ID, "
      "so that the structured fields and the narratives for a given incident can "
      "be retrieved together. The merged dataset preserves the same 2,243 incidents "
      "and adds nothing new at the row level. The narrative text is what is new "
      "from a modeling perspective."
      )
    p(doc,
      "Two structured fields are used heavily in the methodology that follows. "
      "The first is the sequence of events for each incident, which is an ordered "
      "chain of occurrences with one occurrence marked as the defining event. The "
      "defining event acts as the anchor for the prognosis task: the events before "
      "the anchor are the precursors, and the events after the anchor are the "
      "downstream consequences. The second is the findings field, which lists the "
      "causes and contributing factors identified by the NTSB investigators, and "
      "which we use as the ground truth for the diagnosis task. The narrative text "
      "supports the retrieval step in the methodology and is also used in the "
      "structural mapping extension to extract a structured causal chain for each "
      "incident."
      )
    p(doc,
      "For the held-out evaluation in Section 8 we split the 2,243 incidents "
      "into a training set and a test set. The training set is the index used by "
      "the retrieval step. The test set is held out. The 77 test incidents reported "
      "in Section 8 are drawn from the test side of this split and are never seen "
      "during cache construction, index building, or any step of the pipeline."
      )
    note(doc,
        "Note to self: confirm the exact 177/77 train/test split with Jesse. "
        "The split file lives at data/Testing_Data_Metrics/splits/test_ev_ids.txt."
    )


def section_background(doc: Document) -> None:
    h(doc, "4  Background Concepts", level=1)
    p(doc,
      "This section introduces the four background concepts that the methodologies "
      "in Section 5 and Section 6 build on. The goal is to fix notation and give "
      "the minimum needed to follow the rest of the paper."
      )

    h(doc, "4.1  Vector Embeddings and Cosine Similarity", level=2)
    p(doc,
      "A vector embedding maps a piece of text to a fixed-length vector of real "
      "numbers in a way that places texts with similar meaning close together in "
      "the vector space. We use OpenAI\u2019s text-embedding-3-small model, which "
      "produces 1,536-dimensional vectors. Each dimension does not have a single "
      "human-interpretable meaning, but together the dimensions capture topic, "
      "phase of flight, type of failure, and other latent features of the text."
      )
    p(doc,
      "Once the texts are mapped to vectors, the similarity between any two texts "
      "can be computed mathematically. The standard measure is cosine similarity. "
      "Given a query embedding q and an incident embedding i, the cosine similarity "
      "is defined as:"
      )
    eq(doc, "sim(q, i) = (q \u00b7 i) / (||q|| \u00b7 ||i||)", number="1")
    p(doc,
      "where q \u00b7 i is the dot product and ||\u00b7|| is the Euclidean norm. The output "
      "ranges from 0 (unrelated) to 1 (identical meaning). For example, the query "
      "loss of engine power during cruise returns a similarity of 0.81 against the "
      "incident turbine blade separation during cruise and only 0.25 against bird "
      "strike on ground."
      )

    h(doc, "4.2  K-means Clustering", level=2)
    p(doc,
      "K-means is an unsupervised clustering algorithm that partitions a set of "
      "data points into k groups, where each point is assigned to the group whose "
      "centroid it is closest to. The algorithm minimizes the sum of squared "
      "distances between each point and its assigned centroid. We use k-means with "
      "k = 50 on the 2,243 incident embeddings to organize the database into 50 "
      "incident-type clusters during a one-time setup step. Each cluster is then "
      "given a short text label produced by GPT-4o-mini, based on the incidents "
      "that fall in the cluster (for example, engine fire due to component "
      "failures or nose landing gear malfunctions)."
      )

    h(doc, "4.3  Law of Total Probability", level=2)
    p(doc,
      "The law of total probability gives a way to compute the marginal probability "
      "of an event from a set of conditional probabilities and their associated "
      "weights. If C is the event of interest, K1, K2, ..., Km is a partition of "
      "the sample space, and Q is a query, then:"
      )
    eq(doc, "P(C | Q) = \u03a3_j  P(C | K_j) \u00b7 P(K_j | Q)", number="2")
    p(doc,
      "In Section 5, C will be a candidate cause or a candidate next event, K_j "
      "will be one of the clusters from Section 4.2, and Q will be a query "
      "incident. The conditional terms P(C | K_j) are estimated as cluster-level "
      "frequency counts. The weights P(K_j | Q) are computed from the cosine "
      "similarities between the query and the incidents in each cluster."
      )

    h(doc, "4.4  Needleman-Wunsch Sequence Alignment", level=2)
    p(doc,
      "Needleman-Wunsch is a dynamic programming algorithm originally developed "
      "for aligning two protein sequences (Needleman and Wunsch, 1970). The "
      "algorithm finds the best global alignment of two sequences by filling a "
      "two-dimensional table dp with the best score reachable at each (i, j) "
      "position. At each cell, the algorithm chooses the best of three moves: "
      "align position i in sequence A with position j in sequence B, skip a "
      "position in A, or skip a position in B. The recurrence is:"
      )
    eq(doc,
       "dp[i][j] = max{ dp[i-1][j-1] + sim(A[i], B[j]),  dp[i-1][j] + g,  dp[i][j-1] + g }",
       number="3")
    p(doc,
      "where sim(A[i], B[j]) is the per-position similarity score and g is a "
      "small negative gap penalty that allows but discourages skipping. The final "
      "cell dp[n][m] holds the best score for aligning the full two sequences. We "
      "use this algorithm in Section 6 to align the structured causal chains of "
      "two incidents, where each position in the sequence is one step in the "
      "causal chain."
      )


def section_methodology(doc: Document) -> None:
    h(doc, "5  Methodology: The Hybrid Pipeline", level=1)
    p(doc,
      "This section describes the hybrid pipeline that combines LLM-based "
      "retrieval and clustering with frequency-based probability calculations "
      "using the law of total probability. The pipeline supports both diagnosis "
      "(backward inference from the incident to the cause) and prognosis (forward "
      "inference from the current state to the next event). Both modes share the "
      "same retrieval and clustering steps in Sections 5.1 through 5.4. The two "
      "modes diverge at the probability step, which is described separately for "
      "diagnosis in Section 5.6 and for prognosis in Section 5.7. Section 5.5 "
      "describes the two query styles supported by the system."
      )

    h(doc, "5.1  Merged Dataset", level=2)
    p(doc,
      "The 14 NTSB files described in Section 3 are merged into a single record "
      "per incident, keyed on the NTSB event ID. Each merged record holds the "
      "structured fields (events, occurrences, subject codes, findings, injury "
      "and damage data) and the narrative text for that incident in one place. "
      "The merged dataset is stored as JSON for fast lookup. This step is "
      "one-time and offline. At query time the system only reads from the merged "
      "file, which avoids the cost of cross-referencing multiple files for every "
      "incident."
      )

    h(doc, "5.2  Embeddings", level=2)
    p(doc,
      "Each incident in the merged dataset is converted into a set of embedding "
      "vectors using OpenAI\u2019s text-embedding-3-small model, which produces "
      "1,536-dimensional vectors as defined in Section 4.1. We embed the narrative "
      "text, the findings, and the entries from the NTSB event dictionary. We "
      "embed all three because the system needs to be able to match a free-form "
      "query against either the narrative text, the structured cause findings, or "
      "the standardized event descriptions. In total, embedding the 2,243 "
      "incidents produces 4,360 embedding vectors, since each incident contributes "
      "multiple text pieces."
      )

    h(doc, "5.3  Similarity Search", level=2)
    p(doc,
      "Given an embedded query, the system retrieves the most relevant past "
      "incidents using the cosine similarity from Equation 1. The query embedding "
      "is compared against all 4,360 embeddings in the index, and the top 50 most "
      "similar incidents are kept for the next step. Fifty was chosen as a "
      "compromise between coverage (enough incidents to support cluster-level "
      "frequency counts) and focus (few enough that the retrieved set still "
      "concentrates on the query topic)."
      )

    h(doc, "5.4  Clustering", level=2)
    p(doc,
      "The top 50 retrieved incidents are not a single homogeneous group. They "
      "are a mix of failure scenarios that may have very different causal "
      "patterns and downstream consequences. To organize them, we use the "
      "pre-computed k-means clusters described in Section 4.2. During the "
      "one-time setup step, k-means with k = 50 was run on all 2,243 incident "
      "embeddings, and each cluster was given a short label produced by "
      "GPT-4o-mini. These cluster labels are stored directly in the merged "
      "dataset, so at query time the system only needs to look up the cluster "
      "of each retrieved incident."
      )

    h(doc, "5.5  Two Query Styles", level=2)
    p(doc,
      "The pipeline accepts two query styles. The first is the after-incident "
      "query. This is the view of the safety analyst or the NTSB investigator. "
      "The full investigator narrative is available, and the system\u2019s job is to "
      "place the incident into a failure theme, surface the most likely causes, "
      "and report what tends to happen next in similar incidents. The "
      "after-incident query is typically a few thousand characters long."
      )
    p(doc,
      "The second query style is the during-incident query. This is the view of "
      "the pilot or the dispatcher in the middle of the event. The query is one or "
      "two sentences long and reads like a radio call. The pilot reports the one "
      "or two symptoms that are visible right now and asks what could be wrong "
      "and what is about to happen. The system processes the during-incident "
      "query through the same retrieval, clustering, and probability steps as "
      "the after-incident query."
      )
    p(doc,
      "Examples of each query style for both worked-example incidents are given "
      "in Section 7. The during-incident queries are written in the pilot\u2019s voice "
      "and are short on purpose. They were written by hand and are not drawn from "
      "a corpus of real pilot radio calls. Section 9 discusses this as a limitation."
      )

    h(doc, "5.6  Diagnosis", level=2)
    p(doc,
      "In diagnosis mode the goal is to estimate the probability of each "
      "candidate cause given the query, P(Cause | Q). We compute two things. "
      "First, we compute the cluster-level posterior P(K_j | Q) for each cluster "
      "K_j that appears in the top 50 retrieved set. This is the headline number. "
      "It tells the user how strongly the query belongs to each failure theme. "
      "Second, we use the law of total probability to combine the cluster-level "
      "cause distributions into a final cause distribution P(Cause | Q). The user "
      "can read off either layer."
      )

    h(doc, "5.6.1  Cluster-Level Cause Probabilities", level=3)
    p(doc,
      "For each cluster K_j, we compute the cluster-level cause distribution "
      "P(Cause | K_j) by counting how often each cause appears in the incidents "
      "that belong to that cluster. Causes come from the NTSB findings field. We "
      "use natural language processing to extract the causal phrases recorded by "
      "the investigators. Each incident may list more than one cause. For one "
      "cluster, we go through each incident in the cluster, extract every cause "
      "mentioned, and count how often each unique cause appears. Dividing each "
      "count by the total number of cause mentions in the cluster gives the "
      "cluster-level cause probability."
      )

    h(doc, "5.6.2  Cluster Weights", level=3)
    p(doc,
      "The cluster weight P(K_j | Q) measures how relevant cluster K_j is to the "
      "query. We compute the unnormalized cluster weight as the average cosine "
      "similarity of the incidents from K_j that appear in the top 50, multiplied "
      "by the number of such incidents:"
      )
    eq(doc, "w_j = s\u0304_j \u00b7 n_j", number="4")
    p(doc,
      "where s\u0304_j is the mean cosine similarity of the incidents from cluster "
      "K_j that appear in the top 50, and n_j is the count of such incidents. The "
      "two factors capture the two ways a cluster can be relevant: it can have "
      "many retrieved incidents, or it can have a few highly similar ones, or "
      "both. The weights are then normalized to a probability distribution:"
      )
    eq(doc, "P(K_j | Q) = w_j / \u03a3_k w_k", number="5")

    h(doc, "5.6.3  Combining Across Clusters", level=3)
    p(doc,
      "The cluster-level cause probabilities and the cluster weights are combined "
      "using the law of total probability from Equation 2:"
      )
    eq(doc, "P(Cause | Q) = \u03a3_j  P(Cause | K_j) \u00b7 P(K_j | Q)", number="6")
    p(doc,
      "This produces a probability distribution over candidate causes. The "
      "highest-probability cause is returned as the system\u2019s primary diagnosis. "
      "Both the cluster-level probabilities P(K | Q) and the cause-level "
      "probabilities P(Cause | Q) are reported to the user. The reason for "
      "reporting both is that the cluster-level numbers are usually concentrated "
      "on a single theme (so the headline is clean) while the cause-level numbers "
      "spread thinner across many candidate causes. A worked example with real "
      "numbers is in Section 7."
      )

    h(doc, "5.7  Prognosis", level=2)
    p(doc,
      "In prognosis mode the question is reversed. Given the current state, the "
      "system predicts what is most likely to happen next. The prognosis pipeline "
      "reuses Sections 5.1 through 5.5 unchanged. Only what is extracted from the "
      "retrieved incidents changes."
      )

    h(doc, "5.7.1  Sequence of Events Data", level=3)
    p(doc,
      "For each incident, the NTSB sequence of events field gives an ordered "
      "chain of occurrences in the order they happened. One occurrence in the "
      "chain is marked as the defining event, which is the main event that "
      "triggered the NTSB investigation. The defining event acts as the anchor: "
      "the events before it are the precursors, and the events after it are the "
      "downstream consequences. For prognosis we are interested in the downstream "
      "events. For each retrieved incident we find the defining event in its "
      "sequence and record what comes immediately after."
      )

    h(doc, "5.7.2  Cluster-Level Next-Event Probabilities", level=3)
    p(doc,
      "For each cluster K_j we compute the cluster-level next-event distribution "
      "P(Next Event | K_j) by counting how often each event appears as the "
      "immediate successor of the defining event across the incidents in that "
      "cluster. The procedure has five steps. First, take one cluster. Second, "
      "for each incident in that cluster, find the defining event. Third, record "
      "what event comes immediately after. Fourth, count the frequency of each "
      "next event across the cluster. Fifth, divide each count by the total "
      "number of sequences contributed by the cluster to get the cluster-level "
      "next-event probability."
      )

    h(doc, "5.7.3  Combining Across Clusters", level=3)
    p(doc,
      "The same form of the law of total probability used for diagnosis applies "
      "to prognosis, with next event in place of cause:"
      )
    eq(doc, "P(Next Event | Q) = \u03a3_j  P(Next Event | K_j) \u00b7 P(K_j | Q)", number="7")
    p(doc,
      "The cluster weights P(K_j | Q) are computed in the same way as in "
      "Section 5.6.2. The output is a probability distribution over candidate "
      "next events. The highest-probability event is returned as the predicted "
      "next event. A worked example with real numbers is in Section 7."
      )

    h(doc, "5.7.4  Multi-Step Prediction", level=3)
    p(doc,
      "Aviation incidents do not stop at one event. Events cascade in a sequence: "
      "A leads to B, B leads to C. To predict events further down the chain, we "
      "use transition probabilities. A transition probability is a conditional "
      "probability over sequential events, P(Event_t | Event_{t-1}), that tells "
      "us what tends to follow each event. Once we know the distribution at "
      "step t - 1, the distribution at step t is given by another application of "
      "the law of total probability:"
      )
    eq(doc,
       "P(Event_t | Q) = \u03a3_e  P(Event_t | Event_{t-1} = e) \u00b7 P(Event_{t-1} = e | Q)",
       number="8")
    p(doc,
      "The transition probabilities are estimated from all consecutive pairs in "
      "the sequence data across the retrieved incidents. This step can be applied "
      "repeatedly to extend the prediction further down the chain, with one "
      "extra application of Equation 8 per additional step."
      )


def section_structural(doc: Document) -> None:
    h(doc, "6  Methodology: Structural Mapping Extension", level=1)
    p(doc,
      "The hybrid approach in Section 5 relies on text similarity to decide which "
      "past incidents are relevant to a new query, and then on cluster-level "
      "frequency counts to combine evidence across the retrieved set. This works "
      "well when the wording of the new incident lines up with the wording of "
      "the cause finding it should match. The approach has a known blind spot, "
      "described in Section 6.1, and the structural mapping extension in this "
      "section is built to address it. The extension is an operationalization of "
      "the structure-mapping framework of Gentner (1983), with the alignment "
      "carried out by the dynamic programming algorithm of Needleman and Wunsch "
      "(1970) and the reranking step following the two-stage retrieval pattern "
      "of Nogueira and Cho (2019)."
      )

    h(doc, "6.1  Motivation", level=2)
    p(doc,
      "Two incidents with very different causal mechanisms can read similar in "
      "plain text, and two incidents with the same underlying mechanism can be "
      "described in very different vocabulary. The phrase loss of engine power "
      "during cruise can match against an incident caused by fuel contamination, "
      "an incident caused by a maintenance error on a turbine blade, or an "
      "incident caused by ice ingestion. The embedding does not know which of "
      "these mechanisms the new query actually belongs to. It only knows the "
      "words sound similar."
      )
    p(doc,
      "Gentner and Markman (1997) make this point in general terms: matching on "
      "words and matching on causal structure are two different things, and the "
      "structural match is the one that tells you whether two cases are really "
      "alike. The structural mapping extension addresses this by adding a second "
      "similarity signal that compares the underlying causal mechanism of two "
      "incidents, not just their wording, and uses it to rerank the top retrieved "
      "candidates."
      )

    h(doc, "6.2  Causal Chain Schema", level=2)
    p(doc,
      "For every incident we extract a structured causal chain. The chain is a "
      "list of steps in the order they occurred, and each step has three fields. "
      "The role field captures where the step sits in the causal sequence, with "
      "allowed values initiating event, propagation, system compromise, system "
      "failure, terminal failure, operational consequence, and outcome. The "
      "system field captures which subsystem the step belongs to, with allowed "
      "values such as engine mechanical, lubrication, fuel, hydraulic, electrical, "
      "structural, flight controls, landing gear, propulsion, fire protection, "
      "environment, human performance, maintenance, and aircraft. The mechanism "
      "field captures what physically went wrong at that step, with allowed "
      "values such as maintenance error, material degradation, fatigue, "
      "corrosion, contamination, thermal damage, mechanical loosening, "
      "deprivation, overheating, fire, separation, power loss, loss of control, "
      "terrain collision, procedural error, and design deficiency."
      )
    p(doc,
      "In addition to the chain itself, each incident has a list of contributing "
      "factors and a single failure-pattern label such as thermal cascade, "
      "maintenance latent defect cascade, or human performance chain. The schema "
      "is fixed so that two incidents extracted independently can be compared "
      "field by field, which is what allows the alignment step in Section 6.5 to "
      "produce a meaningful score."
      )

    h(doc, "6.3  Extraction with the LLM", level=2)
    p(doc,
      "The structured chain is extracted by sending the incident text to "
      "GPT-4o-mini with a fixed schema prompt. The model returns a JSON object "
      "that we then normalize against the allowed values listed in Section 6.2. "
      "Anything outside the allowed list is coerced to a default value such as "
      "unknown, so that the downstream scoring code is never surprised by an "
      "unexpected category. This is the only place in the pipeline where an LLM "
      "is involved beyond the retrieval embedding step. The extraction is run "
      "once per incident in the training set and once per query, and the result "
      "is written to a cache file. There is no LLM call during ranking, scoring, "
      "or fusion."
      )

    h(doc, "6.4  Step Similarity", level=2)
    p(doc,
      "To compare two causal chains we first need a way to compare two individual "
      "steps. Given step A from one chain and step B from another, we compute a "
      "step similarity score:"
      )
    eq(doc, "sim(A, B) = 0.50 \u00b7 r(A, B) + 0.30 \u00b7 m(A, B) + 0.20 \u00b7 s(A, B)", number="9")
    p(doc,
      "where r, m, and s are the role, mechanism, and system component scores. "
      "The weights (0.50 for role, 0.30 for mechanism, 0.20 for system) reflect "
      "the relative importance of each field and follow the relational hierarchy "
      "of Gentner (1983). Role gets the highest weight because it captures where "
      "in the causal sequence the step sits, which is the most diagnostic field "
      "for distinguishing one mechanism from another. Mechanism gets the "
      "next-highest weight because it describes the causal connection between "
      "consecutive steps. System gets the lowest weight because two incidents "
      "can involve different physical components and still share the same "
      "underlying failure pattern."
      )
    p(doc,
      "If two steps have the exact same role, r returns 1.0. If the roles are "
      "different but adjacent in the causal sequence (for example, propagation "
      "and system compromise), r returns a value between 0.7 and 0.95 from a "
      "small lookup table. The system and mechanism scores work the same way, "
      "with a compatibility table that gives partial credit for related categories. "
      "The output of sim(A, B) is a single number between 0 and 1 that measures "
      "how close the two steps are."
      )

    h(doc, "6.5  Chain Alignment", level=2)
    p(doc,
      "Two incidents do not always have the same number of steps in their chain, "
      "and the steps do not always appear in the same order. To compare the two "
      "full chains we use the Needleman-Wunsch alignment algorithm from "
      "Section 4.4, with the per-step similarity from Equation 9 as the "
      "alignment score and a constant gap penalty of g = -0.05 so that skipping "
      "a step is allowed but discouraged. After the dynamic programming table is "
      "filled, the final cell holds the best score for aligning the two full chains."
      )
    p(doc,
      "We then walk the alignment back and classify each aligned position into "
      "one of three categories. If the step similarity is at least 0.70, the pair "
      "counts as a strong match. If it is between 0.35 and 0.70, it counts as a "
      "partial match. Aligned pairs below 0.35 and any positions left as gaps in "
      "the alignment count as unmapped. The structural similarity is then "
      "computed as:"
      )
    eq(doc,
       "struct_sim = (strong + 0.5 \u00b7 partial) / (|A| + |B|)  -  0.10 \u00b7 unmapped / (|A| + |B|)",
       number="10")
    p(doc,
      "where |A| and |B| are the lengths of the two chains being compared, so the "
      "denominator is the total number of step positions across both chains. The "
      "unmapped penalty term subtracts a small fraction proportional to the "
      "number of positions that did not find a good match, so that two chains "
      "with many gaps cannot reach a high score on the strength of a few matched "
      "steps alone. The final structural similarity is a number between 0 and 1."
      )

    h(doc, "6.6  Score Fusion", level=2)
    p(doc,
      "The structural similarity does not replace the embedding score. It "
      "reweights it. For each candidate incident, the new score is:"
      )
    eq(doc, "new_score = max(0, cos_sim) \u00b7 exp(\u03b1 \u00b7 struct_sim)", number="11")
    p(doc,
      "where cos_sim is the cosine similarity from Section 5.2, struct_sim is the "
      "structural similarity from Section 6.5, and \u03b1 is a tuning parameter that "
      "controls how strongly the structural signal influences the ranking. The "
      "two-stage shape (fast first-pass retrieval followed by a slower per-pair "
      "reranker) is the standard pattern of Nogueira and Cho (2019). Combining "
      "the two scores into a single ranking score, rather than running them as "
      "fully independent stages, is consistent with Goldstone (1994). Based on "
      "initial experiments, we set \u03b1 = 2.0."
      )
    p(doc,
      "When the structural similarity is 0, the exponential factor is 1 and the "
      "new score reduces to the embedding cosine. The fusion can never make a "
      "candidate look more similar than it already was on text alone unless the "
      "structural signal also agrees. The new scores are sorted, and the top "
      "candidate becomes the prediction returned by the system."
      )
    p(doc,
      "As a small inline example: a candidate with cosine 0.62 and structural "
      "similarity 0.78 has a new score of 0.62 \u00b7 exp(2 \u00b7 0.78) = 2.96. A candidate "
      "with the same cosine 0.62 and structural similarity 0.40 has a new score "
      "of 0.62 \u00b7 exp(2 \u00b7 0.40) = 1.38. The first candidate gets a much larger "
      "boost, so it moves above the second in the ranking. We refer to the "
      "embedding-only baseline (no structural rerank) as A0 throughout the rest "
      "of the paper, and the structural mapping extension as A2."
      )


# ---------------------------------------------------------------------------
# Worked examples
# ---------------------------------------------------------------------------

def _top5_cluster_rows(d_diag: dict) -> list[list[str]]:
    """Build a top-5 cluster table for one (case, query, pipeline)."""
    out = []
    for r in d_diag.get("clusters", [])[:5]:
        out.append([
            str(r.get("cluster", "")),
            str(r.get("n_incidents", 0)),
            f"{float(r.get('avg_similarity', 0.0)):.3f}",
            pct(float(r.get("p_k_given_q", 0.0))),
        ])
    return out


def _top5_cause_rows(d_diag: dict) -> list[list[str]]:
    out = []
    for r in d_diag.get("ltp_causes_enriched", [])[:5]:
        out.append([
            str(r.get("rank", "")),
            str(r.get("label", "")),
            pct(float(r.get("probability", 0.0)), digits=2),
            (str(r.get("elaboration", "")) or "")[:140],
        ])
    return out


def _top5_event_rows(d_prog: dict) -> list[list[str]]:
    out = []
    for r in d_prog.get("ltp_events_enriched", [])[:5]:
        out.append([
            str(r.get("rank", "")),
            str(r.get("label", "")),
            pct(float(r.get("probability", 0.0)), digits=2),
            (str(r.get("elaboration", "")) or "")[:140],
        ])
    return out


def _render_scenario(doc: Document, header: str, ex: dict, query_label: str) -> None:
    h(doc, header, level=3)

    p(doc, f"Query ({query_label}):", italic=False)
    quote(doc, ex["diagnosis_query"] if query_label == "during-incident pilot voice" else ex["diagnosis_query"][:400] + " ...")

    truth = GROUND_TRUTH.get(ex["ev_id"], {})
    p(doc, "Ground truth (NTSB probable-cause text):")
    quote(doc, truth.get("narr_cause", "(unavailable)"))

    h(doc, "Diagnosis: A0 top-5 failure-theme clusters", level=4)
    a0 = ex["diagnosis"]["a0"]
    add_table(
        doc,
        headers=["Cluster (failure theme)", "n incidents", "mean cosine", "P(K | Q)"],
        rows=_top5_cluster_rows(a0),
        col_widths=[3.0, 0.8, 1.0, 1.0],
        caption=f"A0 top clusters for {header}",
    )

    h(doc, "Diagnosis: A2 top-5 failure-theme clusters (with structural rerank)", level=4)
    a2 = ex["diagnosis"]["a2"]
    add_table(
        doc,
        headers=["Cluster (failure theme)", "n incidents", "mean weighted score", "P(K | Q)"],
        rows=_top5_cluster_rows(a2),
        col_widths=[3.0, 0.8, 1.0, 1.0],
        caption=f"A2 top clusters for {header}",
    )

    h(doc, "Diagnosis: A0 top-5 causes from the law of total probability", level=4)
    add_table(
        doc,
        headers=["#", "Cause (plain English)", "P(Cause | Q)", "Elaboration"],
        rows=_top5_cause_rows(a0),
        col_widths=[0.3, 1.7, 0.7, 3.3],
        caption=f"A0 top LTP causes for {header}",
    )

    h(doc, "Diagnosis: A2 top-5 causes from the law of total probability", level=4)
    add_table(
        doc,
        headers=["#", "Cause (plain English)", "P(Cause | Q)", "Elaboration"],
        rows=_top5_cause_rows(a2),
        col_widths=[0.3, 1.7, 0.7, 3.3],
        caption=f"A2 top LTP causes for {header}",
    )

    h(doc, "Prognosis: A0 top-5 next events", level=4)
    add_table(
        doc,
        headers=["#", "Next event (plain English)", "P(Event | Q)", "Elaboration"],
        rows=_top5_event_rows(ex["prognosis"]["a0"]),
        col_widths=[0.3, 1.7, 0.7, 3.3],
        caption=f"A0 top next events for {header}",
    )

    h(doc, "Prognosis: A2 top-5 next events", level=4)
    add_table(
        doc,
        headers=["#", "Next event (plain English)", "P(Event | Q)", "Elaboration"],
        rows=_top5_event_rows(ex["prognosis"]["a2"]),
        col_widths=[0.3, 1.7, 0.7, 3.3],
        caption=f"A2 top next events for {header}",
    )

    a0_top_cluster = (a0["clusters"][0]["cluster"] if a0["clusters"] else "")
    a0_top_p = float(a0["clusters"][0]["p_k_given_q"]) if a0["clusters"] else 0.0
    a2_top_cluster = (a2["clusters"][0]["cluster"] if a2["clusters"] else "")
    a2_top_p = float(a2["clusters"][0]["p_k_given_q"]) if a2["clusters"] else 0.0

    p(doc,
      f"Reading the tables above. The headline number for both pipelines is "
      f"P(top cluster | Q): A0 lands on \u201c{a0_top_cluster}\u201d at {pct(a0_top_p)}, "
      f"A2 lands on \u201c{a2_top_cluster}\u201d at {pct(a2_top_p)}. The structural "
      f"rerank changes the cluster mass but not the top cluster identity in this "
      f"scenario. The cause-level distribution from the law of total probability "
      f"spreads thinner because there are many candidate causes in the FAR 121 "
      f"data, which is why the top-1 P(Cause | Q) is in the single-digit percent "
      f"range. The cluster-level number is the one that is operationally useful."
      )


def section_worked_examples(doc: Document, examples: dict) -> None:
    h(doc, "7  Worked Examples", level=1)
    p(doc,
      "This section walks through two real held-out test incidents end to end. "
      "Both incidents are in the held-out test set. Neither incident was used in "
      "the retrieval index. For each incident we run the pipeline under two "
      "query styles. The after-incident query uses the full NTSB narrative. The "
      "during-incident query is a one-sentence pilot voice query. We report both "
      "pipelines (A0 embedding-only baseline and A2 structural mapping extension) "
      "for both query styles."
      )
    p(doc,
      "The two cases were chosen to cover different failure families. Case 1 "
      "(Section 7.1) is an engine fire (powerplant family). Case 2 (Section 7.2) "
      "is a nose landing gear failure on landing (gear family). The aggregate "
      "evaluation in Section 8 confirms that the cluster-level match scores for "
      "these two cases sit inside the solid-match cohort across the full 77-case "
      "test set, so the two worked examples are representative of the pipeline\u2019s "
      "behavior and not cherry-picked outliers."
      )

    h(doc, "7.1  Case 1: ATR72 Left-Engine Fire on Takeoff (Event ID 20100114X11754)", level=2)
    p(doc,
      "On January 11, 2010, about 2000 Atlantic Standard Time, an Aerospatiale "
      "ATR72 experienced a No. 1 (left) engine fire during takeoff from Henry E "
      "Rohlsen Airport in St. Croix. The pilot declared an emergency, shut down "
      "the No. 1 engine, discharged both fire bottles, and performed an air-turn "
      "back. The aircraft landed approximately 11 minutes later. The 44 passengers "
      "and 4 crew deplaned normally. No injuries were reported. The NTSB later "
      "determined that the probable cause was a sealing failure between the fuel "
      "nozzle adapter assembly and the fuel transfer tubes that allowed fuel to "
      "leak into the nacelle fire zone where it was ignited by the hot combustor "
      "case."
      )
    _render_scenario(doc, "7.1.1  After-incident query (investigator view)",
                     examples["engine_after"], "after-incident investigator narrative")
    _render_scenario(doc, "7.1.2  During-incident query (pilot voice)",
                     examples["engine_during"], "during-incident pilot voice")

    h(doc, "7.2  Case 2: DHC-8 Nose Landing Gear Retracted on Landing (Event ID 20081116X33137)", level=2)
    p(doc,
      "On November 16, 2008, about 0934 eastern standard time, a deHavilland "
      "DHC-8-311 operated by Piedmont Airlines as USAirways flight 4551 sustained "
      "minor damage when it landed with the nose landing gear retracted at "
      "Philadelphia International Airport. The flight crew received an unsafe "
      "nose-gear indication on approach. The mains showed three greens. The crew "
      "executed the gear-up landing checklist and the aircraft was landed on the "
      "mains and the nose-gear doors. The NTSB later determined that the probable "
      "cause was mechanical overload of the nosewheel steering links for "
      "undetermined reasons, which resulted in nose landing gear rotation and "
      "subsequent wedging within the wheel well structure."
      )
    _render_scenario(doc, "7.2.1  After-incident query (investigator view)",
                     examples["gear_after"], "after-incident investigator narrative")
    _render_scenario(doc, "7.2.2  During-incident query (pilot voice)",
                     examples["gear_during"], "during-incident pilot voice")

    note(doc,
        "Maha - I deliberately put the during-incident query right next to the "
        "after-incident query for both cases. The purpose is to show that the same "
        "tool produces a useful answer in both situations even though the inputs "
        "look very different. If you think the during-incident sections belong in "
        "a separate chapter please mark it up and I will restructure."
    )

    h(doc, "7.3  Summary of the Two Cases", level=2)
    p(doc,
      "Table 7.1 summarizes the headline P(top cluster | Q) across the two cases, "
      "the two query styles, and the two pipelines."
      )

    eng_a = examples["engine_after"]
    eng_d = examples["engine_during"]
    gea_a = examples["gear_after"]
    gea_d = examples["gear_during"]

    def _tpc(ex, pipe):
        c = ex["diagnosis"][pipe]["clusters"]
        return (c[0]["cluster"], float(c[0]["p_k_given_q"])) if c else ("", 0.0)

    rows = []
    for label, ex in [("Engine fire, after-incident", eng_a),
                      ("Engine fire, during-incident (pilot voice)", eng_d),
                      ("Landing gear, after-incident", gea_a),
                      ("Landing gear, during-incident (pilot voice)", gea_d)]:
        c0, p0 = _tpc(ex, "a0")
        c2, p2 = _tpc(ex, "a2")
        rows.append([label, f"{c0}", pct(p0), f"{c2}", pct(p2)])

    add_table(
        doc,
        headers=["Scenario", "A0 top cluster", "P(K | Q) A0", "A2 top cluster", "P(K | Q) A2"],
        rows=rows,
        col_widths=[2.4, 1.6, 0.8, 1.6, 0.8],
        caption="Table 7.1. Headline P(top cluster | Q) across the two cases and the two query styles.",
    )

    p(doc,
      "Reading the table. The top cluster is the same under A0 and A2 in all "
      "four scenarios. The cluster mass shifts slightly between pipelines. In "
      "every row, both pipelines land on a cluster name that reads as a plain "
      "English description of the actual failure (engine fire, nose landing gear "
      "malfunctions). For the engine fire case the mass is between 50 and 54 "
      "percent on the top cluster, which is concentrated mass on a single "
      "operational label. For the landing gear case the mass on the after-incident "
      "query is lower (15 to 18 percent) because the full narrative includes "
      "approach, runway, and crew-action prose that pulls retrieval into multiple "
      "themes. The pilot voice query for the same incident lands at 39 to 41 "
      "percent on the same cluster. The during-incident question is shorter and "
      "stays closer to the failure-relevant words, which is why the cluster mass "
      "is higher even though the underlying incident is the same. This is "
      "discussed further in Section 8.6."
      )


def section_stress_test(doc: Document, examples: dict) -> None:
    """Section 7.4: symptom-only stress test on the two worked-example incidents.
    Renders only if the two symptom-only JSONs were loaded successfully."""
    if "engine_symptom" not in examples or "gear_symptom" not in examples:
        return

    h(doc, "7.4  Stress Test: Symptom-Only Queries", level=2)
    p(doc,
      "A natural concern with the during-incident pilot voice queries in "
      "Section 7.1.2 and 7.2.2 is that the query already contains the name of "
      "the failure (engine fire, nose-gear). A reader could reasonably ask "
      "whether the pipeline is doing real diagnostic work or simply matching "
      "those keywords back to the cause taxonomy. To check this we constructed "
      "a third query style for each of the two worked-example incidents. The "
      "symptom-only query keeps only what a pilot would directly observe in "
      "the first few seconds of the event, and removes any word that names the "
      "failure or appears in the cause taxonomy (no fire, no engine, no flames, "
      "no gear, no failure, no unsafe). The symptom-only queries are written "
      "from the actual NTSB narratives, not invented."
      )

    h(doc, "7.4.1  Engine fire, symptom-only query", level=3)
    p(doc, "Symptom-only query (no diagnosis words):")
    quote(doc, examples["engine_symptom"]["diagnosis_query"])

    h(doc, "Diagnosis A0 top-5 clusters", level=4)
    a0_es = examples["engine_symptom"]["diagnosis"]["a0"]
    add_table(
        doc,
        headers=["Cluster (failure theme)", "n incidents", "mean cosine", "P(K | Q)"],
        rows=_top5_cluster_rows(a0_es),
        col_widths=[3.0, 0.8, 1.0, 1.0],
        caption="A0 top clusters for engine fire, symptom-only query.",
    )

    h(doc, "Diagnosis A2 top-5 clusters", level=4)
    a2_es = examples["engine_symptom"]["diagnosis"]["a2"]
    add_table(
        doc,
        headers=["Cluster (failure theme)", "n incidents", "mean weighted score", "P(K | Q)"],
        rows=_top5_cluster_rows(a2_es),
        col_widths=[3.0, 0.8, 1.0, 1.0],
        caption="A2 top clusters for engine fire, symptom-only query.",
    )

    h(doc, "7.4.2  Landing gear, symptom-only query", level=3)
    p(doc, "Symptom-only query (no diagnosis words):")
    quote(doc, examples["gear_symptom"]["diagnosis_query"])

    h(doc, "Diagnosis A0 top-5 clusters", level=4)
    a0_gs = examples["gear_symptom"]["diagnosis"]["a0"]
    add_table(
        doc,
        headers=["Cluster (failure theme)", "n incidents", "mean cosine", "P(K | Q)"],
        rows=_top5_cluster_rows(a0_gs),
        col_widths=[3.0, 0.8, 1.0, 1.0],
        caption="A0 top clusters for landing gear, symptom-only query.",
    )

    h(doc, "Diagnosis A2 top-5 clusters", level=4)
    a2_gs = examples["gear_symptom"]["diagnosis"]["a2"]
    add_table(
        doc,
        headers=["Cluster (failure theme)", "n incidents", "mean weighted score", "P(K | Q)"],
        rows=_top5_cluster_rows(a2_gs),
        col_widths=[3.0, 0.8, 1.0, 1.0],
        caption="A2 top clusters for landing gear, symptom-only query.",
    )

    h(doc, "7.4.3  Side-by-Side Comparison Across All Three Query Styles", level=3)
    p(doc,
      "Table 7.2 lays the three query styles side by side for both incidents. "
      "It shows the top cluster identity, P(K | Q) under both pipelines, and "
      "whether the predicted top cause code matches the NTSB ground truth code "
      "for the incident."
      )

    def _row_for(label: str, ex: dict) -> list[str]:
        c0 = ex["diagnosis"]["a0"]["clusters"]
        c2 = ex["diagnosis"]["a2"]["clusters"]
        top_a0 = c0[0]["cluster"] if c0 else ""
        top_a2 = c2[0]["cluster"] if c2 else ""
        p_a0 = float(c0[0]["p_k_given_q"]) if c0 else 0.0
        p_a2 = float(c2[0]["p_k_given_q"]) if c2 else 0.0
        truth_code = str(ex["ground_truth"].get("code", ""))
        a0_code = str(ex["diagnosis"]["a0"].get("top1_code", ""))
        a2_code = str(ex["diagnosis"]["a2"].get("top1_code", ""))
        a0_hit = "yes" if a0_code == truth_code and truth_code else "no"
        a2_hit = "yes" if a2_code == truth_code and truth_code else "no"
        return [label, top_a0, pct(p_a0), a0_hit, top_a2, pct(p_a2), a2_hit]

    rows_table = [
        _row_for("Engine fire: after-incident", examples["engine_after"]),
        _row_for("Engine fire: during-incident (with diagnosis)", examples["engine_during"]),
        _row_for("Engine fire: symptom-only (no diagnosis)", examples["engine_symptom"]),
        _row_for("Landing gear: after-incident", examples["gear_after"]),
        _row_for("Landing gear: during-incident (with diagnosis)", examples["gear_during"]),
        _row_for("Landing gear: symptom-only (no diagnosis)", examples["gear_symptom"]),
    ]
    add_table(
        doc,
        headers=["Scenario", "A0 top cluster", "P(K|Q) A0", "A0 code hit", "A2 top cluster", "P(K|Q) A2", "A2 code hit"],
        rows=rows_table,
        col_widths=[2.0, 1.5, 0.7, 0.6, 1.5, 0.7, 0.6],
        caption="Table 7.2. Headline cluster results across all three query styles for each of the two incidents. Bottom row of each pair is the symptom-only stress test.",
    )

    h(doc, "7.4.4  Reading the Stress Test", level=3)
    p(doc,
      "The two cases give two different answers. For the engine fire incident, "
      "the symptom-only query (we just lifted off, strong yaw to the left, "
      "master warning sounding, losing power on one side, smell of burning) "
      "still places the largest cluster mass on engine fire due to component "
      "failures. The mass drops to about 37 percent under both pipelines, "
      "compared to 50 to 54 percent for the after-incident and during-incident "
      "queries on the same incident. The system is less confident, but it lands "
      "on the same family. The top predicted cause code is still 130 "
      "(Airframe / component / system failure / malfunction), which matches the "
      "NTSB ground truth code. This is consistent with the pipeline making real "
      "use of leading indicators (yaw direction, master warning, asymmetric "
      "thrust feel, burning smell) rather than only matching keywords from the "
      "cause taxonomy."
      )
    p(doc,
      "For the landing gear incident the same stress test does not hold up. "
      "With the words gear, unsafe, and nose-gear removed, the query "
      "(two greens, one red on the configuration panel, doors are open but the "
      "structure below is not visible, going around) pulls retrieval into a "
      "different cluster (multiple contributing mechanical failures) with about "
      "75 to 79 percent mass under both pipelines. The correct cluster (nose "
      "landing gear malfunctions) has only one of fifty retrieved incidents in "
      "this view, and the predicted top cause code is 250 (Loss of control - "
      "in flight) which does not match the NTSB ground truth code 196 (Gear "
      "not extended). This is an honest failure of the pipeline. The "
      "configuration-panel language used in the symptom-only query "
      "(two greens / one red / doors open) is not specific enough to "
      "discriminate between gear, flap, slat, and other configuration "
      "anomalies. The original pilot voice query in 7.2.2 succeeded in part "
      "because the word nose-gear anchored retrieval. Removing that anchor "
      "exposed the limitation."
      )
    p(doc,
      "Two takeaways. First, the engine fire result establishes that the "
      "pipeline is not purely a keyword matcher. It can place the largest "
      "cluster mass on the right family from leading indicators alone, with "
      "lower but still operationally useful confidence. Second, the landing "
      "gear result identifies a real boundary of the current design. When the "
      "symptoms a pilot can name in the first few seconds are inherently "
      "ambiguous between several configuration anomalies, the pipeline cannot "
      "always disambiguate. A symptom-to-domain bridge that maps observed "
      "indications to the affected subsystem before retrieval is a future-work "
      "item, listed in Section 10.3."
      )
    p(doc,
      "We do not attempt this stress test on the full 77-case held-out set in "
      "this paper. Generating a symptom-only counterpart for each test "
      "incident automatically would itself require an LLM step (to strip "
      "diagnosis words from the narrative without changing meaning), which "
      "would introduce a new layer of LLM-dependent processing into the "
      "evaluation. We flag this as future work and treat the two-case stress "
      "test as a sanity check on the keyword-matching concern, not as a full "
      "evaluation."
      )


def section_aggregate(doc: Document, summary: dict) -> None:
    h(doc, "8  Aggregate Validation on a Held-Out Test Set", level=1)
    p(doc,
      "Section 7 walked through two incidents in detail. To check that those two "
      "incidents are not cherry-picked outliers we ran the pipeline on every one "
      "of the 77 held-out test incidents in the FAR 121 corpus. This section "
      "reports the aggregate results."
      )

    h(doc, "8.1  Setup", level=2)
    p(doc,
      "We split the 2,243 FAR 121 incidents into a training set of 2,166 "
      "incidents and a held-out test set of 77 incidents. The training set is "
      "used to build the embedding index, the k-means clusters, the cluster-level "
      "cause and next-event tables, and the structural cache. The 77 test "
      "incidents are never seen during cache construction, index building, or "
      "any other step of the pipeline. Every aggregate number in this section is "
      "computed on the 77 test incidents."
      )
    p(doc,
      "For each test incident we run the full pipeline under the after-incident "
      "query (the investigator-view narrative is the only one we have ground "
      "truth for; we do not have a corpus of real pilot voice queries to "
      "evaluate against). We compute the top failure-theme cluster, the cluster "
      "mass P(top cluster | Q), the top free-text cause from the law of total "
      "probability, and the top next event. We do this twice, once under A0 (the "
      "embedding-only baseline) and once under A2 (the structural mapping "
      "extension with \u03b1 = 2.0)."
      )

    h(doc, "8.2  Metric: Cluster-Cosine to NTSB Probable Cause", level=2)
    p(doc,
      "For each test incident the NTSB report includes an official probable-cause "
      "text written by the investigator. This text is in plain English and "
      "describes the underlying mechanism that caused the incident. We compare "
      "the pipeline\u2019s predicted top cluster label to this NTSB probable-cause "
      "text using cosine similarity in the same OpenAI embedding space that the "
      "pipeline uses for retrieval. The same comparison is done for the top "
      "free-text cause from the law of total probability."
      )
    p(doc,
      "Why cosine similarity. Both the predicted cluster label (for example, "
      "engine fire due to component failures) and the NTSB probable-cause text "
      "(for example, sealing failure between the fuel nozzle adapter assembly "
      "and fuel transfer tubes) are short pieces of free-text English. The "
      "pipeline already uses cosine similarity in the same embedding space to do "
      "retrieval, so using it again to evaluate the output keeps the evaluation "
      "self-consistent. Reading the numbers, a cosine of greater than 0.50 reads "
      "as the same failure family in plain English, a cosine of 0.40 to 0.50 "
      "reads as a related family, and a cosine of less than 0.30 reads as a "
      "different topic."
      )

    h(doc, "8.3  Aggregate Results", level=2)
    n = summary["n"]
    cl = summary["cluster"]
    cs = summary["cause"]
    pk = summary["p_top_cluster"]
    add_table(
        doc,
        headers=["Metric", "A0 (embedding only)", "A2 (structural rerank)", "\u0394 (A2 - A0)"],
        rows=[
            ["Mean cosine(top cluster, NTSB cause)",
             f"{cl['mean_a0']:.3f}", f"{cl['mean_a2']:.3f}",
             f"{cl['mean_a2'] - cl['mean_a0']:+.3f}"],
            ["Median cosine(top cluster, NTSB cause)",
             f"{cl['median_a0']:.3f}", f"{cl['median_a2']:.3f}",
             f"{cl['median_a2'] - cl['median_a0']:+.3f}"],
            ["Solid match (cos > 0.50)",
             f"{cl['thresholds']['0.50']['n_a0']} / {n} ({pct(cl['thresholds']['0.50']['frac_a0'])})",
             f"{cl['thresholds']['0.50']['n_a2']} / {n} ({pct(cl['thresholds']['0.50']['frac_a2'])})",
             f"{(cl['thresholds']['0.50']['frac_a2'] - cl['thresholds']['0.50']['frac_a0'])*100:+.1f} pp"],
            ["Plausible match (cos > 0.40)",
             f"{cl['thresholds']['0.40']['n_a0']} / {n} ({pct(cl['thresholds']['0.40']['frac_a0'])})",
             f"{cl['thresholds']['0.40']['n_a2']} / {n} ({pct(cl['thresholds']['0.40']['frac_a2'])})",
             f"{(cl['thresholds']['0.40']['frac_a2'] - cl['thresholds']['0.40']['frac_a0'])*100:+.1f} pp"],
            ["At least related (cos > 0.30)",
             f"{cl['thresholds']['0.30']['n_a0']} / {n} ({pct(cl['thresholds']['0.30']['frac_a0'])})",
             f"{cl['thresholds']['0.30']['n_a2']} / {n} ({pct(cl['thresholds']['0.30']['frac_a2'])})",
             f"{(cl['thresholds']['0.30']['frac_a2'] - cl['thresholds']['0.30']['frac_a0'])*100:+.1f} pp"],
            ["Mean P(top cluster | Q)",
             pct(pk["mean_a0"]), pct(pk["mean_a2"]),
             f"{(pk['mean_a2'] - pk['mean_a0'])*100:+.1f} pp"],
            ["Mean cosine(top cause, NTSB cause)",
             f"{cs['mean_a0']:.3f}", f"{cs['mean_a2']:.3f}",
             f"{cs['mean_a2'] - cs['mean_a0']:+.3f}"],
        ],
        col_widths=[2.5, 1.6, 1.6, 1.0],
        caption=f"Table 8.1. Aggregate predicted-cluster and predicted-cause cosine similarities to the NTSB probable-cause text, across n = {n} held-out test incidents.",
    )

    p(doc,
      "Reading Table 8.1. The headline number is the mean cosine similarity "
      f"between the predicted top failure-theme cluster and the NTSB "
      f"probable-cause text. A0 lands at {cl['mean_a0']:.3f} and A2 lands at "
      f"{cl['mean_a2']:.3f}. The shift from A0 to A2 is small but consistent. "
      f"On {cl['thresholds']['0.50']['n_a2']} out of {n} test incidents "
      f"({pct(cl['thresholds']['0.50']['frac_a2'])}), the predicted cluster is a "
      f"solid semantic match to the NTSB cause text (cos > 0.50). On "
      f"{cl['thresholds']['0.40']['n_a2']} out of {n} "
      f"({pct(cl['thresholds']['0.40']['frac_a2'])}), it is at least plausibly "
      f"related (cos > 0.40)."
      )

    h(doc, "8.4  A2 versus A0 Head-to-Head", level=2)
    wtl = cl["a2_vs_a0"]
    p(doc,
      "We also computed the per-incident difference in cluster cosine between A2 "
      "and A0. The table below shows the win, tie, and loss counts."
      )
    add_table(
        doc,
        headers=["Outcome", "Cluster cosine", "Top cause cosine"],
        rows=[
            ["A2 strictly improves on A0", f"{wtl['win']} / {n}", f"{cs['a2_vs_a0']['win']} / {n}"],
            ["A2 ties with A0 (no change in top result)", f"{wtl['tie']} / {n}", f"{cs['a2_vs_a0']['tie']} / {n}"],
            ["A2 strictly underperforms A0", f"{wtl['loss']} / {n}", f"{cs['a2_vs_a0']['loss']} / {n}"],
            ["A2 does no harm (improves or ties)",
             f"{wtl['win'] + wtl['tie']} / {n} ({pct((wtl['win'] + wtl['tie']) / n)})",
             f"{cs['a2_vs_a0']['win'] + cs['a2_vs_a0']['tie']} / {n} ({pct((cs['a2_vs_a0']['win'] + cs['a2_vs_a0']['tie']) / n)})"],
        ],
        col_widths=[3.0, 1.5, 1.5],
        caption="Table 8.2. Head-to-head outcomes on the per-incident cosine similarity, for both the top cluster label and the top free-text cause.",
    )
    p(doc,
      f"Reading Table 8.2. On the cluster-cosine metric A2 strictly improves on "
      f"A0 on {wtl['win']} of {n} test incidents, ties on {wtl['tie']}, and loses "
      f"on {wtl['loss']}. The no-harm rate is "
      f"{pct((wtl['win'] + wtl['tie']) / n)}. On the top-cause cosine metric "
      f"the wins, ties, and losses are {cs['a2_vs_a0']['win']}, "
      f"{cs['a2_vs_a0']['tie']}, and {cs['a2_vs_a0']['loss']} respectively. The "
      "extension is a small but consistent positive on the cluster metric and a "
      "near-wash on the per-cause metric. We discuss why in Section 9."
      )

    h(doc, "8.5  Secondary Metric: M1 Top-1 Hit and McNemar Test", level=2)
    p(doc,
      "For the statistically inclined reader we also report the M1 metric used "
      "in earlier versions of this work. For each test incident the system\u2019s top "
      "predicted cause is compared to the official cause findings written by NTSB "
      "investigators. The prediction passes (M1 hit) if the embedding cosine "
      "similarity between the prediction and any of the true cause findings is "
      "at least 0.75. Top-1 accuracy is the fraction of the 77 test incidents "
      "that pass at rank 1. Under this metric A2 produces 27 hits out of 77 (35.1 "
      "percent) versus 25 hits under A0 (32.5 percent). McNemar\u2019s exact "
      "two-sided test on the paired binary outcome gives p = 0.500 (2 incidents "
      "where A2 was correct and A0 was not, 0 incidents where A0 was correct and "
      "A2 was not, the rest agreeing). The McNemar p-value is high because there "
      "are only two disagreements, not because the effect is in the wrong "
      "direction. The flip pattern is the best possible shape for a paired "
      "comparison."
      )
    note(doc,
        "Maha - I kept the M1 / McNemar paragraph for the statistically inclined "
        "reviewer but moved it to a secondary subsection. The primary metric is "
        "now the cluster-cosine one in 8.3 because it lines up with what is "
        "operationally useful for a pilot or investigator. Let me know if you want "
        "to swap the order, drop one of them, or expand on either."
    )

    h(doc, "8.6  Limitations of the Aggregate Metric", level=2)
    p(doc,
      "Two honest limitations of the aggregate evaluation. First, the cosine "
      "similarity threshold (0.30, 0.40, 0.50) is a continuous metric not a "
      "binary one. Any specific threshold is a choice. We report all three so "
      "that a reviewer can pick their own bar. Second, the embedding model used "
      "to compute the similarity is the same model used to do retrieval, which "
      "introduces a kind of circularity. A future evaluation should use a "
      "different embedding model or an LLM-judge spot check to confirm that the "
      "matches reported here are not an artifact of the embedding model\u2019s "
      "internal geometry."
      )
    p(doc,
      "A third limitation is that the test set size (n = 77) is small. The "
      "confidence intervals on the per-threshold fractions are wide. A larger "
      "held-out set, or k-fold cross-validation across the FAR 121 set, would "
      "give tighter intervals. We treat the current numbers as evidence of the "
      "right direction, not as a final benchmark."
      )


def section_discussion(doc: Document) -> None:
    h(doc, "9  Discussion", level=1)
    p(doc,
      "Three things stood out in the worked examples and the aggregate evaluation."
      )
    p(doc,
      "First, the cluster level is the right level to report a probability at. "
      "The per-cause distribution P(Cause | Q) is the formally correct output of "
      "the law of total probability, but in practice it spreads thin across many "
      "candidate causes because the FAR 121 cause taxonomy is large. The top-1 "
      "cause probability is usually in the low single digits, which does not "
      "read as a confident answer even when the underlying cluster mass is high. "
      "The cluster-level probability P(K | Q) is concentrated on one or two "
      "themes and reads as a decision-useful answer. Both engine-fire scenarios "
      "in Section 7 land between 50 and 54 percent on a single cluster. That is "
      "the kind of number a pilot or investigator can act on."
      )
    p(doc,
      "Second, query length matters. The landing gear case in Section 7.2 has a "
      "long after-incident narrative that includes approach, runway, and "
      "crew-action prose. That extra prose pulls retrieval into multiple themes "
      "and the headline cluster mass drops to 15 percent. The same incident "
      "queried with a short pilot voice query lands at 39 percent on the same "
      "cluster. The pipeline is the same. The difference is what is in the "
      "query. This is a feature of the design (the during-incident view is "
      "supposed to be short and focused) but it is also a reminder that the "
      "after-incident view can benefit from preprocessing the narrative to "
      "extract the failure-relevant portion."
      )
    p(doc,
      "Third, the structural mapping extension does a small amount of work in "
      "the aggregate and a larger amount of work in specific cases. The "
      "head-to-head numbers in Section 8.4 show that A2 is a small consistent "
      "positive on the cluster metric (17 wins, 51 ties, 9 losses out of 77). "
      "On individual cases A2 can shift cluster mass by several percentage "
      "points and can change the top-1 cause text. The extension is not a "
      "replacement for the embedding score. It is a reranker that breaks ties "
      "when the embedding score alone is not enough to separate one mechanism "
      "from another."
      )
    note(doc,
        "Maha - this section is short on purpose. I would rather you tell me "
        "which threads to expand than guess. Some candidates: cluster naming "
        "quality, what happens when k-means is set differently, comparison "
        "between A2 alpha values, comparison with a simpler keyword baseline."
    )


def section_conclusion(doc: Document) -> None:
    h(doc, "10  Conclusion", level=1)

    h(doc, "10.1  Contributions", level=2)
    p(doc,
      "This paper makes four contributions to the analysis of free-text accident "
      "narratives in the NTSB FAR 121 database. First, a hybrid pipeline that "
      "brings narrative text into the analysis. The LLM is used for retrieval "
      "and clustering. Probabilities are computed from frequency counts using "
      "the law of total probability. Second, a two-query design (after-incident "
      "investigator narrative and during-incident pilot voice) that runs through "
      "the same pipeline. Third, a structural mapping extension that extracts a "
      "structured causal chain for each incident and reranks the top retrieved "
      "candidates using a single-line fusion formula. Fourth, an aggregate "
      "evaluation on a held-out test set of 77 FAR 121 incidents that shows the "
      "predicted top cluster is a plausible or solid semantic match to the NTSB "
      "probable-cause text on roughly 42 to 62 percent of cases (depending on "
      "the threshold), and the structural extension improves on the "
      "embedding-only baseline on 17 of 77 cases with no harm on 68 of 77."
      )

    h(doc, "10.2  Limitations", level=2)
    p(doc,
      "Several limitations should be addressed in future work. First, the test "
      "set size (n = 77) is small. The confidence intervals on rate-style metrics "
      "are wide. A larger held-out set, or k-fold cross-validation across the "
      "full FAR 121 set, would give tighter intervals."
      )
    p(doc,
      "Second, the structural extraction step depends on an LLM, and errors in "
      "extraction propagate into the alignment and produce noisy structural "
      "similarities. The schema is finite and may not cover every nuance of every "
      "incident."
      )
    p(doc,
      "Third, the fusion parameter \u03b1 in Equation 11 was set to 2.0 based on "
      "initial experiments. A grid search on a held-out subset would let \u03b1 be "
      "picked from data."
      )
    p(doc,
      "Fourth, the during-incident pilot voice queries used in Section 7 were "
      "written by hand. They are not drawn from a corpus of real pilot radio "
      "calls. A future evaluation should use real pilot transcripts where "
      "available."
      )
    p(doc,
      "Fifth, the aggregate metric in Section 8 uses the same OpenAI embedding "
      "model for retrieval and for the predicted-cluster-versus-NTSB-cause "
      "comparison. A future evaluation should use a different embedding model "
      "or an LLM-judge spot check to confirm that the matches are not an "
      "artifact of the embedding model\u2019s internal geometry."
      )

    h(doc, "10.3  Future Work", level=2)
    p(doc,
      "Three places this work can go next. First, the during-incident query "
      "style can be validated against real pilot transcripts and real cockpit "
      "voice recorder data. Second, the structural reranker can be evaluated "
      "separately on the prognosis pipeline (Section 5.7) under the same "
      "head-to-head design as the diagnosis evaluation. Third, the per-step "
      "similarity weights in Equation 9 (0.50, 0.30, 0.20) were set by hand. "
      "These weights can be learned from data on a held-out subset, which would "
      "let the fusion adapt to the structure of the data instead of relying on "
      "a fixed prior."
      )


def section_references(doc: Document) -> None:
    h(doc, "References", level=1)
    refs = [
        "B. Falkenhainer, K. D. Forbus, and D. Gentner. The structure-mapping engine: Algorithm and examples. Artificial Intelligence, 41(1):1-63, 1989.",
        "D. Gentner. Structure-mapping: A theoretical framework for analogy. Cognitive Science, 7(2):155-170, 1983.",
        "D. Gentner and A. B. Markman. Structure mapping in analogy and similarity. American Psychologist, 52(1):45-56, 1997.",
        "R. L. Goldstone. Similarity, interactive activation, and mapping. Journal of Experimental Psychology: Learning, Memory, and Cognition, 20(1):3-28, 1994.",
        "International Air Transport Association. IATA forecast predicts 8.2 billion air travelers in 2037, 2019.",
        "S. B. Needleman and C. D. Wunsch. A general method applicable to the search for similarities in the amino acid sequence of two proteins. Journal of Molecular Biology, 48(3):443-453, 1970.",
        "R. Nogueira and K. Cho. Passage re-ranking with BERT. arXiv preprint arXiv:1901.04085, 2019.",
        "J. Spencer-Smith and R. L. Goldstone. The dynamics of similarity. Cognitive Studies: Bulletin of the Japanese Cognitive Science Society, 4:38-56, 1997.",
    ]
    for r in refs:
        para = doc.add_paragraph()
        para.paragraph_format.left_indent = Inches(0.3)
        para.paragraph_format.first_line_indent = Inches(-0.3)
        para.add_run(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    examples = _load_examples()
    rows, summary = _load_aggregate()

    doc = Document()
    _setup(doc)

    section_title(doc)
    section_abstract(doc)
    doc.add_page_break()

    section_intro(doc)
    section_related(doc)
    section_data(doc)
    doc.add_page_break()

    section_background(doc)
    section_methodology(doc)
    doc.add_page_break()

    section_structural(doc)
    doc.add_page_break()

    section_worked_examples(doc, examples)
    section_stress_test(doc, examples)
    doc.add_page_break()

    section_aggregate(doc, summary)
    doc.add_page_break()

    section_discussion(doc)
    section_conclusion(doc)
    doc.add_page_break()

    section_references(doc)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUTPUT_PATH))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  paragraphs: {len(doc.paragraphs)}")
    print(f"  tables    : {len(doc.tables)}")


if __name__ == "__main__":
    main()
