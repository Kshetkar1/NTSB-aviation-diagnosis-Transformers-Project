"""
Fallback A2 precompute: parse the existing trace file into data/a2.json.

We can't re-run `precompute_a2.py` because `data/processed/embeddings_train.npy`
has been evicted by macOS iCloud (0 disk blocks even though logically 36MB).
The May 20 trace at
`Testing_Structural_Mapping_Slides/outputs/trace_20100114X11754_a2_full.txt`
already captures the full run with α=2.0 against the 177-train index. We
parse it into the same JSON schema as `precompute_a2.py`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
TRACE_PATH = (
    _HERE.parent
    / "Testing_Structural_Mapping_Slides"
    / "outputs"
    / "trace_20100114X11754_a2_full.txt"
)
OUT_PATH = _HERE / "data" / "a2.json"
EV_ID = "20100114X11754"


def _extract_section(text: str, header_marker: str, next_marker: str) -> str:
    """Extract text between two ============ section markers."""
    pat = re.compile(
        rf"={{5,}}\s*\n{re.escape(header_marker)}\s*\n={{5,}}\s*\n(.*?)(?:\n={{5,}}|\Z)",
        re.DOTALL,
    )
    m = pat.search(text)
    if not m:
        return ""
    section = m.group(1)
    next_pat = re.compile(rf"\n={{5,}}\s*\n{re.escape(next_marker)}", re.DOTALL)
    nm = next_pat.search(section)
    if nm:
        section = section[: nm.start()]
    return section


def parse_trace(text: str) -> dict:
    payload: dict = {
        "mode": "held_out",
        "corpus_size": 177,  # 177 training incidents (paper Section 8.1)
        "ev_id": EV_ID,
    }

    # ----- Query (Step 1) -----
    step1 = _extract_section(text, "STEP 1: Query Text", "STEP 2:")
    m_text = re.search(r"^Text:\s*(.*?)(?:\n\nGround truth|\Z)", step1, re.DOTALL | re.MULTILINE)
    query = m_text.group(1).strip() if m_text else ""
    m_truth = re.search(r"^Ground truth cause:\s*(.+)$", step1, re.MULTILINE)
    truth_text = m_truth.group(1).strip() if m_truth else ""
    payload["query"] = query

    # ----- Retrieval (Step 3) -----
    step3 = _extract_section(text, "STEP 3: Top 50 Matches (cosine similarity)", "STEP 4:")
    retrieval = []
    for line in step3.splitlines():
        m = re.match(r"^(\d+)\s+([\d.]+)\s+(\S+)", line)
        if m:
            retrieval.append({
                "rank": int(m.group(1)),
                "ev_id": m.group(3),
                "source": "incident",
                "score": float(m.group(2)),
                "snippet": "",
            })
    payload["retrieval"] = retrieval

    # ----- A0 cluster table (Step 5) -----
    step5 = _extract_section(
        text,
        "STEP 5: P(Cluster|Query) = avg_sim_j * n_j / sum_k (avg_sim_k * n_k)",
        "STEP 6:",
    )
    a0_clusters = []
    for line in step5.splitlines():
        m = re.match(
            r"^(.+?)\s{2,}(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
            line,
        )
        if m:
            label = m.group(1).strip()
            if label.startswith("Cluster") or label.startswith("-"):
                continue
            a0_clusters.append({
                "cluster": label,
                "n_incidents": int(m.group(2)),
                "avg_similarity": float(m.group(3)),
                "weight": float(m.group(4)),
                "p_k_given_q": float(m.group(5)),
            })

    # ----- A0 chain rule causes (Step 7) -----
    step7 = _extract_section(
        text,
        "STEP 7: Chain Rule — P(Cause|Query) = sum P(Cause|Cluster) * P(Cluster|Query)",
        "STEP 8:",
    )
    a0_causes = []
    for line in step7.splitlines():
        m = re.match(r"^(\d+)\s+([\d.]+)\s+(.+)$", line)
        if m and not line.startswith("Rank"):
            a0_causes.append({
                "rank": int(m.group(1)),
                "probability": float(m.group(2)),
                "cause": m.group(3).strip(),
            })

    # ----- A0 coded distribution (Step 8) -----
    step8 = _extract_section(text, "STEP 8: Map to Zhang's 54 Occurrence Codes", "STEP 9:")
    a0_codes = []
    capture = False
    for line in step8.splitlines():
        if line.strip().startswith("Code") and "Label" in line:
            capture = True
            continue
        if "Sum of coded probabilities" in line:
            capture = False
        if capture:
            m = re.match(r"^(\S+)\s+(.+?)\s{2,}([\d.]+)\s*$", line)
            if m:
                code = m.group(1)
                a0_codes.append({
                    "rank": len(a0_codes) + 1,
                    "code": "—" if code == "?" else code,
                    "label": m.group(2).strip(),
                    "probability": float(m.group(3)),
                })

    # Ground truth code (Step 8 footer)
    m_gt = re.search(
        r"Ground truth maps to:\s*code\s+(\d+)\s*=\s*'([^']+)'\s*\(similarity:\s*([\d.]+)\)",
        step8,
    )
    if m_gt:
        gt_code = m_gt.group(1)
        gt_label = m_gt.group(2)
        gt_sim = float(m_gt.group(3))
    else:
        gt_code, gt_label, gt_sim = "—", "", 0.0

    payload["ground_truth"] = {
        "text": truth_text,
        "code": gt_code,
        "label": gt_label,
        "similarity_to_label": gt_sim,
    }

    # ----- A2 (Step 10) -----
    step10 = _extract_section(
        text, "STEP 10: A2 — Structural Mapping Reranking", "STEP 11:"
    )

    # A2 cluster weights
    a2_clusters = []
    in_a2_clusters = False
    for line in step10.splitlines():
        if "A2 cluster weights P(K|Q)" in line:
            in_a2_clusters = True
            continue
        if in_a2_clusters and line.startswith("Total A2 weight"):
            in_a2_clusters = False
            continue
        if in_a2_clusters:
            m = re.match(
                r"^(.+?)\s{2,}(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
                line,
            )
            if m:
                label = m.group(1).strip()
                if label.startswith("Cluster") or label.startswith("-"):
                    continue
                a2_clusters.append({
                    "cluster": label,
                    "n_incidents": int(m.group(2)),
                    "avg_similarity": float(m.group(3)),
                    "weight": float(m.group(4)),
                    "p_k_given_q": float(m.group(5)),
                })

    # A2 chain rule top 20
    a2_causes = []
    in_a2_causes = False
    for line in step10.splitlines():
        if line.startswith("A2 top 20 causes"):
            in_a2_causes = True
            continue
        if in_a2_causes and line.startswith("A2 sum"):
            in_a2_causes = False
        if in_a2_causes:
            m = re.match(r"^(\d+)\s+([\d.]+)\s+(.+)$", line)
            if m and not line.startswith("Rank"):
                a2_causes.append({
                    "rank": int(m.group(1)),
                    "probability": float(m.group(2)),
                    "cause": m.group(3).strip(),
                })

    # A2 coded distribution
    a2_codes = []
    in_a2_codes = False
    for line in step10.splitlines():
        if line.startswith("A2 top 10 coded probabilities"):
            in_a2_codes = True
            continue
        if in_a2_codes and (line.startswith("A0 top-1") or line.startswith("A2 top-1")):
            in_a2_codes = False
        if in_a2_codes:
            m = re.match(r"^(\S+)\s+(.+?)\s{2,}([\d.]+)\s*$", line)
            if m and not line.startswith("Code"):
                code = m.group(1)
                a2_codes.append({
                    "rank": len(a2_codes) + 1,
                    "code": "—" if code == "?" else code,
                    "label": m.group(2).strip(),
                    "probability": float(m.group(3)),
                })

    a0_top1_code = a0_codes[0]["code"] if a0_codes else "—"
    a2_top1_code = a2_codes[0]["code"] if a2_codes else "—"
    payload["a0"] = {
        "clusters": a0_clusters,
        "ltp_causes": a0_causes[:20],
        "coded_distribution": a0_codes[:15],
        "top1_code": a0_top1_code,
    }
    payload["a2"] = {
        "clusters": a2_clusters,
        "ltp_causes": a2_causes[:20],
        "coded_distribution": a2_codes[:15],
        "top1_code": a2_top1_code,
    }
    payload["hit"] = {
        "a0": a0_top1_code == gt_code,
        "a2": a2_top1_code == gt_code,
    }
    return payload


def main() -> None:
    if not TRACE_PATH.exists():
        raise FileNotFoundError(f"Trace not found: {TRACE_PATH}")
    text = TRACE_PATH.read_text()
    print(f"[a2_from_trace] Parsing {TRACE_PATH.name}...")
    payload = parse_trace(text)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"[a2_from_trace] Saved {OUT_PATH}")
    print(
        f"  retrieval: {len(payload['retrieval'])}, "
        f"A0 clusters: {len(payload['a0']['clusters'])}, "
        f"A2 clusters: {len(payload['a2']['clusters'])}, "
        f"A0 causes: {len(payload['a0']['ltp_causes'])}, "
        f"A2 causes: {len(payload['a2']['ltp_causes'])}, "
        f"A0 codes: {len(payload['a0']['coded_distribution'])}, "
        f"A2 codes: {len(payload['a2']['coded_distribution'])}"
    )
    print(
        f"  ground truth code={payload['ground_truth']['code']}, "
        f"A0 top-1={payload['a0']['top1_code']}, "
        f"A2 top-1={payload['a2']['top1_code']}, "
        f"A0 hit={payload['hit']['a0']}, A2 hit={payload['hit']['a2']}"
    )


if __name__ == "__main__":
    main()
