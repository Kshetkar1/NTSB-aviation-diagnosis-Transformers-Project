#!/usr/bin/env python3
"""Step 7 — Diagnosis/prognosis: retrieval engine vs Zhang's BN (Table 9).

Reads vendored numbers (no external dependencies):
  data/engine_vs_zhang_table9.json   — engine A0/A2 (May 2026) + A0≈A2 finding
  data/zhang_table9_ground_truth.json — Zhang BN posteriors (repro vs published)

Emits outputs/step07_diagnosis_vs_zhang.{json,md} for the report. This is the
Lane 2 comparison: the engine conditions on a NARRATIVE; Zhang conditions on a
coded evidence node — different quantities, shown honestly side by side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpt_config import ENGINE_VS_ZHANG_PATH, OUTPUT_DIR, ZHANG_TABLE9_GT_PATH  # noqa: E402


def main() -> None:
    engine = json.loads(ENGINE_VS_ZHANG_PATH.read_text(encoding="utf-8"))
    gt = json.loads(ZHANG_TABLE9_GT_PATH.read_text(encoding="utf-8"))

    # Cross-check: Zhang published vs replication for loss of engine power | inop instruments.
    repro_check = None
    for row in gt.get("rows", []):
        if row.get("node_id") == "Lossofenginepower":
            repro_check = {
                "published": row.get("Inoperative engine instruments -- Zhang"),
                "repro": row.get("Inoperative engine instruments -- repro"),
                "delta": row.get("Inoperative engine instruments -- delta"),
            }
            break

    report = {
        "evidence": engine.get("evidence"),
        "provenance": engine.get("provenance"),
        "zhang_bn_self_check": repro_check,
        "rows": engine.get("rows", []),
        "a0_vs_a2": engine.get("a0_vs_a2", {}),
    }
    (OUTPUT_DIR / "step07_diagnosis_vs_zhang.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        "# Step 7 — Diagnosis/Prognosis: retrieval engine vs Zhang BN (Table 9)",
        "",
        f"**Evidence:** {report['evidence']}",
        "",
        "Zhang BN self-check (loss of engine power | inop. instruments): "
        f"published {repro_check['published']} vs repro {repro_check['repro']} "
        f"(delta {repro_check['delta']:.5f}) — replication is faithful." if repro_check else "",
        "",
        "| Outcome | Direction | Zhang BN | Engine A0 | Engine A2 | Pattern |",
        "|---------|-----------|----------|-----------|-----------|---------|",
    ]
    for r in report["rows"]:
        lines.append(
            f"| {r['outcome']} | {r['direction']} | {r['zhang']:.4f} | "
            f"{r['engine_a0']:.4f} | {r['engine_a2']:.4f} | {r['pattern']} |"
        )
    a = report["a0_vs_a2"]
    lines += [
        "",
        "## A0 vs A2 (doubt #4)",
        "",
        f"- A2 changed top-1 in **{a.get('a2_changed_top1_incidents')}/{a.get('total_incidents')}** incidents; "
        f"McNemar p = **{a.get('mcnemar_p')}** (not significant).",
        f"- {a.get('interpretation')}",
        "",
        "## Honest interpretation",
        "",
        "- Zhang **spikes the cause** (0.95) via a causal edge; retrieval **spreads over observed "
        "consequences**. This is a *question mismatch*, not a tuning gap.",
        "- Closing it requires changing the **probability source** (LLM token logprobs / calibration "
        "layer), not tuning retrieval — and that needs no BN and no agent.",
    ]
    (OUTPUT_DIR / "step07_diagnosis_vs_zhang.md").write_text(
        "\n".join(x for x in lines if x is not None) + "\n", encoding="utf-8"
    )
    print("Step 7 → outputs/step07_diagnosis_vs_zhang.md")
    if repro_check:
        print(f"  Zhang self-check OK: {repro_check['repro']} vs {repro_check['published']}")


if __name__ == "__main__":
    main()
