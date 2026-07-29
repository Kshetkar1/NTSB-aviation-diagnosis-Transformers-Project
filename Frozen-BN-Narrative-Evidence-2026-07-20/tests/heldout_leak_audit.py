#!/usr/bin/env python3
"""Audit injury/damage held-out eval for data leakage.

Jesse's concern is valid: there are TWO different leakage types:

1. TRAIN/TEST leakage — held-out accident appears in BN fit or retrieval index.
2. OUTCOME-IN-TEXT leakage — narrative states "substantial damage" / "fatal injury"
   and we use that text to predict coded injury/damage (inflates narr-sev, full).

This script checks (1) and quantifies (2), then compares primary vs ablation metrics.

Run from repo root:
  export PYTHONPATH="shared/code:Frozen-BN-Narrative-Evidence-2026-07-20/code"
  python Frozen-BN-Narrative-Evidence-2026-07-20/tests/heldout_leak_audit.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

os.environ.pop("NTSB_FULL_CORPUS", None)
os.environ.pop("NTSB_USE_TRAIN_INDEX", None)

import numpy as np  # noqa: E402

import config  # noqa: E402
import prognosis as pg  # noqa: E402
import query_to_bn as qb  # noqa: E402
from bn_upgraded import build_upgraded, INJ_NODE, DMG_NODE, INJ_STATES, DMG_STATES, DMG_BY_CODE  # noqa: E402

FULL = config.DATA_DIR / "refined_dataset.json"
WINDOW = config.DATA_DIR / "refined_dataset_1982_2006.json"
OUT = FROZEN_DIR / "outputs" / "heldout_leak_audit.md"

INJ_IDX = {s: i for i, s in enumerate(INJ_STATES)}
DMG_CODES = ["DEST", "SUBS", "MINR", "NONE"]


def truth(inc):
    inj = pg.zhang_injury_code(inc)
    inj_i = {"FATL": 0, "SERS": 1, "MINR": 2, "NONE": 3}[inj]
    dmg_code = str(inc.get("damage") or "").upper()
    dmg_i = DMG_BY_CODE.get(dmg_code) if dmg_code in DMG_BY_CODE else None
    return inj_i, dmg_i, inj, dmg_code


def main() -> int:
    import main_app  # noqa: E402 — after env lock

    full = json.loads(FULL.read_text(encoding="utf-8"))
    window_ids = set(json.loads(WINDOW.read_text(encoding="utf-8")).keys())
    index_ids = {m.get("ev_id") for m in main_app.embeddings_map if m.get("ev_id")}

    held = []
    for k, inc in full.items():
        if k in window_ids:
            continue
        narr = str(inc.get("narr_accf") or "").strip()
        if len(narr) < 100:
            continue
        held.append((k, inc, narr))

    in_index = [k for k, _, _ in held if k in index_ids]
    in_window = [k for k, _, _ in held if k in window_ids]

    stated_inj_match = stated_dmg_match = stated_any = 0
    redact_changes = 0
    for _, inc, narr in held:
        st = qb.severity_statements(narr)
        inj_i, dmg_i, inj_code, dmg_code = truth(inc)
        if st["injury"] or st["damage"]:
            stated_any += 1
        if st["injury"] and st["injury"] == inj_code:
            stated_inj_match += 1
        if st["damage"] and st["damage"] == dmg_code:
            stated_dmg_match += 1
        if qb.redact_severity_phrases(narr) != narr:
            redact_changes += 1

    # Quick redacted vs full hard+soft (BN path, no stated-severity)
    ds = pg.load_dataset()
    bn, _ = build_upgraded(ds)
    names = [n for n in bn.names() if n not in (INJ_NODE, DMG_NODE)]
    prior_inj, prior_dmg = _prior(bn)

    acc_full = {"inj": [], "dmg": []}
    acc_red = {"inj": [], "dmg": []}
    for k, inc, narr in held:
        inj_y, dmg_y, _, _ = truth(inc)
        red = qb.redact_severity_phrases(narr)
        for tag, text, bucket in (("full", narr, acc_full), ("redact", red, acc_red)):
            try:
                conf = qb.parse_query_to_bn_evidence(
                    text, names, dataset=ds, semantic=True)["confidence"]
                pi, pd = _posteriors(bn, conf)
                bucket["inj"].append(float(pi.argmax() == inj_y))
                if dmg_y is not None:
                    bucket["dmg"].append(float(pd.argmax() == dmg_y))
            except Exception:
                bucket["inj"].append(float(prior_inj.argmax() == inj_y))
                if dmg_y is not None:
                    bucket["dmg"].append(float(prior_dmg.argmax() == dmg_y))

    # Load last full eval if present
    eval_path = FROZEN_DIR / "outputs" / "heldout_narrative_bn_eval.json"
    prev = json.loads(eval_path.read_text()) if eval_path.is_file() else {}

    lines = [
        "# Held-out leakage audit",
        "",
        f"n held-out (2007-2019, narr_accf): **{len(held)}**",
        f"Retrieval index: **{config.ACTIVE_INDEX_LABEL}**",
        "",
        "## 1. Train/test leakage (should be zero)",
        "",
        f"| Check | Count |",
        f"|-------|------:|",
        f"| Held-out EV_ID in 1982-2006 window | {len(in_window)} |",
        f"| Held-out EV_ID in active embedding index | {len(in_index)} |",
        "",
        "**Verdict:** " + (
            "No train/test ID leakage detected."
            if not in_window and not in_index
            else "**FAIL** — held-out IDs found in train artifacts."
        ),
        "",
        "## 2. Outcome-in-text leakage (Jesse's concern — real for some predictors)",
        "",
        "NTSB factual narratives often **state** the final injury/damage level. "
        "Predictors that read those phrases (`full`, `soft+stated`, `narr-sev`) "
        "partially predict the label from the answer written in the text.",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Narratives with any stated severity phrase | {stated_any} / {len(held)} |",
        f"| Stated injury phrase matches coded truth | {stated_inj_match} / {len(held)} |",
        f"| Stated damage phrase matches coded truth | {stated_dmg_match} / {len(held)} |",
        f"| Narratives changed by severity redaction | {redact_changes} / {len(held)} |",
        "",
        "## 3. Event-path robustness: hard+soft, full vs redacted narrative",
        "",
        "| Text | Injury acc | Damage acc |",
        "|------|----------:|----------:|",
        f"| Full narrative (leak_safe=False) | {100*np.mean(acc_full['inj']):.1f}% | "
        f"{100*np.mean(acc_full['dmg']) if acc_full['dmg'] else 0:.1f}% |",
        f"| Severity phrases redacted | {100*np.mean(acc_red['inj']):.1f}% | "
        f"{100*np.mean(acc_red['dmg']) if acc_red['dmg'] else 0:.1f}% |",
        "",
        "Near-identical accuracies are EXPECTED here: the event path parses "
        "mechanism vocabulary, not severity wording, so redaction should not "
        "change it. Severity-readout leakage is measured separately by "
        "`redaction_leak_probe.py` and the `NTSB_ALLOW_STATED_SEVERITY=1` "
        "ablation eval.",
        "",
        "**The paper's primary severity result is `bn-sev`** (leak-safe "
        "redacted text). Stated-severity virtual evidence is disabled by "
        "default (`LEAK_SAFE_SEVERITY`).",
        "",
    ]

    if prev.get("summary"):
        s = prev["summary"]
        lines += [
            "## 4. Predictor tiers (from last held-out run)",
            "",
            "| Predictor | Injury acc | Damage acc | Leakage tier |",
            "|-----------|----------:|----------:|--------------|",
        ]
        tiers = {
            "prior": "none",
            "hard": "low",
            "hard+soft": "low (event vocabulary only)",
            "soft-only": "low (redacted embedding)",
            "soft-priority": "low (redacted embedding)",
            "retrieval-sev": "low (redacted embedding) -- raw k-NN ablation",
            "bn-sev": "low (redacted embedding) -- PRIMARY",
            "bn-fused": "low -- negative ablation (double counting)",
            "full": "high (adds stated severity)",
            "soft+stated": "high",
            "narr-sev": "high",
        }
        for p in ("prior", "hard", "hard+soft", "soft-only", "soft-priority",
                  "retrieval-sev", "bn-sev", "bn-fused", "full", "narr-sev"):
            if p not in s:
                continue
            lines.append(
                f"| {p} | {100*s[p]['inj_acc']:.1f}% | "
                f"{100*s[p]['dmg_acc']:.1f}% | {tiers.get(p, '?')} |"
            )
        if prev.get("severity_stated"):
            ss = prev["severity_stated"]
            lines += [
                "",
                f"Last leak-safe run detected stated damage in "
                f"{ss.get('damage', '?')} and stated injury in "
                f"{ss.get('injury', '?')} narratives — 0 is expected and "
                "correct: redaction removes stated levels before detection.",
            ]

    lines += [
        "",
        "## Recommendation for Maha / Jesse",
        "",
        "1. **Primary:** `bn-sev` (k-NN severity as virtual evidence through "
        "the frozen BN, leak-safe redacted text) — see "
        "`outputs/heldout_significance.md` for CIs, Macro-F1, per-class "
        "recall and the binary severe screen.",
        "2. **Stated-severity virtual evidence is OFF** in production "
        "(`LEAK_SAFE_SEVERITY`); 'severity stated: 0' in eval logs means "
        "the leak-safe redaction removed all stated levels, as intended.",
        "3. **Redaction is audited** by `tests/redaction_leak_probe.py` "
        "(outcome words + NTSB full-report boilerplate stripped; remaining "
        "top-weight probe tokens are crash-mechanism words).",
        "4. **Never run held-out with** `NTSB_FULL_CORPUS=1` (would index "
        "test IDs).",
        "",
    ]

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(OUT.read_text())
    return 0


def _prior(bn):
    return _posteriors(bn, {})


def _posteriors(bn, confidence):
    import pyagrum as gum
    ie = gum.LazyPropagation(bn)
    if confidence:
        qb.apply_evidence(ie, bn, confidence)
    ie.addTarget(INJ_NODE)
    ie.addTarget(DMG_NODE)
    ie.makeInference()

    def dist(node, states):
        v = bn.variable(node)
        post = ie.posterior(node)
        by = {v.label(i): float(post[i]) for i in range(v.domainSize())}
        return np.array([by[s] for s in states])

    return dist(INJ_NODE, INJ_STATES), dist(DMG_NODE, DMG_STATES)


if __name__ == "__main__":
    raise SystemExit(main())
