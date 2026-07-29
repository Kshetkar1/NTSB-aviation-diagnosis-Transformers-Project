#!/usr/bin/env python3
"""Generate a seeded random sample of category-mapping decisions for HAND audit.

The era-fair diagnosis eval (tests/diagnosis_heldout_eval.py) maps legacy
1982-2006 finding subjects to CICTT top-level categories with keyword rules.
Jesse's standard: you must be able to defend every mapping yourself. This
writes a CSV of mapping decisions -- 50 legacy window findings (weighted by
how often each distinct subject occurs) and 25 held-out CICTT findings --
with an empty `verdict` column to fill in by hand (ok / wrong / unsure).

Run:
  /Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 \
      tests/diagnosis_mapping_audit_sample.py
"""
from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_DIR = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "shared" / "code", FROZEN_DIR / "code",
           FROZEN_DIR / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagnosis_heldout_eval import categorize_legacy, categorize_cictt  # noqa: E402

FULL = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset.json"
WINDOW = REPO_ROOT / "shared" / "data" / "processed" / "refined_dataset_1982_2006.json"
OUT = FROZEN_DIR / "outputs" / "mapping_audit_sample.csv"

SEED = 42
N_LEGACY, N_CICTT = 50, 25


def main() -> int:
    full = json.loads(FULL.read_text())
    window_ids = set(json.loads(WINDOW.read_text()).keys())
    rng = random.Random(SEED)

    legacy = []           # (subject, person, count)
    counts: Counter = Counter()
    persons = {}
    for k in window_ids:
        for f in (full.get(k) or {}).get("findings") or []:
            if f.get("Cause_Factor") not in ("C", "F"):
                continue
            s = str(f.get("finding_description") or "").strip()
            if not s:
                continue
            counts[s.lower()] += 1
            persons.setdefault(s.lower(),
                               str(f.get("person_description") or ""))
    subjects = list(counts)
    weights = [counts[s] for s in subjects]
    picked = set()
    while len(picked) < min(N_LEGACY, len(subjects)):
        picked.add(rng.choices(subjects, weights=weights)[0])
    for s in sorted(picked):
        legacy.append((s, persons[s], counts[s]))

    cictt = set()
    for k, inc in full.items():
        if k in window_ids:
            continue
        for f in inc.get("findings") or []:
            if f.get("Cause_Factor") not in ("C", "F"):
                continue
            s = str(f.get("finding_description") or "").strip()
            if s:
                cictt.add(s)
    cictt_sample = rng.sample(sorted(cictt), min(N_CICTT, len(cictt)))

    # Never clobber a coded audit: verdicts are hand-entered and
    # irreplaceable. Regenerate only with --force (writes a fresh template).
    if OUT.exists() and "--force" not in sys.argv:
        content = OUT.read_text()
        if any(v in content for v in (",ok,", ",wrong,", ",unsure,")):
            print(f"REFUSING to overwrite {OUT}: it contains filled-in "
                  "verdicts. Use --force to regenerate a blank template.")
            return 1

    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["era", "finding_subject", "person", "window_count",
                    "mapped_category", "verdict (ok/wrong/unsure)", "notes"])
        for s, p, n in legacy:
            w.writerow(["legacy 1982-2006", s, p, n,
                        categorize_legacy(s, p) or "UNMAPPED", "", ""])
        for s in sorted(cictt_sample):
            w.writerow(["cictt 2007-2019", s, "", "",
                        categorize_cictt(s) or "UNMAPPED", "", ""])
    print(f"wrote {OUT} ({len(legacy)} legacy + {len(cictt_sample)} CICTT rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
