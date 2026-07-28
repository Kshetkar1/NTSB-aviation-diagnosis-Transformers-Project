"""
Cause Normalization: Before vs After
======================================
Shows the concrete effect of normalizing cause strings
(.strip().lower()) in the cluster analysis function.

Before fix: "Engine Failure" and "engine failure" and "engine failure "
            are counted as 3 different causes.

After fix:  all three map to "engine failure" → counted as 1 cause
            with frequency 3.

The source of truth is embeddings_map.json, which is what
calculate_cause_probabilities_per_cluster() reads at runtime.

Output:
  tests/05_cause_normalization_proof.html

Run from project root:
    python tests/05_cause_normalization_proof.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from collections import Counter
from pathlib import Path
from config import EMBEDDINGS_MAP_PATH

OUT_HTML = Path("tests/05_cause_normalization_proof.html")

print("Loading embeddings map ...", flush=True)
with open(EMBEDDINGS_MAP_PATH) as f:
    emb_map = json.load(f)

# Collect all raw cause strings from all_causes in the embeddings map
# (this is exactly what calculate_cause_probabilities_per_cluster reads)
raw_causes = []
for m in emb_map:
    dd = m.get("diagnostic_data") or m.get("bayesian_data", {})
    if dd:
        for c in dd.get("all_causes", []):
            if c and c.strip():
                raw_causes.append(c)

print(f"  {len(raw_causes):,} total cause strings found in embeddings map", flush=True)

# ── BEFORE: no normalization ────────────────────────────────────────────────
counter_before = Counter(raw_causes)
unique_before  = len(counter_before)

# ── AFTER: strip + lower ────────────────────────────────────────────────────
normalized     = [c.strip().lower() for c in raw_causes]
counter_after  = Counter(normalized)
unique_after   = len(counter_after)

duplicates_collapsed = unique_before - unique_after
pct_reduction = duplicates_collapsed / unique_before * 100

print(f"\nBEFORE normalization: {unique_before:,} unique cause strings", flush=True)
print(f"AFTER  normalization: {unique_after:,} unique cause strings", flush=True)
print(f"Collapsed {duplicates_collapsed:,} duplicate variants ({pct_reduction:.1f}% reduction)", flush=True)

# Find the most dramatic examples (same base cause, many case/whitespace variants)
from collections import defaultdict
variants_map = defaultdict(list)
for raw in counter_before:
    norm = raw.strip().lower()
    variants_map[norm].append(raw)

# Keep only causes that had multiple variants (case/whitespace differences)
multi_variants = {
    norm: variants
    for norm, variants in variants_map.items()
    if len(variants) > 1
}

print(f"\nCauses with multiple case/whitespace variants: {len(multi_variants):,}", flush=True)

# Top 20 most-collapsed causes by total count gain
def count_gain(norm, variants):
    total = sum(counter_before[v] for v in variants)
    return total

top_examples = sorted(
    multi_variants.items(),
    key=lambda x: count_gain(x[0], x[1]),
    reverse=True
)[:20]

# ── Build HTML ───────────────────────────────────────────────────────────────
example_rows = ""
for norm, variants in top_examples:
    total_count = sum(counter_before[v] for v in variants)
    after_count = counter_after[norm]
    variant_list = "".join(
        f'<li style="font-size:11px;color:#8b949e;font-family:monospace">'
        f'{repr(v)} <span style="color:#58a6ff">×{counter_before[v]}</span></li>'
        for v in sorted(variants)
    )
    example_rows += f"""
    <tr>
      <td style="font-family:monospace;font-size:12px;color:#79c0ff">{norm}</td>
      <td style="text-align:center;color:#e74c3c">{len(variants)}</td>
      <td style="text-align:center;color:#2ecc71;font-weight:bold">{after_count}</td>
      <td><ul style="margin:0;padding-left:14px">{variant_list}</ul></td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Cause Normalization: Before vs After</title>
  <style>
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ font-family:'Segoe UI',Arial,sans-serif; background:#0d1117;
            color:#c9d1d9; padding:32px; max-width:1100px; margin:0 auto; }}
    h1   {{ color:#58a6ff; font-size:20px; margin-bottom:6px; }}
    .sub {{ color:#8b949e; font-size:13px; margin-bottom:28px; line-height:1.7; }}
    h2   {{ color:#e6edf3; font-size:15px; margin:28px 0 12px; }}

    .stats {{
      display:flex; gap:20px; margin-bottom:32px; flex-wrap:wrap;
    }}
    .stat-box {{
      flex:1; min-width:180px; background:#161b22; border-radius:8px;
      padding:18px 22px; border-top:3px solid var(--c);
    }}
    .stat-val  {{ font-size:28px; font-weight:bold; color:var(--c); }}
    .stat-label{{ font-size:12px; color:#8b949e; margin-top:4px; }}

    table {{ width:100%; border-collapse:collapse; }}
    th {{ padding:10px 14px; background:#161b22; color:#58a6ff;
          font-size:12px; border-bottom:2px solid #30363d; text-align:left; }}
    td {{ padding:10px 14px; border-bottom:1px solid #21262d;
          vertical-align:top; font-size:13px; }}
    tr:hover td {{ background:#161b22; }}

    .verdict {{
      background:#161b22; border-left:4px solid #2ecc71;
      padding:18px 22px; border-radius:6px; margin-bottom:28px;
    }}
    .verdict b {{ color:#2ecc71; }}
  </style>
</head>
<body>
  <h1>Cause Normalization: Before vs After</h1>
  <p class="sub">
    When counting cause frequencies in cluster analysis, the <b>raw cause strings</b>
    from NTSB findings contain case and whitespace inconsistencies.<br>
    Without normalization, the same cause is counted as multiple distinct causes,
    diluting its true frequency and distorting the probability calculation.<br>
    Fix: <code>.strip().lower()</code> applied before counting.
  </p>

  <div class="verdict">
    Applying <code>.strip().lower()</code> collapsed
    <b>{duplicates_collapsed:,} duplicate cause variants</b> ({pct_reduction:.1f}% reduction),
    reducing unique cause strings from <b>{unique_before:,} → {unique_after:,}</b>
    across {len(raw_causes):,} total cause/finding records in the dataset.
  </div>

  <div class="stats">
    <div class="stat-box" style="--c:#e74c3c">
      <div class="stat-val">{unique_before:,}</div>
      <div class="stat-label">Unique causes BEFORE<br>(no normalization)</div>
    </div>
    <div class="stat-box" style="--c:#2ecc71">
      <div class="stat-val">{unique_after:,}</div>
      <div class="stat-label">Unique causes AFTER<br>(strip + lowercase)</div>
    </div>
    <div class="stat-box" style="--c:#f39c12">
      <div class="stat-val">{duplicates_collapsed:,}</div>
      <div class="stat-label">Duplicate variants collapsed<br>({pct_reduction:.1f}% reduction)</div>
    </div>
    <div class="stat-box" style="--c:#58a6ff">
      <div class="stat-val">{len(multi_variants):,}</div>
      <div class="stat-label">Causes with multiple<br>case/whitespace variants</div>
    </div>
  </div>

  <h2>Top 20 Most-Affected Causes (with their raw variants)</h2>
  <p style="font-size:12px;color:#8b949e;margin-bottom:12px">
    Each row shows one normalized cause, how many raw variants existed,
    the consolidated count after normalization, and the actual variant strings.
  </p>
  <table>
    <thead>
      <tr>
        <th style="width:35%">Normalized Cause</th>
        <th style="width:10%;text-align:center">Variants<br>(before)</th>
        <th style="width:10%;text-align:center">Count<br>(after)</th>
        <th>Raw Variant Strings</th>
      </tr>
    </thead>
    <tbody>
      {example_rows}
    </tbody>
  </table>
</body>
</html>"""

OUT_HTML.write_text(html, encoding="utf-8")
print(f"\nSaved: {OUT_HTML}", flush=True)

import subprocess
try:
    subprocess.Popen(["open", str(OUT_HTML)])
except Exception:
    import webbrowser
    webbrowser.open(OUT_HTML.resolve().as_uri())

print("Done.", flush=True)
