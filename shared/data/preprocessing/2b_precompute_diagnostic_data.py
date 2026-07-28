"""
Pre-Computation Script for Diagnostic Data Generation

This script calculates empirical cause frequencies using BOTH structured 
findings AND narrative causes for maximum diagnostic coverage (58% vs 14%).

Process:
1. Load merged_dataset.json (REFINED_DATA_PATH / MERGED_DATA_PATH)
2. Extract causes from BOTH sources:
   - Structured findings (Cause_Factor='C')
   - Narrative causes (narr_cause field with keyword extraction)
3. Calculate P(cause) = Count(cause) / Total_incidents (empirical frequency)
4. Load embeddings_map.json
5. Augment each incident entry with diagnostic data from BOTH sources
6. Save the augmented embeddings_map.json

Note: This uses empirical frequency estimation, NOT Bayesian inference.
"""

import argparse
import json
from collections import Counter, defaultdict
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import (
    REFINED_DATA_PATH,
    TRAIN_MERGED_DATA_PATH,
    EMBEDDINGS_MAP_PATH,
    EMBEDDINGS_MAP_TRAIN_PATH,
    CAUSE_STATS_PATH,
    CAUSE_STATS_TRAIN_PATH,
)

_cli = argparse.ArgumentParser(description="Augment embeddings_map with per-incident diagnostic_data + cause_statistics.")
_cli.add_argument(
    "--train",
    action="store_true",
    help="Use merged_dataset_train.json and embeddings_map_train.json; write train-only outputs (no overwrite of full map).",
)
_args = _cli.parse_args()

_merged_path = TRAIN_MERGED_DATA_PATH if _args.train else REFINED_DATA_PATH
_map_path = EMBEDDINGS_MAP_TRAIN_PATH if _args.train else EMBEDDINGS_MAP_PATH
_cause_stats_path = CAUSE_STATS_TRAIN_PATH if _args.train else CAUSE_STATS_PATH

print("="*70)
print("PRE-COMPUTING DIAGNOSTIC DATA (BOTH SOURCES)" + (" — TRAIN" if _args.train else ""))
print("="*70)

# ============ DYNAMIC CAUSE EXTRACTION FOR NARRATIVES ============
def extract_causes_from_narrative(narr_cause):
    """
    Dynamically extract cause phrases from narrative text without hardcoded categories.
    
    Strategy: Extract sentences/phrases that contain causal language,
    focusing on the actual text rather than mapping to predefined categories.
    
    Returns list of extracted cause phrases (up to 100 chars each).
    """
    if not narr_cause or len(narr_cause.strip()) < 10:
        return []
    
    # Split into sentences
    sentences = re.split(r'[.!?]', narr_cause)
    causes_found = []
    
    # Causal indicator keywords (words that often precede or indicate a cause)
    causal_indicators = [
        'failure', 'failed', 'malfunction', 'error', 'inadequate', 'improper',
        'loss of', 'lack of', 'unable to', 'did not', 'fatigue', 'corrosion',
        'contamination', 'starvation', 'exhaustion', 'defect', 'impairment',
        'deterioration', 'mismanagement', 'oversight', 'omission', 'collision',
        'strike', 'fracture', 'separation', 'rupture', 'overload', 'overheat'
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 15:
            continue
        
        sentence_lower = sentence.lower()
        
        # Check if sentence contains causal language
        for indicator in causal_indicators:
            if indicator in sentence_lower:
                # Extract a reasonable-length cause phrase
                cause_phrase = sentence[:100] if len(sentence) > 100 else sentence
                causes_found.append(cause_phrase)
                break  # Only add each sentence once
    
    # If no causes found via indicators, use the first sentence as the cause
    # (narratives typically start with the primary cause)
    if not causes_found and sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) >= 15:
            causes_found.append(first_sentence[:100])
    
    return causes_found

# ============ STEP 1: LOAD DATA ============
print("\n📂 Loading data files...")

with open(_merged_path, 'r') as f:
    refined_data = json.load(f)
print(f"✅ Loaded {len(refined_data)} incidents from merged dataset ({_merged_path.name})")

with open(_map_path, 'r') as f:
    embeddings_map = json.load(f)
print(f"✅ Loaded {len(embeddings_map)} entries from {_map_path.name}")

# ============ STEP 2: EXTRACT AND COUNT ALL CAUSES (BOTH SOURCES) ============
print("\n🔍 Extracting root causes from BOTH structured findings AND narratives...")

cause_counter = Counter()
finding_counter = Counter()
narrative_cause_counter = Counter()
combined_cause_counter = Counter()
incidents_per_cause = defaultdict(list)  # Track which incidents have each cause

structured_count = 0
narrative_count = 0
both_count = 0

for ev_id, incident in refined_data.items():
    has_structured = False
    has_narrative = False
    
    # SOURCE 1: Extract from structured findings (marked with 'C' for Cause)
    findings = incident.get('findings', [])
    for finding in findings:
        desc = finding.get('finding_description', 'Unknown')
        factor = finding.get('Cause_Factor', '')
        if factor:
            factor = factor.strip()
        
        # Count all findings
        finding_counter[desc] += 1
        
        # Track causal findings
        if factor == 'C':
            cause_counter[desc] += 1
            combined_cause_counter[desc] += 1
            incidents_per_cause[desc].append(ev_id)
            has_structured = True
    
    # SOURCE 2: Extract from narrative causes
    narr_cause = incident.get('narr_cause', '')
    if narr_cause:
        narrative_causes = extract_causes_from_narrative(narr_cause)
        for cause in narrative_causes:
            narrative_cause_counter[cause] += 1
            combined_cause_counter[cause] += 1
            incidents_per_cause[cause].append(ev_id)
            has_narrative = True
    
    # Track coverage statistics
    if has_structured and has_narrative:
        both_count += 1
    elif has_structured:
        structured_count += 1
    elif has_narrative:
        narrative_count += 1

total_with_causes = structured_count + narrative_count + both_count
coverage_pct = (total_with_causes / len(refined_data)) * 100

print(f"\n📊 Cause Statistics (BOTH SOURCES):")
print(f"  Structured findings only: {structured_count} incidents ({structured_count/len(refined_data)*100:.1f}%)")
print(f"  Narrative causes only: {narrative_count} incidents ({narrative_count/len(refined_data)*100:.1f}%)")
print(f"  Both sources: {both_count} incidents ({both_count/len(refined_data)*100:.1f}%)")
print(f"  TOTAL COVERAGE: {total_with_causes} incidents ({coverage_pct:.1f}%)")
print(f"  Unique causal findings (structured): {len(cause_counter)}")
print(f"  Unique narrative causes: {len(narrative_cause_counter)}")
print(f"  Combined unique causes: {len(combined_cause_counter)}")

# ============ STEP 3: CALCULATE BASE PROBABILITIES (COMBINED) ============
print("\n🧮 Calculating base probabilities from combined sources...")

total_incidents = len(refined_data)
cause_probabilities = {}

for cause, count in combined_cause_counter.items():
    cause_probabilities[cause] = count / total_incidents

print(f"✅ Calculated probabilities for {len(cause_probabilities)} combined causes")

# Show top 10 most common causes from COMBINED sources
print(f"\n🏆 Top 10 Most Probable Causes (COMBINED):")
for i, (cause, prob) in enumerate(sorted(cause_probabilities.items(), 
                                          key=lambda x: x[1], 
                                          reverse=True)[:10], 1):
    count = combined_cause_counter[cause]
    print(f"  {i:2d}. P={prob:.4f} ({count:3d}x) {cause[:60]}")

# ============ STEP 4: AUGMENT EMBEDDINGS MAP ============
print("\n🔧 Augmenting embeddings_map with diagnostic data...")

augmented_count = 0
skipped_count = 0

for i, entry in enumerate(embeddings_map):
    # Only augment incident entries (not dictionary entries)
    if entry.get('source') != 'incident':
        continue
    
    ev_id = entry.get('ev_id')
    if not ev_id or ev_id not in refined_data:
        skipped_count += 1
        continue
    
    incident = refined_data[ev_id]
    
    # Extract diagnostic data
    findings_list = []
    cause_factors = []
    causal_findings = []
    
    # SOURCE 1: Structured findings
    for finding in incident.get('findings', []):
        desc = finding.get('finding_description', '')
        factor = finding.get('Cause_Factor', '')
        if factor:
            factor = factor.strip()
        
        if desc:
            findings_list.append(desc)
        if factor:
            cause_factors.append(factor)
        if factor == 'C' and desc:
            causal_findings.append(desc)
    
    # SOURCE 2: Narrative causes
    narr_cause = incident.get('narr_cause', '')
    narrative_extracted_causes = []
    if narr_cause:
        narrative_extracted_causes = extract_causes_from_narrative(narr_cause)
    
    # COMBINED: Union of both sources
    all_causes = list(set(causal_findings + narrative_extracted_causes))
    
    # Create diagnostic_data structure with BOTH sources
    diagnostic_data = {
        'findings': findings_list,
        'cause_factors': cause_factors,
        'causal_findings': causal_findings,  # Structured findings marked with 'C'
        'narrative_causes': narrative_extracted_causes,  # Extracted from narr_cause
        'all_causes': all_causes,  # Combined from both sources
        'narr_cause': narr_cause,
        'has_diagnostic_data': len(all_causes) > 0
    }
    
    # Add base probabilities for this incident's causes (from combined)
    if all_causes:
        diagnostic_data['cause_base_probs'] = {
            cause: cause_probabilities.get(cause, 0.0)
            for cause in all_causes
        }
    
    # Attach to entry
    entry['diagnostic_data'] = diagnostic_data
    augmented_count += 1

print(f"✅ Augmented {augmented_count} incident entries")
print(f"   Skipped {skipped_count} entries (not incidents or not found)")

# ============ STEP 5: SAVE AUGMENTED DATA ============
print("\n💾 Saving augmented embeddings_map...")

# Create backup of target map
if _map_path.exists():
    backup_path = _map_path.with_suffix('.json.backup')
    with open(_map_path, 'r') as f_in:
        with open(backup_path, 'w') as f_out:
            f_out.write(f_in.read())
    print(f"📋 Created backup: {backup_path}")

# Save augmented version
with open(_map_path, 'w') as f:
    json.dump(embeddings_map, f, indent=2)

print(f"✅ Saved augmented {_map_path.name}")

# ============ STEP 6: SAVE GLOBAL STATISTICS ============
print("\n📊 Saving global cause statistics...")

global_stats = {
    'total_incidents': total_incidents,
    'unique_causes': len(cause_counter),
    'total_causal_findings': sum(cause_counter.values()),
    'cause_probabilities': cause_probabilities,
    'cause_counts': dict(cause_counter),
    'incidents_per_cause': {k: len(v) for k, v in incidents_per_cause.items()}
}

with open(_cause_stats_path, 'w') as f:
    json.dump(global_stats, f, indent=2)

print(f"✅ Saved global statistics to {_cause_stats_path.name}")

# ============ VERIFICATION ============
print("\n" + "="*70)
print("VERIFICATION - COMBINED SOURCES")
print("="*70)

# Check a few augmented entries showing both sources
print("\n🔍 Sample augmented entries (showing BOTH sources):")
sample_count = 0
for entry in embeddings_map:
    if entry.get('source') == 'incident' and entry.get('diagnostic_data', {}).get('has_diagnostic_data'):
        ev_id = entry.get('ev_id')
        bd = entry['diagnostic_data']
        print(f"\n  Incident: {ev_id}")
        print(f"    Structured findings: {len(bd.get('causal_findings', []))}")
        if bd.get('causal_findings'):
            print(f"      Example: {bd['causal_findings'][0][:60]}")
        print(f"    Narrative causes: {len(bd.get('narrative_causes', []))}")
        if bd.get('narrative_causes'):
            print(f"      Extracted: {', '.join(bd['narrative_causes'])}")
        print(f"    COMBINED total causes: {len(bd.get('all_causes', []))}")
        if bd.get('narr_cause'):
            narr_preview = bd['narr_cause'][:80] + "..." if len(bd['narr_cause']) > 80 else bd['narr_cause']
            print(f"    Original narrative: {narr_preview}")
        
        sample_count += 1
        if sample_count >= 3:
            break

print("\n" + "="*70)
print("✅ PRE-COMPUTATION COMPLETE!")
print("="*70)
print("\nNext steps:")
print("  1. Implement weighted diagnosis in main_app.py")
print("  2. Update Streamlit UI to display diagnostic results")
print("  3. Test with sample queries")

