"""
Data Analysis Script for Dr. Smith's Bayesian Diagnosis System

This script analyzes the NTSB refined dataset to understand:
1. How many incidents have findings/causes documented
2. What types of findings exist
3. How rich the causal data is
4. Distribution of cause factors

This validates we have sufficient root cause data for diagnosis.
"""

import json
from collections import Counter, defaultdict

print("="*70)
print("NTSB DATA ANALYSIS FOR BAYESIAN DIAGNOSIS SYSTEM")
print("="*70)

# Load the refined dataset
print("\nLoading refined_dataset.json...")
with open('refined_dataset.json', 'r') as f:
    refined_data = json.load(f)

print(f"✅ Loaded {len(refined_data)} incidents")

# ============ PART 1: OVERVIEW ============
print("\n" + "="*70)
print("PART 1: DATASET OVERVIEW")
print("="*70)

total_incidents = len(refined_data)
incidents_with_narratives = 0
incidents_with_narr_cause = 0
incidents_with_narr_accf = 0
incidents_with_findings = 0
incidents_with_sequences = 0

for ev_id, incident in refined_data.items():
    if incident.get('narr_cause'):
        incidents_with_narr_cause += 1
    if incident.get('narr_accf'):
        incidents_with_narr_accf += 1
    if incident.get('narr_cause') or incident.get('narr_accf'):
        incidents_with_narratives += 1
    if incident.get('findings') and len(incident['findings']) > 0:
        incidents_with_findings += 1
    if incident.get('sequence_of_events') and len(incident['sequence_of_events']) > 0:
        incidents_with_sequences += 1

print(f"\n📊 Dataset Statistics:")
print(f"  Total incidents: {total_incidents}")
print(f"  Incidents with narratives: {incidents_with_narratives} ({incidents_with_narratives/total_incidents*100:.1f}%)")
print(f"  Incidents with narr_cause: {incidents_with_narr_cause} ({incidents_with_narr_cause/total_incidents*100:.1f}%)")
print(f"  Incidents with narr_accf: {incidents_with_narr_accf} ({incidents_with_narr_accf/total_incidents*100:.1f}%)")
print(f"  Incidents with findings: {incidents_with_findings} ({incidents_with_findings/total_incidents*100:.1f}%)")
print(f"  Incidents with event sequences: {incidents_with_sequences} ({incidents_with_sequences/total_incidents*100:.1f}%)")

# ============ PART 2: FINDINGS ANALYSIS ============
print("\n" + "="*70)
print("PART 2: FINDINGS AND CAUSE FACTORS ANALYSIS")
print("="*70)

finding_descriptions = Counter()
cause_factors = Counter()
findings_per_incident = []
causal_findings = []  # Findings marked with "C" for Cause

for ev_id, incident in refined_data.items():
    findings = incident.get('findings', [])
    findings_per_incident.append(len(findings))
    
    for finding in findings:
        desc = finding.get('finding_description', 'Unknown')
        factor = finding.get('Cause_Factor', '')
        if factor:
            factor = factor.strip()
        
        finding_descriptions[desc] += 1
        if factor:
            cause_factors[factor] += 1
        
        # Track causal findings (those marked with "C")
        if factor == 'C':
            causal_findings.append(desc)

print(f"\n📈 Findings Statistics:")
print(f"  Total unique finding descriptions: {len(finding_descriptions)}")
print(f"  Total findings across all incidents: {sum(finding_descriptions.values())}")
print(f"  Average findings per incident: {sum(findings_per_incident)/len(findings_per_incident):.2f}")

print(f"\n🎯 Cause Factor Distribution:")
for factor, count in cause_factors.most_common():
    print(f"  '{factor}': {count} occurrences")

print(f"\n🏆 Top 20 Most Common Finding Descriptions:")
for i, (desc, count) in enumerate(finding_descriptions.most_common(20), 1):
    pct = count / incidents_with_findings * 100
    print(f"  {i:2d}. {desc[:70]}")
    print(f"      Count: {count} ({pct:.1f}% of incidents with findings)")

# ============ PART 3: CAUSAL FINDINGS ANALYSIS ============
print("\n" + "="*70)
print("PART 3: CAUSAL FINDINGS (Marked with 'C')")
print("="*70)

causal_counter = Counter(causal_findings)
print(f"\n📌 Causal Findings Statistics:")
print(f"  Total causal findings (factor='C'): {len(causal_findings)}")
print(f"  Unique causal finding types: {len(causal_counter)}")

print(f"\n🔍 Top 15 Most Common ROOT CAUSES:")
for i, (cause, count) in enumerate(causal_counter.most_common(15), 1):
    prob = count / total_incidents
    print(f"  {i:2d}. {cause[:65]}")
    print(f"      Count: {count}, P(cause) = {prob:.4f} ({prob*100:.2f}%)")

# ============ PART 4: NARRATIVE CAUSE ANALYSIS ============
print("\n" + "="*70)
print("PART 4: NARRATIVE CAUSE (narr_cause) ANALYSIS")
print("="*70)

# Sample a few narr_cause to understand the format
sample_causes = []
for ev_id, incident in refined_data.items():
    if incident.get('narr_cause'):
        sample_causes.append(incident['narr_cause'])
        if len(sample_causes) >= 3:
            break

print(f"\n📝 Sample Narrative Causes (first 3):")
for i, cause in enumerate(sample_causes, 1):
    truncated = cause[:200] + "..." if len(cause) > 200 else cause
    print(f"\n  {i}. {truncated}")

# ============ PART 5: DATA QUALITY ASSESSMENT ============
print("\n" + "="*70)
print("PART 5: DATA QUALITY FOR DIAGNOSIS")
print("="*70)

# Check how many incidents have BOTH findings AND narratives
both_count = 0
for ev_id, incident in refined_data.items():
    has_findings = incident.get('findings') and len(incident['findings']) > 0
    has_narrative = incident.get('narr_cause') or incident.get('narr_accf')
    if has_findings and has_narrative:
        both_count += 1

print(f"\n✅ Data Richness:")
print(f"  Incidents with BOTH findings AND narratives: {both_count} ({both_count/total_incidents*100:.1f}%)")
print(f"  Incidents with ONLY findings: {incidents_with_findings - both_count}")
print(f"  Incidents with ONLY narratives: {incidents_with_narratives - both_count}")
print(f"  Incidents with NEITHER: {total_incidents - incidents_with_findings - incidents_with_narratives + both_count}")

# ============ PART 6: RECOMMENDATION ============
print("\n" + "="*70)
print("RECOMMENDATION FOR DR. SMITH'S APPROACH")
print("="*70)

narrative_rich = incidents_with_narratives / total_incidents > 0.5
findings_rich = incidents_with_findings / total_incidents > 0.3
causal_data_available = len(causal_counter) > 20

print(f"\n💡 Assessment:")
print(f"  Narrative-rich dataset: {narrative_rich} ({incidents_with_narratives/total_incidents*100:.1f}% have narratives)")
print(f"  Findings-rich dataset: {findings_rich} ({incidents_with_findings/total_incidents*100:.1f}% have findings)")
print(f"  Diverse causal data: {causal_data_available} ({len(causal_counter)} unique causes identified)")

print(f"\n🎯 Recommendation:")
if narrative_rich and findings_rich:
    print("  ✅ EXCELLENT DATA FOR DR. SMITH'S APPROACH!")
    print("  Your dataset has:")
    print("    - Rich narratives for semantic embeddings")
    print("    - Detailed findings for root cause extraction")
    print("    - Diverse set of documented causes")
    print("\n  Implementation Strategy:")
    print("    1. Use embeddings from narratives to find similar incidents")
    print("    2. Extract findings and cause factors from similar incidents")
    print("    3. Weight causes by cosine similarity")
    print("    4. Return probabilistic diagnosis")
elif narrative_rich:
    print("  ✅ GOOD DATA - Focus on narrative-based diagnosis")
    print("  Use narr_cause field primarily, supplement with findings where available")
elif findings_rich:
    print("  ✅ GOOD DATA - Focus on findings-based diagnosis")
    print("  Use findings and cause factors primarily")
else:
    print("  ⚠️  LIMITED DATA - May need to supplement with other approaches")

print("\n" + "="*70)
print("Analysis complete! Ready to proceed with implementation.")
print("="*70)

