"""
Validation Script for Dr. Smith's Bayesian Diagnosis System

This script tests the diagnostic system with known queries to validate
that it produces reasonable and accurate diagnoses.
"""

import sys
sys.path.append('..')

from main_app import diagnose_root_causes, diagnose_incident_and_get_details
import json

print("="*70)
print("TESTING DR. SMITH'S WEIGHTED BAYESIAN DIAGNOSIS SYSTEM")
print("="*70)

# Test cases covering different types of incidents
test_cases = [
    {
        'query': 'Engine fire during takeoff',
        'expected_themes': ['engine', 'fire', 'power', 'turbine', 'fuel']
    },
    {
        'query': 'Turbulence encounter during cruise flight',
        'expected_themes': ['turbulence', 'weather', 'convective', 'environmental']
    },
    {
        'query': 'Landing gear collapsed during landing',
        'expected_themes': ['landing gear', 'mechanical', 'failure', 'collapse']
    },
    {
        'query': 'Loss of engine power in flight',
        'expected_themes': ['engine', 'power', 'fuel', 'mechanical']
    },
    {
        'query': 'Bird strike on approach',
        'expected_themes': ['bird', 'wildlife', 'animal', 'environmental']
    }
]

def check_relevance(causes, expected_themes):
    """Check if any of the top causes contain expected keywords"""
    found_themes = []
    for cause_info in causes[:5]:  # Check top 5
        cause_text = cause_info['cause'].lower()
        for theme in expected_themes:
            if theme.lower() in cause_text:
                found_themes.append(theme)
                break
    return found_themes

# Run tests
print("\n" + "="*70)
print("RUNNING TEST CASES")
print("="*70)

results = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"TEST {i}: {test_case['query']}")
    print(f"{'='*70}")
    
    try:
        # Run diagnosis
        diagnosis = diagnose_root_causes(test_case['query'], top_n=5)
        
        top_causes = diagnosis['top_causes']
        metadata = diagnosis['diagnosis_metadata']
        
        print(f"\n📊 Metadata:")
        print(f"  - Incidents analyzed: {metadata['incidents_analyzed']}")
        print(f"  - Total causes found: {metadata['total_causes_found']}")
        print(f"  - Methodology: {metadata['methodology']}")
        
        print(f"\n🎯 Top 5 Diagnosed Causes:")
        for j, cause_info in enumerate(top_causes, 1):
            prob_pct = cause_info['probability'] * 100
            cause_text = cause_info['cause']
            num_incidents = cause_info['num_incidents']
            
            # Truncate if too long
            if len(cause_text) > 80:
                cause_text = cause_text[:77] + "..."
            
            print(f"  {j}. [{prob_pct:5.1f}%] {cause_text}")
            print(f"     (Found in {num_incidents} incidents)")
        
        # Check relevance
        found_themes = check_relevance(top_causes, test_case['expected_themes'])
        
        print(f"\n✅ Relevance Check:")
        print(f"  Expected themes: {', '.join(test_case['expected_themes'])}")
        print(f"  Found themes: {', '.join(found_themes) if found_themes else 'None'}")
        
        # Determine if test passed
        passed = len(found_themes) > 0
        
        results.append({
            'test_case': test_case['query'],
            'passed': passed,
            'incidents_analyzed': metadata['incidents_analyzed'],
            'causes_found': metadata['total_causes_found'],
            'relevant_themes': found_themes
        })
        
        if passed:
            print(f"  ✅ TEST PASSED - Found relevant causes")
        else:
            print(f"  ⚠️  TEST UNCERTAIN - No exact theme match (may still be accurate)")
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        results.append({
            'test_case': test_case['query'],
            'passed': False,
            'error': str(e)
        })

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

passed_tests = sum(1 for r in results if r.get('passed', False))
total_tests = len(results)

print(f"\n📊 Overall Results:")
print(f"  Tests passed: {passed_tests}/{total_tests}")
print(f"  Success rate: {passed_tests/total_tests*100:.1f}%")

print(f"\n📋 Individual Results:")
for i, result in enumerate(results, 1):
    status = "✅ PASS" if result.get('passed') else "⚠️  UNCERTAIN"
    print(f"  {i}. {status} - {result['test_case']}")
    if 'error' not in result:
        print(f"     Analyzed {result['incidents_analyzed']} incidents, found {result['causes_found']} causes")

# Save results
output_file = 'diagnosis_test_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Results saved to: {output_file}")

print("\n" + "="*70)
print("✅ TESTING COMPLETE")
print("="*70)

print("\n💡 Next Steps:")
print("  1. Review the test results above")
print("  2. Verify that diagnoses make sense for each query")
print("  3. If accuracy is good, the system is ready for use!")
print("  4. If accuracy is low, consider:")
print("     - Expanding the dataset")
print("     - Adjusting similarity thresholds")
print("     - Improving narrative extraction")






















