"""Quick test of the diagnosis system"""
import sys
sys.path.append('..')

from main_app import diagnose_incident_and_get_details

print("Testing diagnosis system with one query...")
print("="*70)

query = "turbulence during flight"
print(f"Query: {query}\n")

try:
    scores, details, diagnosis = diagnose_incident_and_get_details(query)
    
    if scores:
        print(f"\n✅ Found {len(scores)} similar incidents")
        
        if diagnosis and diagnosis.get('weighted_causes'):
            print(f"\n✅ Diagnosis successful!")
            print(f"   Analyzed {diagnosis['total_incidents_analyzed']} incidents")
            print(f"   Found {len(diagnosis['weighted_causes'])} potential causes")
            
            print(f"\n🎯 Top 3 Causes:")
            for i, cause_info in enumerate(diagnosis['weighted_causes'][:3], 1):
                prob_pct = cause_info['probability'] * 100
                print(f"   {i}. {prob_pct:.1f}% - {cause_info['cause'][:70]}")
                print(f"      (Found in {cause_info['num_incidents']} incidents)")
            
            print("\n✅ System is working correctly!")
        else:
            print("\n⚠️  No diagnosis results (may be normal if no causal data)")
    else:
        print(f"\n❌ Error: {details}")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("="*70)






















