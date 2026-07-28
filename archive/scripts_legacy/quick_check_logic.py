
from main_app import diagnose_with_conditional_probabilities
import sys

try:
    print("Testing Diagnosis Logic...")
    result = diagnose_with_conditional_probabilities("engine fire during takeoff", top_n=5, top_n_incidents=10)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print("\nSuccess! Top Causes:")
        for cause in result['weighted_causes'][:3]:
            print(f"- {cause['cause']}: {cause['probability']:.4f}")
            if cause.get('cluster_breakdown'):
                print("  Breakdown available")
                
except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
    import traceback
    traceback.print_exc()
