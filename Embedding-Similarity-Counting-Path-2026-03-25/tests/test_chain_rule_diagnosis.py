"""
Test script for Conditional Probability Diagnosis (Chain Rule approach).

This tests the implementation that:
1. Clusters top 50 incidents by type (using LLM)
2. Calculates P(Cause|Cluster) from historical frequencies
3. Applies chain rule: P(Cause|Query) = P(Cause|Cluster) × P(Cluster|Query)

Note: This is NOT Bayesian inference - it uses empirical conditional probabilities.
"""

import sys
from pathlib import Path

# Add parent directory to path to import main_app
sys.path.insert(0, str(Path(__file__).parent.parent))

from main_app import (
    diagnose_with_conditional_probabilities,
    calculate_similarity_weighted_diagnosis,
    get_embedding,
    find_top_matches
)


def test_chain_rule_vs_similarity_weighted(query):
    """
    Compare conditional probability approach with similarity-weighted approach.
    """
    print("=" * 80)
    print(f"COMPARISON TEST: '{query}'")
    print("=" * 80)
    
    # Method 1: Conditional Probability (Chain Rule)
    print("\n" + "─" * 80)
    print("METHOD 1: CONDITIONAL PROBABILITY DIAGNOSIS (CHAIN RULE)")
    print("─" * 80)
    
    chain_rule_results = diagnose_with_conditional_probabilities(query, top_n=10, top_n_incidents=50)
    
    if chain_rule_results.get('weighted_causes'):
        print(f"\n✅ Results from {chain_rule_results['total_incidents_analyzed']} incidents in {chain_rule_results['total_clusters']} clusters:")
        print(f"\nTop 10 Most Likely Causes:")
        for i, cause_info in enumerate(chain_rule_results['weighted_causes'][:10], 1):
            prob_pct = cause_info['probability'] * 100
            cause_text = cause_info['cause'][:80]
            print(f"\n{i}. {prob_pct:.2f}% - {cause_text}")
            print(f"   Found in {cause_info['num_incidents']} incidents across {cause_info['num_clusters']} clusters")
            
            # Show cluster breakdown
            if cause_info.get('cluster_breakdown'):
                print(f"   Top contributing clusters:")
                for cluster_info in cause_info['cluster_breakdown'][:2]:
                    contribution_pct = cluster_info['contribution'] * 100
                    print(f"     • '{cluster_info['type']}': +{contribution_pct:.2f}%")
    else:
        print("❌ No causes found using chain rule method")
    
    # Method 2: Similarity-Weighted Approach
    print("\n" + "─" * 80)
    print("METHOD 2: SIMILARITY-WEIGHTED DIAGNOSIS")
    print("─" * 80)
    
    query_embedding = get_embedding(query)
    top_scores, top_matches = find_top_matches(query_embedding)
    similarity_results = calculate_similarity_weighted_diagnosis(top_scores, top_matches, top_n_incidents=50)
    
    if similarity_results.get('weighted_causes'):
        print(f"\n✅ Results from {similarity_results['total_incidents_analyzed']} incidents:")
        print(f"\nTop 10 Most Likely Causes:")
        for i, cause_info in enumerate(similarity_results['weighted_causes'][:10], 1):
            prob_pct = cause_info['probability'] * 100
            cause_text = cause_info['cause'][:80]
            print(f"\n{i}. {prob_pct:.2f}% - {cause_text}")
            print(f"   Found in {cause_info['num_incidents']} similar incidents")
    else:
        print("❌ No causes found using similarity-weighted method")
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)
    
    return chain_rule_results, similarity_results


def test_conditional_probability(query):
    """
    Test only the conditional probability approach with detailed output.
    """
    print("=" * 80)
    print(f"CONDITIONAL PROBABILITY DIAGNOSIS TEST: '{query}'")
    print("=" * 80)
    
    results = diagnose_with_conditional_probabilities(query, top_n=15, top_n_incidents=50)
    
    if results.get('error'):
        print(f"❌ Error: {results['error']}")
        return results
    
    # Show cluster information
    if results.get('clusters'):
        print("\n📊 CLUSTER ANALYSIS:")
        print("─" * 80)
        for cluster_type, analysis in sorted(
            results['clusters'].items(), 
            key=lambda x: x[1]['total_incidents'], 
            reverse=True
        ):
            print(f"\nCluster: '{cluster_type}'")
            print(f"  • {analysis['total_incidents']} incidents")
            print(f"  • Avg similarity: {analysis['avg_similarity']:.3f}")
            print(f"  • Top causes in this cluster:")
            
            top_causes = sorted(
                analysis['causes'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:3]
            
            for cause, prob in top_causes:
                cause_text = cause[:60]
                print(f"     - {prob*100:.1f}%: {cause_text}")
    
    # Show final diagnosis
    print("\n🎯 FINAL DIAGNOSIS (Conditional Probability with Chain Rule):")
    print("─" * 80)
    
    if results.get('weighted_causes'):
        for i, cause_info in enumerate(results['weighted_causes'][:15], 1):
            prob_pct = cause_info['probability'] * 100
            cause_text = cause_info['cause'][:70]
            print(f"\n{i}. {prob_pct:.2f}% - {cause_text}")
            print(f"   Evidence: {cause_info['num_incidents']} incidents, {cause_info['num_clusters']} clusters")
    else:
        print("❌ No causes identified")
    
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    # Test queries
    test_queries = [
        "engine lost power during flight",
        "landing gear collapsed on landing",
        "bird strike on takeoff"
    ]
    
    # Choose test mode
    import sys
    
    if len(sys.argv) > 1:
        # Use command line argument as query
        query = ' '.join(sys.argv[1:])
        test_conditional_probability(query)
    else:
        # Run comparison test on first query
        print("\n🧪 Running comparison test on sample query...")
        print("(To test a custom query, run: python test_chain_rule_diagnosis.py 'your query here')\n")
        
        test_chain_rule_vs_similarity_weighted(test_queries[0])
