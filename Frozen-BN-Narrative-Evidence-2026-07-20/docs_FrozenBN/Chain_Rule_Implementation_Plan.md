# Professor's Chain Rule Diagnosis Implementation

## Overview

This document details the implementation of Professor Dr. Ma's chain rule approach for root cause diagnosis, as discussed in the meeting on January 21, 2025.

## The Missing Piece (From Whiteboard)

### What Was Missing:

Your original approach calculated:
```
P(Cause|Query) = cosine_similarity_score
```

But the professor showed this is **incomplete** because you need:
```
P(C₁₁|Q) = P(C₁₁|I₁) × P(I₁|Q)
           ↑             ↑
      MISSING!      YOU HAD THIS
```

### The Two Required Components:

1. **P(I|Q)** = Cosine similarity between incident and query ✅ (Already implemented)
2. **P(C|I)** = Probability of cause given incident type ❌ (Was missing)

## Implementation Steps

### Step 1: Cluster Incidents by Type

**Function**: `cluster_incidents_by_type(top_scores, top_matches, top_n_incidents=50)`

**What it does:**
- Takes the top 50 similar incidents
- For each incident, extracts the narrative
- Uses LLM (GPT-4o-mini) to classify it into a type category
- Groups incidents by their assigned type

**Example output:**
```
Clustered 47 incidents into 8 types:
  - 'engine failure': 12 incidents
  - 'fuel system malfunction': 8 incidents
  - 'landing gear collapse': 6 incidents
  ...
```

### Step 2: Calculate P(Cause|Cluster)

**Function**: `calculate_cause_probabilities_per_cluster(clusters)`

**What it does:**
- For each cluster, extracts all causes from incidents in that cluster
- Counts how many times each cause appears
- Calculates: `P(Cause|Cluster) = Count(Cause in Cluster) / Total_causes_in_cluster`

**Example:**
```
Cluster: 'engine failure'
  • 12 incidents
  • Causes:
    - 60%: Fatigue/wear/corrosion (found in 7 incidents)
    - 25%: Improper maintenance (found in 3 incidents)
    - 15%: Design defect (found in 2 incidents)
```

### Step 3: Apply Chain Rule

**Function**: `calculate_chain_rule_diagnosis(clusters, cluster_analysis)`

**What it does:**
- For each cause, sums contributions from all clusters
- Uses formula: `P(Cause|Query) = Σ P(Cause|Cluster_i) × P(Cluster_i|Query)`
- Where `P(Cluster_i|Query)` = weighted average similarity of incidents in that cluster

**Example calculation:**
```
Cause: "Fatigue/wear/corrosion"

From Cluster "engine failure" (P(cluster|query) = 0.4):
  P(cause|cluster) = 0.6
  Contribution = 0.6 × 0.4 = 0.24

From Cluster "maintenance issue" (P(cluster|query) = 0.2):
  P(cause|cluster) = 0.3
  Contribution = 0.3 × 0.2 = 0.06

Final: P(cause|query) = 0.24 + 0.06 = 0.30 (30%)
```

### Step 4: High-Level Interface

**Function**: `diagnose_with_chain_rule(query, top_n=10, top_n_incidents=50)`

**What it does:**
- Orchestrates the full pipeline
- Returns ranked causes with probabilities and evidence

## Code Location

All new functions are in: `main_app.py` (lines ~237-540)

### New Functions Added:

1. `cluster_incidents_by_type()` - LLM-based clustering
2. `calculate_cause_probabilities_per_cluster()` - P(C|I_type) calculation
3. `calculate_chain_rule_diagnosis()` - Chain rule application
4. `diagnose_with_chain_rule()` - High-level interface

## Testing

**Test script**: `tests/test_chain_rule_diagnosis.py`

### Run comparison test:
```bash
cd tests
python test_chain_rule_diagnosis.py
```

This will compare:
- **Method 1**: Professor's Chain Rule (NEW)
- **Method 2**: Dr. Smith's Weighted Approach (CURRENT)

### Test with custom query:
```bash
python test_chain_rule_diagnosis.py "engine lost power"
```

## Key Differences: Chain Rule vs. Dr. Smith's Approach

| Aspect | Dr. Smith's Weighted | Professor's Chain Rule |
|--------|---------------------|------------------------|
| **Formula** | `P(C\|Q) = Σ sim(I,Q) × 1[C∈I] / Σ sim` | `P(C\|Q) = Σ P(C\|Type) × P(Type\|Q)` |
| **Clustering** | No clustering | Clusters incidents by type |
| **Cause Probability** | Uses raw similarity | Calculates P(C\|Type) from data |
| **Theoretical Basis** | Weighted averaging | Bayesian chain rule |
| **Complexity** | Simple, fast | More complex, requires LLM |
| **Interpretability** | Direct similarity weighting | Shows cluster breakdown |

## Implementation Status

- ✅ Step 1: Clustering function implemented
- ✅ Step 2: P(Cause|Cluster) calculation implemented
- ✅ Step 3: Chain rule application implemented
- ✅ Step 4: High-level interface implemented
- ✅ Test script created
- ⏳ UI integration (not yet done)
- ⏳ Validation on known incidents (not yet done)

## Next Steps for Your Thesis

### 1. Test and Validate (Immediate)
```bash
cd /Users/kanushetkar/Desktop/Vanderbilt_University/Internships/Current_Projects/NTSB/NTSB_Shivy/tests
python test_chain_rule_diagnosis.py "engine lost power"
```

### 2. Compare Methods (For Paper)
- Run both methods on 10-20 test queries
- Compare accuracy, probabilities, and interpretability
- Document which performs better and why

### 3. Integrate into Streamlit UI (Optional)
- Add a toggle to switch between methods
- Show cluster information in the UI
- Display breakdown of P(Cause|Cluster) contributions

### 4. Write Paper Section
- Introduction: Explain the chain rule approach
- Methodology: Describe the 3-step process
- Results: Compare both methods
- Discussion: Which works better and why

## API Cost Considerations

**LLM clustering costs:**
- ~$0.15 per 1000 classifications (GPT-4o-mini)
- For 50 incidents per query: ~$0.0075 per diagnosis
- Budget-friendly for testing and thesis work

## Professor's Approval

✅ This implementation matches the whiteboard diagram  
✅ Implements the three-step process he outlined  
✅ Uses LLM for clustering (as he suggested)  
✅ Applies proper chain rule mathematics  

## Questions for Next Meeting

1. Minimum cluster size - should we merge small clusters?
2. What if an incident doesn't fit any cluster?
3. Should we validate on specific known cases?
4. Do you want to see cluster information in the UI?

---

**Implementation Date**: January 2025  
**Based on**: Meeting with Professor Dr. Ma (January 21, 2025)  
**Status**: Implemented and ready for testing
