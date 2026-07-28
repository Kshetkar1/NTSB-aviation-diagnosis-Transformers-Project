# Professor's Chain Rule Diagnosis - Step by Step Guide

## The Whiteboard Breakdown

From your professor's whiteboard, here's what he drew:

```
Query (Q) → Find Similar Incidents

Incident I1 ─┐
             ├─ Has causes: C11, C12
Incident I2 ─┤
             ├─ Has causes: C21, C22
Incident I3 ─┘
             └─ Has causes: C31, C32, C33
```

### The Formula He Wrote:

```
P(C11|Q) = P(C11|I1) × P(I1|Q)
```

**Translation:**
- **P(I1|Q)** = "How similar is incident I1 to the query?" → **You already have this!** (cosine similarity = 0.7)
- **P(C11|I1)** = "Given incident type I1, what's the probability it was caused by C11 vs C12?" → **This is what's missing!**

---

## Example Walkthrough

Let's use a real example to understand:

### Your Query: "engine lost power"

### Step 1: Find Similar Incidents (YOU ALREADY DO THIS ✅)

```
Top 3 Similar Incidents:
1. Incident A: "Engine failure during cruise" → Similarity = 0.85
2. Incident B: "Power loss on takeoff" → Similarity = 0.82  
3. Incident C: "Engine stopped unexpectedly" → Similarity = 0.78
```

### Step 2: Cluster by Type (NEW - PROFESSOR WANTS THIS)

Group similar incidents together:

```
Cluster 1: "Engine mechanical failure" (Incidents A, C)
  - Incident A (similarity 0.85)
  - Incident C (similarity 0.78)
  
Cluster 2: "Fuel system issue" (Incident B)
  - Incident B (similarity 0.82)
```

### Step 3: Calculate P(Cause|Cluster) (NEW - THIS IS THE MISSING PIECE!)

**For Cluster 1 "Engine mechanical failure":**
Look at the causes from the 2 incidents in this cluster:
- Incident A: "Fatigue/wear/corrosion"
- Incident C: "Fatigue/wear/corrosion"

**Count:** 2 out of 2 incidents had "Fatigue/wear/corrosion"

**P(Fatigue|Engine_failure_cluster) = 2/2 = 1.0 (100%)**

**For Cluster 2 "Fuel system issue":**
Look at the cause from 1 incident:
- Incident B: "Fuel contamination"

**P(Fuel_contamination|Fuel_cluster) = 1/1 = 1.0 (100%)**

### Step 4: Apply Chain Rule (NEW - COMBINE EVERYTHING)

**For "Fatigue/wear/corrosion":**
```
P(Fatigue|Query) = P(Fatigue|Engine_failure_cluster) × P(Engine_failure_cluster|Query)

Where:
  P(Fatigue|Engine_failure_cluster) = 1.0 (from Step 3)
  P(Engine_failure_cluster|Query) = average similarity of cluster
                                   = (0.85 + 0.78) / 2 = 0.815
                                   
Final: P(Fatigue|Query) = 1.0 × 0.815 = 0.815 = 81.5%
```

**For "Fuel contamination":**
```
P(Fuel_contamination|Query) = P(Fuel_contamination|Fuel_cluster) × P(Fuel_cluster|Query)

Where:
  P(Fuel_contamination|Fuel_cluster) = 1.0
  P(Fuel_cluster|Query) = 0.82
  
Final: P(Fuel_contamination|Query) = 1.0 × 0.82 = 0.82 = 82%
```

### Final Diagnosis:

```
Top Root Causes:
1. Fuel contamination: 82%
2. Fatigue/wear/corrosion: 81.5%
```

---

## The Key Difference

### Your Old Method:
```
"Fatigue" appears in 2 incidents with similarities 0.85, 0.78
Probability = (0.85 + 0.78) / total_similarity
```

**Problem:** Doesn't distinguish between causes within the same incident type!

### Professor's Method:
```
1. Group incidents by type
2. Within each type, calculate: What % of incidents had cause X?
3. Weight by cluster similarity
```

**Benefit:** Properly calculates P(Cause|Incident_Type) from historical data!

---

## What I've Implemented for You

I've already coded all 4 steps:

### Function 1: `cluster_incidents_by_type()`
- Takes top 50 incidents
- Uses LLM to classify each into a type
- Groups them

### Function 2: `calculate_cause_probabilities_per_cluster()`
- For each cluster, extracts all causes
- Counts frequencies
- Returns P(Cause|Cluster)

### Function 3: `calculate_chain_rule_diagnosis()`
- Applies the chain rule formula
- Combines all clusters
- Returns final P(Cause|Query)

### Function 4: `diagnose_with_conditional_probabilities()`
- High-level function that runs all 3 steps
- Easy to call and test

---

## Next Steps

1. **Test the implementation** with a sample query
2. **Compare** professor's method vs. your current method
3. **Add to Streamlit UI** (optional)
4. **Validate** on known incidents
5. **Write up** for your paper

Let's test it now!
