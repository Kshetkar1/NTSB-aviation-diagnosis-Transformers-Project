# Slide 18 & 19: Chain Rule Worked Example

---

## **📊 SLIDE 18: Chain Rule - Worked Example**

### **Slide Content:**

```
Complete Example: Calculating P("Improper Maintenance" | Query)

Setup - Top 50 incidents grouped into 3 clusters:
• Engine Failure: 12 incidents, avg similarity = 0.85
• Fuel System: 8 incidents, avg similarity = 0.75
• Landing Gear: 5 incidents, avg similarity = 0.60

Step 1: Calculate P(Cluster|Query) for each cluster

Cluster Weights:
• Engine Failure: 0.85 × 12 = 10.2
• Fuel System: 0.75 × 8 = 6.0
• Landing Gear: 0.60 × 5 = 3.0
• Total Weight = 10.2 + 6.0 + 3.0 = 19.2

P(Cluster|Query):
• P(Engine Failure|Query) = 10.2 / 19.2 = 0.53 (53%)
• P(Fuel System|Query) = 6.0 / 19.2 = 0.31 (31%)
• P(Landing Gear|Query) = 3.0 / 19.2 = 0.16 (16%)
```

### **🗣️ Speaker Notes for Slide 18:**

> "Let me walk through a complete example with real numbers so you can see exactly how this works.
>
> **The Scenario:** We've retrieved our top 50 similar incidents and grouped them into three clusters: Engine Failure with 12 incidents, Fuel System with 8 incidents, and Landing Gear with 5 incidents. Each cluster has an average similarity score showing how closely those incidents match the query.
>
> **Step 1 - Calculate cluster relevance:** First, we need to figure out how relevant each cluster is to this query. We calculate the weight by multiplying similarity times count.
>
> Engine Failure: 0.85 times 12 equals 10.2.
>
> Fuel System: 0.75 times 8 equals 6.0.
>
> Landing Gear: 0.60 times 5 equals 3.0.
>
> The total weight is 19.2.
>
> Now we divide each by the total to get probabilities. Engine Failure: 10.2 divided by 19.2 equals 53%. Fuel System: 31%. Landing Gear: 16%. These percentages tell us how much each cluster should influence the final answer."

---

## **📊 SLIDE 19: Continued**

### **Slide Content:**

```
Step 2: Get P("Improper Maintenance"|Cluster) from earlier calculation

• P(Improper Maintenance | Engine Failure) = 0.35 (35%)
• P(Improper Maintenance | Fuel System) = 0.20 (20%)
• P(Improper Maintenance | Landing Gear) = 0.10 (10%)

Step 3: Apply Chain Rule

P("Improper Maintenance" | Query) =
  (0.35 × 0.53) + (0.20 × 0.31) + (0.10 × 0.16)
  = 0.186 + 0.062 + 0.016
  = 0.264 = 26.4%

Final Answer: "Improper Maintenance" has 26.4% probability as root cause
```

### **🗣️ Speaker Notes for Slide 19:**

> "**Step 2 - Get the within-cluster probabilities:** Remember when we counted how often each cause appeared within each cluster? We found that 'Improper Maintenance' appears 35% of the time in Engine Failure incidents, 20% in Fuel System incidents, and 10% in Landing Gear incidents. Those are the P(Cause|Cluster) values we calculated earlier.
>
> **Step 3 - Apply the chain rule:** Now we multiply and add. From the Engine Failure cluster, we get 0.35 times 0.53, which equals 0.186. From Fuel System, 0.20 times 0.31 equals 0.062. From Landing Gear, 0.10 times 0.16 equals 0.016.
>
> Add them all up: 0.186 plus 0.062 plus 0.016 equals 0.264, or 26.4%.
>
> **What this means:** The final probability that 'Improper Maintenance' is the root cause is 26.4%. This number combines evidence from all three incident types, weighted by how similar they are to the query. The Engine Failure cluster contributed the most because it had high similarity and many incidents. We repeat this calculation for every possible cause to get our complete diagnosis.
>
> Now let me show you the system in action with a live demo."

---

## **📋 Quick Reference Summary:**

**Flow:**

1. Calculate cluster weights (similarity × count)
2. Calculate P(Cluster|Query) by dividing by total weight
3. Use pre-calculated P(Cause|Cluster) values
4. Multiply and sum: Σ P(Cause|Cluster) × P(Cluster|Query)
5. Get final probability for each cause

**Key Numbers in Example:**

- Total weight: 19.2
- Engine Failure cluster: 53% relevance, 35% improper maintenance → contributes 0.186
- Fuel System cluster: 31% relevance, 20% improper maintenance → contributes 0.062
- Landing Gear cluster: 16% relevance, 10% improper maintenance → contributes 0.016
- **Final: 26.4% probability for "Improper Maintenance"**

---

_These slides demonstrate the complete chain rule calculation from start to finish, showing where every number comes from and how they combine to produce the final diagnosis._
