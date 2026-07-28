# Terminology Update: Removing "Bayesian" References

## Date: January 2025

## Reason for Change

After reviewing the implementation with Professor Dr. Ma, it became clear that the system is **NOT performing Bayesian inference**. Instead, it uses:

1. **Empirical frequency estimation** from historical data
2. **Similarity-weighted averaging** based on cosine similarity
3. **Chain rule of probability** (which is just math, not specifically Bayesian)
4. **Frequentist conditional probabilities**

**What it's NOT:**
- ❌ Bayesian Networks with DAGs
- ❌ MCMC sampling
- ❌ Prior/posterior distributions
- ❌ Bayesian updating

**What it IS:**
- ✅ Similarity-based retrieval
- ✅ Historical case-based reasoning
- ✅ Empirical frequency counting
- ✅ Weighted aggregation

---

## Changes Made

### 1. **Function Names in `main_app.py`**

| Old Name | New Name |
|----------|----------|
| `calculate_weighted_diagnosis()` | `calculate_similarity_weighted_diagnosis()` |
| `diagnose_with_chain_rule()` | `diagnose_with_conditional_probabilities()` |

### 2. **Variable Names Throughout**

| Old Name | New Name |
|----------|----------|
| `bayesian_data` | `diagnostic_data` |
| `BAYESIAN_STATS_PATH` | `CAUSE_STATS_PATH` |

### 3. **File Names**

| Old Name | New Name |
|----------|----------|
| `2b_precompute_bayesian_data.py` | `2b_precompute_diagnostic_data.py` |
| `bayesian_cause_statistics.json` | `cause_statistics.json` |
| `test_chain_rule_diagnosis.py` | *(updated internally but kept name)* |

### 4. **Section Headers**

| Old Name | New Name |
|----------|----------|
| "Dr. Smith's Weighted Bayesian Diagnosis Functions" | "Similarity-Based Diagnosis Functions" |
| "Professor's Chain Rule Diagnosis Functions" | "Conditional Probability Diagnosis (Chain Rule Approach)" |

### 5. **Methodology Strings**

| Old Description | New Description |
|----------------|-----------------|
| "Similarity-weighted empirical Bayesian diagnosis" | "Similarity-weighted historical case analysis" |
| "Professor's Chain Rule: P(Cause\|Query) = ..." | "Conditional probability with chain rule: P(Cause\|Query) = ..." |
| "Bayesian Network approach" | "Empirical conditional probability approach" |

---

## How to Describe the System (For Your Paper)

### ✅ **Correct Terminology:**

- "Similarity-based diagnosis system"
- "Retrieval-augmented diagnosis"
- "Historical case-based reasoning"
- "Empirical conditional probability estimation"
- "Frequency-based probabilistic diagnosis"
- "Weighted evidence synthesis"

### ❌ **Avoid These Terms:**

- "Bayesian diagnosis" (unless you actually implement Bayesian inference)
- "Bayesian networks" (you're not using graphical models)
- "Likelihood" in this context (professor explicitly said to avoid this)
- "Posterior probabilities" (you're not computing posteriors)

---

## What Your System Actually Does

### Method 1: Similarity-Weighted Diagnosis

```python
P(Cause|Query) = Σ similarity(I,Q) × indicator(Cause in I) / Σ similarity
```

**Description**: "We weight historical case frequencies by their similarity to the query incident."

### Method 2: Conditional Probability with Chain Rule

```python
P(Cause|Query) = Σ P(Cause|Cluster) × P(Cluster|Query)
```

Where:
- `P(Cause|Cluster)` = empirical frequency (count/total)
- `P(Cluster|Query)` = average similarity score

**Description**: "We estimate conditional probabilities from historical frequencies and apply the chain rule."

---

## For Your Paper - Suggested Text

### Introduction/Methodology Section:

> "We develop a **similarity-based diagnosis system** that leverages historical NTSB incident data to identify probable root causes. The system uses **semantic embeddings** to find historically similar incidents, then applies **empirical conditional probability estimation** to rank potential causes.
>
> While our approach employs the **chain rule of probability** and calculates conditional probabilities P(Cause|IncidentType), it is fundamentally a **frequentist, case-based reasoning method** rather than Bayesian inference. Probabilities are derived directly from historical case frequencies rather than through Bayesian updating of prior distributions.
>
> This design prioritizes **computational efficiency** (real-time inference), **interpretability** (clear evidence trails to supporting incidents), and **robustness** (handles sparse/incomplete data naturally)."

---

## Impact on Existing Data

**Good News:** The data structures in `embeddings_map.json` are unchanged!

- The field is still called `diagnostic_data` now (was `bayesian_data`)
- But since we're renaming everything, we need to:
  1. Re-run `2b_precompute_diagnostic_data.py` to regenerate with new field names
  2. Or update the Streamlit app to look for either name (backward compatibility)

---

## Testing After Rename

Run these to ensure everything works:

```bash
# Test the similarity-weighted approach
cd /Users/kanushetkar/Desktop/Vanderbilt_University/Internships/Current_Projects/NTSB/NTSB_Shivy
python main_app.py

# Test the conditional probability approach
cd tests
python test_chain_rule_diagnosis.py "engine lost power"

# Test the Streamlit app
streamlit run streamlit_app.py
```

---

## Summary

**What changed:** Naming and terminology to accurately reflect what the system does

**What didn't change:** The actual algorithms, math, or logic

**Why it matters:** 
1. Academic honesty - calling it what it is
2. Avoids confusion with actual Bayesian methods
3. Makes the paper/thesis more defensible
4. Professor explicitly requested this clarification

---

**Status:** ✅ Complete

**Next Steps:** 
1. Re-run preprocessing to update field names in data
2. Test all functions with new names
3. Update any remaining documentation
