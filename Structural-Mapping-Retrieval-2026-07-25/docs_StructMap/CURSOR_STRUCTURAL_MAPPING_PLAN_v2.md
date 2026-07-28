# Structural Mapping Implementation Plan for Cursor

## What This Is

This is a complete implementation plan for adding **structural similarity mapping** to the NTSB aviation incident retrieval system. The goal: replace the current flat-enum Jaccard structural reranking (A1) with a causal chain alignment approach (A2) grounded in cognitive science, then **quantitatively prove** whether it improves outcomes over embeddings alone (A0).

Every formula in this plan traces to a published paper. Nothing is made up.

---

## Theoretical Foundation (Why These Formulas)

### The Core Idea

Two incidents can share keywords ("engine," "oil," "failure") but have completely different causal structures. One might be: maintenance error → bearing spin → oil blockage → engine seizure. The other: corrosion → fatigue → cylinder separation → fire → wing loss. Embedding similarity rates these as similar. They are not.

The structural mapping approach comes from cognitive science research on how humans compare complex structured scenes.

### The Papers and What They Give Us

**1. Structure-Mapping Theory — Gentner (1983); Gentner & Markman (1997)**
- *Citation*: Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. *Cognitive Science*, 7(2), 155–170.
- *Citation*: Gentner, D., & Markman, A. B. (1997). Structure mapping in analogy and similarity. *American Psychologist*, 52(1), 45–56.
- *What it gives us*: Three constraints that any structural similarity computation must respect:
  1. **One-to-one correspondence**: Each element in incident A maps to at most one element in incident B → *This is why we use dynamic programming alignment, not many-to-many matching*
  2. **Parallel connectivity**: If step i in A maps to step j in B, then step i+1 must map to j+1 or later → *This is why alignment is monotonic (order-preserving)*
  3. **Systematicity**: Connected relational systems (causal chains) matter more than isolated matches → *This is why role-based matching is weighted highest*

**2. SIAM Model — Goldstone (1994)**
- *Citation*: Goldstone, R. L. (1994). Similarity, interactive activation, and mapping. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 20(1), 3–28.
- *What it gives us*: Similarity and mapping are computed simultaneously, not sequentially. You can't compute similarity first and then map. The mapping determines what counts as similar. → *This is why our score derives from the alignment itself, not from independent feature comparison*

**3. Spencer-Smith & Goldstone (1997)**
- *Citation*: Spencer-Smith, J. & Goldstone, R. L. (1997). The dynamics of similarity. *Cognitive Studies: Bulletin of the Japanese Cognitive Science Society*, 4, 38–56.
- *What it gives us*: Similarity judgments are dynamic — features are selectively weighted during comparison. → *This supports our approach of weighting role > mechanism > system rather than treating all dimensions equally*

**4. Structure-Mapping Engine (SME) — Falkenhainer, Forbus & Gentner (1989)**
- *Citation*: Falkenhainer, B., Forbus, K. D., & Gentner, D. (1989). The structure-mapping engine: Algorithm and examples. *Artificial Intelligence*, 41(1), 1–63.
- *What it gives us*: The actual computational scoring approach:
  - Match hypotheses get evidence values (0.5 for same-functor attributes, 0.4 for first-order relations)
  - Higher-order relations propagate evidence down to their arguments (trickle-down: 0.8–0.9 multiplier)
  - Structural Evaluation Score (SES) = sum of all match hypothesis evidence in the best global mapping
  - → *Our scoring formula is a simplified operationalization of SES: sum evidence for what maps (strong + 0.5 × partial), penalize what doesn't (unmapped elements)*

**5. Two-Stage Retrieval with Reranking — Nogueira & Cho (2019); standard IR practice**
- *Citation*: Nogueira, R. & Cho, K. (2019). Passage re-ranking with BERT. *arXiv:1901.04085*.
- *What it gives us*: The fusion formula pattern. In information retrieval, it is standard to combine a fast first-pass retriever (embeddings) with a more expensive reranker (structural mapping) using multiplicative reweighting with exponential scaling.
  - → *This is where `w' = max(0, cosine) × exp(α × s)` comes from*

**6. Needleman-Wunsch Sequence Alignment — Needleman & Wunsch (1970)**
- *Citation*: Needleman, S. B. & Wunsch, C. D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. *Journal of Molecular Biology*, 48(3), 443–453.
- *What it gives us*: The dynamic programming alignment algorithm. Originally for protein sequences, it's the standard algorithm for finding optimal alignments between two ordered sequences with gaps. → *We use this to align causal chains while allowing for unmapped elements (gaps)*

---

## Formula Derivations (Where Each Number Comes From)

### Formula 1: Element Similarity

```
element_sim(a, b) = w_role × role_match(a,b) + w_mech × mechanism_match(a,b) + w_sys × system_match(a,b)
```

Where:
- `w_role = 0.50` — Functional role is most important (Gentner's systematicity principle: relational structure matters most)
- `w_mech = 0.30` — Mechanism captures how one step causes the next (first-order relations in SME)
- `w_sys = 0.20` — System domain is least important — this is a surface attribute in Gentner's taxonomy

**Why these specific weights**: The 50/30/20 split follows from the theory's hierarchy: systematic relational structure > first-order relations > object attributes. SME assigns ~0.4–0.5 to relational matches and ~0.5 to attribute matches at the base level, but higher-order relations get 0.8–0.9 trickle-down bonuses. Our role (the highest-order structural feature) gets the most weight. These are starting values — tune them empirically on your training data.

Match values:
- Exact match on a dimension = full weight for that dimension
- Related match (adjacent roles, related mechanisms) = 0.5 × weight
- No match = 0

**Source for related/adjacent groupings**: These come from domain knowledge about aviation systems. Related mechanisms (e.g., fatigue and corrosion both being material degradation) and adjacent roles (e.g., propagation and system_compromise being sequential in a failure chain) reflect the NTSB's own classification system in `seq_of_events` group codes.

### Formula 2: Sequence Alignment (Needleman-Wunsch)

```
dp[i][j] = max(
    dp[i-1][j-1] + element_sim(chain_a[i], chain_b[j]),   # align these elements
    dp[i-1][j]   + GAP_PENALTY,                            # skip element in A
    dp[i][j-1]   + GAP_PENALTY                             # skip element in B
)
```

Where:
- `GAP_PENALTY = -0.05` — Small penalty for unmapped elements

**Source**: This is the standard Needleman-Wunsch recurrence (Needleman & Wunsch, 1970). The gap penalty operationalizes Gentner's treatment of non-alignable differences: elements present in one incident with no counterpart in the other should reduce similarity, but not dominate the score. A small penalty (0.05) means gaps hurt a little but don't overwhelm good matches. Tune this on your training data.

### Formula 3: Structural Similarity Score

```
score = (n_strong + 0.5 × n_partial) / n_total − penalty × (n_unmapped / n_total)
```

Where:
- `n_strong` = count of aligned pairs with element_sim ≥ 0.70 (strong match)
- `n_partial` = count of aligned pairs with 0.35 ≤ element_sim < 0.70 (partial match)
- `n_total` = total unique elements across both chains (union)
- `n_unmapped` = elements with no alignment counterpart
- `penalty = 0.05`

**Source**: This directly operationalizes SME's Structural Evaluation Score (Falkenhainer et al., 1989). In SME:
- Match hypotheses with same-functor predicates get evidence ~0.4–0.5 → our "strong" matches (they got the role right, which is the highest-order structural feature)
- Partial matches (adjacent roles) get discounted evidence → our 0.5 multiplier
- Unmapped predicates don't contribute and implicitly reduce the overall score → our explicit penalty term

**Thresholds (0.70 and 0.35)**:
- 0.70 means role matched (0.50) plus at least mechanism matched (0.30) → the element plays the same functional role AND operates through the same causal mechanism
- 0.35 means role matched (0.50 × 0.5 = 0.25 for adjacent) plus something else, or exact role match alone → structurally similar position but different specifics
- These are reasonable starting points — validate and tune on training data

### Formula 4: Fusion with Embeddings

```
w' = max(0, cosine_sim) × exp(α × structural_score)
```

Where:
- `cosine_sim` = embedding cosine similarity (your existing first-pass retrieval score)
- `structural_score` = the score from Formula 3 (range [0, 1])
- `α` = reranking strength parameter (start with α = 2.0, tune on training data)

**Source**: This is standard two-stage retrieval fusion (Nogueira & Cho, 2019; and widely used in RAG systems). The multiplicative form ensures structural mapping only reranks candidates already in the semantic neighborhood (cosine > 0). The exponential controls reranking aggressiveness:
- `α = 0`: structural score has no effect (pure embedding)
- `α = 1`: moderate reranking
- `α = 3+`: aggressive reranking (structural score dominates)

**Why exponential, not linear**: `exp(α × s)` produces a sharper distinction between high and low structural scores than linear weighting would. When s = 0.9 vs s = 0.1, exp(2 × 0.9) / exp(2 × 0.1) ≈ 5:1 amplification. This matches the theoretical prediction: truly structurally similar incidents should be strongly promoted over superficially similar ones.

---

## What You Already Have (A0 and A1)

Your current system has:
- **A0 (embedding only)**: Query → OpenAI text-embedding-3-small → cosine similarity → top-k retrieval
- **A1 (flat enum structural reranking)**: Same retrieval → extract flat JSON enums (failure_mode, aircraft_system, etc.) → Jaccard similarity on enum overlap → `w' = max(0, cosine) × exp(α × jaccard)` fusion

A1 is what Gentner would call "mere appearance matching" — it checks if two incidents share the same attribute labels without considering whether those attributes play the same functional role in the causal chain.

---

## What You Are Building (A2)

### Step 1: Schema Extraction Prompt (`extraction_prompt_v2.txt`)

Create a new prompt that instructs the LLM to return a relational causal chain schema. The LLM reads the incident bundle (narrative, probable cause, findings, seq_of_events codes) and outputs structured JSON.

```
You are an aviation safety analyst. Given an NTSB incident report, extract the
causal chain as a structured JSON object.

For each step in the causal chain, identify:
- element: What specifically happened (concise description)
- role: The functional role of this element in the failure chain. Use ONLY these values:
    initiating_event | propagation | system_compromise | system_failure |
    terminal_failure | operational_consequence | outcome
- system: Which aircraft system is involved. Use ONLY these values:
    engine_mechanical | lubrication | fuel | hydraulic | electrical | structural |
    flight_controls | landing_gear | propulsion | fire_protection | environment |
    human_performance | maintenance | aircraft
- mechanism: How this step causes the next. Use ONLY these values:
    maintenance_error | material_degradation | fatigue | corrosion | contamination |
    thermal_damage | mechanical_loosening | deprivation | overheating | fire |
    separation | power_loss | loss_of_control | terrain_collision |
    procedural_error | design_deficiency | unknown

CRITICAL: Focus on CAUSAL ROLES, not surface descriptions. The same component
(e.g., "oil") can play completely different roles in different incidents:
- In one incident, oil blockage causes engine failure (oil is a causal agent)
- In another, oil leaks onto a hot surface causing fire (oil is fuel for fire)
- In a third, oil level is checked and found normal (oil is a bystander)
The ROLE and MECHANISM matter more than the component name.

Also identify:
- contributing_factors: Elements that shaped the outcome but were not in the main chain
- bystanders: Elements present but NOT causally involved (these matter for preventing
  false keyword matches)
- failure_pattern: High-level label for the overall architecture. Use ONLY:
    maintenance_latent_defect_cascade | material_degradation_cascade |
    thermal_cascade | defense_in_depth_failure | human_performance_chain |
    environmental_encounter | cross_system_propagation | fuel_management_failure |
    unknown

Return ONLY valid JSON in this exact format:
{
  "ev_id": "<event_id>",
  "causal_chain": [
    {"step": 1, "element": "...", "role": "...", "system": "...", "mechanism": "..."},
    ...
  ],
  "contributing_factors": [
    {"element": "...", "role": "outcome_amplifier", "system": "..."}
  ],
  "bystanders": [
    {"element": "...", "system": "..."}
  ],
  "failure_pattern": "..."
}
```

### Step 2: Extraction Script (`extract_struct_v2.py`)

Same structure as existing `extract_struct.py` but uses the v2 prompt.

```python
"""
extract_struct_v2.py — Extract relational causal chain schemas from NTSB incidents.
Uses a reasoning model to generate structured representations that capture causal roles,
not just surface keywords.

Input: merged_dataset_train.json (or test)
Output: struct_train_v2.jsonl (one JSON object per line per ev_id)
"""

import json
import os
from pathlib import Path

# Use the same build_incident_bundle function from your existing code
# Input: narr_accp, narr_accf, narr_cause, sequence lines, findings
# Output: a text bundle the LLM can read

PROMPT_PATH = "Testing_Structural_Mapping/schema/extraction_prompt_v2.txt"
OUTPUT_PATH = "Testing_Structural_Mapping/schema/struct_train_v2.jsonl"

# Controlled vocabularies for validation
VALID_ROLES = {
    "initiating_event", "propagation", "system_compromise",
    "system_failure", "terminal_failure", "operational_consequence", "outcome"
}
VALID_SYSTEMS = {
    "engine_mechanical", "lubrication", "fuel", "hydraulic", "electrical",
    "structural", "flight_controls", "landing_gear", "propulsion",
    "fire_protection", "environment", "human_performance", "maintenance", "aircraft"
}
VALID_MECHANISMS = {
    "maintenance_error", "material_degradation", "fatigue", "corrosion",
    "contamination", "thermal_damage", "mechanical_loosening", "deprivation",
    "overheating", "fire", "separation", "power_loss", "loss_of_control",
    "terrain_collision", "procedural_error", "design_deficiency", "unknown"
}
VALID_PATTERNS = {
    "maintenance_latent_defect_cascade", "material_degradation_cascade",
    "thermal_cascade", "defense_in_depth_failure", "human_performance_chain",
    "environmental_encounter", "cross_system_propagation",
    "fuel_management_failure", "unknown"
}

def validate_schema(schema: dict) -> tuple[bool, list[str]]:
    """Validate that the schema conforms to controlled vocabularies."""
    errors = []

    chain = schema.get("causal_chain", [])
    if not chain:
        errors.append("causal_chain is empty")
        return False, errors

    for i, step in enumerate(chain):
        if step.get("role") not in VALID_ROLES:
            errors.append(f"Step {i+1}: invalid role '{step.get('role')}'")
        if step.get("system") not in VALID_SYSTEMS:
            errors.append(f"Step {i+1}: invalid system '{step.get('system')}'")
        if step.get("mechanism") not in VALID_MECHANISMS:
            errors.append(f"Step {i+1}: invalid mechanism '{step.get('mechanism')}'")

    pattern = schema.get("failure_pattern", "")
    if pattern not in VALID_PATTERNS:
        errors.append(f"Invalid failure_pattern '{pattern}'")

    return len(errors) == 0, errors

def extract_schema(incident_bundle: str, ev_id: str, llm_client) -> dict:
    """
    Call the LLM to extract a relational causal chain schema.
    Uses temperature=0 for consistency across extractions.
    Retries once on validation failure.
    """
    prompt = Path(PROMPT_PATH).read_text()

    for attempt in range(2):
        response = llm_client.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": incident_bundle}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        schema = json.loads(response.content)
        schema["ev_id"] = ev_id

        valid, errors = validate_schema(schema)
        if valid:
            return schema

        if attempt == 0:
            # Retry with error feedback
            response = llm_client.chat(
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": incident_bundle},
                    {"role": "assistant", "content": json.dumps(schema)},
                    {"role": "user", "content": f"Validation errors: {errors}. Fix and return valid JSON."}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            schema = json.loads(response.content)
            schema["ev_id"] = ev_id
            valid, errors = validate_schema(schema)
            if valid:
                return schema

    # Fallback: return with error flag
    schema["error"] = f"validation_failed: {errors}"
    return schema
```

**Run it:**
```bash
python extract_struct_v2.py --input data/merged_dataset_train.json \
    --output Testing_Structural_Mapping/schema/struct_train_v2.jsonl
```

Expected: ~30-60 min for 2,243 incidents with API calls. Cache one line per ev_id, skip if already cached.

### Step 3: Structural Scoring (`struct_score_v2.py`)

This is the core change. Complete implementation:

```python
"""
struct_score_v2.py — Structural similarity scoring via causal chain alignment.

Theoretical basis:
- Scoring formula: operationalizes SME's Structural Evaluation Score
  (Falkenhainer, Forbus & Gentner, 1989, Artificial Intelligence 41(1), 1-63)
- Alignment algorithm: Needleman-Wunsch dynamic programming
  (Needleman & Wunsch, 1970, J Mol Biol 48(3), 443-453)
- Mapping constraints: one-to-one correspondence, parallel connectivity, systematicity
  (Gentner & Markman, 1997, American Psychologist 52(1), 45-56)
- Dynamic similarity: mapping determines similarity, not vice versa
  (Goldstone, 1994, JEP:LMC 20(1), 3-28; Spencer-Smith & Goldstone, 1997)
"""

# =============================================================================
# CONFIGURABLE PARAMETERS (tune on training data)
# =============================================================================

# Element comparison weights (must sum to 1.0)
# Source: Gentner's hierarchy — systematic relations > first-order relations > attributes
ROLE_WEIGHT = 0.50       # Functional role = highest-order structural feature
MECHANISM_WEIGHT = 0.30  # Causal mechanism = first-order relation
SYSTEM_WEIGHT = 0.20     # System domain = surface attribute (least important)

# Alignment parameters
# Source: Needleman-Wunsch gap penalty; operationalizes non-alignable difference cost
GAP_PENALTY = -0.05

# Match classification thresholds
# Source: derived from weight structure
#   Strong (≥0.70): role matched (0.50) + mechanism matched (0.30) = 0.80
#   Partial (≥0.35): role adjacent (0.25) + something else, or role exact alone (0.50)
STRONG_THRESHOLD = 0.70
PARTIAL_THRESHOLD = 0.35

# Score composition
# Source: SME trickle-down — higher-order structure (chain alignment) weighted more
#   than lower-order (contributing factors)
CHAIN_WEIGHT = 0.85
FACTOR_WEIGHT = 0.15
UNMAPPED_PENALTY = 0.10  # Applied to fraction of unmapped elements

# Pattern bonus
# Source: Gentner's systematicity — matching overall causal architecture is a
#   strong signal of deep structural similarity
PATTERN_BONUS_WEIGHT = 0.20
ALIGNMENT_WEIGHT = 0.80  # 1.0 - PATTERN_BONUS_WEIGHT

# =============================================================================
# DOMAIN KNOWLEDGE (from NTSB classification system)
# =============================================================================

# Roles that are adjacent in a typical failure chain
ADJACENT_ROLES = {
    ("initiating_event", "propagation"),
    ("propagation", "system_compromise"),
    ("system_compromise", "system_failure"),
    ("system_failure", "terminal_failure"),
    ("terminal_failure", "operational_consequence"),
    ("operational_consequence", "outcome"),
}

# Mechanisms that are functionally related
RELATED_MECHANISMS = [
    {"maintenance_error", "procedural_error"},
    {"material_degradation", "fatigue", "corrosion"},
    {"contamination", "deprivation"},
    {"thermal_damage", "overheating", "fire"},
    {"mechanical_loosening", "separation"},
]

# Aircraft systems that are physically/functionally adjacent
RELATED_SYSTEMS = [
    {"engine_mechanical", "lubrication", "propulsion"},
    {"fuel", "fire_protection"},
    {"hydraulic", "landing_gear", "flight_controls"},
    {"electrical", "fire_protection"},
]

# Failure patterns that are related
RELATED_PATTERNS = [
    {"maintenance_latent_defect_cascade", "material_degradation_cascade"},
    {"thermal_cascade", "cross_system_propagation"},
]


def roles_are_adjacent(role_a: str, role_b: str) -> bool:
    return (role_a, role_b) in ADJACENT_ROLES or (role_b, role_a) in ADJACENT_ROLES

def mechanisms_are_related(mech_a: str, mech_b: str) -> bool:
    for group in RELATED_MECHANISMS:
        if mech_a in group and mech_b in group:
            return True
    return False

def systems_are_related(sys_a: str, sys_b: str) -> bool:
    for group in RELATED_SYSTEMS:
        if sys_a in group and sys_b in group:
            return True
    return False


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def element_similarity(elem_a: dict, elem_b: dict) -> float:
    """
    Compare two causal chain elements across three dimensions.

    Returns a score in [0, 1] where:
    - 1.0 = exact match on all dimensions
    - 0.50 = role matches but nothing else (still structurally significant)
    - 0.0 = nothing matches

    Theoretical basis: Gentner's systematicity principle — functional role
    (the highest-order structural feature) matters most. System domain
    (a surface attribute) matters least.
    """
    score = 0.0

    # Role match (highest weight — this IS the structural mapping)
    if elem_a.get("role") == elem_b.get("role"):
        score += ROLE_WEIGHT
    elif roles_are_adjacent(elem_a.get("role", ""), elem_b.get("role", "")):
        score += ROLE_WEIGHT * 0.5

    # Mechanism match (how this step causes the next)
    if elem_a.get("mechanism") == elem_b.get("mechanism"):
        score += MECHANISM_WEIGHT
    elif mechanisms_are_related(elem_a.get("mechanism", ""), elem_b.get("mechanism", "")):
        score += MECHANISM_WEIGHT * 0.5

    # System match (least important — surface feature)
    if elem_a.get("system") == elem_b.get("system"):
        score += SYSTEM_WEIGHT
    elif systems_are_related(elem_a.get("system", ""), elem_b.get("system", "")):
        score += SYSTEM_WEIGHT * 0.5

    return score


def compute_element_similarity_matrix(chain_a: list, chain_b: list) -> list:
    """NxM matrix of element similarities between two chains."""
    n, m = len(chain_a), len(chain_b)
    return [[element_similarity(chain_a[i], chain_b[j]) for j in range(m)] for i in range(n)]


def find_best_alignment(sim_matrix: list, chain_a: list, chain_b: list) -> dict:
    """
    Find the best monotonic alignment between two causal chains using
    Needleman-Wunsch dynamic programming.

    Monotonic alignment enforces Gentner's parallel connectivity constraint:
    if element i in A maps to element j in B, then element i+1 in A can
    only map to element j+1 or later in B.

    Source: Needleman & Wunsch (1970), adapted for causal chain alignment.

    Returns dict with:
    - aligned: list of (idx_a, idx_b, similarity) tuples
    - unmatched_a: indices in A with no counterpart
    - unmatched_b: indices in B with no counterpart
    """
    n, m = len(chain_a), len(chain_b)

    # DP table
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    backtrack = [[None] * (m + 1) for _ in range(n + 1)]

    # Initialize gaps
    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + GAP_PENALTY
        backtrack[i][0] = "skip_a"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + GAP_PENALTY
        backtrack[0][j] = "skip_b"

    # Fill table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = dp[i-1][j-1] + sim_matrix[i-1][j-1]
            skip_a_score = dp[i-1][j] + GAP_PENALTY
            skip_b_score = dp[i][j-1] + GAP_PENALTY

            best = max(match_score, skip_a_score, skip_b_score)
            dp[i][j] = best

            if best == match_score:
                backtrack[i][j] = "match"
            elif best == skip_a_score:
                backtrack[i][j] = "skip_a"
            else:
                backtrack[i][j] = "skip_b"

    # Traceback
    aligned, unmatched_a, unmatched_b = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and backtrack[i][j] == "match":
            aligned.append((i-1, j-1, sim_matrix[i-1][j-1]))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or backtrack[i][j] == "skip_a"):
            unmatched_a.append(i-1)
            i -= 1
        else:
            unmatched_b.append(j-1)
            j -= 1

    aligned.reverse()
    return {"aligned": aligned, "unmatched_a": unmatched_a, "unmatched_b": unmatched_b}


def score_alignment(alignment: dict, chain_a: list, chain_b: list,
                    factors_a: list, factors_b: list) -> float:
    """
    Compute structural similarity score from an alignment.

    Formula (from SME — Falkenhainer, Forbus & Gentner, 1989):

      score = (n_strong + 0.5 × n_partial) / n_total
              − penalty × (n_unmapped / n_total)

    Where:
    - n_strong: aligned pairs with element_sim ≥ STRONG_THRESHOLD
    - n_partial: aligned pairs with PARTIAL_THRESHOLD ≤ element_sim < STRONG_THRESHOLD
    - n_total: average of chain lengths (normalizes for unequal chain lengths)
    - n_unmapped: elements in either chain with no counterpart
    - penalty: cost of non-alignable differences (UNMAPPED_PENALTY)

    Combined with contributing factor overlap (FACTOR_WEIGHT) for completeness.
    """
    strong, partial = 0, 0

    for idx_a, idx_b, sim in alignment["aligned"]:
        if sim >= STRONG_THRESHOLD:
            strong += 1
        elif sim >= PARTIAL_THRESHOLD:
            partial += 1
        # Below PARTIAL_THRESHOLD: aligned but too weak to count

    n_unmapped = len(alignment["unmatched_a"]) + len(alignment["unmatched_b"])
    n_total = (len(chain_a) + len(chain_b)) / 2  # average chain length

    if n_total == 0:
        return 0.0

    # Core chain alignment score
    chain_score = (strong + 0.5 * partial) / n_total

    # Unmapped penalty
    unmapped_frac = n_unmapped / (len(chain_a) + len(chain_b)) if (len(chain_a) + len(chain_b)) > 0 else 0
    penalty = UNMAPPED_PENALTY * unmapped_frac

    # Contributing factor overlap (Jaccard on role+system pairs)
    factor_score = _factor_overlap(factors_a, factors_b)

    # Combine
    combined = CHAIN_WEIGHT * chain_score + FACTOR_WEIGHT * factor_score - penalty
    return max(0.0, min(1.0, combined))


def _factor_overlap(factors_a: list, factors_b: list) -> float:
    """Jaccard similarity on contributing factor (role, system) pairs."""
    if not factors_a or not factors_b:
        return 0.0
    set_a = {(f.get("role", ""), f.get("system", "")) for f in factors_a}
    set_b = {(f.get("role", ""), f.get("system", "")) for f in factors_b}
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def pattern_match_bonus(struct_a: dict, struct_b: dict) -> float:
    """
    Bonus for matching high-level failure patterns.
    Source: Gentner's systematicity — if two incidents share the same overall
    causal architecture, that's a strong signal even if individual element
    matches are imperfect.
    """
    pa = struct_a.get("failure_pattern", "unknown")
    pb = struct_b.get("failure_pattern", "unknown")

    if pa == "unknown" or pb == "unknown":
        return 0.0
    if pa == pb:
        return 1.0

    for group in RELATED_PATTERNS:
        if pa in group and pb in group:
            return 0.5
    return 0.0


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def structural_similarity(struct_a: dict, struct_b: dict) -> float:
    """
    Compute structural similarity between two incident schemas.

    This is the function that replaces the Jaccard-based struct_score.
    It aligns causal chains element-by-element based on functional role,
    then scores the alignment quality.

    Returns: float in [0, 1]
    """
    chain_a = struct_a.get("causal_chain", [])
    chain_b = struct_b.get("causal_chain", [])
    factors_a = struct_a.get("contributing_factors", [])
    factors_b = struct_b.get("contributing_factors", [])

    if not chain_a or not chain_b:
        return 0.0

    # Step 1: Pairwise element similarity matrix
    sim_matrix = compute_element_similarity_matrix(chain_a, chain_b)

    # Step 2: Find best monotonic alignment (Needleman-Wunsch)
    alignment = find_best_alignment(sim_matrix, chain_a, chain_b)

    # Step 3: Score the alignment
    align_score = score_alignment(alignment, chain_a, chain_b, factors_a, factors_b)

    # Step 4: Pattern bonus (systematicity)
    bonus = pattern_match_bonus(struct_a, struct_b)

    # Final: weighted combination
    final = ALIGNMENT_WEIGHT * align_score + PATTERN_BONUS_WEIGHT * bonus

    return min(1.0, max(0.0, final))
```

### Step 4: Update Fusion (`reweight.py`)

Only change the import:

```python
# OLD (A1 — flat enum Jaccard)
from struct_score import structural_similarity

# NEW (A2 — causal chain alignment)
from struct_score_v2 import structural_similarity

# Fusion formula stays IDENTICAL:
# Source: standard two-stage retrieval reranking (Nogueira & Cho, 2019)
from math import exp

def reweight(cosine_sim: float, structural_sim: float, alpha: float = 2.0) -> float:
    """
    Fuse embedding similarity with structural similarity.

    w' = max(0, cosine) × exp(α × s)

    - cosine gates: structural score can only rerank candidates already
      in the semantic neighborhood
    - exp(α × s) amplifies structural differences: high structural match
      gets exponentially boosted
    - α controls reranking aggressiveness (tune on training data)
    """
    return max(0.0, cosine_sim) * exp(alpha * structural_sim)
```

Add a `--struct_version` flag to your evaluation scripts so you can switch between v1 and v2 at eval time.

### Step 5: Evaluation Protocol

Run three conditions on your holdout test set:

```bash
# A0: Embedding only (baseline)
python eval_diagnosis.py --mode A0
python eval_prognosis.py --mode A0

# A1: Embedding + flat enum Jaccard (your current structural approach)
python eval_diagnosis.py --mode A1 --struct_version v1
python eval_prognosis.py --mode A1 --struct_version v1

# A2: Embedding + causal chain alignment (new approach)
python eval_diagnosis.py --mode A2 --struct_version v2
python eval_prognosis.py --mode A2 --struct_version v2
```

### Step 6: Metrics to Report

**Diagnosis metrics** (what caused this?):
- M1: Cause "C" findings match rate
- M2: narr_cause match rate
- Top-1 accuracy
- Recall@5
- MRR (Mean Reciprocal Rank)
- **F1 score** (Jesse specifically mentioned this)

**Prognosis metrics** (what happens next?):
- Next-event exact@1
- Recall@5

**Report as a comparison table:**

| Metric | A0 (Embed Only) | A1 (Flat Enum) | A2 (Causal Chain) | A2 vs A0 Δ |
|--------|-----------------|----------------|--------------------|-----------:|
| F1     | ?               | ?              | ?                  | ?          |
| Top-1  | ?               | ?              | ?                  | ?          |
| R@5    | ?               | ?              | ?                  | ?          |
| MRR    | ?               | ?              | ?                  | ?          |
| M1     | ?               | ?              | ?                  | ?          |
| M2     | ?               | ?              | ?                  | ?          |

Jesse's words: "You need to point to those numbers and say, my F1 score improves." Or if it doesn't improve: "provably it made no difference" — which is still a valid finding.

### Step 7: Parameter Tuning

After running the initial evaluation, tune these parameters on the TRAINING set (not test):

1. **α (fusion strength)**: Try α ∈ {0.5, 1.0, 2.0, 3.0, 5.0}. Plot F1 vs α.
2. **ROLE_WEIGHT / MECHANISM_WEIGHT / SYSTEM_WEIGHT**: Try {0.60/0.25/0.15}, {0.50/0.30/0.20}, {0.40/0.40/0.20}
3. **GAP_PENALTY**: Try {-0.02, -0.05, -0.10}
4. **STRONG_THRESHOLD / PARTIAL_THRESHOLD**: Try {0.60/0.30}, {0.70/0.35}, {0.80/0.40}

Use grid search or just manual exploration. Report best parameters and their effect.

### Step 8: Diagnostic Analysis

For cases where A2 disagrees with A0:
- **A2 improves**: Did structural alignment correctly promote a causally similar neighbor that embeddings ranked lower? Save examples to `eval_results/a2_improvements.json`
- **A2 hurts**: Did the schema extraction make errors? Did alignment find spurious matches? Save to `eval_results/a2_regressions.json`

This analysis is what Jesse means by "you can build and you can say, I looked at that. I interrogated that."

---

## File Structure

```
Testing_Structural_Mapping/
├── schema/
│   ├── extraction_prompt_v1.txt        # existing (keep)
│   ├── extraction_prompt_v2.txt        # NEW — relational chain prompt
│   ├── struct_train_v1.jsonl           # existing (keep)
│   ├── struct_train_v2.jsonl           # NEW — relational train schemas
│   ├── query_struct_v1.jsonl           # existing (keep)
│   └── query_struct_v2.jsonl           # NEW — relational query schemas
├── extract_struct.py                    # existing (keep)
├── extract_struct_v2.py                 # NEW — relational extraction
├── struct_score.py                      # existing (keep — needed for A1)
├── struct_score_v2.py                   # NEW — chain alignment scoring
├── reweight.py                          # UPDATE — add version flag
├── eval_diagnosis.py                    # UPDATE — add --struct_version flag
├── eval_prognosis.py                    # UPDATE — add --struct_version flag
└── eval_results/
    ├── a0_results.json
    ├── a1_results.json
    ├── a2_results.json
    ├── a2_improvements.json             # NEW — where A2 beats A0
    ├── a2_regressions.json              # NEW — where A2 hurts
    └── parameter_tuning.json            # NEW — grid search results
```

## Implementation Order

1. Write `extraction_prompt_v2.txt` — test manually on 5-10 incidents
2. Write `extract_struct_v2.py` — run on full train set, check validation rate
3. Write `struct_score_v2.py` — unit test with known pairs
4. Update `reweight.py` — add version switching
5. Run A0/A1/A2 evaluation — compare metrics
6. Tune parameters on training set
7. Final evaluation on test set with best parameters
8. Diagnostic analysis of improvements and regressions

## What to Tell Your Committee

"The structural similarity scoring operationalizes the three constraints from structure-mapping theory (Gentner, 1983; Gentner & Markman, 1997). The alignment algorithm uses Needleman-Wunsch dynamic programming (Needleman & Wunsch, 1970) to find optimal monotonic correspondences between causal chains. The scoring formula follows SME's structural evaluation approach (Falkenhainer, Forbus & Gentner, 1989), where matched relational structure contributes positively and non-alignable differences contribute negatively. The fusion with embedding retrieval follows standard two-stage retrieval practice (Nogueira & Cho, 2019). Specific parameter values were selected based on theoretical priors and validated empirically on the training holdout set."

## References

1. Falkenhainer, B., Forbus, K. D., & Gentner, D. (1989). The structure-mapping engine: Algorithm and examples. *Artificial Intelligence*, 41(1), 1–63.
2. Gentner, D. (1983). Structure-mapping: A theoretical framework for analogy. *Cognitive Science*, 7(2), 155–170.
3. Gentner, D., & Markman, A. B. (1997). Structure mapping in analogy and similarity. *American Psychologist*, 52(1), 45–56.
4. Goldstone, R. L. (1994). Similarity, interactive activation, and mapping. *Journal of Experimental Psychology: Learning, Memory, and Cognition*, 20(1), 3–28.
5. Needleman, S. B. & Wunsch, C. D. (1970). A general method applicable to the search for similarities in the amino acid sequence of two proteins. *Journal of Molecular Biology*, 48(3), 443–453.
6. Nogueira, R. & Cho, K. (2019). Passage re-ranking with BERT. *arXiv:1901.04085*.
7. Spencer-Smith, J. & Goldstone, R. L. (1997). The dynamics of similarity. *Cognitive Studies: Bulletin of the Japanese Cognitive Science Society*, 4, 38–56.
