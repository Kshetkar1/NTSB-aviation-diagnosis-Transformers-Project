# Aviation Incident Diagnosis Engine: Transformers for Safety Analysis

**Author:** Kanu Shetkar
**Repository:** https://github.com/Kshetkar1/NTSB-aviation-diagnosis-Transformers-Project

---

## Abstract

This project demonstrates how transformer embeddings, LLM function calling, and weighted Bayesian analysis can diagnose probable root causes of aviation incidents. Rather than relying on LLMs to generate safety-critical facts (which risks hallucinations), we use transformers as intelligent orchestrators—coordinating semantic search with deterministic diagnostic tools to produce reliable, explainable results from historical NTSB data.

**Key Innovation:** Similarity-weighted evidence aggregation where transformer attention mechanisms meet Bayesian reasoning.

---

## 1. Problem Statement & Overview (10 pts)

### The Challenge

Aviation safety analysts must search thousands of NTSB incident reports to identify probable root causes. **Keyword search fails** because semantically similar descriptions use different words:
- "engine fire during takeoff" ≠ "smoke from engine compartment on departure"
- Yet both describe the same phenomenon

**Why not just ask an LLM?** LLMs can hallucinate facts—unacceptable for safety-critical applications.

### The Solution: Hybrid Transformer Architecture

```
┌─────────────────────────────────────────┐
│   USER QUERY: "engine fire during takeoff"  │
└──────────────────┬──────────────────────┘
                   ↓
         ┌─────────────────────┐
         │   LLM AGENT         │
         │   (GPT-4o-mini)     │
         │                     │
         │  1. Generate Report │
         │  2. Call Tool      │───→ Function Calling
         │  3. Synthesize      │
         └──────────┬──────────┘
                    ↓
         ┌─────────────────────────┐
         │   DIAGNOSTIC TOOL       │
         │   • Semantic Search     │
         │   • Weighted Bayesian   │
         │   • Cosine Similarity   │
         └──────────┬──────────────┘
                    ↓
         ┌─────────────────────────┐
         │   NTSB DATA (1536-dim)  │
         └─────────────────────────┘
```

**Architecture Principles:**
1. Transformers for semantic understanding (embeddings)
2. LLMs as orchestrators (not fact generators)
3. Deterministic tools for verifiable facts
4. Weighted aggregation (attention-inspired)

---

## 2. Methodology: Transformers in Action (50 pts)

### 2.1 Transformer Embeddings: Semantic Vector Spaces

**Theoretical Foundation** (Phuong & Hutter, 2022 - Formal Algorithms):

```
Algorithm: Token Embedding
────────────────────────────
Input:  v ∈ V (token)
Output: e ∈ ℝ^d_e (vector)
Parameters: W_e ∈ ℝ^(d_e × |V|)

return e = W_e[:, v]
```

**Our Extension:** Document-level embeddings using OpenAI text-embedding-3-small:

```python
def get_embedding(text: str) -> np.ndarray:
    """Maps incident narrative to semantic vector space (ℝ^1536)"""
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
```

**Why it works:** Transformer self-attention learns semantic relationships during pre-training.

**Example:**
```
Query: "engine fire during takeoff" → [0.023, -0.145, ..., 0.034] (1536-dim)

Top Matches (by semantic meaning):
1. "fire in engine compartment on departure"  → 0.87 similarity
2. "smoke observed from engine during climb"  → 0.82 similarity
3. "engine failure shortly after takeoff"     → 0.79 similarity
```

### 2.2 Attention-Inspired Similarity Search

**Attention Mechanism** (Vaswani et al., 2017):
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The attention scores `QK^T` measure **relatedness** between query and keys.

**Our Adaptation:** Cosine similarity (normalized dot product):

```
Algorithm: Semantic Search
──────────────────────────
Input:  q ∈ ℝ^d (query embedding)
        D ∈ ℝ^(N×d) (database embeddings)
Output: Ranked incidents

For each incident i:
    similarity[i] = (q · D[i]) / (||q|| × ||D[i]||)

return top_k incidents by similarity
```

**Implementation:**
```python
def find_top_matches(query_embedding, embeddings, top_k=50):
    """O(N·d) vectorized similarity search"""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    return similarities[sorted_indices[:top_k]]
```

**Connection to Attention:** Both use dot products to measure relationships, both aggregate information weighted by relevance.

### 2.3 Weighted Bayesian Diagnosis

**The Problem:** Given 50 similar incidents, how to aggregate evidence from multiple root causes?

**Naive Approach:** Count frequency (treats incident at rank 1 same as rank 50)
**Our Approach:** Similarity-weighted probability

```
Algorithm: Weighted Bayesian Diagnosis
──────────────────────────────────────
Input:  [(s_1, C_1), (s_2, C_2), ..., (s_n, C_n)]
        where s_i = similarity, C_i = causes

1. For each cause c:
   evidence[c] = Σ(similarity where c appears)

2. total = Σ(all similarities)

3. P(c) = evidence[c] / total

Output: Ranked causes by probability
```

**Mathematical Formulation:**
```
P_weighted(cause_j) = Σ_{i: cause_j ∈ C_i} sim(q, d_i) / Σ_{i=1}^{n} sim(q, d_i)
```

**Why better:** More similar incidents contribute more (like attention weights).

### 2.4 LLM Function Calling: Tool Orchestration

**Function Calling Paradigm:**
```
Traditional: User → LLM → [generated diagnosis] ❌ (hallucination risk)

Function Calling:
User → LLM → "I need to call diagnostic tool"
         ↓
    Tool → [factual statistical results]
         ↓
    LLM → [synthesizes + explains] ✅ (facts from tool)
```

**Three-Step Agent:**
```python
def run_diagnosis_agent(user_query):
    # Step 1: Generate detailed report (shows LLM capability)
    report = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Generate synopsis: {user_query}"}]
    ).choices[0].message.content

    # Step 2: Tool call (demonstrates reasoning + function calling)
    tool_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Diagnose: {report}"}],
        tools=tools,
        tool_choice="auto"
    )

    # Execute tool
    results = diagnose_root_causes(report)

    # Step 3: Synthesize (demonstrates interpretation)
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "tool", "content": json.dumps(results)}]
    ).choices[0].message.content

    return {report, results, final}
```

**Why This Matters:**
- **Separation:** LLM handles understanding/explanation, Tool handles facts
- **Reliability:** Diagnostic facts retrieved, not generated
- **Transparency:** Tool results separately visible

---

## 3. Implementation & Demo (20 pts)

### Key Algorithms

**Algorithm 1: Embedding Generation**
```python
# O(1) API call
embedding = client.embeddings.create(
    input=[text],
    model="text-embedding-3-small"
).data[0].embedding
```

**Algorithm 2: Weighted Diagnosis**
```python
# O(n·c) where n=incidents, c=avg causes
def calculate_weighted_diagnosis(scores, matches, top_n=50):
    cause_evidence = defaultdict(list)

    for score, match in zip(scores[:top_n], matches[:top_n]):
        for cause in match['causes']:
            cause_evidence[cause].append(score)

    total = sum(sum(scores) for scores in cause_evidence.values())

    return [{
        'cause': cause,
        'probability': sum(scores) / total,
        'num_incidents': len(scores)
    } for cause, scores in cause_evidence.items()]
```

### Live Demo (Streamlit)

**Demo Flow:**
1. Enter: "engine fire during takeoff"
2. **Output 1:** LLM-generated detailed report
3. **Output 2:** Tool results (Top 10 causes with probabilities)
   - Example: "Fatigue/wear/corrosion: 32.4% (23 incidents, avg sim: 0.81)"
4. **Output 3:** LLM synthesis explaining results

**Run:** `streamlit run streamlit_app.py`

---

## 4. Assessment & Evaluation (15 pts)

### Model Architecture

| Component | Model | Purpose |
|-----------|-------|---------|
| **LLM** | GPT-4o-mini | Report generation, orchestration, synthesis |
| **Embeddings** | text-embedding-3-small | Semantic similarity (1536 dims) |
| **Context** | 128K tokens | Multi-turn reasoning |

### Intended Uses ✅ & Limitations ❌

**Appropriate:**
- ✅ Research and historical pattern analysis
- ✅ Educational demonstration of hybrid AI
- ✅ Supporting (not replacing) expert analysis

**Not Intended:**
- ❌ Real-time flight operations decisions
- ❌ Regulatory compliance without validation
- ❌ Definitive causal determinations (provides probabilities)

### Ethical Considerations

**Data Bias:** Historical NTSB data may reflect reporting biases
**Mitigation:** Weighted approach accounts for relevance, show supporting evidence

**LLM Hallucinations:** LLMs can generate false information
**Mitigation:** Tool provides factual data, LLM only interprets (not invents)

**Safety-Critical Domain:** Aviation decisions have life-or-death consequences
**Mitigation:** Educational use only, requires expert validation

---

## 5. Model & Data Cards (5 pts)

### Model Card: GPT-4o-mini

| Property | Value |
|----------|-------|
| **Provider** | OpenAI |
| **Architecture** | Decoder-only transformer |
| **Context** | 128,000 tokens |
| **Capabilities** | Generation, function calling, reasoning |
| **Usage** | Report generation, orchestration, synthesis |
| **Docs** | https://platform.openai.com/docs/models/gpt-4o-mini |

### Model Card: text-embedding-3-small

| Property | Value |
|----------|-------|
| **Dimensions** | 1536 |
| **Max Input** | 8191 tokens |
| **Use Case** | Semantic similarity, dense retrieval |
| **Usage** | Converting narratives to vectors |
| **Docs** | https://platform.openai.com/docs/guides/embeddings |

### Data Card: NTSB Aviation Database

| Property | Value |
|----------|-------|
| **Source** | National Transportation Safety Board |
| **License** | Public domain (U.S. government data) |
| **Size** | Thousands of incidents |
| **Format** | Structured JSON (post-processing) |
| **Access** | https://www.ntsb.gov/Pages/AviationQuery.aspx |

---

## 6. Critical Analysis (10 pts)

### Impact

**Aviation Safety:**
- Enables rapid pattern recognition (seconds vs. hours)
- Semantic search finds relevant cases with different terminology
- Evidence-based recommendations grounded in historical precedent

**Transformer Research:**
- Demonstrates practical hybrid architecture (embeddings + function calling + tools)
- Shows LLM hallucination mitigation for safety-critical domains
- Example of LLMs as orchestrators, not oracles

**Technical Innovation:**
- Similarity-weighted Bayesian combines semantic search with statistical aggregation
- Attention-inspired diagnosis applies transformer concepts outside model architecture

### What This Reveals

**About Transformers:**
- ✅ Embeddings excel at semantic understanding
- ✅ Function calling extends capabilities safely
- ✅ Multi-step reasoning handles complex tasks
- ❌ Generation alone insufficient for safety-critical facts

**About Hybrid AI:**
- LLM understanding + deterministic accuracy > either alone
- Use LLMs as coordinators, not knowledge sources

### Limitations & Next Steps

**Current Limitations:**
1. No quantitative evaluation (no ground truth validation)
2. No temporal analysis (doesn't track trends over time)
3. No confidence intervals (probabilities lack uncertainty)

**Future Work:**
- **Short-term:** Expert validation study, performance optimization (ANN search)
- **Medium-term:** Temporal analysis, causal chain visualization
- **Long-term:** Predictive modeling, multi-modal analysis

---

## 7. Documentation & Resources (5 pts)

### Setup

```bash
git clone https://github.com/Kshetkar1/NTSB-aviation-diagnosis-Transformers-Project.git
cd NTSB-aviation-diagnosis-Transformers-Project
pip install -r requirements.txt

# Add OpenAI API key
echo "OPENAI_API_KEY=your-key" > .env

# Run app
streamlit run streamlit_app.py
```

### Course-Relevant Papers

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   https://arxiv.org/abs/1706.03762
   *Relevance: Attention mechanism foundation*

2. **"Formal Algorithms for Transformers"** (Phuong & Hutter, 2022)
   https://arxiv.org/abs/2207.09238
   *Relevance: Embeddings (Alg 1), Attention (Alg 4-5), GPT (Alg 10)*

3. **OpenAI Function Calling Guide**
   https://platform.openai.com/docs/guides/function-calling
   *Relevance: Tool use implementation*

4. **OpenAI Embeddings Guide**
   https://platform.openai.com/docs/guides/embeddings
   *Relevance: Semantic search*

---

## Summary: Transformer Concepts Applied

| Transformer Concept | Our Application |
|---------------------|-----------------|
| Token Embeddings | Document embeddings for incidents |
| Attention Mechanism | Cosine similarity for relevance |
| Softmax Aggregation | Weighted probability calculation |
| Decoder Architecture | LLM generates reports (GPT-style) |
| Function Calling | Tool orchestration |
| Multi-step Reasoning | 3-step agent workflow |

**This project demonstrates how transformer architectures can be responsibly deployed in safety-critical domains by partnering LLM capabilities with deterministic analysis tools.**

---

**Repository:** https://github.com/Kshetkar1/NTSB-aviation-diagnosis-Transformers-Project
**Author:** Kanu Shetkar
