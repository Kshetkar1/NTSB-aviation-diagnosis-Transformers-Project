# Presentation Script: Aviation Incident Diagnosis Engine
## 12 Minute Word-for-Word Script

**Total Words:** ~1,896 (12:38 minutes at 150 words/min)
**Author:** Kanu Shetkar

---

## Instructions:
- Read this script word-for-word
- Follow [SCREEN] directions to know what to show
- Practice timing: Should be 12 minutes total
- Don't rush - clear articulation is important

---

## **INTRODUCTION (25 seconds - 60 words)**

[SCREEN: Show README_PRESENTATION.md title]

Good morning. I'm Kanu Shetkar, presenting an Aviation Incident Diagnosis Engine that combines transformer embeddings with LLM agents for safety analysis.

This demonstrates how transformer architectures—embeddings, attention, and function calling—enable real-world safety applications while mitigating LLM hallucination risks that make pure generation unreliable for critical systems.

---

## **SECTION 1: PROBLEM STATEMENT (50 seconds - 130 words)**

[SCREEN: Scroll to "1. Problem Statement & Overview"]

Aviation safety analysts must search the NTSB database—over eighty thousand aviation incidents since 1962—to investigate patterns. Manual analysis takes hours. Traditional keyword search fails because semantically identical descriptions use different words: "engine fire during takeoff" won't match "smoke from engine compartment on departure"—same phenomenon, different words.

Why not just ask an LLM? LLMs hallucinate facts, generating plausible diagnoses not grounded in real data. For life-or-death aviation decisions, this is unacceptable.

[SCREEN: Point to architecture diagram]

My hybrid architecture delivers diagnoses in seconds, not hours. It uses transformers for semantic understanding but deterministic tools for facts. Three components: an LLM agent for orchestration, a diagnostic tool for semantic search and weighted Bayesian analysis, and the NTSB database with pre-computed embeddings.

---

## **SECTION 2: METHODOLOGY (4.5 minutes - 650 words)**

[SCREEN: Scroll to "2. Methodology"]

Let me explain how transformer concepts enable this system.

### **Part A: Transformer Embeddings (50 seconds - 120 words)**

[SCREEN: Show "2.1 Transformer Embeddings"]

First, transformer embeddings. From Phuong and Hutter's Formal Algorithms paper, token embeddings map discrete vocabulary to continuous vector representations in d-e dimensional real space.

[SCREEN: Point to code snippet]

I extend this to document-level embeddings using text-embedding-3-small, producing fifteen-hundred-thirty-six dimensional vectors. One API call converts an incident narrative into semantic coordinates.

Why does this work? Transformer self-attention learns semantic relationships during pre-training.

[SCREEN: Point to example]

My query "engine fire during takeoff" finds incidents with different wording—"fire in engine compartment," "smoke from engine"—all semantically related. Cosine similarity scores quantify closeness in vector space.

### **Part B: Attention-Inspired Similarity (1 minute - 150 words)**

[SCREEN: Scroll to "2.3 Attention-Inspired Similarity"]

Second, attention-inspired similarity search. Let me connect this to the attention mechanism from Vaswani et al.'s "Attention Is All You Need."

[SCREEN: Point to Attention formula]

Attention equals softmax of Q times K-transpose divided by square root of d-k, times V. The Q-K dot product measures query-key relationships. The square-root-d-k scaling is crucial—it prevents softmax saturation in high dimensions, stabilizing gradients.

I apply this same dot product principle with cosine similarity: q dot d divided by norms—a normalized dot product measuring document-level relationships, just like attention measures token-level relationships.

[SCREEN: Point to algorithm]

My implementation computes similarities for all N incidents, returning top-k matches. Both attention and my search use dot products to measure relationships and weight evidence by relevance.

### **Part B2: Transformer Architecture Fundamentals (1 minute - 150 words)**

[SCREEN: Scroll to "2.2 Transformer Architecture: Key Components"]

Let me explain the transformer fundamentals that make my system possible.

[SCREEN: Point to Multi-Head Attention algorithm]

Multi-head attention computes H parallel attention functions with different learned projections Q-h, K-h, V-h, then concatenates results. Why multiple heads? Different heads specialize: one captures syntactic patterns, another captures synonyms, another positional relationships.

This is exactly why text-embedding-3-small works for aviation safety—its multi-head architecture learned to capture diverse semantic relationships. When I embed "engine fire," different attention heads activate for combustion semantics, aircraft component relationships, and emergency terminology.

[SCREEN: Point to Layer Normalization]

Layer normalization stabilizes training in deep networks by normalizing activations across dimensions. Both GPT-4-o-mini and text-embedding-3-small are deep transformers with twenty-plus layers. Without layer norm preventing gradient vanishing, training these deep models would fail—the embeddings we rely on wouldn't exist.

### **Part C: Similarity-Weighted Aggregation (30 seconds - 75 words)**

[SCREEN: Scroll to "2.4 Similarity-Weighted Aggregation"]

Third, similarity-weighted aggregation. How do we combine evidence from fifty different incidents with different root causes?

[SCREEN: Point to formula]

The weighted probability of cause j equals the sum of similarities where that cause appears, divided by the sum of all similarities.

This is the same weighted aggregation principle as attention—more similar incidents contribute more to the final probability, just like high-attention tokens contribute more to transformer outputs.

### **Part D: LLM Function Calling (1.5 minutes - 225 words)**

[SCREEN: Scroll to "2.5 LLM Function Calling"]

Finally, LLM function calling—this is how the system orchestrates everything.

In traditional LLM usage, you ask a question and the LLM generates an answer. But for safety-critical applications, that's dangerous because the LLM might hallucinate facts.

Function calling solves this problem by giving the LLM access to reliable tools.

[SCREEN: Point to the three-step workflow diagram]

My agent has three steps. Step one: the LLM generates a detailed incident report from a brief query. This demonstrates its generation capability—it's creating a plausible, NTSB-style report.

Step two: the LLM recognizes it needs factual data to make a diagnosis. So it calls my diagnostic tool—this is function calling. The tool executes the semantic search and weighted Bayesian analysis I just explained, and returns statistical results from real historical data.

Step three: the LLM receives those tool results and synthesizes them into a human-readable summary. It's interpreting and explaining the statistics, but it's not inventing the diagnostic facts—those came from the tool.

[SCREEN: Point to code]

This is the actual implementation. Three API calls: one for generation, one with tools defined for the tool call, and one for synthesis.

Why does this matter? Because it demonstrates separation of concerns. The LLM handles understanding and explanation. The tool handles facts. Neither can hallucinate the diagnosis because it's retrieved from historical data, not generated.

---

## **SECTION 3: DEMO (1.5 minutes - 210 words)**

[SCREEN: Switch to Streamlit app at localhost:8501]

Let me demonstrate the system. I've built a Streamlit interface showing all three steps.

[SCREEN: Enter query and click "Run Diagnostic Agent"]

Query: "engine fire during takeoff."

[SCREEN: Scroll to Output 1]

Step one: LLM-generated incident synopsis. Notice the detailed NTSB-style report—aircraft type, flight phase, event summary. This demonstrates transformer text generation.

[SCREEN: Scroll to Output 2]

Step two: diagnostic tool results—factual data from history, not LLM hallucinations.

[SCREEN: Point to top cause]

Top cause: "Fatigue, wear, corrosion" at thirty-two-point-four percent probability, based on twenty-three similar incidents with zero-point-eight-one average similarity.

[SCREEN: Point to distribution]

Each probability uses the weighted aggregation I explained. More similar incidents contribute more weight.

This demonstrates three transformer concepts: embeddings enabled semantic search finding fifty similar incidents, attention-inspired similarity ranked them, weighted aggregation calculated probabilities.

[SCREEN: Scroll to Output 3]

Step three: LLM synthesis. The LLM interprets statistical results in plain English, providing context but not inventing facts—everything is grounded in tool output.

This is the complete hybrid architecture: LLM orchestration with deterministic factual grounding.

---

## **SECTION 4: EVALUATION (1 minute - 180 words)**

[SCREEN: Switch back to README, scroll to "4. Assessment & Evaluation"]

Let me cover evaluation and model cards.

[SCREEN: Point to Performance Evaluation table]

I evaluated semantic search against keyword baseline using five test queries. Methodology: for each query, I manually reviewed top-ten results and labeled them relevant or irrelevant based on whether they described similar incident types.

Results: semantic search achieved eighty-eight-point-three percent average precision versus forty-two percent for keyword matching—a two-point-one-times improvement. For "engine fire during takeoff," semantic search found "smoke from engine compartment on departure"—keyword search missed it. This validates transformer embeddings capture semantics, not just lexical matching.

[SCREEN: Point to model architecture table]

Two models power this: GPT-4-o-mini for generation, orchestration, and synthesis, and text-embedding-3-small for semantic similarity with fifteen-hundred-thirty-six dimensions.

[SCREEN: Point to Intended Uses]

This system is appropriate for research, historical pattern analysis, and educational demonstration of hybrid AI. It's not intended for real-time flight operations or regulatory compliance without validation. It provides probabilities, not certainties—it supports expert analysis but doesn't replace it.

[SCREEN: Point to Ethical Considerations]

Ethical considerations: The historical data may reflect reporting biases. My mitigation is the weighted approach that accounts for relevance. LLM hallucinations are mitigated because the diagnostic facts come from the tool, not generation. For safety-critical domains, this is educational use only and requires expert validation.

[SCREEN: Briefly show the model cards tables]

Complete model and data cards are in the README with licenses, documentation links, and specifications.

---

## **SECTION 5: CRITICAL ANALYSIS (55 seconds - 135 words)**

[SCREEN: Scroll to "6. Critical Analysis"]

Impact: diagnoses in seconds versus hours, grounded in historical data.

Three key insights about transformers:

[SCREEN: Point to "What This Reveals"]

First, transfer learning works. Transformers trained on general text transfer to specialized aviation domains without fine-tuning, suggesting they capture fundamental semantic structures.

Second, the hallucination-accuracy tradeoff. LLM generation enables synthesis but risks hallucination. Embeddings enable factual retrieval without generation. Solution: use both—LLMs for orchestration, embeddings for grounding.

Third, attention as a universal pattern. Weighted aggregation appears at token-level, document-level, and evidence-level, suggesting attention is fundamental to information processing.

[SCREEN: Point to Limitations]

Limitations: small-scale evaluation, no temporal analysis, no uncertainty quantification.

[SCREEN: Point to Future Work]

Future work: expert validation with NTSB analysts, FAISS for scalability, temporal trend analysis.

---

## **SECTION 6: WRAP-UP (20 seconds - 50 words)**

[SCREEN: Scroll to bottom showing Repository link]

In summary: transformer architectures—embeddings, attention, and function calling—enable responsible safety-critical AI applications.

[SCREEN: Show GitHub URL]

Code, model cards, and documentation: https://github.com/Kshetkar1/NTSB-aviation-diagnosis-Transformers-Project.

Thank you. Questions?

---

## **END OF PRESENTATION**

**Total Word Count:** ~1,830 words
**Expected Duration:** 12 minutes
**Rubric Coverage:** All 8 sections covered ✓

---

## **Timing Breakdown:**

- Introduction: 25 sec (~60 words)
- Problem Statement: 50 sec (~120 words)
- Methodology: 4.5 min (~650 words) - HIGHEST VALUE (50 pts)
  - Part A: Transformer Embeddings (50 sec - 120 words)
  - Part B: Attention & Similarity (1 min - 140 words)
  - Part B2: Multi-Head Attention (1 min - 140 words)
  - Part C: Similarity-Weighted Aggregation (30 sec - 75 words)
  - Part D: LLM Function Calling (1.5 min - 175 words)
- Demo: 1.5 min (~210 words)
- Evaluation: 1 min (~180 words) - HIGH VALUE (15 pts)
- Critical Analysis: 50 sec (~120 words)
- Wrap-up: 20 sec (~50 words)
- **Total: ~12 minutes (1,830 words)**

---

## **Q&A Preparation:**

**Likely Questions:**

1. **"Why not fine-tune an LLM on NTSB data?"**
   → Cost: Fine-tuning requires massive compute. Flexibility: Embeddings allow easy updates. Hallucination risk: Fine-tuned models still generate, not retrieve. My approach: Retrieval is deterministic and verifiable.

2. **"How do you validate accuracy?"**
   → Currently: Informal validation (results match domain knowledge). Transparency: Show supporting evidence (incident IDs, similarity scores). Future: Expert review study, ground truth comparison.

3. **"Why weighted vs. simple counting?"**
   → Accounts for relevance, not just frequency. More similar incidents contribute more (like attention weights). Example: 95% similar incident weighs more than 62% similar.

4. **"Could this work with other models?"**
   → Yes! Architecture is model-agnostic. Any model with embeddings + function calling works. Trade-offs: open-source (cost) vs. proprietary (quality).

5. **"How scalable with millions of incidents?"**
   → Current: Order N brute-force search. Optimization: Approximate nearest neighbors (FAISS, Annoy). With ANN: Sub-linear search, handles millions efficiently.
