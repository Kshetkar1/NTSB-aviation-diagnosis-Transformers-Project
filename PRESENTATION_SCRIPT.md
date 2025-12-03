# Presentation Script: Aviation Incident Diagnosis Engine
## 13 Minute Word-for-Word Script

**Total Words:** ~1,943 (12:57 minutes at 150 words/min)
**Author:** Kanu Shetkar

---

## Instructions:
- Read this script word-for-word
- Follow [SCREEN] directions to know what to show
- Practice timing: Should be 12 minutes total
- Don't rush - clear articulation is important

---

## **INTRODUCTION (25 seconds - 65 words)**

[SCREEN: Show README_PRESENTATION.md title]

December 28th, 2016. American Airlines Flight 383 catches fire during takeoff at Chicago O'Hare. All passengers evacuate safely, but investigators face a challenge: searching eighty thousand historical incidents to find similar engine fire cases. This manual analysis takes hours.

I'm Kanu Shetkar. My Aviation Incident Diagnosis Engine does this search in seconds using transformer embeddings and LLM agents, demonstrating how transformers enable safety-critical applications while mitigating hallucination risks.

---

## **SECTION 1: PROBLEM STATEMENT (50 seconds - 130 words)**

[SCREEN: Scroll to "1. Problem Statement & Overview"]

Aviation safety analysts must search the NTSB database—over eighty thousand aviation incidents since 1962—to investigate patterns. Manual analysis takes hours. Traditional keyword search fails because semantically identical descriptions use different words: "engine fire during takeoff" won't match "smoke from engine compartment on departure"—same phenomenon, different words.

Why not just ask an LLM? LLMs hallucinate facts, generating plausible diagnoses not grounded in real data. For life-or-death aviation decisions, this is unacceptable.

[SCREEN: Point to architecture diagram]

My hybrid architecture delivers diagnoses in seconds, not hours. It uses transformers for semantic understanding but deterministic tools for facts. Three components: an LLM agent for orchestration, a diagnostic tool for semantic search and weighted Bayesian analysis, and the NTSB database with pre-computed embeddings.

---

## **SECTION 2: METHODOLOGY (4.25 minutes - 695 words)**

[SCREEN: Scroll to "2. Methodology"]

Let me explain how transformer concepts enable this system.

### **Part A: Transformer Embeddings (50 seconds - 120 words)**

[SCREEN: Show "2.1 Transformer Embeddings"]

First, transformer embeddings solve the core aviation safety problem. Traditional keyword search fails because "engine fire during takeoff" and "smoke from engine compartment on departure" share zero keywords, yet describe identical failure modes.

[SCREEN: Point to code snippet]

I use text-embedding-3-small to convert incident narratives into fifteen-hundred-thirty-six dimensional semantic coordinates. Why does this work? The transformer's self-attention mechanism learned during pre-training that "fire," "smoke," and "combustion" are semantically related—and that "takeoff" and "departure" describe the same flight phase.

[SCREEN: Point to example]

My query "engine fire during takeoff" now finds incidents with completely different wording—"fire in engine compartment," "smoke from engine," "combustion event during climb"—because they're close in semantic vector space, not keyword space. This is the fundamental advantage: transformers capture meaning, not just words.

### **Part B: Transformer Architecture Fundamentals (45 seconds - 110 words)**

[SCREEN: Scroll to "2.2 Transformer Architecture: Why Transfer Learning Works"]

Second, why does this transfer learning work without aviation-specific fine-tuning? The multi-head attention architecture is key.

[SCREEN: Point to Multi-Head Attention algorithm]

When I embed "engine fire," different attention heads activate for different semantic dimensions—one captures combustion-related terms like "smoke" and "flame," another captures aircraft component relationships like "engine," "nacelle," "powerplant," and a third captures emergency terminology like "abort," "shutdown," "evacuation." This specialization happens automatically because the architecture learned to decompose meaning into multiple parallel representations.

This is exactly why semantic search achieves eighty-eight percent precision versus forty-two percent for keywords—the architecture captures relationships that simple word matching cannot.

### **Part C: Attention-Inspired Similarity Search (40 seconds - 95 words)**

[SCREEN: Scroll to "2.3 Attention-Inspired Similarity"]

Third, how I adapted attention principles for document-level similarity search. The attention mechanism uses dot products to measure token relationships—Q times K-transpose.

[SCREEN: Point to Attention formula]

I adapt this core intuition to the document level with cosine similarity: query dot product incident divided by their norms. While attention uses softmax-weighted aggregation, my approach uses normalized dot products to measure how aligned two incident vectors are in semantic space.

[SCREEN: Point to algorithm]

My implementation computes cosine similarity against all eighty thousand NTSB incidents, then returns the top fifty matches ranked by relevance.

### **Part D: Similarity-Weighted Aggregation (1 minute - 145 words)**

[SCREEN: Scroll to "2.4 Similarity-Weighted Aggregation"]

Fourth, similarity-weighted aggregation—this is my key innovation combining transformers with Bayesian reasoning.

Here's the problem: I have fifty similar incidents, each with different root causes. How do I combine this evidence? Why not just count causes?

[SCREEN: Point to formula]

Because relevance matters more than frequency. Imagine ten incidents cite "pilot error" but they're only sixty percent similar to my query—that's a total weight of six-point-zero. Now imagine five incidents cite "mechanical failure" but they're ninety-five percent similar—that's a weight of four-point-seven-five.

Simple counting would say "pilot error" is more likely because ten is greater than five. But my weighted approach correctly recognizes that the five highly-similar mechanical failures are more relevant evidence than ten loosely-similar pilot errors.

This is weighted Bayesian reasoning: I'm computing posterior probabilities where more relevant incidents contribute more weight—exactly like attention mechanisms weight tokens by relevance. The formula is: weighted probability of cause j equals sum of similarities where that cause appears, divided by sum of all similarities.

### **Part E: LLM Function Calling (1.5 minutes - 225 words)**

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

Now let me demonstrate this system in action.

[SCREEN: Switch to Streamlit app at localhost:8501]

I've built a Streamlit interface showing all three steps.

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

## **SECTION 5: CRITICAL ANALYSIS (1 minute 10 seconds - 172 words)**

[SCREEN: Scroll to "6. Critical Analysis"]

Impact: diagnoses in seconds versus hours, grounded in historical data.

Three key insights about transformers:

[SCREEN: Point to "What This Reveals"]

First, transfer learning works. Transformers trained on general text transfer to specialized aviation domains without fine-tuning, suggesting they capture fundamental semantic structures.

Second, the hallucination-accuracy tradeoff. LLM generation enables synthesis but risks hallucination. Embeddings enable factual retrieval without generation. Solution: use both—LLMs for orchestration, embeddings for grounding.

Third, attention as a universal pattern. Weighted aggregation appears at token-level, document-level, and evidence-level, suggesting attention is fundamental to information processing.

What surprised me during this project: I expected embeddings to struggle with rare aviation terms like "nacelle" or "turbofan spool." But the model handled them perfectly because multi-head attention composes meaning from context, not just vocabulary. This validates compositional semantics over word memorization.

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

**Total Word Count:** ~1,943 words
**Expected Duration:** 12:57 minutes (at 150 wpm)
**Rubric Coverage:** All 8 sections covered ✓

---

## **Timing Breakdown:**

- Introduction: 25 sec (~65 words) ⭐ REAL INCIDENT HOOK
- Problem Statement: 50 sec (~130 words)
- Methodology: 4.25 min (~695 words) - HIGHEST VALUE (50 pts)
  - Part A: Transformer Embeddings (50 sec - 120 words)
  - Part B: Transformer Architecture Fundamentals (45 sec - 110 words)
  - Part C: Attention-Inspired Similarity Search (40 sec - 95 words) ⭐ FIXED PRECISION
  - Part D: Similarity-Weighted Aggregation (1 min - 145 words) ⭐ EXPANDED
  - Part E: LLM Function Calling (1.5 min - 225 words)
- Demo: 1.5 min (~215 words)
- Evaluation: 1 min (~180 words) - HIGH VALUE (15 pts)
- Critical Analysis: 1 min 10 sec (~172 words) ⭐ ADDED REFLECTION
- Wrap-up: 20 sec (~50 words)
- **Total: ~12:57 minutes (1,943 words)**

**Note:** Under 13-minute limit. Improvements: (1) Opening hook with AA383 incident, (2) Fixed cosine≠attention precision, (3) Added "what surprised me" reflection showing compositional semantics insight.

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
