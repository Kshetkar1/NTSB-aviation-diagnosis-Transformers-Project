# Presentation Script: Aviation Incident Diagnosis Engine
## 13 Minute Word-for-Word Script

**Total Words:** ~1,300 (13:00 minutes at 150 words/min)
**Author:** Kanu Shetkar

---

## Instructions:
- Read only the "SAY THIS" sections word-for-word
- Follow screen directions at the start of each section
- Practice timing: Should be ~13 minutes total
- Don't rush - clear articulation is important

---

## **INTRODUCTION (25 seconds - 65 words)**

**📺 SCREEN:**
- Show README_PRESENTATION.md title

**SAY THIS:**

December 28th, 2016. American Airlines Flight 383 catches fire during takeoff at Chicago O'Hare. All passengers evacuate safely, but investigators face a challenge: searching eighty thousand historical incidents to find similar engine fire cases. This manual analysis takes hours.

I'm Kanu Shetkar. My Aviation Incident Diagnosis Engine does this search in seconds using transformer embeddings and LLM agents, demonstrating how transformers enable safety-critical applications while mitigating hallucination risks.

---

## **SECTION 1: PROBLEM STATEMENT (55 seconds - 145 words)**

**📺 SCREEN:**
- Scroll to "1. Problem Statement & Overview"
- Point to architecture diagram when mentioned

**SAY THIS:**

Aviation safety analysts must search the NTSB database—over eighty thousand aviation incidents since 1962—to investigate patterns. Manual analysis takes hours. Traditional keyword search fails because semantically identical descriptions use different words: "engine fire during takeoff" won't match "smoke from engine compartment on departure"—same phenomenon, different words.

Why not just ask an LLM? LLMs hallucinate facts, generating plausible diagnoses not grounded in real data. For life-or-death aviation decisions, this is unacceptable.

My hybrid architecture delivers diagnoses in seconds, not hours. As you can see in this workflow diagram, a query comes in, goes through three steps—generate, diagnose, synthesize—and outputs the final diagnosis. Three components power this: an LLM agent for orchestration, a diagnostic tool for semantic search and similarity-weighted aggregation, and the NTSB database with pre-computed embeddings.

---

## **SECTION 2: METHODOLOGY (3 minutes 5 seconds - 500 words)**

**📺 SCREEN:**
- Note the formal algorithms link at top of section

**SAY THIS (5 sec):**

Before diving in, note that all formal algorithm specifications are available in the linked PDF document.

---

### **Part A: Transformer Embeddings (45 seconds - 120 words)**

**📺 SCREEN:**
- Scroll to "2.1 Transformer Embeddings"
- Point to code snippet when mentioned
- Point to example "engine fire during takeoff"

**SAY THIS:**

Let me explain how transformer concepts enable this system.

First, transformer embeddings. I use text-embedding-3-small to convert incident narratives into fifteen-hundred-thirty-six dimensional semantic coordinates. How did the transformer learn these semantic relationships?

During pre-training on billions of text examples, the model learned to predict masked words. To correctly predict "fire" from surrounding context, it had to learn that "smoke," "combustion," and "flames" appear in similar contexts. This forces the architecture to position semantically related words close together in vector space—not through manual rules, but through statistical patterns in language.

My query "engine fire during takeoff" now finds incidents with completely different wording—"fire in engine compartment," "smoke from engine," "combustion event during climb"—because they're close in semantic vector space, not keyword space. This is the fundamental advantage: transformers capture meaning, not just words.

---

### **Part B: Transformer Architecture Fundamentals (20 seconds - 50 words)**

**📺 SCREEN:**
- Scroll to "2.2 Transformer Architecture: Why Transfer Learning Works"
- Point to Multi-Head Attention algorithm when mentioned

**SAY THIS:**

Second, why does this transfer learning work without aviation-specific fine-tuning? Multi-head attention is key—here's the formal algorithm. Different attention heads specialize in different semantic dimensions—combustion terms, aircraft components, emergency terminology. This automatic decomposition of meaning explains why semantic search achieves eighty-eight percent precision versus forty-two percent for keywords.

---

### **Part C: Attention-Inspired Similarity Search (25 seconds - 60 words)**

**📺 SCREEN:**
- Scroll to "2.3 Attention-Inspired Similarity"
- Point to visual diagram when mentioned

**SAY THIS:**

Third, adapting attention principles for document-level similarity. This visual shows the process: the query embedding is compared against incident embeddings using cosine similarity—notice how fire-related incidents score high, point-eight-seven and point-eight-two, while unrelated ones score low, point-two-three. Just as attention uses dot products—Q times K-transpose—I use cosine similarity to measure how aligned query and incident vectors are in semantic space.

---

### **Part D: Similarity-Weighted Aggregation (1 minute 20 seconds - 190 words)**

**📺 SCREEN:**
- Scroll to "2.4 Similarity-Weighted Aggregation"
- Point to formula when mentioned

**SAY THIS:**

Fourth, similarity-weighted aggregation—this is my key innovation adapting attention-style weighting to diagnostic reasoning.

Here's the problem: I have forty-three similar incidents, each with different root causes. How do I combine this evidence? Why not just count causes?

Because relevance matters more than frequency. Imagine ten incidents cite "pilot error" but they're only sixty percent similar to my query—that's a total weight of six-point-zero. Now imagine five incidents cite "mechanical failure" but they're ninety-five percent similar—that's a weight of four-point-seven-five.

Simple counting would say "pilot error" is more likely because ten is greater than five. But my weighted approach correctly recognizes that the five highly-similar mechanical failures are more relevant evidence than ten loosely-similar pilot errors.

Why this approach? Because similarity scores represent evidence strength. A ninety-five percent similar incident is much stronger evidence than a sixty percent similar one—it should contribute proportionally more to the probability. This treats similarity as confidence weights, just like attention mechanisms weight tokens by relevance.

The formula: weighted probability of cause j equals sum of similarities where that cause appears, divided by sum of all similarities. This ensures probabilities sum to one while respecting evidence quality.

---

### **Part E: LLM Function Calling (50 seconds - 110 words)**

**📺 SCREEN:**
- Scroll to "2.5 LLM Function Calling"
- Point to three-step workflow diagram when mentioned

**SAY THIS:**

Finally, LLM function calling—this orchestrates the system while preventing hallucination.

My agent has three steps. Step one: generate an NTSB-style incident report.

Step two: call the diagnostic tool for factual data. This is function calling—the LLM retrieves the diagnosis, doesn't generate it. The tool executes semantic search and weighted aggregation.

Step three: synthesize results into plain English, grounded in tool output, not generated.

Three API calls implement this: generation, tool call with diagnostic function defined, and synthesis.

This demonstrates separation of concerns: the LLM handles orchestration and explanation, the tool handles facts. The diagnosis cannot be hallucinated because it's retrieved from historical data, not generated.

---

## **SECTION 3: DEMO (1 minute - 150 words)**

**📺 SCREEN:**
- Switch to Streamlit app at localhost:8501
- Enter query "engine fire during takeoff" and click "Run Diagnostic Agent"
- Focus on Output 3 (detailed diagnostic data)

**SAY THIS:**

Let me demonstrate this system in action. Query: "engine fire during takeoff."

The Streamlit interface shows three outputs. Output one: LLM-generated NTSB-style incident synopsis.

Output two: LLM synthesis interpreting the statistics in plain English—contextualizing probabilities and explaining implications.

Output three—the critical diagnostic results—shows factual data from historical analysis, not hallucinations.

The system found forty-three similar historical incidents. The top three causes each have four percent probability: maintenance personnel installation issues, engine turbine section failure, and turbine section fatigue and corrosion—each appearing in three of the forty-three similar incidents.

Each probability uses similarity-weighted aggregation—more similar incidents contribute more weight.

This demonstrates the complete transformer pipeline: embeddings enabled semantic search finding forty-three relevant cases, attention-inspired similarity ranked them by relevance, and weighted aggregation calculated evidence-based probabilities. LLM orchestration with deterministic factual grounding.

---

## **SECTION 4: EVALUATION (1 minute - 140 words)**

**📺 SCREEN:**
- Switch back to README, scroll to "4. Assessment & Evaluation"
- Point to Performance Evaluation table
- Point to model architecture table
- Point to Intended Uses section
- Point to Ethical Considerations section
- Briefly show model cards tables at end

**SAY THIS:**

Let me cover evaluation and model cards.

I evaluated semantic search against keyword baseline using five test queries, manually reviewing top-ten results. Semantic search achieved eighty-eight percent precision versus forty-two percent for keywords—a two-times improvement. This improvement was consistent across all test queries, suggesting systematic advantage. This validates transformer embeddings capture semantics, not just lexical matching.

Two models power this system. GPT-4-o-mini for generation and orchestration, text-embedding-3-small for semantic similarity. The NTSB aviation database is public domain U.S. government data available at NTSB.gov. Complete model and data cards are in the README.

This system is for research and education, not real-time flight operations. It supports expert analysis but doesn't replace it.

Ethical considerations: Historical data may reflect biases, mitigated by weighted relevance. Diagnostic facts come from the tool, not generation, preventing hallucination.

---

## **SECTION 5: CRITICAL ANALYSIS (1 minute - 150 words)**

**📺 SCREEN:**
- Scroll to "6. Critical Analysis"

**SAY THIS:**

Impact: diagnoses in seconds versus hours, grounded in historical data.

Four key insights: First, transfer learning works—transformers trained on general text transfer to aviation without fine-tuning, capturing fundamental semantic structures.

Second, the hallucination-accuracy tradeoff. LLM generation enables synthesis but risks hallucination. Embeddings enable factual retrieval. Solution: hybrid architecture—LLMs for orchestration, embeddings for grounding.

Third, attention appears as a universal pattern. We see weighted aggregation by relevance at token-level in multi-head attention, document-level in cosine similarity, and evidence-level in similarity-weighted diagnosis. This suggests attention is a fundamental computational principle, not just a neural network technique.

What surprised me: I expected embeddings to struggle with rare aviation terms like "nacelle" or "empennage." But the model handled them perfectly because multi-head attention composes meaning from context, not just vocabulary. This validates compositional semantics over word memorization.

Current limitations: small-scale evaluation with five test queries, no temporal trend analysis, and no uncertainty quantification.

Future work: validate with NTSB analysts, add temporal trend analysis, and implement predictive modeling.

---

## **SECTION 6: WRAP-UP (30 seconds - 75 words)**

**📺 SCREEN:**
- Scroll to bottom showing Repository link
- Show GitHub URL prominently

**SAY THIS:**

Remember American Airlines Flight 383? What took investigators hours to analyze—searching eighty thousand incidents for similar engine fires—this system does in seconds. And unlike pure LLMs, every diagnosis is grounded in real historical data, not hallucinated.

Transformer architectures—embeddings, attention, and function calling—don't just enable AI applications. They enable responsible, verifiable, safety-critical AI.

All code, model cards, and documentation are available at the GitHub repository shown here.

Thank you. I'm happy to take questions.

---

## **END OF PRESENTATION**

**Total Duration:** ~13:00 minutes
**Word Count:** ~1,300 words
