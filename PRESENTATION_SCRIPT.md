# Presentation Script: Aviation Incident Diagnosis Engine
## 10-11 Minute Word-for-Word Script

**Total Words:** ~1,650 (11 minutes at 150 words/min)
**Author:** Kanu Shetkar

---

## Instructions:
- Read this script word-for-word
- Follow [SCREEN] directions to know what to show
- Practice timing: Should be 10-11 minutes total
- Don't rush - clear articulation is important

---

## **INTRODUCTION (30 seconds - 75 words)**

[SCREEN: Show README_PRESENTATION.md title]

Good morning. I'm Kanu Shetkar, and I'm presenting an Aviation Incident Diagnosis Engine that combines transformer embeddings with LLM agents for safety-critical pattern analysis.

This project demonstrates how modern transformer architectures—specifically embeddings, attention mechanisms, and function calling—can be applied to real-world safety analysis while avoiding the hallucination problems that make pure LLM solutions unreliable for critical applications.

---

## **SECTION 1: PROBLEM STATEMENT (1 minute - 150 words)**

[SCREEN: Scroll to "1. Problem Statement & Overview"]

Let me explain the problem. Aviation safety analysts need to investigate incidents by searching through thousands of NTSB reports. The challenge is that traditional keyword search completely fails at this task.

For example, if you search for "engine fire during takeoff," you won't find reports that say "smoke from engine compartment on departure"—even though they're describing the exact same phenomenon. The words are different, but the meaning is identical.

Now, you might ask: why not just ask an LLM like ChatGPT? The problem is that LLMs can hallucinate facts. They might generate a plausible-sounding diagnosis that isn't based on any real historical data. For aviation safety, where decisions can be life-or-death, this is completely unacceptable.

My solution is a hybrid architecture that uses transformers for semantic understanding, but relies on deterministic tools for the actual facts.

[SCREEN: Point to architecture diagram in README]

The system has three components: an LLM agent that orchestrates everything, a diagnostic tool that does semantic search and Bayesian analysis, and the NTSB historical database with pre-computed embeddings.

---

## **SECTION 2: METHODOLOGY (5 minutes - 750 words)**

[SCREEN: Scroll to "2. Methodology"]

Now let me explain how transformer concepts enable this system. This is where the connection to our course material becomes clear.

### **Part A: Transformer Embeddings (1 minute - 150 words)**

[SCREEN: Show "2.1 Transformer Embeddings"]

First, transformer embeddings. From the Formal Algorithms for Transformers paper by Phuong and Hutter, we know that the token embedding algorithm maps discrete vocabulary elements to continuous vector representations in d-e dimensional real space.

[SCREEN: Point to the algorithm pseudocode]

I extend this concept from token-level to document-level embeddings. I use OpenAI's text-embedding-3-small model, which produces fifteen-hundred-thirty-six dimensional vectors—essentially coordinates in a semantic space.

[SCREEN: Point to code snippet]

Here's the actual implementation. One API call converts an entire incident narrative into a vector that captures its semantic meaning.

Why does this work? Because transformer self-attention learns semantic relationships during pre-training. Similar meanings produce similar vectors.

[SCREEN: Point to the example]

Look at this real example. My query "engine fire during takeoff" finds three incidents with completely different wording—"fire in engine compartment," "smoke observed from engine," "engine failure"—but all semantically related. The cosine similarity scores show how close these meanings are in vector space.

### **Part B: Attention-Inspired Similarity (1 minute - 150 words)**

[SCREEN: Scroll to "2.2 Attention-Inspired Similarity"]

Second, attention-inspired similarity search. Let me connect this to the attention mechanism from Vaswani et al.'s "Attention Is All You Need."

[SCREEN: Point to Attention formula]

Recall that attention is softmax of Q times K-transpose, divided by the square root of d-k, times V. The key part is Q times K-transpose—this dot product measures how related the query and key vectors are.

I use the same mathematical concept with cosine similarity.

[SCREEN: Point to cosine similarity formula]

Cosine similarity equals q dot d, divided by the norm of q, times the norm of d. This is essentially a normalized dot product—exactly like attention scores, but applied at the document level instead of the token level.

[SCREEN: Point to algorithm]

My implementation is order N times d complexity—a vectorized similarity search using numpy. For each of the N incidents in the database, I compute the similarity with the query embedding, then return the top k matches.

The connection to attention is clear: both use dot products to measure relationships, and both aggregate information weighted by relevance scores.

### **Part C: Weighted Bayesian Diagnosis (1.5 minutes - 225 words)**

[SCREEN: Scroll to "2.3 Weighted Bayesian"]

Third, the weighted Bayesian diagnosis. This is where I combine semantic search with statistical analysis.

Here's the problem: I've found fifty similar incidents, and each one has different root causes listed. How do I determine which causes are most likely for the new incident?

The naive approach is just frequency counting—count how often each cause appears. But that treats the incident at rank one, with ninety-five percent similarity, exactly the same as the incident at rank fifty with only sixty-two percent similarity. That doesn't make sense.

My approach is similarity-weighted probability.

[SCREEN: Point to algorithm]

For each cause, I sum up the similarity scores of all incidents where that cause appears. Then I divide by the total of all similarity scores. This gives me a probability that accounts for relevance, not just frequency.

[SCREEN: Point to mathematical formula]

The mathematical formulation is: the weighted probability of cause j equals the sum of similarities where cause j appears, divided by the sum of all similarities.

Why is this better? Because more similar incidents contribute more to the final probability—just like how high-attention tokens contribute more to transformer outputs. This is the same weighted aggregation principle, applied to evidence aggregation instead of token aggregation.

### **Part D: LLM Function Calling (1.5 minutes - 225 words)**

[SCREEN: Scroll to "2.4 LLM Function Calling"]

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

## **SECTION 3: DEMO (2 minutes - 300 words)**

[SCREEN: Switch to Streamlit app, have it already loaded at localhost:8501]

Now let me show you this system in action. I've built a Streamlit web interface that demonstrates all three steps.

[SCREEN: Make sure query is visible in input box]

I'm going to enter the query: "engine fire during takeoff."

[SCREEN: Click "Run Diagnostic Agent" button]

The system is now running through the three-step workflow.

[SCREEN: When results appear, scroll to Output 1]

Here's step one: the LLM-generated incident synopsis. Notice how it's created a detailed, plausible NTSB-style report from my brief query. It includes aircraft type, flight phase, event summary—all the elements you'd expect in a real incident report. This demonstrates the transformer's text generation capability.

[SCREEN: Scroll to Output 2 - Tool Results]

Step two: the diagnostic tool results. This is the crucial part. These are factual results from historical data—not LLM hallucinations.

[SCREEN: Point to top cause]

Notice the top cause: "Fatigue, wear, corrosion" with thirty-two-point-four percent probability. This is based on twenty-three similar historical incidents, with an average similarity score of zero-point-eight-one.

[SCREEN: Point to other causes]

You can see the complete distribution—each probability is calculated using that weighted Bayesian approach I explained. More similar incidents contributed more to these probabilities.

This demonstrates three transformer concepts in action: embeddings enabled the semantic search to find these fifty similar incidents, attention-inspired similarity ranked them, and weighted aggregation calculated the probabilities.

[SCREEN: Scroll to Output 3]

Step three: LLM synthesis. The LLM has taken those statistical results and explained them in plain English. It's interpreting the data, providing context, but not inventing facts. You can compare what it says to the raw tool output above—it's grounded in the actual data.

[SCREEN: Go back to top of Streamlit briefly]

This demonstrates the complete hybrid architecture: LLM orchestration with deterministic tools providing factual grounding.

---

## **SECTION 4: EVALUATION (1 minute - 150 words)**

[SCREEN: Switch back to README, scroll to "4. Assessment & Evaluation"]

Let me briefly cover evaluation and model cards, which the rubric requires.

[SCREEN: Point to model architecture table]

The system uses two models: GPT-4-o-mini for the LLM agent—generation, orchestration, and synthesis—and text-embedding-3-small for semantic similarity, with fifteen-hundred-thirty-six dimensions.

[SCREEN: Point to Intended Uses]

This system is appropriate for research, historical pattern analysis, and educational demonstration of hybrid AI. It's not intended for real-time flight operations or regulatory compliance without validation. It provides probabilities, not certainties—it supports expert analysis but doesn't replace it.

[SCREEN: Point to Ethical Considerations]

Ethical considerations: The historical data may reflect reporting biases. My mitigation is the weighted approach that accounts for relevance. LLM hallucinations are mitigated because the diagnostic facts come from the tool, not generation. For safety-critical domains, this is educational use only and requires expert validation.

[SCREEN: Briefly show the model cards tables]

Complete model and data cards are in the README with licenses, documentation links, and specifications.

---

## **SECTION 5: CRITICAL ANALYSIS (1 minute - 150 words)**

[SCREEN: Scroll to "6. Critical Analysis"]

Let me address the impact and limitations.

The impact on aviation safety: this enables rapid pattern recognition—diagnoses that would take hours of manual search now complete in seconds. Semantic search finds relevant cases even with different terminology. Recommendations are grounded in historical precedent.

For transformer research, this demonstrates a practical hybrid architecture. It shows how to mitigate LLM hallucination in safety-critical domains. The key insight: use LLMs as orchestrators, not oracles.

The technical innovation is similarity-weighted Bayesian analysis—combining semantic search with statistical aggregation using attention-inspired weighted aggregation.

[SCREEN: Point to Limitations]

Current limitations: No formal ground truth evaluation yet. No temporal analysis to track trends over time. No confidence intervals on the probabilities.

[SCREEN: Point to Future Work]

Future directions include expert validation studies, temporal analysis to track how failure modes change over time, causal chain visualization using network graphs, and developing a formal evaluation framework with aviation safety experts.

---

## **SECTION 6: WRAP-UP (30 seconds - 75 words)**

[SCREEN: Scroll to bottom showing Resources and Repository link]

In summary, this project demonstrates how transformer architectures—through embeddings, attention mechanisms, and function calling—can be responsibly applied to safety-critical domains.

All the code, model cards, and documentation are available in the GitHub repository.

[SCREEN: Show the GitHub URL clearly]

The link is https://github.com/Kshetkar1/NTSB-aviation-diagnosis-Transformers-Project.

I'm ready for questions. Thank you.

---

## **END OF PRESENTATION**

**Total Word Count:** ~1,650 words
**Expected Duration:** 10-11 minutes
**Rubric Coverage:** All 8 sections covered ✓

---

## **Timing Breakdown:**

- Introduction: 30 sec
- Problem Statement: 1 min
- Methodology: 5 min (biggest section - 50 points)
- Demo: 2 min
- Evaluation: 1 min
- Critical Analysis: 1 min
- Wrap-up: 30 sec
- **Total: 11 minutes**

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
