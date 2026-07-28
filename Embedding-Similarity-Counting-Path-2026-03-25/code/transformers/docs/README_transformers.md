# Aviation Incident Diagnosis Engine: LLM Agents for Safety Analysis

**How AI Can Help Save Lives by Learning from History**

Imagine if every time an airplane had an incident, we could instantly search through decades of similar cases to identify the most likely causes. This project demonstrates how modern AI—specifically Large Language Models (LLMs) and transformer technology—can be combined with traditional data analysis to create a powerful diagnostic tool for aviation safety.

**Project:** Aviation Incident Diagnosis Engine  
**Author:** [Your Name]  
**Course:** Transformers & Large Language Models  
**Date:** [Presentation Date]  
**Duration:** 15 minutes (10-11 min presentation + Q&A)

---

## What You'll Learn

This project demonstrates:

- **How transformer embeddings enable semantic search** (finding things by meaning, not keywords)
- **How LLMs can use "function calling"** to extend their capabilities with reliable tools
- **How multi-step reasoning** allows AI to tackle complex problems systematically
- **Why hybrid AI systems** (combining AI with traditional methods) are more reliable than AI alone

---

## Table of Contents

1. [Problem Statement & Overview](#problem-statement--overview)
2. [Methodology: LLM Agents & Semantic Search](#methodology-llm-agents--semantic-search)
3. [Implementation & Demo](#implementation--demo)
4. [Assessment & Evaluation](#assessment--evaluation)
5. [Critical Analysis](#critical-analysis)
6. [Documentation & Setup](#documentation--setup)
7. [Resource Links](#resource-links)

---

## Problem Statement & Overview

### The Challenge: Why This Matters

Imagine you're an aviation safety analyst. A plane has an incident—maybe an engine fire during takeoff. Your job is to figure out what went wrong and why. You need to search through **thousands** of historical incident reports, looking for similar cases that might reveal the root cause.

**The problem?** This is incredibly time-consuming. Worse, you might miss important connections because two incidents might describe the same problem using completely different words. "Engine fire during takeoff" and "smoke observed from engine compartment on departure" are describing the same thing, but a keyword search wouldn't find both.

**What if AI could help?** But here's the catch: AI language models are great at understanding language and generating text, but they can "hallucinate"—make up facts that sound plausible but aren't true. For safety-critical applications like aviation, we can't afford that.

### The Solution: A Hybrid Approach

This project solves this problem by creating a **smart partnership** between AI and traditional data analysis:

1. **AI understands your question** - When you type "engine fire during takeoff," the AI understands what you mean, even if you phrase it differently than historical reports
2. **Traditional tools find the facts** - Instead of letting AI guess, we use proven statistical methods to search through real historical data
3. **AI explains the results** - The AI takes those factual results and explains them in plain English

**Think of it like this:** Instead of asking an AI to be an expert on everything (which it isn't), we ask it to be a smart assistant that knows when to look things up in a reliable database.

### How It Works: The System Flow

The system follows a clear three-step process:

**Step 1: Generate a Detailed Report**

- You enter a brief description: "engine fire during takeoff"
- The AI expands this into a detailed, professional incident report
- This detailed report helps the system understand exactly what you're looking for

**Step 2: Find Similar Historical Cases**

- The system searches through thousands of historical NTSB reports
- It finds incidents that are semantically similar (meaning similar, not just word-for-word matches)
- It calculates probabilities: "Based on 50 similar incidents, here are the most likely root causes"

**Step 3: Synthesize and Explain**

- The AI takes those statistical results and explains them clearly
- You get both the raw data AND a human-readable summary

**Why This Works:**

- ✅ **Accurate:** Uses real historical data, not AI guesses
- ✅ **Fast:** Finds relevant cases in seconds, not hours
- ✅ **Understandable:** Explains results in plain language
- ✅ **Reliable:** Combines AI intelligence with proven statistical methods

---

## Methodology: LLM Agents & Semantic Search

### Connection to Transformers & LLM Course Content

This project demonstrates several key concepts from our transformers course:

#### 1. **Transformer Embeddings: How AI Understands Meaning**

**What This Means (In Simple Terms):**
Think of how you understand language. When someone says "car broke down," you know that's similar to "vehicle malfunction" even though the words are completely different. Transformers do something similar—they convert text into mathematical representations (called "embeddings") that capture meaning, not just words.

**Course Connection:**
Transformers learn these semantic representations through self-attention mechanisms (as covered in our course). The embedding layer captures semantic meaning in high-dimensional vector spaces—essentially, it creates a "map" where similar meanings are close together.

**How We Use It:**

- We use OpenAI's embedding model to convert each incident report into a 1536-dimensional vector (think of it as coordinates on a map)
- When you search for "engine fire during takeoff," we convert your query to the same type of vector
- We then find reports whose vectors are "close" to yours—meaning they're semantically similar

**Real Example:**

```
Your Query: "engine fire during takeoff"

Similar Incidents Found (by meaning, not keywords):
- "fire in engine compartment on departure" (87% similar)
- "smoke observed from engine during initial climb" (82% similar)
- "engine failure shortly after takeoff" (79% similar)
```

The AI understands that "fire," "smoke," and "failure" in the context of "engine" and "takeoff" are all describing related problems, even though they use different words. This is the power of transformer embeddings!

#### 2. **LLM Function Calling: AI That Knows When to Ask for Help**

**What This Means (In Simple Terms):**
Imagine you're working with a brilliant but forgetful assistant. They're great at writing and explaining, but they don't have perfect memory. Function calling is like giving them a phone book—when they need specific information, they can "call" a reliable source instead of guessing.

**Course Connection:**
Modern LLMs can extend their capabilities beyond their training data by calling external functions. This is a form of "tool use" that enables LLMs to access real-time information and perform deterministic computations—a key concept we've covered in our transformers course.

**How It Works in Our System:**

1. **You ask a question:** "What causes engine fires during takeoff?"
2. **AI generates a detailed report** - The LLM creates a professional incident synopsis (this shows its generation capability)
3. **AI decides it needs data** - The LLM recognizes it needs factual information, so it calls our diagnostic tool (this shows reasoning)
4. **Tool provides facts** - Our deterministic tool searches the database and returns statistical results
5. **AI explains the results** - The LLM takes those facts and explains them clearly

**Why This Is Powerful:**

- **Without Function Calling:** The AI would either need to memorize thousands of incidents (impossible) or make educated guesses (unreliable)
- **With Function Calling:** The AI acts as an intelligent coordinator—it knows when to use tools for facts and when to use its own capabilities for explanation
- This demonstrates the **hybrid AI** paradigm: LLMs as smart orchestrators, not all-knowing oracles

#### 3. **Multi-Step Reasoning: AI That Thinks Through Problems**

**What This Means (In Simple Terms):**
Just like humans break complex problems into steps, our AI system does the same. It doesn't try to do everything at once—it follows a logical sequence, remembering what happened in each step.

**Course Connection:**
Transformers can perform complex reasoning by breaking tasks into sequential steps, maintaining context across multiple interactions. This is similar to "chain-of-thought" reasoning we've discussed in class.

**Our Three-Step Process:**

**Step 1: Understand and Expand**

- You give a brief description: "engine fire during takeoff"
- The AI expands this into a detailed, professional report
- This helps the system understand exactly what you're looking for

**Step 2: Find the Facts**

- The AI recognizes it needs factual data
- It calls our diagnostic tool (like looking something up in a database)
- The tool searches historical incidents and calculates probabilities

**Step 3: Explain the Results**

- The AI receives the statistical results
- It synthesizes them into a clear, human-readable explanation
- You get both the raw data AND the interpretation

**The Key Insight:** The AI maintains context throughout all three steps—it remembers what you asked, what it found, and can explain how the results relate to your original question. This demonstrates sophisticated multi-turn reasoning capabilities.

#### 4. **Attention Mechanisms: How AI Focuses on What Matters**

**What This Means (In Simple Terms):**
When you read a sentence, you naturally focus on the important words and how they relate to each other. Transformers do something similar through "attention"—they learn which parts of the input are most relevant.

**Course Connection:**
Self-attention in transformers learns relationships between tokens (words). Similarly, embedding-based similarity measures relationships between entire sequences (sentences or documents). Both use the same fundamental idea: measuring "how related" pieces of information are.

**How We Apply This:**

- We use cosine similarity to measure how related two incident reports are
- The mathematical formula: `similarity = (A · B) / (||A|| × ||B||)`
- This is mathematically similar to attention weights—both measure relationships, just at different scales
- **In practice:** Reports about similar incidents have high similarity scores, just like related words get high attention weights

### Detailed Methodology: How We Built It

#### Phase 1: Organizing the Data

**The Challenge:** NTSB data comes in many separate files—narratives, aircraft information, events, findings, etc. We needed to bring it all together.

**What We Did:**

- Combined all data sources into one unified file
- Linked everything by incident ID (like a unique case number)
- Preserved the chronological order of events (what happened first, second, etc.)

**Result:** One clean, organized database where everything about each incident is connected

#### Phase 2: Converting Text to Meaning (Embeddings)

**The Challenge:** We need to search by meaning, not keywords. "Engine fire" and "fire in engine compartment" should match, even though the words are different.

**What We Did:**

- Used OpenAI's embedding model to convert each incident report into a mathematical representation
- Each report becomes a 1536-dimensional vector (think of it as coordinates on a map of meanings)
- Processed everything in batches for efficiency

**The Result:**

- Every incident report is now represented as a vector that captures its meaning
- Reports with similar meanings have vectors that are close together
- This enables semantic search—finding things by meaning, not just keywords

#### Phase 3: Finding Similar Incidents

**The Challenge:** When you search for "engine fire during takeoff," we need to find all similar incidents, even if they're described differently.

**How It Works:**

1. Convert your query to an embedding (same process as Phase 2)
2. Compare your query's embedding with all stored incident embeddings
3. Calculate similarity scores (how "close" they are in meaning-space)
4. Rank results and return the most similar incidents

**The Math Behind It:**
We use cosine similarity, which measures the angle between two vectors. Think of it like comparing directions on a compass—two vectors pointing in similar directions have high similarity.

**Why This Works:**

- **Normalized scale:** Scores range from 0 (completely unrelated) to 1 (identical meaning)
- **Interpretable:** A score of 0.87 means 87% similar in meaning
- **Robust:** Works well even with high-dimensional embeddings (1536 dimensions in our case)

#### Phase 4: Calculating Probable Root Causes

**The Challenge:** We have 50 similar incidents, each with different root causes. How do we determine which causes are most likely for your specific incident?

**The Solution: Weighted Probability**
Instead of just counting how often each cause appears, we weight each incident by how similar it is to your query. More similar incidents contribute more to the final probability.

**How It Works:**

1. Find the top 50 most similar incidents
2. Extract the root causes from each incident
3. Weight each cause by its incident's similarity score
4. Calculate final probabilities

**The Formula:**

```
Probability of Cause = Sum of (Similarity × Presence) / Sum of All Similarities

Where:
- Similarity = how similar the incident is (0 to 1)
- Presence = 1 if the cause appears in that incident, 0 if not
```

**Why Weighted Instead of Simple Counting?**

- **More accurate:** An incident that's 95% similar should count more than one that's 60% similar
- **Better relevance:** We're not just looking at frequency—we're looking at relevance-weighted frequency
- **Empirical:** Based on actual historical patterns, not theoretical assumptions

#### Phase 5: The AI Agent That Coordinates Everything

**The Challenge:** We need to coordinate three different tasks—understanding your question, finding the facts, and explaining the results. This requires intelligent orchestration.

**How We Built It:**

We define a "tool" that the AI can call—our diagnostic function. The AI knows:

- What the tool does (finds probable root causes)
- What information it needs (a detailed incident description)
- What it will get back (a list of causes with probabilities)

**The Three-Step Agent Workflow:**

1. **Generation:** You give a brief description, AI expands it into a detailed report

   - This helps the system understand exactly what you're looking for
   - The AI uses its language understanding capabilities

2. **Tool Call:** AI recognizes it needs factual data, so it calls our diagnostic tool

   - The AI decides when to use the tool (demonstrating reasoning)
   - The tool provides deterministic, verifiable results

3. **Synthesis:** AI takes the statistical results and explains them clearly
   - Converts numbers and probabilities into human-readable insights
   - Provides context and interpretation

**Why This Multi-Step Approach?**

- **Separation of concerns:** Each step has a clear purpose
- **Intelligent decision-making:** The AI decides when tools are needed
- **Demonstrates reasoning:** Shows chain-of-thought capabilities
- **Reliability:** Facts come from tools, explanations come from AI

---

## Implementation & Demo

### Code Structure

```
NTSB_Shivy/
├── agent.py              # LLM agent orchestration
├── main_app.py           # Core diagnosis engine
├── streamlit_app.py      # Web interface
└── Data/
    ├── 01_create_refined_dataset.py
    ├── 2_generate_embeddings.py
    └── [data files]
```

### Key Functions

#### 1. Embedding Generation (`main_app.py`)

```python
def get_embedding(text):
    """Generates embedding using OpenAI API"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding
```

**What This Demonstrates:**

- Transformer embeddings capture semantic meaning
- Single API call converts text to 1536-dimensional vector
- Preprocessing (removing newlines) ensures clean input

#### 2. Semantic Search (`main_app.py`)

```python
def find_top_matches(query_embedding):
    """Finds most similar incidents using cosine similarity"""
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    return similarities[sorted_indices], [embeddings_map[i] for i in sorted_indices]
```

**What This Demonstrates:**

- Vector similarity search in high-dimensional space
- Efficient numpy operations for large-scale search
- Returns both scores and metadata

#### 3. Weighted Diagnosis (`main_app.py`)

```python
def calculate_weighted_diagnosis(top_scores, top_matches, top_n_incidents=50):
    """Calculates similarity-weighted probabilities"""
    cause_evidence = defaultdict(list)

    for score, match in zip(top_scores[:top_n_incidents], top_matches[:top_n_incidents]):
        if match.get('source') != 'incident':
            continue
        causes = match.get('bayesian_data', {}).get('all_causes', [])
        for cause in causes:
            cause_evidence[cause].append((score, match['ev_id']))

    # Calculate weighted probabilities
    total_similarity = sum(score for evidence_list in cause_evidence.values()
                          for score, _ in evidence_list)

    weighted_causes = []
    for cause, evidence_list in cause_evidence.items():
        cause_similarity_sum = sum(score for score, _ in evidence_list)
        probability = cause_similarity_sum / total_similarity if total_similarity > 0 else 0
        weighted_causes.append({
            'cause': cause,
            'probability': probability,
            'num_incidents': len(evidence_list)
        })

    return sorted(weighted_causes, key=lambda x: x['probability'], reverse=True)
```

**What This Demonstrates:**

- Statistical aggregation of evidence
- Weighted probability calculation
- Handling of edge cases (empty evidence)

#### 4. LLM Agent (`agent.py`)

```python
def run_diagnosis_agent(user_query):
    """Multi-step LLM agent workflow"""

    # Step 1: Generate report
    report_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": report_generation_prompt}]
    )
    generated_report = report_response.choices[0].message.content

    # Step 2: Tool call
    tool_call_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": analysis_prompt}],
        tools=tools,
        tool_choice="auto"
    )

    # Execute tool
    tool_results = diagnose_root_causes(generated_report)

    # Step 3: Synthesize
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_with_tool_results
    )
    final_summary = final_response.choices[0].message.content

    return {
        "generated_report": generated_report,
        "tool_results": tool_results,
        "final_summary": final_summary
    }
```

**What This Demonstrates:**

- Function calling API usage
- Multi-turn conversation management
- Tool integration pattern

### Live Demo Instructions

**Setup:**

```bash
# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key in agent.py and main_app.py
# (Replace API_KEY variable)

# Run Streamlit app
streamlit run streamlit_app.py
```

**Demo Flow:**

1. Open browser to `http://localhost:8501`
2. Enter query: "engine fire during takeoff"
3. Click "Run Diagnostic Agent"
4. Show three result sections:
   - LLM-generated report (demonstrates generation)
   - Tool results (demonstrates deterministic analysis)
   - Final summary (demonstrates synthesis)

**What to Highlight:**

- **LLM Generation:** Notice the detailed, plausible report
- **Tool Results:** Show top 10 causes with probabilities
- **Synthesis:** LLM interprets and explains the results

---

## Assessment & Evaluation

### Model Versions & Architecture

**LLM Models:**

- **Primary:** GPT-4o-mini (OpenAI)

  - Context window: 128K tokens
  - Used for: Report generation, tool orchestration, synthesis
  - Model card: https://platform.openai.com/docs/models/gpt-4o-mini

- **Embeddings:** text-embedding-3-small (OpenAI)
  - Dimensions: 1536
  - Used for: Semantic similarity search
  - Model card: https://platform.openai.com/docs/guides/embeddings

**Architecture:**

- Multi-agent system with three-stage pipeline
- Hybrid approach: LLM reasoning + deterministic tools
- Pre-computed embeddings for efficiency

### Intended Uses

**Primary Use Cases:**

- Research and analysis of historical aviation incidents
- Educational demonstration of LLM agent capabilities
- Pattern analysis in aviation safety data
- Supporting safety analysts in root cause identification

**Limitations:**

- Probabilistic, not deterministic (provides likelihoods, not certainties)
- Based on historical patterns, not real-time analysis
- Requires OpenAI API access (cost considerations)
- Not intended for real-time flight operations decisions

**Appropriate Use:**

- ✅ Research and educational purposes
- ✅ Historical pattern analysis
- ✅ Supporting (not replacing) expert analysis
- ❌ Real-time flight operations
- ❌ Regulatory compliance without validation

### Licenses

- **Code:** Educational/research use
- **NTSB Data:** Public domain (U.S. government data)
- **OpenAI API:** Subject to OpenAI Terms of Service
  - API usage: https://openai.com/api/policies/terms/
  - Rate limits and pricing apply

### Ethical & Bias Considerations

**Data Bias:**

- Historical NTSB data may reflect reporting biases
- Certain incident types may be over/under-represented
- Geographic and temporal biases possible
- **Mitigation:** Weighted approach accounts for similarity, not just frequency

**Model Limitations:**

- LLM-generated reports are hypothetical and may contain hallucinations
- **Mitigation:** Tool-based diagnosis uses factual historical data
- LLM synthesis interprets but doesn't generate diagnostic facts

**Aviation Safety Implications:**

- Provides probabilistic analysis, not definitive answers
- Should not replace expert human analysis
- Results should be validated against domain expertise
- **Responsible Use:** Educational and research purposes only

**Privacy:**

- Uses publicly available NTSB incident data
- No personal information processed
- All data is anonymized incident reports

**Transparency:**

- Methodology is documented and reproducible
- Mathematical formulas are explicit
- Tool results separated from LLM synthesis

---

## Critical Analysis

### Impact of This Project

**Aviation Safety:**

- Provides systematic approach to analyzing historical incident patterns
- Enables rapid identification of probable root causes based on similarity
- Could support safety analysts in identifying common failure modes
- **Potential Impact:** Faster pattern recognition, supporting preventive safety measures

**LLM Research & Education:**

- Demonstrates practical application of LLM agents in domain-specific contexts
- Shows how to combine LLM capabilities with traditional data analysis
- Provides working example of function calling and tool integration
- **Impact:** Educational value in understanding LLM limitations and mitigation strategies

**Technical Innovation:**

- Combines semantic search (embeddings) with statistical analysis (weighted Bayesian)
- Integrates LLM reasoning with deterministic diagnostic tools
- Creates hybrid system leveraging strengths of both approaches
- **Innovation:** Demonstrates "LLMs as orchestrators" paradigm

### What This Project Reveals or Suggests

**About LLM Capabilities:**

- LLMs excel at generation and synthesis but need tools for factual accuracy
- Function calling enables LLMs to extend beyond training data
- Multi-step reasoning allows handling complex, multi-part tasks
- **Revelation:** LLMs are powerful orchestrators but require careful integration with domain-specific tools

**About Aviation Incident Patterns:**

- Similar incidents often share common root causes (validates similarity-weighted approach)
- Historical data contains rich diagnostic information when properly analyzed
- Semantic similarity can identify relevant cases even with different surface-level descriptions
- **Suggestion:** Pattern-based analysis is valuable for safety-critical domains

**About Hybrid AI Systems:**

- Combining LLM reasoning with deterministic tools creates more reliable systems
- Weighted approaches can balance multiple evidence sources effectively
- **Suggestion:** Future AI systems should leverage both generative and analytical capabilities

### Next Steps & Future Directions

**Short-Term Enhancements:**

1. **Evaluation Framework:** Develop quantitative metrics for diagnosis accuracy
2. **User Validation:** Test with aviation safety experts to validate results
3. **Performance Optimization:** Implement caching and approximate nearest neighbor search

**Medium-Term Research:**

1. **Temporal Analysis:** Incorporate time-based trends in incident patterns
2. **Causal Chain Visualization:** Develop network graphs showing causal relationships
3. **Multi-Model Comparison:** Compare different embedding models and LLM variants

**Long-Term Vision:**

1. **Real-Time Integration:** Explore integration with live flight data (with appropriate safeguards)
2. **Domain Expansion:** Apply methodology to other safety-critical domains
3. **Explainable AI:** Enhance interpretability of LLM decisions and tool selections

**Research Questions:**

- How do different embedding models affect diagnosis accuracy?
- Can we predict incident likelihood before they occur?
- What is the optimal balance between LLM reasoning and tool-based analysis?
- How can we improve LLM tool selection and parameter tuning?

---

## Documentation & Setup

### Quick Start

**Prerequisites:**

- Python 3.8 or higher
- OpenAI API key

**Installation:**

```bash
# Clone repository
git clone <repository-url>
cd NTSB_Shivy

# Install dependencies
pip install -r requirements.txt

# Set up API keys
# Create a .env file (copy from .env.example) and add:
# OPENAI_API_KEY=your-key-here
# Or set environment variable: export OPENAI_API_KEY='your-key-here'
```

**Running the Application:**

```bash
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

### Usage Guide

#### Method 1: Web Interface (Recommended)

1. Launch Streamlit app: `streamlit run streamlit_app.py`
2. Enter incident description (e.g., "engine fire during takeoff")
3. Click "Run Diagnostic Agent"
4. View results:
   - LLM-generated incident synopsis
   - Final diagnosis summary
   - Top 10 probable root causes with probabilities

#### Method 2: Python API

**Direct Diagnosis:**

```python
from main_app import diagnose_root_causes

results = diagnose_root_causes("engine fire during takeoff", top_n=10)
for cause in results['top_causes']:
    print(f"{cause['cause']}: {cause['probability']*100:.1f}%")
```

**LLM Agent:**

```python
from agent import run_diagnosis_agent

results = run_diagnosis_agent("engine fire during takeoff")
print(results['final_summary'])
```

### Data Processing (Optional)

If you need to regenerate embeddings or process new data:

```bash
cd Data

# Step 1: Consolidate data
python 01_create_refined_dataset.py

# Step 2: Generate embeddings (requires OpenAI API key)
python 2_generate_embeddings.py
python 2b_precompute_bayesian_data.py
```

**Note:** Embedding generation may take several hours for large datasets and requires OpenAI API access.

### Project Structure

```
NTSB_Shivy/
├── agent.py                          # LLM agent orchestration
├── main_app.py                       # Core diagnosis engine
├── streamlit_app.py                  # Web interface
├── Project_plan_updated.md           # Complete technical documentation
├── streamlit_app_sequence_diagram.md # System architecture diagram
├── requirements.txt                  # Python dependencies
└── Data/
    ├── 01_create_refined_dataset.py  # Data consolidation
    ├── 2_generate_embeddings.py      # Embedding generation
    ├── 2b_precompute_bayesian_data.py # Bayesian preprocessing
    ├── refined_dataset.json          # Consolidated incident data
    ├── embeddings.npy                # Vector embeddings
    ├── embeddings_map.json           # Embedding metadata
    └── bayesian_cause_statistics.json # Pre-computed statistics
```

---

## Resource Links

### Papers & Research

1. **Attention Is All You Need** (Vaswani et al., 2017)

   - Foundation for transformer architecture
   - URL: https://arxiv.org/abs/1706.03762
   - **Relevance:** Embeddings and attention mechanisms

2. **OpenAI Embeddings Documentation**

   - text-embedding-3-small model details
   - URL: https://platform.openai.com/docs/guides/embeddings
   - **Relevance:** Semantic search implementation

3. **OpenAI Function Calling**

   - Tool use and function calling guide
   - URL: https://platform.openai.com/docs/guides/function-calling
   - **Relevance:** LLM agent implementation

4. **Bayesian Networks for Aviation Safety**
   - Reference: BN-NTSB RESS 2021.pdf (included in repository)
   - **Relevance:** Bayesian analysis methodology

### Code Bases & Libraries

1. **OpenAI Python SDK**

   - Repository: https://github.com/openai/openai-python
   - License: MIT
   - **Usage:** LLM API and embeddings

2. **Streamlit**

   - Repository: https://github.com/streamlit/streamlit
   - License: Apache 2.0
   - **Usage:** Web interface framework

3. **scikit-learn**
   - Repository: https://github.com/scikit-learn/scikit-learn
   - License: BSD
   - **Usage:** Cosine similarity calculations

### Data Sources

1. **NTSB Aviation Accident Database**
   - Source: National Transportation Safety Board
   - License: Public domain (U.S. government data)
   - URL: https://www.ntsb.gov/Pages/AviationQuery.aspx

### Documentation References

1. **OpenAI API Documentation**

   - URL: https://platform.openai.com/docs
   - **Usage:** API integration, model specifications

2. **Streamlit Documentation**
   - URL: https://docs.streamlit.io/
   - **Usage:** UI component development

---

## Model & Data Cards

### Model Card: GPT-4o-mini

**Model Information:**

- **Name:** GPT-4o-mini
- **Provider:** OpenAI
- **Version:** Latest (as of project date)
- **Type:** Large Language Model (Transformer-based)
- **Context Window:** 128K tokens

**Intended Use:**

- Text generation (incident reports)
- Tool orchestration (function calling)
- Information synthesis (summarizing results)

**Limitations:**

- May generate plausible but incorrect information (hallucination)
- Requires API access (cost considerations)
- Not suitable for real-time safety-critical decisions

**Mitigation Strategies:**

- Tool-based fact-checking (diagnostic tool provides factual data)
- Structured outputs (function calling ensures correct format)
- Human validation (results reviewed by domain experts)

### Model Card: text-embedding-3-small

**Model Information:**

- **Name:** text-embedding-3-small
- **Provider:** OpenAI
- **Dimensions:** 1536
- **Type:** Text embedding model

**Intended Use:**

- Semantic similarity search
- Finding related incidents based on meaning

**Performance Characteristics:**

- Captures semantic relationships
- Normalized vectors (cosine similarity appropriate)
- Efficient for large-scale search

### Data Card: NTSB Aviation Incident Database

**Dataset Information:**

- **Source:** National Transportation Safety Board (NTSB)
- **License:** Public domain (U.S. government data)
- **Time Period:** Historical incidents (varies by data subset)
- **Size:** Thousands of incidents (exact count depends on subset)

**Data Characteristics:**

- **Format:** Structured JSON after preprocessing
- **Fields:** Narratives, event sequences, findings, causes
- **Preprocessing:** Consolidated from multiple sources, merged on event ID

**Data Quality:**

- **Strengths:** Comprehensive, official source, detailed narratives
- **Limitations:** Reporting biases possible, temporal/geographic variations
- **Bias Considerations:** Historical data may reflect reporting practices of different eras

**Usage:**

- Used for similarity search and pattern analysis
- Pre-computed embeddings for efficiency
- Bayesian statistics pre-extracted

---

## Presentation Flow (10-11 minutes)

### Suggested Timing

1. **Problem & Overview** (1-2 min)

   - Aviation incident diagnosis challenge
   - Why hybrid AI approach
   - System architecture overview

2. **Methodology Deep Dive** (4-5 min) - **FOCUS AREA**

   - Transformer embeddings & semantic search
   - LLM function calling & tool use
   - Multi-step reasoning
   - Weighted Bayesian diagnosis

3. **Implementation & Demo** (2-3 min)

   - Code structure overview
   - Live demo walkthrough
   - Key functions explained

4. **Results & Critical Analysis** (2 min)

   - Impact on aviation safety
   - What this reveals about LLMs
   - Future directions

5. **Q&A Preparation** (remaining time)
   - Be ready to discuss:
     - Why function calling vs. fine-tuning?
     - How embeddings enable semantic search
     - Trade-offs of hybrid approach
     - Limitations and ethical considerations

### Key Points to Emphasize

1. **LLM as Orchestrator:** Not replacing tools, but intelligently coordinating them
2. **Semantic Understanding:** Embeddings capture meaning, not just keywords
3. **Hybrid Approach:** Combines strengths of LLMs (generation) and deterministic tools (accuracy)
4. **Practical Application:** Real-world problem with measurable impact potential

---

## Questions for Discussion

**Potential Q&A Topics:**

1. **Why use function calling instead of fine-tuning the LLM on incident data?**

   - Fine-tuning would require massive dataset and compute
   - Function calling allows leveraging existing high-quality models
   - Tool provides verifiable, deterministic results

2. **How do you ensure the LLM doesn't hallucinate diagnoses?**

   - Tool-based analysis uses factual historical data
   - LLM only synthesizes tool results, doesn't generate facts
   - Separation of concerns: generation vs. analysis

3. **What's the advantage of weighted Bayesian approach over simple frequency counting?**

   - Accounts for similarity, not just occurrence
   - More similar incidents contribute more to diagnosis
   - Better reflects actual relevance

4. **Could this work with other transformer models?**

   - Yes, any model with embeddings and function calling
   - Different models may have different trade-offs
   - OpenAI chosen for API availability and quality

5. **How scalable is this approach?**
   - Embedding search: O(n) where n = number of incidents
   - Can optimize with approximate nearest neighbors
   - Pre-computation enables fast inference

---

_This project demonstrates how modern LLM capabilities (function calling, embeddings, multi-step reasoning) can be integrated with traditional data analysis methods to create practical, reliable systems for domain-specific applications._
