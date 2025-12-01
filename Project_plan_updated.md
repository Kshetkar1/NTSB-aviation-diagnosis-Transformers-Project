# Aviation Incident Diagnosis Engine: Updated Project Plan (V3)

This document reflects the current state of the Aviation Incident Diagnosis Engine project, documenting what has been implemented and the architecture that has evolved from the original plan.

---

## **Project Overview**

The Aviation Incident Diagnosis Engine is a multi-component system that uses LLM agents, semantic search, and weighted Bayesian analysis to diagnose root causes of aviation incidents based on historical NTSB data. The system combines:

- **Semantic similarity search** using OpenAI embeddings
- **Weighted Bayesian diagnosis** based on similarity-weighted historical evidence
- **LLM agent orchestration** for intelligent report generation and synthesis
- **Streamlit web interface** for user interaction

---

## **Phase 1: Data Consolidation** ✅ COMPLETED

**Status:** Complete

**Implementation:** `Data/01_create_refined_dataset.py`

**What Was Done:**

- Consolidated multiple data sources (narratives, aircraft info, events, findings, etc.) into a single structured JSON file
- Merged data on `ev_id` (event ID) as the primary key
- Organized event sequences chronologically for each incident
- Created `Data/refined_dataset.json` containing all structured incident data

**Key Features:**

- Single source of truth for all incident data
- Chronological event sequences preserved
- All relevant metadata (narratives, causes, findings) linked to incidents

---

## **Phase 2: Embedding Generation** ✅ COMPLETED

**Status:** Complete

**Implementation:** `Data/2_generate_embeddings.py` and `Data/2b_precompute_bayesian_data.py`

**What Was Done:**

- Generated embeddings for both historical incidents and master event dictionary
- Created unified embedding memory (`Data/embeddings.npy` and `Data/embeddings_map.json`)
- Pre-computed Bayesian diagnostic data (causes from findings and narratives)
- Enhanced mapping metadata to include diagnostic information

**Key Features:**

- Combined embeddings from incidents and dictionary entries
- Metadata mapping tracks source, ev_id, and diagnostic data
- Pre-computed cause extraction for faster diagnosis
- Batch processing for efficient API usage

**Files Created:**

- `Data/embeddings.npy` - Vector embeddings array
- `Data/embeddings_map.json` - Metadata mapping for each embedding
- `Data/bayesian_cause_statistics.json` - Pre-computed cause statistics

---

## **Phase 3: Core Diagnosis Engine** ✅ COMPLETED

**Status:** Complete

**Implementation:** `main_app.py`

**What Was Done:**

- Implemented semantic similarity search using cosine similarity
- Built weighted Bayesian diagnosis function (Dr. Smith's approach)
- Created high-level diagnostic interface
- Added verification functions for mathematical accuracy

### **Core Functions:**

1. **`get_embedding(text)`**

   - Generates embeddings for query text using OpenAI API
   - Uses `text-embedding-3-small` model

2. **`find_top_matches(query_embedding)`**

   - Calculates cosine similarity between query and all stored embeddings
   - Returns sorted matches with scores and metadata

3. **`calculate_weighted_diagnosis(top_scores, top_matches, top_n_incidents=50)`**

   - Implements similarity-weighted Bayesian diagnosis
   - Formula: `P_weighted(cause) = Σ(similarity × presence) / Σ(similarity)`
   - Returns weighted probabilities for each root cause

4. **`diagnose_root_causes(query, top_n=10)`**

   - High-level interface for diagnosis
   - Returns top N causes with probabilities and supporting evidence

5. **`verify_math_for_one_pair(text_a, text_b)`**

   - Manual cosine similarity verification
   - Compares manual calculation with sklearn for accuracy

### **Key Features:**

- Similarity-weighted probability calculation
- Support for both structured findings and narrative-extracted causes
- Metadata tracking (incidents analyzed, methodology used)
- Mathematical verification capabilities

---

## **Phase 4: LLM Agent Orchestration** ✅ COMPLETED

**Status:** Complete

**Implementation:** `agent.py`

**What Was Done:**

- Created multi-step LLM agent using OpenAI's function calling
- Integrated diagnostic tool as a callable function
- Implemented three-step process: report generation → tool analysis → synthesis

### **Agent Workflow:**

1. **Step 1: Generate Hypothetical Report**

   - Takes brief user query
   - LLM generates detailed NTSB-style incident synopsis
   - Creates plausible, factual-sounding report for analysis

2. **Step 2: Diagnostic Tool Call**

   - LLM receives generated report
   - Decides to call `diagnose_root_causes` tool
   - Tool executes similarity search and weighted diagnosis
   - Returns structured results (top causes with probabilities)

3. **Step 3: Synthesize Final Summary**

   - LLM receives tool results
   - Synthesizes comprehensive diagnosis summary
   - Provides human-readable interpretation

### **Key Features:**

- Function calling integration with OpenAI API
- Tool definition with structured parameters
- Graceful handling of tool call decisions
- Multi-step reasoning and synthesis

---

## **Phase 5: Streamlit Web Interface** ✅ COMPLETED

**Status:** Complete

**Implementation:** `streamlit_app.py`

**What Was Done:**

- Created interactive web interface using Streamlit
- Integrated agent execution with UI
- Displayed results in organized sections

### **UI Components:**

1. **User Input Section**

   - Text area for incident description
   - Default example: "engine fire during takeoff"
   - Button to trigger agent execution

2. **Agent Execution Log**

   - Shows progress with spinner
   - Displays three main result sections:
     - LLM-Generated Incident Synopsis
     - LLM's Final Diagnosis Summary
     - Detailed Diagnostic Data (from Tool)

3. **Results Display**

   - Top 10 causes with probabilities
   - Progress bars for visual probability representation
   - Metrics showing percentage probabilities
   - Supporting incident counts

### **Key Features:**

- Wide layout for better readability
- Organized information hierarchy
- Visual probability indicators
- Metadata display (incidents analyzed)

---

## **Current Architecture**

```
┌─────────────────┐
│   Streamlit UI  │
│  (streamlit_app)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Agent      │
│  (agent.py)     │
│                 │
│  1. Generate    │
│     Report      │
│                 │
│  2. Call Tool   │──┐
│                 │  │
│  3. Synthesize  │  │
└─────────────────┘  │
                     │
                     ▼
         ┌───────────────────────┐
         │  Diagnostic Tool      │
         │  (main_app.py)        │
         │                       │
         │  • Semantic Search    │
         │  • Weighted Diagnosis │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  Embeddings     │    │  Refined Dataset│
│  (embeddings.npy│    │  (refined_data  │
│   + map.json)   │    │   .json)        │
└─────────────────┘    └─────────────────┘
```

---

## **Key Design Decisions**

1. **Weighted Bayesian Approach**

   - Uses similarity scores as weights for probability calculation
   - More similar incidents contribute more to final probabilities
   - Empirical approach based on historical evidence

2. **LLM Agent Pattern**

   - Separates report generation from diagnosis
   - Allows LLM to synthesize and interpret results
   - Provides human-readable summaries

3. **Pre-computed Diagnostic Data**

   - Bayesian cause statistics pre-computed for performance
   - Faster diagnosis without real-time extraction
   - Balances accuracy and speed

4. **Unified Embedding Memory**

   - Single embedding space for incidents and dictionary
   - Enables semantic search across all knowledge
   - Efficient similarity calculations

---

## **File Structure**

```
NTSB_Shivy/
├── agent.py                          # LLM agent orchestration
├── main_app.py                       # Core diagnosis engine
├── streamlit_app.py                  # Web interface
├── streamlit_app_original.py         # Original UI version
├── Project_plan.md                   # Original project plan
├── Project_plan_updated.md           # This file
├── requirements.txt                  # Python dependencies
└── Data/
    ├── 01_create_refined_dataset.py  # Data consolidation
    ├── 2_generate_embeddings.py      # Embedding generation
    ├── 2b_precompute_bayesian_data.py # Bayesian data pre-computation
    ├── refined_dataset.json          # Consolidated incident data
    ├── embeddings.npy                # Vector embeddings
    ├── embeddings_map.json           # Embedding metadata
    └── bayesian_cause_statistics.json # Pre-computed cause stats
```

---

## **Future Enhancements & Next Steps**

### **Potential Improvements:**

1. **Performance Optimization**

   - [ ] Implement caching for frequently queried incidents
   - [ ] Optimize embedding search with approximate nearest neighbors (ANN)
   - [ ] Add parallel processing for batch queries

2. **Enhanced Diagnosis**

   - [ ] Add temporal analysis (trends over time)
   - [ ] Implement causal chain visualization
   - [ ] Add confidence intervals for probabilities

3. **User Experience**

   - [ ] Add export functionality (PDF reports, CSV data)
   - [ ] Implement query history and saved diagnoses
   - [ ] Add comparison mode (compare multiple incidents)
   - [ ] Interactive visualizations (network graphs, timelines)

4. **Model Improvements**

   - [ ] Fine-tune embeddings on aviation domain data
   - [ ] Experiment with different embedding models
   - [ ] Add ensemble methods for diagnosis

5. **Testing & Validation**

   - [ ] Unit tests for core functions
   - [ ] Integration tests for agent workflow
   - [ ] Validation against known incident outcomes
   - [ ] Performance benchmarking

6. **Documentation**

   - [ ] API documentation
   - [ ] User guide
   - [ ] Developer documentation
   - [ ] Mathematical methodology paper

---

## **Dependencies**

Key Python packages:

- `streamlit` - Web interface
- `openai` - LLM API and embeddings
- `numpy` - Numerical operations
- `scikit-learn` - Cosine similarity calculations
- `pandas` - Data manipulation (for data processing scripts)

---

## **Usage**

### **Running the Streamlit App:**

```bash
streamlit run streamlit_app.py
```

### **Using the Diagnosis Engine Directly:**

```python
from main_app import diagnose_root_causes

results = diagnose_root_causes("engine fire during takeoff", top_n=10)
print(results['top_causes'])
```

### **Running the Agent:**

```python
from agent import run_diagnosis_agent

results = run_diagnosis_agent("engine fire during takeoff")
print(results['final_summary'])
```

---

## **Methodology Notes**

### **Weighted Bayesian Diagnosis**

The core diagnosis uses a similarity-weighted approach:

- **Similarity Score**: Cosine similarity between query embedding and incident embeddings
- **Weight**: Similarity score serves as the weight for each incident's evidence
- **Probability Calculation**:
  ```
  P(cause) = Σ(similarity × presence) / Σ(similarity)
  ```
- **Top N Incidents**: Analyzes top 50 most similar incidents by default

### **Why This Approach?**

1. **Empirical**: Based on actual historical data, not theoretical models
2. **Weighted**: More similar incidents contribute more to the diagnosis
3. **Transparent**: Clear mathematical foundation
4. **Scalable**: Works with large datasets efficiently

---

## **Conclusion**

The Aviation Incident Diagnosis Engine has successfully evolved from the original plan into a comprehensive system combining semantic search, weighted Bayesian analysis, and LLM agent orchestration. The current implementation provides:

- ✅ Robust data consolidation and embedding generation
- ✅ Accurate similarity-weighted diagnosis
- ✅ Intelligent LLM agent workflow
- ✅ User-friendly web interface

The system is ready for use and can be extended with the enhancements outlined above.
