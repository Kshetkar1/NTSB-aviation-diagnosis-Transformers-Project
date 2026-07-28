# Sequence Diagram: Streamlit App Flow

```mermaid
sequenceDiagram
    participant User
    participant StreamlitUI as Streamlit UI
    participant Agent as run_diagnosis_agent()
    participant OpenAI1 as OpenAI API<br/>(Report Generation)
    participant DiagnosticTool as diagnose_root_causes()
    participant OpenAI2 as OpenAI API<br/>(Tool Call)
    participant OpenAI3 as OpenAI API<br/>(Final Summary)
    participant Database as NTSB Database<br/>(Embeddings)

    User->>StreamlitUI: Enter incident description
    User->>StreamlitUI: Click "Run Diagnostic Agent"

    StreamlitUI->>Agent: run_diagnosis_agent(user_query)

    Note over Agent: Step 1: Generate Hypothetical Report
    Agent->>OpenAI1: Create chat completion<br/>(report generation prompt)
    OpenAI1-->>Agent: generated_report

    Note over Agent: Step 2: Analyze with Diagnostic Tool
    Agent->>OpenAI2: Create chat completion<br/>(analysis prompt + tools)
    OpenAI2-->>Agent: tool_call (diagnose_root_causes)

    Agent->>DiagnosticTool: diagnose_root_causes(query, top_n)
    DiagnosticTool->>Database: Query similar incidents<br/>(cosine similarity)
    Database-->>DiagnosticTool: Similar incidents + causes
    DiagnosticTool-->>Agent: tool_results<br/>(top_causes with probabilities)

    Agent->>OpenAI2: Append tool response to messages
    Agent->>OpenAI3: Create chat completion<br/>(synthesize final summary)
    OpenAI3-->>Agent: final_summary

    Agent-->>StreamlitUI: Return results dict<br/>{generated_report, tool_results, final_summary}

    StreamlitUI->>User: Display Agent Execution Log
    StreamlitUI->>User: Display LLM-Generated Incident Synopsis
    StreamlitUI->>User: Display LLM's Final Diagnosis Summary
    StreamlitUI->>User: Display Detailed Diagnostic Data<br/>(Top 10 causes with probabilities)
```

## Component Descriptions

- **User**: The end user interacting with the Streamlit web interface
- **Streamlit UI**: The Streamlit application (`streamlit_app.py`) handling UI rendering and user interactions
- **run_diagnosis_agent()**: The main agent function (`agent.py`) orchestrating the multi-step diagnosis process
- **OpenAI API (Report Generation)**: First LLM call to generate a hypothetical incident report
- **OpenAI API (Tool Call)**: Second LLM call that decides to use the diagnostic tool
- **OpenAI API (Final Summary)**: Third LLM call to synthesize the final diagnosis summary
- **diagnose_root_causes()**: The diagnostic tool function (`main_app.py`) that analyzes the incident
- **NTSB Database (Embeddings)**: The database of historical NTSB reports with pre-computed embeddings

## Flow Summary

1. User enters an incident description and clicks the button
2. Streamlit calls the agent function
3. **Step 1**: Agent generates a hypothetical incident report using OpenAI
4. **Step 2**: Agent calls OpenAI with tools, which triggers the diagnostic tool
5. The diagnostic tool queries the NTSB database for similar incidents
6. Tool returns structured results (top causes with probabilities)
7. **Step 3**: Agent synthesizes a final summary using OpenAI
8. Results are displayed in the Streamlit UI across multiple sections
