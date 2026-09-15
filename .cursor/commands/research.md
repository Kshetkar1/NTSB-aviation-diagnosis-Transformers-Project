Run a research task using a CLI subagent.

Instructions:
1) Define the research question (e.g., "How does auth work?", "Find all API endpoints").
2) Create `.cursor/state/gsd/RESEARCH_QUERY.md` with the question.
3) Spawn the **RESEARCHER** subagent:
   - Prompt: "Read .cursor/state/gsd/RESEARCH_QUERY.md. Explore the codebase to answer the question. Write findings to .cursor/state/gsd/RESEARCH_RESULTS.md."
4) Wait for completion.
5) Review `.cursor/state/gsd/RESEARCH_RESULTS.md`.
