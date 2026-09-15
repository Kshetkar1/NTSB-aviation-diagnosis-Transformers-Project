---
name: experiment-loop
description: >
  Autonomous experimentation skill for data science workflows.
  Hypothesis -> design -> execute (Jupyter) -> analyze -> journal -> iterate.
  Runs without blocking the user. "Done" = a question answered with evidence, not code shipped.
---

# Experiment Loop Skill

## Purpose

Convert a vague research question into a structured experiment cycle that the agent runs autonomously. The loop continues until the question is answered with evidence or a dead-end is reached and escalated.

This is NOT the RALF implementation loop. RALF's "done" = gates pass. Experiment Loop's "done" = **a question answered with evidence**.

## When to Use

- User says "investigate", "test whether", "figure out why", "explore", "experiment", "try different", "what happens if", "compare", or similar exploratory/research language.
- User asks a question about data, model behavior, or system performance that requires running code and looking at results.
- User provides a notebook and asks for analysis, iteration, or exploration.

## The Loop

```
HYPOTHESIZE -> DESIGN -> EXECUTE -> ANALYZE -> JOURNAL -> DECIDE
     ^                                                       |
     |___________________ iterate ___________________________|
```

### Step 1: HYPOTHESIZE

Before touching any code:
- **Question**: Restate the research question in one sentence.
- **Hypothesis**: State a falsifiable prediction. Use the form: "If [condition], then [expected observation], because [reasoning]."
- **Success signal**: What specific output (number, plot shape, comparison) would confirm or reject this hypothesis?
- **Null hypothesis**: What does "no effect" or "inconclusive" look like?

Write this to the experiment journal (see State section).

### Step 2: DESIGN

Plan the experiment before executing:
- **What to run**: Which notebook (existing or new)? Which cells? What parameters?
- **What to measure**: Specific metrics, outputs, or visual patterns to look for.
- **Controls**: What stays constant? What varies? If comparing, what's the baseline?
- **Risks**: What could invalidate results (data leakage, wrong split, stale cache, random seed)?

Keep the experiment as small as possible. Test one thing at a time.

### Step 3: EXECUTE

Run the notebook or cells. Prefer the smallest execution unit that tests the hypothesis.

**Execution methods (in order of preference):**

1. **Edit + shell execution with papermill** (for parameterized runs):
   ```bash
   papermill notebooks/experiment.ipynb notebooks/output/experiment_run01.ipynb \
     -p learning_rate 0.01 -p epochs 50
   ```
   Papermill saves the executed notebook with all outputs. Read the output notebook afterwards.

2. **Edit notebook cells directly** (for interactive exploration):
   - Use the EditNotebook tool to modify cells.
   - Use shell to execute: `jupyter nbconvert --to notebook --execute notebooks/experiment.ipynb --output experiment_executed.ipynb`
   - Read the executed notebook to inspect outputs.

3. **Python script extraction** (for heavy computation):
   - Extract the core logic into a `.py` script.
   - Run it via shell, capture stdout/stderr.
   - Script should write results to a known location (CSV, JSON, or plots to `reports/figures/`).

**After execution:**
- Read the output notebook or result files.
- For notebooks: inspect cell outputs (text, tables, error tracebacks).
- For images/plots: note the file paths -- the user can view them. Describe what you expect to see vs. what you actually see based on any text/numeric outputs.

### Step 4: ANALYZE

Interpret the results honestly:
- **Observation**: What did the outputs show? Be precise (numbers, trends, shapes).
- **vs. Hypothesis**: Does this confirm, reject, or partially support the hypothesis?
- **Surprises**: Anything unexpected? Anomalies? Errors?
- **Confounds**: Could something else explain the result (bug, data issue, wrong metric)?

Do NOT cherry-pick or rationalize. If the result is unclear, say so.

### Step 5: JOURNAL

Append to the experiment journal (`.cursor/state/experiments/JOURNAL.md`):

```markdown
## Experiment E<NNN>: <title>
**Date**: <YYYY-MM-DD>
**Question**: <one sentence>
**Hypothesis**: <falsifiable prediction>
**Method**: <what was run, with what parameters>
**Result**: <what was observed -- precise numbers/descriptions>
**Interpretation**: <confirms/rejects/inconclusive + reasoning>
**Next**: <what follows from this result>
```

Also update `.cursor/state/experiments/STATUS.md` with current state.

### Step 6: DECIDE

Based on the result, choose one:

- **ITERATE**: The result suggests a refinement. Formulate the next hypothesis and loop back to Step 1. Continue autonomously.
- **PIVOT**: The approach is fundamentally wrong. Formulate a new direction. Summarize why in the journal. Loop back to Step 1 with the new direction.
- **REPORT**: The question is answered (confirmed or rejected with evidence). Write a summary finding to the journal. Stop and present results to user.
- **ESCALATE**: You're stuck -- ambiguous results after 3+ iterations, need domain expertise, or need data/resources you can't access. Stop and present what you've tried, what you've learned, and what you need from the user.

**Autonomy budget**: Run up to **5 iterations** without user input. After 5, always REPORT or ESCALATE regardless of state. The user can say "continue" to grant more iterations.

## State

All experiment state lives in `.cursor/state/experiments/`:

| File | Purpose |
|------|---------|
| `JOURNAL.md` | Append-only experiment log. Never delete entries. |
| `STATUS.md` | Current experiment: question, iteration count, status (running/paused/done) |
| `HYPOTHESES.md` | Running list of hypotheses tested with outcomes (quick reference) |
| `notebooks/` | Output notebooks from papermill runs (gitignored) |

Create these files if they don't exist. Never overwrite JOURNAL.md -- only append.

## Experiment Numbering

- Experiments are numbered E001, E002, etc. (global counter across all questions).
- Read `JOURNAL.md` to find the last experiment number and increment.
- If no journal exists, start at E001.

## Interaction with User Notebooks

- **User's notebooks are sacred.** Do NOT modify notebooks in `notebooks/` unless the user explicitly asks.
- Instead, create experiment-specific copies or use papermill to produce output notebooks in `.cursor/state/experiments/notebooks/`.
- If you need to add analysis cells, create a new notebook or append to an experiment notebook -- never edit the user's source notebooks.

## Context Management

- Before starting, read the project README and any existing experiment journal to understand context.
- Read MEMORY.md if it exists -- it may contain relevant learnings from implementation work.
- For large notebooks (>50 cells), read only the cells relevant to the hypothesis.
- Summarize dataframe outputs rather than dumping raw tables.

## Using Subagents in the Experiment Loop

For complex experiments, the orchestrator may delegate parts of the loop to subagents via the Task tool:

- **Research phase**: Spawn a `researcher` subagent (model=fast, readonly=true) to survey existing code/data before designing an experiment.
- **Analysis phase**: If the output is large or requires specialized interpretation, spawn an `explore` subagent (model=fast, readonly=true) to summarize findings.
- **Council on pivot decisions**: If results are ambiguous after 2+ iterations, spawn `challenger` + `explorer` subagents in parallel (both fast, readonly) to get a council perspective on whether to pivot.

The experiment loop itself runs in the orchestrator (not as a subagent) because it needs to:
- Edit notebooks (requires write access)
- Execute shell commands (papermill, nbconvert)
- Make multi-step decisions across iterations

## Integration with GSD/RALF

- Experiment results can inform GSD specs. If an experiment reveals a design constraint, note it in the journal and mention it when running `/gsd-spec`.
- If an experiment requires code changes, switch to RALF for implementation. The experiment loop resumes after RALF completes.
- Experiment JOURNAL.md is never archived by `/reset-state` -- it's a permanent research record.

## Example Flow

**User**: "Does graph expansion (1-hop vs 2-hop) actually improve retrieval quality?"

**Agent (autonomous)**:
1. **H: E001**: "2-hop graph expansion will return more relevant nodes than 1-hop, measured by average similarity score of top-5 results."
2. **Design**: Run retrieval benchmark with 10 sample queries, compare faiss-only vs 1-hop vs 2-hop. Measure mean similarity score and unique node count.
3. **Execute**: Create experiment notebook, run with papermill.
4. **Analyze**: 2-hop returns 40% more nodes but mean similarity drops 15%. Mixed signal.
5. **Journal**: Record finding.
6. **Decide**: ITERATE -- refine hypothesis to test whether 2-hop helps for *specific query types*.
7. **H: E002**: "2-hop expansion improves retrieval for multi-topic queries but hurts single-topic queries."
8. ... (continues autonomously until answered or budget exhausted)
