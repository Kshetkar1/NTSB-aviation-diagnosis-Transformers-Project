# Experiment Loop

Run an autonomous experimentation cycle to answer a research question.

## Instructions

1) Read `.cursor/skills/experiment-loop/SKILL.md` and follow it exactly.

2) Read existing context:
   - `.cursor/state/experiments/JOURNAL.md` (if exists) -- prior experiments
   - `.cursor/state/experiments/STATUS.md` (if exists) -- any in-progress experiment
   - `.cursor/state/ralf/MEMORY.md` (if exists) -- project learnings

3) If the user provided a research question, start the loop from Step 1 (HYPOTHESIZE).
   If resuming a paused experiment, read STATUS.md and continue from where it left off.

4) Run autonomously for up to 5 iterations. REPORT or ESCALATE after 5.

5) After stopping (REPORT or ESCALATE), present:
   - A concise summary of what was learned
   - Key evidence (numbers, comparisons)
   - Confidence level (high/medium/low)
   - Recommended next steps (if any)

## Tips

- Start with "experiment: [your question]" for a clean invocation.
- Say "continue" to grant more iterations if the agent paused at the budget limit.
- Say "pivot to [new direction]" to redirect mid-experiment.
- The experiment journal at `.cursor/state/experiments/JOURNAL.md` is your permanent research record.
