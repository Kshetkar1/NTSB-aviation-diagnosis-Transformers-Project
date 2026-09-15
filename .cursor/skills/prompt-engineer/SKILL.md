---
name: prompt-engineer
description: optimization skill to generate high-fidelity subagent prompts.
---

# Prompt Engineer Skill

## Purpose
Transform a vague task into a rigorous prompt for a CLI subagent.

## Prompt Construction Rules
1.  **Role & Constraint:** Always specify the agent's role (Implementer/Verifier) and read constraints (SKILL.md).
2.  **Context Loading:** Explicitly list *which* files to read first. Don't say "read relevant files." Say "Read src/config.py and src/main.py".
3.  **Memory Injection:** Always include content from `.cursor/state/ralf/MEMORY.md` if it exists.
4.  **Acceptance Criteria:** Copy exact AC from `TASKS.yaml`.
5.  **Chain of Thought:** Instruct the agent to "Think step-by-step in comments before writing code."

## Template
"""
You are the [ROLE] subagent.
1. Read .cursor/skills/[SKILL]/SKILL.md.
2. Read .cursor/state/ralf/MEMORY.md (Project Memory).

CONTEXT:
[List specific files]

TASK:
[Title]
[Description]

ACCEPTANCE CRITERIA:
[List AC]

INSTRUCTIONS:
- Think before acting.
- [Specific instruction]
- Write summary to notes.md.
"""
