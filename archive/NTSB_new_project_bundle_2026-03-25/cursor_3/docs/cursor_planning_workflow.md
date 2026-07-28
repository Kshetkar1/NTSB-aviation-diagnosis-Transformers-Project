> Iteration Rule: Every plan is produced **provisionally**, then red-teamed, then refined via targeted questions. Plans must show a Δ-log (what changed and why) between versions until LOCKED.


# Shivam’s Planning Workflow Guideline

This document defines the reasoning workflow Cursor should follow
whenever Shivam requests a new **plan**, **experiment**, or **system design**.

It teaches Cursor how to think — not just what to build.

---

## 🧭 I. Context and Problem Understanding
**Goal:** Establish clarity before building.

Cursor should:
1. Restate the **core objective** clearly.
2. Identify the **domain or system context** (tools, models, datasets, etc.).
3. Explain **why** this task exists — what decision or value it supports.
4. Note any **constraints**, **dependencies**, or **unknowns**.

_Output:_ short contextual summary and key constraints.

---

## 🧩 II. System and Environment Analysis
**Goal:** Understand what already exists.

Cursor should:
1. Map relevant modules, components, or subsystems.
2. Explain how these connect (data flow, APIs, dependencies).
3. Identify reusable patterns or existing frameworks.
4. Highlight unclear integration points or missing data.

_Output:_ “Current State Analysis” — concise, factual overview.

---

## 🧪 III. Reasoning and Approach
**Goal:** Design the logical path forward.

Cursor should:
1. Generate possible approaches or hypotheses.
2. Justify the chosen approach based on logic, feasibility, or prior work.
3. Define **metrics**, **evaluation methods**, and **expected outcomes**.
4. Explain tradeoffs when applicable.

_Output:_ “Approach” and “Reasoning” sections explaining *why* and *how*.

---

## 🧱 IV. Implementation Blueprint
**Goal:** Define the execution structure.

Cursor should:
1. Specify file and folder layout.
2. Describe what each component or script will contain.
3. Reference existing examples for consistency.
4. List helper utilities, configs, and parameters.

_Output:_ structured breakdown of what to create.

---

## 📘 V. Supporting Assets
**Goal:** Extend beyond code.

Cursor should:
1. Include documentation deliverables (`README.md`, notebooks, etc.).
2. Add analysis or visualization components if relevant.
3. Describe result formats, outputs, and how to interpret them.

_Output:_ “Supporting Files” section.

---

## ✅ VI. Verification and Next Steps
**Goal:** Ensure logical completeness.

Cursor should:
1. Recheck alignment with original objective.
2. List validation and success criteria.
3. Mention potential risks or future improvements.
4. Summarize next actionable steps.

_Output:_ short “Verification / Next Steps” note.

---

## 🎯 VII. Tone and Presentation
- Use structured, calm, analytical phrasing.
- Show reasoning before building.
- Use minimal but meaningful icons (🧩 🧭 🧪 🔍) to indicate conceptual phases.
- Avoid verbosity; emphasize clarity, coherence, and traceability.

---

### Example Output Structure
When Shivam provides a query, structure the response as:

# Objective
# Context / Current State
# Approach / Reasoning
# Implementation Blueprint
# Supporting Assets
# Verification / Next Steps

Each section should have clear reasoning and concise deliverables.

---

**End of Guideline**