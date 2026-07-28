# 🔍 Deep Codebase Audit Command

Use this command to perform a **Deep Reasoning Codebase Audit** - evaluating logical correctness, problem-solution alignment, and design reasoning.

## What This Does

This command triggers the Deep Codebase Audit protocol, which will:
1. **Context Acquisition** - Search for design docs, plans, and README to understand intent
2. **Map the Project** - Build a mental model of the codebase structure
3. **Logical Proof Protocol** - Define invariants and prove step-by-step that code maintains them
4. **Adversarial Audit** - Try to find where the code might break
5. **Reasoning Analysis** - Evaluate the logic behind design decisions
6. **Output Report** - Provide prioritized findings (P0/P1/P2) with evidence

## How to Use

After typing `/audit`, specify what to audit:

**Examples:**
- "Audit the reward calculation logic"
- "Deep audit the entire state space implementation"
- "Audit this code against the LOCKED PLAN in docs/plans/reward_shaping.md"
- "Check if the PPO training loop matches the design spec"

## What Gets Checked

- **Problem-Solution Alignment** - Does code solve the stated problem?
- **Logical Correctness** - Are there bugs, edge cases, or logic errors?
- **Data Handling** - Input validation, data leakage (ML), silent failures
- **Architecture Clarity** - Can someone understand the reasoning?
- **Notebook/Results Consistency** - Do outputs match code behavior?
- **Plan Compliance** - Does implementation match the original plan?

## Output Format

You'll receive:
- **Executive Summary** with confidence score
- **Issues & Gaps** (P0/P1/P2 prioritized)
- **Reasoning Analysis** of design decisions
- **Prioritized Next Steps**

## Tips

- If you have a LOCKED PLAN, mention it: "Audit against docs/plans/X.md"
- Be specific: "Audit the reward function" vs "audit everything"
- After audit, use `/planning-workflow` if major refactoring is needed

---

**Ready?** Describe what you want audited below:
