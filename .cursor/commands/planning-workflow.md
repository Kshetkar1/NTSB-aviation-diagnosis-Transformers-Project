# 🧭 Planning Workflow Command

Use this command to initiate the **Iterative Planning Directive** for new features, experiments, or system designs.

## What This Does

This command triggers the Planning Directive workflow, which will:
1. **Audit Snapshot** - Understand the current state and constraints
2. **Create Plan v1** - Draft a provisional plan
3. **Self-Critique** - Challenge the plan's assumptions
4. **Ask Targeted Questions** - Get clarification on key decisions
5. **Iterate** - Refine until you say "lock plan"
6. **Output LOCKED PLAN** - Create the Source of Truth for future audits

## How to Use

After typing `/planning-workflow`, describe what you want to plan:

**Examples:**
- "Plan a reward shaping mechanism for my RL environment"
- "Design an experiment to test different hyperparameter configurations"
- "Create architecture for a distributed training system"
- "Plan a refactoring of the state space representation"

## Workflow After Planning

Once you have a **LOCKED PLAN**:
1. **Implement** the code following the plan
2. **Audit** your implementation using: `/audit` or "audit this code"
3. **Verify** the code matches the plan's intent
4. **Iterate** if needed

## Tips

- Be specific about scope and constraints
- Mention any dependencies or existing code to consider
- The planning directive will ask clarifying questions - answer them to refine the plan
- Say **"lock plan"** when you're ready to finalize

---

**Ready?** Describe what you want to plan below:
