# ✅ Check Plan Compliance Command

Use this command to **verify that your implementation matches a LOCKED PLAN** or design document.

## What This Does

This command:
1. **Finds the Plan** - Locates the LOCKED PLAN or design doc you specify
2. **Reads Implementation** - Examines the actual code
3. **Compares** - Checks if code matches plan's objectives, steps, and success criteria
4. **Intent Verification** - Proves that the implementation logically implies the plan's objectives
5. **Reports Gaps** - Identifies what's missing, what's different, or what's extra

## When to Use

- **After Implementation** - "I built this feature, does it match the plan?"
- **Mid-Development Checkpoint** - "Am I still on track with the plan?"
- **Pre-Merge Verification** - "Does this PR fulfill the original plan?"

## How to Use

After typing `/check-plan`, specify the plan and what to check:

**Examples:**
- "Check if the reward shaping implementation matches docs/plans/reward_shaping.md"
- "Verify the PPO training matches the LOCKED PLAN"
- "Does the current state space match the design spec in docs/design.md?"
- "Check plan compliance for the entire environment module"

## What Gets Checked

- **Objective Alignment** - Does code achieve the plan's objective?
- **Step Completion** - Are all planned steps implemented?
- **Success Criteria** - Do metrics/evaluation match the plan?
- **Architecture Compliance** - Does structure match the design?
- **Missing Features** - What did the plan say but code doesn't have?
- **Extra Features** - What's in code but not in plan (scope creep?)

## Output Format

- **Compliance Score** - How well does it match? (High/Medium/Low)
- **Matches** - What aligns correctly
- **Gaps** - What's missing from the plan
- **Deviations** - What's different (and whether that's good or bad)
- **Recommendations** - Should you update the plan or fix the code?

## Tips

- If you don't specify a plan, I'll search for `LOCKED_PLAN.md` or `*.plan.md` files
- Mention if deviations are intentional: "I changed approach X, verify it still meets objectives"
- Use `/audit` for deeper logical analysis beyond plan compliance

---

**Ready?** Describe the plan and what to check below:
