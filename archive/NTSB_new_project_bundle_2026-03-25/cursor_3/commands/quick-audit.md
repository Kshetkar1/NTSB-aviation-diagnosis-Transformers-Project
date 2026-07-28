# ⚡ Quick Audit Command

Use this command for a **lightweight, focused audit** of a specific file, function, or small module.

## What This Does

A streamlined audit that:
- Focuses on a single file/function/module (not the whole codebase)
- Skips the full context acquisition phase
- Performs quick logic checks and edge case identification
- Provides immediate feedback without extensive analysis

## When to Use

- **Quick sanity check** before committing
- **Single function review** - "Does this function make sense?"
- **Spot check** - "Is there an obvious bug here?"
- **Pre-merge review** - Quick check of your changes

## How to Use

After typing `/quick-audit`, specify what to check:

**Examples:**
- "Quick audit the `calculate_reward` function in environment/rewards.py"
- "Spot check the state normalization logic"
- "Quick review this file: rl_sc_cluster_utils/environment/actions.py"
- "Does this function handle edge cases correctly?"

## What Gets Checked (Focused)

- **Logic correctness** - Does it do what it claims?
- **Edge cases** - What inputs might break it?
- **Obvious bugs** - Off-by-one, null checks, etc.
- **Code clarity** - Is the intent clear?

## Output Format

Concise findings:
- **Quick Summary** (1-2 sentences)
- **Issues Found** (if any) with location and fix suggestion
- **Confidence** - "Looks good" or "Needs attention"

## Tips

- Use `/audit` for comprehensive, deep analysis
- Use `/quick-audit` for fast checks on small pieces
- Great for checking a function you just wrote

---

**Ready?** Describe what you want quickly audited below:
