# 🚀 Smart PR Command

Use this command to open a Pull Request that is verified, summarized, and logically sound.

## What This Does

This is the "Publisher" of your workflow. It ensures that what you are sharing is ready for review.
1. **Plan Compliance:** Runs `/check-plan` to verify alignment with the LOCKED PLAN.
2. **History Aggregation:** Summarizes individual Changelog entries into a cohesive PR description.
3. **Proof Verification:** checks for a "Proof of Correctness" for critical features.

## How to Use

After typing `/smart-pr`:

1. I will run a final check on the current branch.
2. I will read the `docs/about/changelog.md` entries associated with this branch.
3. I will draft a PR title and body.
4. I will use `gh pr create` to open it.

## The Protocol

**1. Pre-Flight Check**
- Run `/check-plan` logic.
- Are all tests passing? (Check `make test` output if available).
- Are docs synced?

**2. Narrative Generation**
- Instead of "Fixed bugs", I will write:
  > "This PR implements the Reward Shaping mechanism defined in Plan #3.
  > **Key Decisions:**
  > - Switched to additive shaping (see Changelog #4) because multiplicative was unstable.
  > - Refactored State Space to support new metrics."

**3. Logical Proof Check**
- For P0 features, I will ask: "Do we have a logical proof that this implementation meets the safety constraints?"
- Include this proof in the PR description.

## Output

- A drafted PR description.
- A command to create the PR.

---

**Ready?** Say "Smart PR" to package your work.
