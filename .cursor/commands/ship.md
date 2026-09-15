Prepare the work to ship.

Instructions:
1) Ensure `/verify` passes.
2) Summarize changes:
   - What was built
   - Why it matters (tie to PRD success metrics)
   - How to test
   - Risks + mitigations

3) Write:
   - `.cursor/state/ralf/SHIP.md` as a PR description draft.

4) **Auto-Update Changelog:**
   - Read `CHANGELOG.md` (or create if missing).
   - Append a new entry under an "Unreleased" or current date header.
   - Use the summary from SHIP.md.
   - **Do not overwrite** existing history; append/prepend safely.

SHIP.md must include:
- Title
- Context
- Summary of changes
- Verification
- Screenshots/logs if relevant
- Rollback plan
