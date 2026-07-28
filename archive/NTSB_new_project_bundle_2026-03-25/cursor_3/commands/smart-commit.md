# 🧠 Smart Commit Command

Use this command to perform a **Synchronized Commit** that ensures code, documentation, and history are always aligned.

## What This Does

This is the "Gatekeeper" of your repository. It orchestrates three steps before allowing a commit:
1. **Docs Synchronization:** Checks if changes in code (e.g., `rewards.py`) require updates in documentation (e.g., `docs/environment.md`).
2. **Decision Logging:** prompts the creation of a narrative entry for the Changelog/Decision Log.
3. **Commit:** Generates a semantic commit message and commits the changes.

## How to Use

After typing `/smart-commit` (or just asking "Smart commit these changes"):

1. I will analyze your `git diff --staged`.
2. I will ask: "Did we update the docs?" If not, I will draft the update for you.
3. I will ask: "Why did we make this change?" to draft the Changelog entry.
4. I will propose the final commit (Code + Docs + Log).

## The Protocol

**1. Analyze Changes**
- Identify modified modules.
- Check corresponding documentation files.

**2. Enforce Documentation**
- IF code changed BUT docs didn't: STOP.
- Draft the documentation update and ask to stage it.

**3. Log the Decision**
- Create an entry in `docs/about/changelog.md` (or similar).
- Format:
  - **Context:** What was happening?
  - **Decision:** What did we change?
  - **Reasoning:** Why? (The "Logical Proof" of the change).

**4. Execute Commit**
- Generate Conventional Commit message (`feat:`, `fix:`, `refactor:`).
- Run `git commit`.

## Tips

- Use this INSTEAD of `git commit` for all non-trivial changes.
- If I find no docs to update, I will proceed to the log.
- You can say: "Smart commit: implemented reward shaping, docs already updated."

---

**Ready?** Stage your files and say "Smart commit".
