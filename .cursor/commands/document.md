# 📝 Document Command

Use this command to update documentation based on code changes. It acts as the "Scribe" for your repository.

## What This Does

It performs **Differential Documentation**:
1. Reads your `git diff` or specified file changes.
2. Reads the corresponding documentation file (e.g., `code.py` -> `docs/code.md`).
3. Identifies gaps: "Code does X, docs say Y."
4. Drafts the update.

## How to Use

**Manual Mode:**
- Type `/document` followed by a file or "all".
- "Document `rewards.py`"
- "Update docs for the `Agent` class"

**Integrated Mode (via `/smart-commit`):**
- Automatically triggered when you try to commit code changes without doc changes.

## Logic

1. **Map:** Match source file to doc file.
2. **Compare:** Is the doc outdated?
   - Missing parameters?
   - Changed logic?
   - New side effects?
3. **Draft:** Create the markdown update.
4. **Link:** Reference the Plan or Decision that caused this change.

## Tips

- "Document this function" works for small scopes.
- "Update all docs" scans the entire `docs/` folder against `src/` (slow but thorough).

---

**Ready?** Tell me what to document.
