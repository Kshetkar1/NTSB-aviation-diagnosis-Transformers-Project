Run verification gates and report results.

Instructions:
1) Look for repo-defined gate commands in:
   - README
   - package scripts
   - pyproject.toml / tox.ini / Makefile
2) If found: run those.
3) Else: fall back to defaults from `.cursor/rules/30-cli-and-gates.md`.
4) Write:
   - `.cursor/state/ralf/GATES.md` with:
     - commands run
     - pass/fail
     - failure output excerpt
     - fixes applied (if any)
