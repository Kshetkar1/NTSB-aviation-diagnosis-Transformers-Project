# GitHub Workflow Reference

> Detailed reference for the GitHub workflow protocol.
> The concise always-on rule is in `.cursor/rules/github_workflow.mdc`.

## Branch Naming Conventions

- **Feature branches**: `feature/<feature-name>` (e.g., `feature/reward-shaping`)
- **Stage branches**: `feature/<parent>-stage<N>` (e.g., `feature/rl_environment-stage1`)
- **Base branch**: `main` (default) or a feature parent branch for staged work

## Commit Message Best Practices

- **Use conventional commits**: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`
- **Reference issues**: Include `fixes #X` or `closes #Y` when applicable
- **Be descriptive**: Explain what and why, not just what
- **Keep focused**: One logical change per commit
- **Auto-close keywords**: `fixes`, `closes`, `resolves` followed by `#<issue-number>`

## Verification Commands

```bash
# Check GitHub Actions status
gh run list --branch <branch-name> --limit 5

# View specific workflow run
gh run view <run-id> --log

# Check if issues were closed
gh issue list --state closed --limit 10

# View PR status
gh pr view <pr-number>

# Check PR checks
gh pr checks <pr-number>
```

## Expected GitHub Actions Behavior

When pushing to feature branches or creating PRs:
- **CI workflow** (`.github/workflows/ci.yml`) should:
  - Run `make test` on Python 3.12
  - Run `make docs` to build documentation
  - Both jobs should pass

- **Issue closing workflow** (`.github/workflows/close-issues.yml`) should:
  - Parse commit messages for keywords (`fixes #X`, `closes #Y`, etc.)
  - Automatically close referenced issues **when PRs are merged** (not on push)
  - Run on `pull_request` events with `type: closed` and `merged: true`

## Post-Push Verification Checklist

### A) GitHub Actions CI Status
- Check workflow runs: `gh run list --branch <branch-name>`
- Verify test job passes
- Verify docs job passes
- Check for failures: `gh run view <run-id>` for details

### B) Issue Auto-Closing
- Issues are closed when PRs are **merged**, not on push
- After PR merge: check commit messages for `fixes #X`, `closes #Y`, etc.
- Verify issue status: `gh issue view <issue-number>`

### C) Pre-commit Hooks (if applicable)
- Check commit status: verify hooks ran successfully
- Review any auto-fixes: check if pre-commit made formatting changes

## Troubleshooting

If verification fails:
1. **Check GitHub Actions logs**: `gh run view <run-id> --log`
2. **Verify workflow files**: Ensure `.github/workflows/*.yml` are correct
3. **Check branch protection**: Verify branch allows pushes
4. **Review commit messages**: Ensure they match expected format
5. **Check permissions**: Verify GitHub token has necessary permissions
