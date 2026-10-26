# GitHub Collaboration Workflow

This is the day-to-day playbook for collaborating on Agentic Eval Toy App via
GitHub. For coding standards and testing guidelines, see
[CONTRIBUTING.md](../CONTRIBUTING.md).

## Ownership

The repository maintainers are responsible for scope, release decisions and
final merge. GitHub auto-assigns reviewers through
[`.github/CODEOWNERS`](../.github/CODEOWNERS).

| Component | Owners |
|-----------|--------|
| Repository | @nicholaslimzixian, @mqf20 |

## For internal team members

### Working on a feature, fix or maintenance task

1. Open or confirm an issue for the work.
2. Create a focused branch from the latest `main` using a standard prefix such
   as `feat/`, `fix/`, `docs/`, `refactor/`, `test/` or `chore/`.
3. Push early and open a draft pull request targeting `main` when useful for
   team visibility.
4. Link the pull request to its issue.
5. Keep unrelated changes out of the branch.

### Code review and merge

1. Run `make ci` before requesting review.
2. Request review from a code owner.
3. Address feedback with additional commits; do not force-push shared branches.
4. Merge only after approval and required CI checks pass:
   - Squash-merge a focused single-concern branch.
   - Use a merge commit only when meaningful intermediate history must remain.
5. Delete the source branch after merge.

## For external contributors

1. Fork the repository.
2. Branch from the latest upstream `main`.
3. Follow the standards in [CONTRIBUTING.md](../CONTRIBUTING.md).
4. Open a pull request from the fork targeting upstream `main` and link the
   relevant issue.
5. A maintainer reviews and merges the pull request.

## Quick reference

| Step | Command or action |
|------|-------------------|
| Update local `main` | `git pull --ff-only origin main` |
| Create branch | `git switch -c <prefix>/<description>` |
| Run checks | `make ci` |
| Open pull request | Target `main` and link the issue |
| Merge | After code-owner approval and passing CI |
