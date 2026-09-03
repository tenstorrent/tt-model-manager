---
name: feature-branch-pr
description: >-
  Make a change in tt-model-manager the team way: branch off `main` with a
  `fix/<slug>` / `feat/<slug>` (or `<username>/<slug>`) branch, keep the diff
  minimal and single-concern, add a regression test, run the full offline
  pytest suite before pushing, leave every other branch untouched, and open a
  **draft** PR against `main` with a human, professional title and description.
  Use whenever the user asks to start a feature/fix/branch, make a change and
  open a pull request, or "do this properly on a new branch" in this repo.
  Never pushes to `main` and never merges its own PR.
---

# tt-model-manager Feature Branch + PR Workflow

Follow this when asked to make a change and/or open a PR in
`tt-model-manager`. The goal is a small, verified, single-concern change that
touches nothing it shouldn't. [AGENTS.md](../../../AGENTS.md) is binding —
read its design invariants before touching code; a change that violates one
is wrong even if tests pass.

## Guardrails (always true)

- **Base everything on `main`.** This repo has no `dev` branch — features and
  fixes branch off `main` and PRs target `main`.
- **Never commit on `main` directly, never push to `main`, never merge your
  own PR.** PRs open as **draft**; a human reviews and clicks merge.
- **Never `git push --force`** (or `--force-with-lease`) to a shared branch.
- **Leave other branches alone.** No checkout-and-edit of unrelated branches,
  no rebasing or deleting branches you didn't create for this task.
- **One concern per PR.** One bug or one feature — no drive-by refactors, no
  bundled cleanups.
- **Respect the AGENTS.md design invariants** (self-contained bundles, HF
  distribution, weights-as-pointer, supported schemas, console.py-only
  output, rendered install.sh/run.sh). If the requested change would break
  one, stop and tell the user instead of implementing it.

## Workflow checklist

Copy this and tick as you go:

```
- [ ] 1. Pick the branch name
- [ ] 2. Branch off main
- [ ] 3. Reproduce, then make the minimal change
- [ ] 4. Add a regression test
- [ ] 5. Run the full offline suite (pytest)
- [ ] 6. Clean up instrumentation
- [ ] 7. Stage only intended files
- [ ] 8. Commit (imperative + why + trailer)
- [ ] 9. Push + open a draft PR against main
- [ ] 10. Return to the original branch
```

### 1. Pick the branch name

The AGENTS.md convention is `fix/<slug>` or `feat/<slug>` (short kebab-case:
`fix/stale-vllm-bundle`, `feat/serve-refresh`). A `<username>/<slug>` prefix
(from `git config user.name` / email local-part, e.g. `jashan/serve-repairs`)
is also in active use — if the user has a preference, use it; otherwise
default to `fix/` / `feat/` / `docs/`.

### 2. Branch off main

Always start from the latest `main`, and remember where you came from:

```bash
ORIG=$(git branch --show-current)        # so you can return in step 10
git fetch origin
git checkout -b <type>/<short-kebab-slug> origin/main
```

Confirm `git status` is clean before editing. Don't branch off whatever
happens to be checked out.

### 3. Reproduce first, then make the minimal change

- For a bug: reproduce it before fixing it, so the regression test in step 4
  is honest.
- Touch only the files required for the stated task. No reformatting, no
  unrelated renames, no dependency bumps "while we're here."
- If two approaches would both work, take the more minimal one — fewer files,
  fewer lines, less new machinery.
- **Reuse, don't reinvent**: extend `hub.py`, `runtime.py`, `packaging.py`,
  `manifest.py`, `metal.py`/`device.py`, `localdb.py` rather than adding
  parallel code paths.
- **All terminal output goes through `console.py`** — no new `typer.secho`,
  no scattered `print` (see [docs/cli_output.md](../../../docs/cli_output.md)).
- The bundle's `install.sh`/`run.sh` are **generated** by
  `render_install_sh`/`render_run_sh` in `packaging.py` — change them there
  (with a test), never in a staged bundle.
- If dependencies change, regenerate the lockfile: `uv lock`, then verify
  with `uv lock --check`.

### 4. Add a regression test

Every bug fix gets a test that **fails before and passes after**. New
functionality gets unit tests; end-to-end changes get integration tests. The
offline suite must not need hardware, network, or HF credentials.

### 5. Run the full offline suite

Verification is mandatory before any push:

```bash
uv sync --locked --extra test
uv run --locked --extra test pytest
```

(or `python -m pytest` inside the repo's `.venv`). Expected: **all pass** —
no hardware, no network. Record the result (e.g. "full suite: N passed") for
the PR body.

**Serve/device-path changes** additionally need hardware validation —
package → pull → serve → `curl`, per
[docs/self_contained_packages.md](../../../docs/self_contained_packages.md)
(Testing) and the checkpoints in AGENTS.md ("Application startup complete"
before the curl, coherent text after). If hardware isn't available, say so
explicitly in the PR rather than implying it was checked.

**Docs-only changes**: pytest still must stay green; also validate the
artifacts you changed (links resolve, commands are correct).

### 6. Clean up instrumentation

Remove anything added only to validate: debug prints, temporary logging,
scratch test files, commented-out experiments. The committed diff should
contain only the intended change plus its tests.

### 7. Stage only intended files

Never `git add -A` blindly. Stage explicit paths, then audit:

```bash
git add <path> <path>
git status
git diff --cached
```

No wheels or large binaries in git, no local scratch dirs, no `.env`.

### 8. Commit

Imperative subject (this repo uses Conventional-Commits-style subjects:
`fix(hub): …`, `feat(serve): …`, `docs: …`), a short body explaining *why*,
and — per AGENTS.md — end with the trailer:

```
fix(serve): repair a pulled package whose image was deleted

A pulled v5.1 package whose container image was removed from the local
daemon failed to serve with an opaque docker error. Re-pull the image by
digest when it is missing instead.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

### 9. Push and open a draft PR against main

```bash
git push -u origin <branch>
gh pr create --draft --base main \
  --title "<type>(<scope>): <human, specific subject>" \
  --body "<what/why + test evidence + tracking issue link if any>"
```

PR body: a few plain sentences on what changed and why, the test evidence
(e.g. "full suite: 142 passed"), hardware validation result if applicable,
and a link to the tracking issue if one applies. Short and human — no walls
of text. Example:

```
Serving a pulled v5.1 package failed with an opaque docker error when the
local image had been deleted. Re-pull the image by digest when missing.

- Regression test in tests/test_serve.py (fails before, passes after)
- Full offline suite: 142 passed
- Fixes #51
```

**Leave the PR as a draft and leave merging to a human** unless the user
explicitly tells you otherwise.

### 10. Return to the original branch

Leave the tree as you found it:

```bash
git checkout "$ORIG"
```

## Anti-patterns

- **Don't** branch off whatever branch happens to be checked out — always
  `origin/main`.
- **Don't** bundle unrelated cleanups into the PR. Minimal, single-concern.
- **Don't** push without running the full offline suite, or claim
  verification (especially hardware validation) you didn't run.
- **Don't** ship a fix without a regression test that failed before it.
- **Don't** force-push, rebase shared branches, or modify branches you
  didn't create.
- **Don't** push to `main`, open the PR ready-for-review by default, or
  merge your own PR.
- **Don't** silently truncate or skip integrity checks / version gates to
  make something pass.
