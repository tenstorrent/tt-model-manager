<!-- SPDX-License-Identifier: Apache-2.0 -->
# CLI output — house rules

`tt-model`'s terminal output is rendered through one module,
[`src/tt_kernel/console.py`](../src/tt_kernel/console.py), ported from the `cli-design`
skill in [`.claude/skills/cli-design/`](../.claude/skills/cli-design/). Read
`SKILL.md` there for the full design language and `reference/patterns.md` for the
failure/progress patterns. This file is the short version plus what is specific to us.

## The one rule

**Every line the user sees is a decision you made. Nothing reaches the terminal because a
subprocess happened to print it.** Raw `pip`/`hf`/vLLM output is *evidence*, not UI —
capture it, keep it, and surface a sentence you wrote.

## The vocabulary

| Glyph | Meaning | How |
|---|---|---|
| `✓` | success | `step()` collapsing, or `milestone()` |
| `○` | benign no-op / expected skip | `handle.skip("reason")`, `note()` |
| `✗` | failure | `handle.fail()` + a diagnosis card |
| `◉` | current phase | `stepper_line()` |
| `!` | actionable warning | `note(..., marker="!", style="warning")` |

Elapsed time is appended by `step()` only when ≥0.8s, so fast steps stay quiet.

## Folding

`show_detail()` is the single predicate: `verbose or not in_phase()`. Gate routine "done"
lines on it. **Never gate failures, prompts, or actionable warnings** — they are why the
user is watching. On our phase-less commands the predicate is `True` today, so a gated
failure would look fine now and vanish the day that command gains a phase.

## Machine-readable output bypasses Rich

`serve --print`, `run --print`, `start --print`, `search --json` and `info` must go through
`console.raw()`, not `console.print()`. Rich wraps at terminal width and parses `[...]` as
markup; either corrupts a pasteable command line or a JSON document. There is a
`COLUMNS=40` test for this — see `tests/test_cli_output.py`.

`legacy_serve.py`'s `print()`s belong to the served process, not the CLI, and stay as-is.

## Subprocesses

- `step()` captures the block's stdout/stderr and reveals it **only on failure**.
- `contextlib.redirect_stdout` is **Python-level only** — a child process inherits fd 1 and
  will paint over the spinner. Background children need
  `stdout=DEVNULL, stderr=STDOUT` and a log file.
- Third-party progress bars are a second live writer on the same row. Ours are suppressed
  in-phase (`HF_HUB_DISABLE_PROGRESS_BARS`) and bridged into the activity row via
  `tqdm_class`; do not re-enable them.
- **Stop the spinner before handing the terminal to a foreground child.** `serve` execs
  vLLM into the foreground; a still-ticking ticker and vLLM will fight for the row.

## Progress denominators

Only show a bar for a total you actually know. `pip` tells us package counts from its
`Collecting` / `Installing collected packages` lines but not total bytes, so we count
packages and show bytes as a plain counter. An honest `61/104 packages · 412 MB` beats an
invented percentage; `progress_bar()` returns `""` for an unknown total by design.

## Failures: diagnose, don't dump

Classify in a **pure** function (text in, dict out, unit-testable), then render one card:
cause in the title, one line of evidence, the **consequence** (fatal, or does the run
continue?), then `Try:` actions. Render it **after** the step collapses — inside the
capturing block it is swallowed and re-emitted uncoloured.

Do not offer to fix the user's machine. `tt-model` does not own the process holding a port
or the tt-metal build; naming the problem is the job.

## Before you commit

```bash
pytest -q
tt-model doctor | cat -v | grep -c '\^\['      # non-TTY: expect 0
COLUMNS=40 tt-model serve <id> --print          # one unwrapped line
COLUMNS=80 tt-model doctor; COLUMNS=120 tt-model doctor
tt-model pull <id> -v                           # folded detail returns
```

Anything animated needs a real PTY (`pty.spawn`) — assert the spinner advances, the row is
erased (`\r\033[2K`) before the result line, and no exit path leaves the terminal dirty.
