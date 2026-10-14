<!-- SPDX-License-Identifier: Apache-2.0 -->
# CLI output: house rules

`tt-model`'s terminal output is rendered through one module,
[`src/tt_kernel/console.py`](../src/tt_kernel/console.py), ported from the `cli-design`
skill in [`.claude/skills/cli-design/`](../.claude/skills/cli-design/). Read
`SKILL.md` there for the full design language and `reference/patterns.md` for the
failure and progress patterns. This file is the short version plus what is specific to
`tt-model`.

## The one rule

**Every line the user sees is a line you chose to show.** Nothing should reach the terminal
only because a subprocess printed it. Raw `pip`, `hf`, and vLLM output is evidence to
capture and keep; surface it on failure, and otherwise show a sentence you wrote.

## The vocabulary

| Glyph | Meaning | How |
|---|---|---|
| `✓` | success | `step()` collapsing, or `milestone()` |
| `○` | benign no-op / expected skip | `handle.skip("reason")`, `note()` |
| `✗` | failure | `handle.fail()` + a diagnosis card |
| `◉` | current phase | `stepper_line()` |
| `!` | actionable warning | `note(..., marker="!", style="warning")` |

Elapsed time is appended by `step()` only when it is 0.8 s or more, so fast steps stay quiet.

## Folding

`show_detail()` is the single predicate: `verbose or not in_phase()`. Gate routine "done"
lines on it. **Do not gate failures, prompts, or actionable warnings.** They are why the
user is watching. On the phase-less commands the predicate is always `True`, so a gated
failure would look fine now and vanish the day that command gains a phase.

## Machine-readable output bypasses Rich

`serve --print`, `search --json`, and `info` must go through `console.raw()`, not
`console.print()`. Rich wraps at terminal width and parses `[...]` as markup; either corrupts a
pasteable command line or a JSON document. There is a `COLUMNS=40` test for this in
`tests/test_cli_output.py`.

The bundle's own `run.sh` and the vLLM server it launches print for themselves. That output
belongs to the served process rather than the CLI, and stays as-is.

## Subprocesses

- `step()` captures the block's stdout and stderr and reveals it **only on failure**.
- `contextlib.redirect_stdout` is **Python-level only**. A child process inherits fd 1 and
  paints over the spinner. Background children need `stdout=DEVNULL, stderr=STDOUT` and a
  log file.
- Third-party progress bars are a second live writer on the same row. `tt-model` suppresses
  them in-phase (`HF_HUB_DISABLE_PROGRESS_BARS`) and bridges them into the activity row via
  `tqdm_class`; do not re-enable them.
- **Stop the spinner before handing the terminal to a foreground child.** `serve` execs
  vLLM into the foreground; a still-ticking ticker and vLLM fight for the row.

## The serve boot checklist

`serve` on a container package watches the boot through `console.checklist()`: a vertical
list where only the active row is live (`⠹ label  ▕████░░░░▏ 12/32 · 38%  0:42`) and every
finished row is printed once. The rows come from `boot_progress.BootTracker`, a pure parser
over the container log (`tests/fixtures/boot_logs` are real boots). Raw log lines reach the
terminal only under `-v`. On success `view.clear()` erases the whole list (a known count of
rows, so a fixed number of cursor-ups) and re-prints the `!` warnings. On failure the list
stays and a `failure_card` built by `boot_progress.diagnose_boot` follows. Only one checklist
may be active and `step()` must not run inside it: do `step()`-shaped work first, then open
the list.

## Progress denominators

Only show a bar for a total you actually know. `pip` reports package counts from its
`Collecting` and `Installing collected packages` lines but not total bytes, so count packages
and show bytes as a plain counter. `61/104 packages · 412 MB` is accurate; an invented
percentage is not. `progress_bar()` returns `""` for an unknown total by design.

## Failures: render a diagnosis card

Classify in a **pure** function (text in, dict out, unit-testable), then render one card:
cause in the title, one line of evidence, the **consequence** (fatal, or does the run
continue?), then `Try:` actions. Render it **after** the step collapses. Inside the
capturing block it is swallowed and re-emitted uncolored.

Do not offer to fix the user's machine. `tt-model` does not own the process holding a port
or the TT-Metalium™ build. The card's job is to name the problem.

## Before you commit

```bash
pytest -q
tt-model list | cat -v | grep -c '\^\['         # non-TTY: expect 0 (a structured offline command)
COLUMNS=40 tt-model serve <id> --print          # one unwrapped line
COLUMNS=80 tt-model list; COLUMNS=120 tt-model list
tt-model pull <id> -v                           # folded detail returns
```

Anything animated needs a real PTY (`pty.spawn`). Assert the spinner advances, the row is
erased (`\r\033[2K`) before the result line, and no exit path leaves the terminal dirty.
