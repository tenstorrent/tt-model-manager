# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Portable CLI output layer — copy this file into a new project.

Everything a setup/launcher CLI needs to stay readable while it drives messy
tools: a theme, a capturing `step()`, a phase stepper, panels, an in-place
activity row, and a subprocess streamer that turns tool chatter into one live
line. Only dependency is `rich`.

    from console import (activity, console, note, notice_panel, phase,
                         ready_panel, run_with_activity, set_verbose,
                         show_detail, step)

    set_verbose("-v" in sys.argv)
    register_phases(["Checks", "Build", "Launch"])
    with phase("Checks"):
        with step("Docker") as s:
            s.detail("28.5.1")
    with phase("Build"):
        rc, out = run_with_activity(["docker", "compose", "pull"], label="Pulling images")

Adapt freely; the shapes matter more than the code. See SKILL.md for the rules
this implements and reference/patterns.md for the failure-handling patterns.
"""

import atexit
import contextlib
import io
import os
import signal
import shutil
import subprocess
import sys
import threading
import time
from rich.box import ROUNDED
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── theme ────────────────────────────────────────────────────────────────────
# One palette, used by name. Swap the accent for your brand colour.
THEME = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "muted": "dim",
    "accent": "color(99)",
    "accent.bold": "bold color(99)",
})

console = Console(theme=THEME, highlight=False, soft_wrap=False)
# A second console bound to the REAL stdout: writes here bypass step()'s capture,
# so spinners and progress bars stay visible inside a captured block.
_real_console = Console(theme=THEME, file=sys.__stdout__, highlight=False)

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
PANEL_WIDTH = 78          # pin panels so a terminal resize can't reflow art
_lock = threading.RLock()  # serializes every raw escape write to the terminal

_VERBOSE = False
_IN_PHASE = False


def set_verbose(value):
    global _VERBOSE
    _VERBOSE = bool(value)


def is_verbose():
    return _VERBOSE


def in_phase():
    return _IN_PHASE


def show_detail():
    """The single folding predicate: gate every routine 'done' line on this.

    Inside a phase on a normal run the collapsed phase line is the confirmation,
    so routine output is hidden; `-v` un-hides it. Failures, prompts, and
    actionable warnings must NOT be gated on this.
    """
    return _VERBOSE or not _IN_PHASE


def _isatty():
    """True only for a genuine tty — stricter than Rich's is_terminal, which a
    forced-colour CI can flip true on a pipe."""
    try:
        return bool(sys.__stdout__) and sys.__stdout__.isatty()
    except Exception:
        return False


# ── formatting helpers ───────────────────────────────────────────────────────
def fmt_duration(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60)}s"


def fmt_bytes(num):
    """Decimal units, matching what Docker/curl report."""
    for unit, size in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if num >= size:
            return f"{num / size:.1f} {unit}"
    return f"{int(num)} B"


def progress_bar(done, total, width=14):
    """Determinate bar; empty string when the total is unknown (say so instead
    of faking a percentage)."""
    if total <= 0:
        return ""
    filled = max(0, min(width, round(width * done / total)))
    return "▕" + "█" * filled + "░" * (width - filled) + "▏"


# ── step(): one calm line per operation ──────────────────────────────────────
class _StepHandle:
    def __init__(self):
        self.failed = False
        self.skipped = False
        self.detail_text = ""
        self.start = time.monotonic()

    def fail(self):
        self.failed = True

    def skip(self, detail=""):
        self.skipped = True
        self.detail_text = detail or self.detail_text

    def detail(self, text):
        self.detail_text = text or ""


def _render_result(label, handle):
    suffix = f"  [muted]{handle.detail_text}[/muted]" if handle.detail_text else ""
    elapsed = time.monotonic() - handle.start
    if elapsed >= 0.8 and not handle.skipped:
        suffix += f"  [muted]{fmt_duration(elapsed)}[/muted]"
    if handle.failed:
        return f"[error]✗ {label}[/error]{suffix}"
    if handle.skipped:
        return f"[muted]○ {label}[/muted]{suffix}"
    return f"[success]✓[/success] {label}{suffix}"


@contextlib.contextmanager
def step(label, spinner=True, log_file=None):
    """Run an operation as a single line: `label…` (spinning) → `✓ label  1.2s`.

    The block's stdout/stderr are captured and revealed ONLY if it fails (or
    appended to `log_file` if given). Handle: .detail("28.5.1"), .skip("nothing
    to do"), .fail(). Pass spinner=False when the block may prompt — a spinner
    would fight the prompt for the row.
    """
    handle = _StepHandle()

    if _VERBOSE:                      # verbose: stream everything, still mark ✓/✗
        _real_console.print(f"[muted]{label}…[/muted]")
        try:
            yield handle
        except BaseException:
            handle.failed = True
            _real_console.print(_render_result(label, handle))
            raise
        _real_console.print(_render_result(label, handle))
        return

    buf = io.StringIO()

    def emit():
        _real_console.print(_render_result(label, handle))
        if handle.failed:             # failure: the captured detail is the evidence
            sys.__stdout__.write(buf.getvalue())
            sys.__stdout__.flush()
        elif log_file:
            try:
                with open(log_file, "a") as f:
                    f.write(buf.getvalue())
            except Exception:
                pass

    if spinner and _isatty():
        # A hand-rolled single-line spinner: rewrite the whole row every frame
        # (`\r\033[2K`), so any stray write beneath it self-heals on the next
        # tick, and erase the row before printing the result.
        stop = threading.Event()

        def spin():
            frame = 0
            f = _real_console.file
            while not stop.is_set():
                glyph = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
                text = Text()
                text.append(f"{glyph} ", style="accent")
                text.append(f"{label}…", style="muted")
                text.no_wrap, text.overflow = True, "crop"
                with _lock:
                    with _real_console.capture() as cap:
                        _real_console.print(text, end="", crop=True)
                    f.write("\r\033[2K" + cap.get())
                    f.flush()
                frame += 1
                stop.wait(0.1)

        ticker = threading.Thread(target=spin, daemon=True)
        ticker.start()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                yield handle
        except BaseException:
            handle.failed = True
            raise
        finally:
            stop.set()
            ticker.join(timeout=0.5)
            with _lock:
                _real_console.file.write("\r\033[2K")
                _real_console.file.flush()
            emit()
        return

    _real_console.print(f"[muted]{label}…[/muted]")
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            yield handle
    except BaseException:
        handle.failed = True
        raise
    finally:
        if _real_console.is_terminal:
            _real_console.file.write("\033[A\033[2K")   # overwrite the "label…" row
            _real_console.file.flush()
        emit()


# ── phases: the run's spine ──────────────────────────────────────────────────
_phases = []        # [{"title": str, "status": "pending|active|done|failed"}]


def register_phases(titles):
    """Declare the run's phases once. A FIXED count, so `k/N` never drifts with
    flags — the user can trust the denominator."""
    global _phases
    _phases = [{"title": t, "status": "pending"} for t in titles]


_PHASE_MARKERS = {
    "done": "[success]✓[/success]",
    "failed": "[error]✗[/error]",
    "skipped": "[muted]⊘[/muted]",
}


def stepper_line():
    """`✓ Checks ── ⊘ Configure ── ◉ Build ── ○ Launch` — done / skipped / current /
    pending, where colour carries the progress (green fills in left to right).

    A skipped phase is `⊘`, deliberately NOT the pending `○`: the two mean opposite
    things (decided against vs. not reached yet) and a run that renders them alike
    reads as if it stalled.
    """
    parts = []
    for i, p in enumerate(_phases):
        if p["status"] == "done":
            parts.append(f"[success]✓ {p['title']}[/success]")
        elif p["status"] == "active":
            parts.append(f"[accent.bold]◉ {p['title']}[/accent.bold]")
        elif p["status"] == "failed":
            parts.append(f"[error]✗ {p['title']}[/error]")
        elif p["status"] == "skipped":
            parts.append(f"[muted]⊘ {p['title']}[/muted]")
        else:
            parts.append(f"[dim]○ {p['title']}[/dim]")
        if i < len(_phases) - 1:
            parts.append("[success] ── [/success]" if p["status"] == "done" else "[dim] ── [/dim]")
    line = Text.from_markup("".join(parts))
    line.no_wrap, line.overflow = True, "crop"
    return line


def rule_width():
    """How wide every rule and panel is. Capped, not full-width: at 160 columns a
    terminal-wide rule is a slab that outweighs the content under it, and the panels are
    already pinned to PANEL_WIDTH — furniture that disagrees on width reads as clutter."""
    return min(shutil.get_terminal_size((80, 24)).columns, PANEL_WIDTH)


def rule_line():
    return "[muted]" + "─" * rule_width() + "[/muted]"


def header_rows():
    """The pinned header, one markup string per terminal row.

    Row 1 is the stepper; row 2 is a full-width rule marking the boundary between the fixed
    header and the scrolling body — a rule that stopped short of the edge read as content
    rather than as a frame.
    """
    with _real_console.capture() as cap:
        _real_console.print(stepper_line(), end="", crop=True)
    return [cap.get(), rule_line()]


def pin_height():
    """Rows the pinned header occupies: the stepper plus its rule."""
    return 2


class _PinnedStepper:
    """Hold the stepper on row 1 while the phase bodies scroll underneath it.

    DECSTBM (`ESC [ top;bottom r`) narrows the terminal's scroll region to the rows below
    the header. Ordinary printing then cannot reach row 1, so the stepper never has to be
    reprinted between phases — `repaint()` redraws it in place and the body keeps scrolling
    past it.

    The hard rule: the region MUST be released on every exit path. A process that dies
    inside one leaves the user's shell scrolling in a box until they type `reset`. Hence
    atexit, a SIGTERM handler, and an explicit `release()` before the terminal is handed to
    a foreground child — which would otherwise print into our box under a stale header.

    TTY only, and off under -v (a verbose transcript is meant to be piped into a file, and
    a scroll region mangles that) or when TT_MODEL_NO_PIN is set.
    """

    def __init__(self):
        self._on = False
        self._prev_term = None

    def active(self):
        return self._on

    def engage(self):
        if self._on or not _isatty() or _VERBOSE or os.environ.get("TT_MODEL_NO_PIN"):
            return
        rows = shutil.get_terminal_size((80, 24)).lines
        height = pin_height()
        if rows < height + 4:           # too short to give up the header and still read
            return
        self._on = True
        with _lock:
            f = _real_console.file
            # Start from a blank canvas, because DECSTBM homes the cursor: fencing the
            # region and then jumping to its top row would land the cursor on whatever is
            # already displayed there and overprint it line by line. Scrolling the screen
            # up first is the non-destructive way to get that canvas — everything on it
            # moves into scrollback rather than being erased.
            f.write("\n" * rows)
            f.write(f"\033[{height + 1};{rows - 1}r")
            f.write(f"\033[{height + 1};1H")
            f.flush()
        atexit.register(self.release)
        with contextlib.suppress(ValueError, OSError):   # no-op off the main thread
            self._prev_term = signal.signal(signal.SIGTERM, self._on_term)
        with contextlib.suppress(AttributeError, ValueError, OSError):
            signal.signal(signal.SIGWINCH, self._on_resize)

    def repaint(self):
        """Redraw the header without disturbing the body's cursor."""
        if not self._on:
            return
        with _lock:
            f = _real_console.file
            out = ["\0337"]                  # DECSC: the body's cursor is restored below
            for i, row in enumerate(header_rows(), 1):
                if "[" in row and not row.startswith("\033"):
                    with _real_console.capture() as cap:
                        _real_console.print(Text.from_markup(row), end="", crop=True)
                    row = cap.get()
                out.append(f"\033[{i};1H\033[2K" + row)
            out.append("\0338")
            f.write("".join(out))
            f.flush()

    def release(self):
        """Give the whole screen back. Safe to call twice."""
        if not self._on:
            return
        self._on = False
        with _lock:
            f = _real_console.file
            # Resetting the region homes the cursor, exactly as setting it does. Without
            # the DECSC/DECRC pair the shell's next prompt lands on row 1-2 — on top of the
            # run's own output — instead of below it.
            f.write("\0337\033[r\0338\n")
            f.flush()
        with contextlib.suppress(ValueError, OSError):
            if self._prev_term is not None:
                signal.signal(signal.SIGTERM, self._prev_term)

    def _on_resize(self, *_):
        """Re-fence after a resize: the old region is in the old geometry's rows."""
        if not self._on:
            return
        rows = shutil.get_terminal_size((80, 24)).lines
        height = pin_height()
        if rows < height + 4:
            self.release()
            return
        with _lock:
            _real_console.file.write(f"\0337\033[{height + 1};{rows - 1}r\0338")
            _real_console.file.flush()
        self.repaint()

    def _on_term(self, signum, frame):
        self.release()
        if callable(self._prev_term):
            self._prev_term(signum, frame)
        else:
            raise SystemExit(128 + signum)


pinned = _PinnedStepper()


def pin_stepper():
    """Opt in to the pinned header for this run (no-op when it cannot apply).

    Call this BEFORE printing anything. It claims the screen, so output produced earlier
    in the run would be scrolled away — and output produced *between* the first print and
    this call gets overprinted by whatever lands in the region next.
    """
    pinned.engage()
    pinned.repaint()


def show_stepper():
    """Put the stepper in front of the user: repainted in place when we own the header,
    printed inline otherwise — so a piped log still records where the run had reached."""
    if pinned.active():
        pinned.repaint()
    else:
        console.print(stepper_line())


@contextlib.contextmanager
def phase(title):
    """Bracket one phase: print the stepper, label the body with a rule, and
    collapse to `✓ Phase k/N · Title  1.2s` (or ✗ on an exception).

    Upgrade path: TT-Studio pins the stepper to row 1 with a DECSTBM scroll
    region (`\\033[3;{rows-1}r`) so it stays visible while the body scrolls
    underneath. Start with this simpler inline version; add the region only once
    the flow is settled, and always reset it (`\\033[r`) on every exit path.
    """
    global _IN_PHASE
    entry = next((p for p in _phases if p["title"] == title), None)
    if entry is None:
        _phases.append({"title": title, "status": "pending"})
        entry = _phases[-1]
    entry["status"] = "active"
    index, total = _phases.index(entry) + 1, len(_phases)
    start = time.monotonic()

    show_stepper()
    console.print(Rule(f"[bold accent]{title}[/bold accent]", align="left",
                       style="muted", characters="─"), width=rule_width())
    _IN_PHASE = True
    try:
        yield entry
    except BaseException:
        entry["status"] = "failed"
        raise
    else:
        # Only an untouched phase becomes "done" — a body that called mark_skipped()
        # has already recorded the truth, and overwriting it would claim work that
        # never happened.
        if entry["status"] == "active":
            entry["status"] = "done"
    finally:
        _IN_PHASE = False
        marker = _PHASE_MARKERS.get(entry["status"], "[success]✓[/success]")
        # A skipped phase reports WHY, not how long: its duration measures the decision,
        # not the work, and printing `0.0s` next to it invites the reader to trust it.
        elapsed = time.monotonic() - start
        if entry["status"] == "skipped":
            tail = entry.get("why") or "skipped"
        else:
            # A phase that only rendered a result someone else computed clocks in at 0ms,
            # and "0ms" next to a check invites the reader to doubt it ran at all. Below the
            # threshold, say nothing rather than something meaningless.
            tail = fmt_duration(elapsed) if elapsed >= 0.01 else ""
        suffix = f"  [muted]{tail}[/muted]" if tail else ""
        console.print(f"{marker} [muted]Phase {index}/{total} ·[/muted] "
                      f"[bold accent]{title}[/bold accent]{suffix}")
        if pinned.active():
            pinned.repaint()


def mark_skipped(entry, why=""):
    """Mark the RUNNING phase as skipped rather than done (call with `phase()`'s entry).

    For a phase that was entered, found nothing to do, and should not claim a ✓.
    """
    entry["status"] = "skipped"
    entry["why"] = why


def skip_phase(title, why=""):
    """Record a phase as skipped WITHOUT entering it, and still print its line.

    For a phase decided against before it starts (a flag turned it off, a prerequisite
    made it moot). It prints rather than silently advancing because a step the user
    never sees is indistinguishable from one that failed quietly — and it keeps k/N
    honest, since the denominator already counted this phase.
    """
    entry = next((p for p in _phases if p["title"] == title), None)
    if entry is None:
        _phases.append({"title": title, "status": "pending"})
        entry = _phases[-1]
    entry["status"], entry["why"] = "skipped", why
    index, total = _phases.index(entry) + 1, len(_phases)
    console.print(f"{_PHASE_MARKERS['skipped']} [muted]Phase {index}/{total} ·[/muted] "
                  f"[bold accent]{title}[/bold accent]  [muted]{why or 'skipped'}[/muted]")
    if pinned.active():
        pinned.repaint()


def _render_lines(renderable, width):
    """Render to ANSI text at a fixed width, matching the live console's colour support.

    A private Console with a StringIO file, so this works inside step()'s capture without
    fighting it for sys.stdout.
    """
    buf = io.StringIO()
    tmp = Console(theme=THEME, highlight=False, soft_wrap=False, width=width, file=buf,
                  force_terminal=True if console.is_terminal else None,
                  # Inherit no_color, don't infer it: NO_COLOR on a real tty leaves
                  # is_terminal True, so inferring would re-add the styling Rich stripped.
                  no_color=console.no_color or not console.is_terminal)
    tmp.print(renderable)
    return buf.getvalue().rstrip("\n").split("\n")


def _body_print(renderable):
    """Print into the phase body without landing on the live activity row, and without
    trailing whitespace.

    Two things, both easy to get wrong separately:

    1. The activity ticker leaves the cursor mid-row (no trailing newline), so any body
       print must erase that row first; the ticker repaints it on its next tick.
       Everything shares `_lock`, so the two writers can't interleave mid-escape-sequence.
    2. `Padding` fills each line to the render width — which is what keeps a wrapped cell's
       indent, but also meant every row trailed spaces out to the terminal edge. At 160
       columns that is ~80 trailing spaces per line in the user's copy buffer. So render at
       a fixed width, then rstrip each line: the indent survives, the padding does not.
    """
    lines = _render_lines(renderable, rule_width())
    with _lock:
        if activity.running() and _isatty():
            _real_console.file.write("\r\033[2K")
            _real_console.file.flush()
        # console.file, not _real_console: inside a captured step() this must stay captured.
        console.file.write("\n".join(line.rstrip() for line in lines) + "\n")
        console.file.flush()


def note(text, marker="○", style="muted"):
    """A short note in the phase body's gutter — why something was skipped, what
    happens instead. Padded (not string-indented) so a wrapped line keeps the
    indent, and rendered as Text so tool-derived content can't trip markup."""
    prefix = f"{marker} " if marker else "  "
    _body_print(Padding(Text(f"{prefix}{text}", style=style), (0, 0, 0, 2)))


def milestone(text, style="success", marker="✓"):
    """One real milestone inside a phase body (`  ✓ chroma pulled`)."""
    _body_print(Padding(Text(f"{marker} {text}", style=style), (0, 0, 0, 2)))


# ── the pinned activity row ──────────────────────────────────────────────────
class _Activity:
    """One in-place line at the bottom: `⠹ <label>`. Proof of life during a long
    silent step, updated by a background ticker so it spins even when the tool
    prints nothing for minutes. TTY only; a no-op when piped."""

    def __init__(self):
        self.label = ""
        self._stop = None
        self._ticker = None
        self._frame = 0

    def start(self, label=""):
        self.label = label
        if not _isatty() or _VERBOSE or self._ticker is not None:
            return
        self._stop = threading.Event()
        self._ticker = threading.Thread(target=self._loop, daemon=True)
        self._ticker.start()

    def set(self, label):
        self.label = label or ""

    def running(self):
        return self._ticker is not None

    def _loop(self):
        f = _real_console.file
        while not self._stop.is_set():
            glyph = SPINNER_FRAMES[self._frame % len(SPINNER_FRAMES)]
            text = Text()
            text.append(f"{glyph} ", style="accent")
            text.append(self.label, style="muted")
            text.no_wrap, text.overflow = True, "crop"
            with _lock:
                with _real_console.capture() as cap:
                    _real_console.print(text, end="", crop=True)
                f.write("\r\033[2K" + cap.get())
                f.flush()
            self._frame += 1
            self._stop.wait(0.1)

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._ticker is not None and self._ticker is not threading.current_thread():
            self._ticker.join(timeout=0.5)
        self._ticker = self._stop = None
        if _isatty():
            with _lock:
                _real_console.file.write("\r\033[2K")
                _real_console.file.flush()


activity = _Activity()


def run_with_activity(cmd, cwd=None, env=None, label="Working", parse=None):
    """Stream a subprocess, keeping ONE live line instead of its output.

    `parse(line)` is your pure aggregator: return a string to update the activity
    label, a ("milestone", text) tuple to emit a ✓ line, or None to ignore. Every
    line is still collected and returned, so a failure can be diagnosed from the
    full output. Raw lines reach the terminal only under --verbose.

    Returns (returncode, full_output).
    """
    process = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, universal_newlines=True,
    )
    lines = []
    activity.start(label)
    try:
        for line in process.stdout:
            lines.append(line)
            if _VERBOSE:
                _body_print(Text(f"  {line.rstrip()}", style="dim"))
            result = parse(line) if parse else None
            if isinstance(result, tuple) and result and result[0] == "milestone":
                milestone(result[1])
            elif isinstance(result, str) and result:
                activity.set(result)
    finally:
        activity.stop()
    process.wait()
    return process.returncode, "".join(lines)


# ── cards ────────────────────────────────────────────────────────────────────
def notice_panel(title, lines, border_style="warning"):
    """Content-sized callout for warnings, errors, and diagnosis cards."""
    body = Text()
    for i, line in enumerate(lines):
        if i:
            body.append("\n")
        body.append_text(Text.from_markup(line) if isinstance(line, str) else line)
    return Panel(body, title=title, title_align="left", box=ROUNDED,
                 border_style=border_style, padding=(1, 2), expand=False)


def ready_panel(title, rows, footer_lines=()):
    """The end-of-run summary: endpoints, mode, and the hints that make the next
    action discoverable (stop / logs / info). Make it re-viewable behind a flag
    (`--info`) by extracting the renderer and probing live state — never by
    duplicating the assembly."""
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(style="muted", no_wrap=True)
    table.add_column()
    for row in rows:
        label, value = row[0], row[1]
        status = f"  [muted]{row[2]}[/muted]" if len(row) > 2 else ""
        table.add_row(label, f"{value}{status}")
    body = [table]
    if footer_lines:
        body.append(Text())
        for line in footer_lines:
            body.append(Text.from_markup(line))
    group = Table.grid()
    group.add_column()
    for item in body:
        group.add_row(item)
    return Panel(group, title=f"[bold accent]{title}[/bold accent]", title_align="left",
                 box=ROUNDED, border_style="accent", padding=(1, 2), width=PANEL_WIDTH)


def failure_card(name, diagnosis, log_file=None, consequence=None):
    """Render a diagnosis dict — {cause, detail, evidence, actions} — as the
    standard failure card. Build the dict in a pure function so the classification
    is unit-testable; see reference/patterns.md."""
    lines = [f"[error]{diagnosis['detail']}[/error]"]
    if diagnosis.get("evidence"):
        lines.append(f"[muted]Log · {diagnosis['evidence'][:120]}[/muted]")
    if consequence:
        lines += ["", f"[warning]{consequence}[/warning]"]
    lines += ["", "[info]Try:[/info]"]
    lines += [f"[muted]  {action}[/muted]" for action in diagnosis.get("actions", ())]
    if log_file and not any(log_file in a for a in diagnosis.get("actions", ())):
        lines.append(f"[muted]  tail -50 {log_file}[/muted]")
    return notice_panel(f"[error]{name} — {diagnosis['cause']}[/error]", lines,
                        border_style="error")


def terminal_width():
    return shutil.get_terminal_size(fallback=(80, 24)).columns


# ── the machine-readable carve-out ───────────────────────────────────────────
def raw(text=""):
    """Print text that a machine will consume, bypassing Rich entirely.

    `serve --print` emits a shell-pasteable `KEY=V ... argv` line and `search --json`
    / `info` emit JSON; users pipe both straight into a shell or a parser. The Rich
    console wraps at terminal width and interprets `[...]` as markup, either of which
    silently corrupts them — a wrapped command line is not runnable and wrapped JSON is
    not parseable. So these go to stdout untouched, at any COLUMNS.

    Never use this for human-facing status; that belongs to step()/note()/cards.
    """
    print(text)


def set_no_color(value=True):
    """Strip colour/styling from both consoles (honours --no-color).

    Rebinds rather than mutating: a Console's colour system is fixed at construction, so
    the only reliable way to turn styling off after import is to build new ones.
    """
    global console, _real_console
    if not value:
        return
    console = Console(theme=THEME, highlight=False, soft_wrap=False, no_color=True)
    _real_console = Console(theme=THEME, file=sys.__stdout__, highlight=False, no_color=True)


def check_table():
    """A borderless table for check/verdict rows (`doctor`, install's Verify, `start`).

    Rich owns the column widths here rather than f-string padding: a version like
    `0.1.dev14190+g24516a94b.empty` is 29 characters and blew a hand-padded layout onto a
    wrapped line, which is exactly the unreadable run-on this replaces.

    Rows come in through `check_row()`, which puts the verdict glyph in the SAME cell as
    the label it judges. As its own column the glyph sat a full column-gap from its label
    and read as floating — and collapsing the padding to fix that flattened the gap between
    every other column too, so the rows read as sentences instead of a table.
    """
    table = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2),
                  expand=False, collapse_padding=True)
    # Column priority matters on a narrow terminal. Rich shrinks columns to fit, and left
    # to itself it squeezed the verdict glyph away entirely at COLUMNS=40 — losing the one
    # character the row exists to convey. So the verdict+name column is fixed, and the
    # requirement column is the only one allowed to give: it folds, then truncates.
    table.add_column(no_wrap=True, min_width=6)                        # ✓/✗/! + component
    table.add_column(no_wrap=True, overflow="ellipsis", max_width=18)  # what we found
    table.add_column(style="muted", overflow="fold", max_width=48)     # requirement
    return table


def check_row(table, mark, name, found="", need=""):
    """Add one verdict row. `mark` is the glyph markup (or "" for an unjudged fact)."""
    table.add_row(f"{mark} {name}" if mark else name, found, need)


def print_table(table, indent=2):
    """Print a check table indented as a block, wrapped cells included.

    Indent via Padding, never by prefixing spaces onto the strings: a hand-indented cell
    loses its indent the moment Rich wraps it, which is how a "requires numpy>=2" note
    ended up starting at column 0 on its second line.

    Width-capped like every other piece of furniture, so a wide terminal does not stretch
    the rows (and their trailing whitespace) across the screen.
    """
    # Through _body_print, not console.print: that is where the width cap and the
    # trailing-whitespace rstrip live, and a table is the widest thing in a phase body.
    _body_print(Padding(table, (0, 0, 0, indent)))


def hint(text, indent=4):
    """A muted continuation line under a row or table — no marker, indent preserved on wrap."""
    _body_print(Padding(Text(text, style="muted"), (0, 0, 0, indent)))


def fmt_version(version, keep_local=False):
    """A version trimmed for a status row.

    PEP 440 puts build metadata after a "+": `0.1.dev14190+g24516a94b.empty` is 29
    characters of which the last 17 are a git hash and a build-target tag. That is
    provenance, not the answer to "which version is installed", and at COLUMNS=40 it
    crowded the verdict glyph off the row. `doctor --verbose` and `info` still show the
    full string; pass keep_local=True where provenance is the point.
    """
    if not version:
        return "not found"
    if keep_local or "+" not in version:
        return version
    return version.split("+", 1)[0]


def steps_panel_lines(title, phases):
    """The upfront roadmap: what this run will do, before it starts doing it.

    Numbers are right-aligned accent so the list reads as a sequence rather than a menu.
    Shown once, so a long install is a known quantity from the first second.
    """
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(style="muted", overflow="fold")
    for i, (name, detail) in enumerate(phases, 1):
        grid.add_row(f"[accent.bold]{i}[/accent.bold]", f"[bold]{name}[/bold]", detail)
    return Panel(grid, title=f"[bold accent]{title} · {len(phases)} steps[/bold accent]",
                 title_align="left", box=ROUNDED, border_style="accent",
                 padding=(1, 2), expand=False)


def secret(prompt_text):
    """Read a secret without echoing it.

    getpass, not input(): a token pasted into a terminal that echoes ends up in scrollback
    and shell history. Never call this inside a capturing step() — the prompt would be
    swallowed and the CLI would look like it had hung.
    """
    import getpass

    if activity.running():
        activity.stop()
    try:
        return getpass.getpass(prompt_text)
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""


def stepper_line_for(title):
    """The stepper with ``title`` marked active, without opening a phase context.

    For the last step of a run that hands the terminal to a long-lived foreground child:
    the user still needs to see where they are, but a phase whose spinner is alive when the
    child starts printing would fight it for the row — and a "✓ Phase 4/4" line printed
    when the child exits hours later is noise, not information.
    """
    for p in _phases:
        if p["status"] == "active":
            p["status"] = "done"
    for p in _phases:
        if p["title"] == title:
            p["status"] = "active"
            break
    return stepper_line()


def _read_choice(prompt_text, count, default=1):
    """The read loop behind every menu: a number, empty for the default, or q."""
    while True:
        try:
            raw = input(f"{prompt_text} [1-{count}, or q to quit] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return None
        if raw == "":
            return default - 1
        if raw.lower() in ("q", "quit", "n", "no"):
            return None
        if raw.isdigit() and 1 <= int(raw) <= count:
            return int(raw) - 1
        console.print(f"[muted]Enter a number from 1 to {count}, or q.[/muted]")


def choose(prompt_text, options, default=1):
    """A numbered menu. Returns the chosen index (0-based), or None if declined.

    A plain numbered list read with input(), not a cursor-driven picker: it degrades to
    readable text on a dumb terminal, works over ssh, and needs no extra dependency. The
    caller is responsible for only calling this when stdin is interactive — a menu on a
    piped stdin reads EOF and silently takes the default.
    """
    if activity.running():
        activity.stop()
    for i, label in enumerate(options, 1):
        console.print(f"  [accent.bold]{i}[/accent.bold]  {label}")
    console.print()
    return _read_choice(prompt_text, len(options), default)


def choose_rows(prompt_text, rows, default=1):
    """`choose()` over COLUMNS instead of pre-joined strings.

    Each row is `(marker, name, meta, note)`. Columns are why this exists: glued into one
    string, a long name pushes every following field out of alignment, and a blocked entry's
    reason ends up trailing a parenthesis where it reads as part of the name. Given its own
    column, the reason lines up under the others and the eye can skip it.

    The marker column is reserved even when every row is fine, so adding one ✗ row does not
    re-indent the whole menu.
    """
    if activity.running():
        activity.stop()
    grid = Table.grid(padding=(0, 2))
    grid.add_column(justify="right", no_wrap=True)                  # number
    grid.add_column(no_wrap=True, width=1)                          # ✓/✗ marker
    grid.add_column(no_wrap=True, style="bold")                     # name
    grid.add_column(no_wrap=True, style="muted")                    # what it is
    grid.add_column(overflow="fold", style="warning")               # why it cannot run
    for i, (marker, name, meta, note) in enumerate(rows, 1):
        grid.add_row(f"[accent.bold]{i}[/accent.bold]",
                     "[error]✗[/error]" if marker else "",
                     Text(name), Text(meta or ""), Text(note or ""))
    console.print(Padding(grid, (0, 0, 0, 2)))
    console.print()
    return _read_choice(prompt_text, len(rows), default)
