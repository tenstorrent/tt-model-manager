# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Turn a served container's boot log into a handful of named steps.

Pure: text in, events out. Nothing here touches the terminal, docker, or the clock, so
the whole matrix is unit-testable against captured logs (``tests/fixtures/boot_logs``).

A vLLM boot on a Tenstorrent card prints a few thousand lines in 1-10 minutes. The user
needs about six of them: engine up, device open, weights loaded, KV cache configured,
warmup done, server listening. :class:`BootTracker` reads the stream and reports when
each of those starts and finishes, plus any real progress the log gives away (tqdm
fragments such as ``12/32``). It never invents a percentage: a phase with no total is
reported without one.

The patterns come from real boots (a ``vllm-fork`` Llama-3.1-8B boot that reached ready
and a ``vllm-plugin`` Qwen3 boot that crashed on a held device). Log order is the truth:
entering a later phase finishes the current one, so a phase never gets stuck "active"
because the tool changed the wording of its completion line.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

Event = Tuple  # ("start", key, label) | ("done", key, label, detail) |
#                ("progress", done, total) | ("detail", text) | ("ready",)


def _rx(*patterns: str) -> Tuple["re.Pattern[str]", ...]:
    return tuple(re.compile(p) for p in patterns)


@dataclass(frozen=True)
class Phase:
    key: str
    label: str        # while active: "opening Tenstorrent device"
    done_label: str   # once finished: "Tenstorrent device opened"
    start: Tuple["re.Pattern[str]", ...]
    done: Tuple["re.Pattern[str]", ...] = ()
    #: line -> a short detail for the row ("2 chips · mesh (1, 2)"), or None
    detail: Optional[Callable[[str], Optional[str]]] = None


#: tqdm's textual bar: ` 12%|█▎        | 4/32 [00:12<01:22,  2.96s/it]`. Only the counts
#: are trusted; the percentage is re-derived so a rounding quirk cannot show 100% at 31/32.
TQDM_RE = re.compile(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)")

#: side-channel chatter from the tt-inference-server wrapper; says nothing about the boot
IGNORE_RE = _rx(r"utils\.prompt_client", r"utils\.cache_monitor")

#: lines that name a boot-failure cause. Kept aside from the rolling tail: a crash in the
#: engine core is followed by ~250 lines of traceback from BOTH processes, so by the time
#: the container exits the line that says WHY has long scrolled out of any sane tail.
CAUSE_RE = _rx(r"Sysmem mapped at unexpected NOC address", r"CHIP_IN_USE",
               r"stale process holding", r"unrecognized arguments", r"error: argument",
               r"no such option", r"Address already in use", r"EngineCore failed to start",
               r"Out of memory|OutOfMemory|OOM")


def _detail_engine(line: str) -> Optional[str]:
    # tt-metal's UMD logs "firmware bundle version: 19.13.1" in the same window; only a
    # vLLM-shaped line may name the engine version.
    if "| UMD |" in line or "firmware" in line:
        return None
    m = re.search(r"(?:vLLM server version|LLM engine \(v|\]\s+.*\bversion)\s*v?(\d+\.\d+[\w.+-]*)",
                  line)
    return f"vLLM {m.group(1)}" if m else None


def _detail_device(line: str) -> Optional[str]:
    m = re.search(r"multidevice with (\d+) devices? and grid \(([^)]*)\)", line)
    if m:
        return f"{m.group(1)} chips · mesh ({m.group(2)})"
    m = re.search(r"Fabric initialized on (\d+) devices", line)
    if m:
        return f"{m.group(1)} chips"
    m = re.search(r"Opening local chip ids/PCIe ids: \{([^}]*)\}", line)
    if m:
        n = len([c for c in m.group(1).split(",") if c.strip()])
        return f"{n} chip{'s' if n != 1 else ''}"
    return None


def _detail_kv(line: str) -> Optional[str]:
    m = re.search(r"KV cache size: ([\d,]+) tokens", line)
    return f"{m.group(1)} tokens" if m else None


def _detail_warmup(line: str) -> Optional[str]:
    m = re.search(r"init engine .* took ([\d.]+) seconds", line)
    return f"{float(m.group(1)):.0f}s of warmup" if m else None


#: the stock vLLM boot (``vllm-plugin``) and the tt-inference-server runner around it
#: (``vllm-fork``) print the same tt-metal / vLLM lines; only the ready line differs.
VLLM_PHASES: Tuple[Phase, ...] = (
    Phase("engine", "starting the engine", "engine initialised",
          start=_rx(r"Available plugins for group vllm\.platform_plugins",
                    r"Platform plugin tt is activated",
                    r"vLLM server version", r"Initializing a V1 LLM engine"),
          detail=_detail_engine),
    Phase("device", "opening Tenstorrent device", "Tenstorrent device opened",
          start=_rx(r"Opening user mode device driver",
                    r"Attempting to open mesh device",
                    r"Starting devices in cluster"),
          done=_rx(r"multidevice with \d+ devices? and grid .* is created"),
          detail=_detail_device),
    Phase("weights", "loading weights", "weights loaded",
          start=_rx(r"Checkpoint directory:", r"Loading checkpoint shards",
                    r"Fetching \d+ files", r"Loading safetensors", r"Loading weights")),
    Phase("kv", "configuring KV cache", "KV cache configured",
          start=_rx(r"KV cache size", r"Allocating TT kv caches", r"num_gpu_blocks"),
          detail=_detail_kv),
    Phase("warmup", "warming up the model", "model warmed up",
          start=_rx(r"Warming up prefill", r"Starting decode warmup",
                    r"Done Compiling Model", r"Capturing .*Trace"),
          done=_rx(r"init engine .* took [\d.]+ seconds"),
          detail=_detail_warmup),
    Phase("server", "starting API server", "API server ready",
          start=_rx(r"Starting vLLM API server", r"Warming up chat template",
                    r"Uvicorn running on", r"Started server process")),
)

#: a uvicorn app around a tt-metal diffusion pipeline; far fewer landmarks, all optional
TT_DIT_PHASES: Tuple[Phase, ...] = (
    Phase("server", "starting the server", "server process started",
          start=_rx(r"Started server process", r"Waiting for application startup")),
    Phase("device", "opening Tenstorrent device", "Tenstorrent device opened",
          start=_rx(r"Opening user mode device driver", r"Starting devices in cluster"),
          done=_rx(r"Fabric initialized on \d+ devices"),
          detail=_detail_device),
    Phase("weights", "loading weights", "weights loaded",
          start=_rx(r"Fetching \d+ files", r"Loading pipeline", r"safetensors",
                    r"Loading weights", r"Loading checkpoint")),
    Phase("warmup", "warming up the pipeline", "pipeline warmed up",
          start=_rx(r"[Ww]arm(?:ing)? ?up", r"Capturing trace", r"Compiling")),
)


class BootTracker:
    """Feed it log lines; it tells you which step the boot is on.

    ``feed`` returns the events one line produced (usually none). ``current`` is the
    active phase or None; ``tail`` is the last lines seen, for a failure card.
    """

    def __init__(self, phases: Sequence[Phase], ready_probe: str, tail_lines: int = 40):
        self.phases: List[Phase] = list(phases)
        self.ready_probe = ready_probe
        self.tail: Deque[str] = deque(maxlen=tail_lines)
        self.notable: List[str] = []   # cause-naming lines, in order (see CAUSE_RE)
        self._index = -1            # index into phases of the current (or last) phase
        self._done = True           # is the current phase finished?
        self._progress: Dict[str, Tuple[int, int]] = {}
        self._detail: Dict[str, str] = {}
        self.ready = False

    # -- queries --------------------------------------------------------------------

    @property
    def current(self) -> Optional[Phase]:
        if self._index < 0 or self._done:
            return None
        return self.phases[self._index]

    def evidence(self) -> List[str]:
        """What a diagnosis should read: the cause-naming lines, then the tail."""
        return [*self.notable, *self.tail]

    def detail_for(self, key: str) -> Optional[str]:
        """The best detail we have for a phase: an extracted fact, else its final count."""
        if key in self._detail:
            return self._detail[key]
        if key in self._progress:
            done, total = self._progress[key]
            return f"{done}/{total}"
        return None

    # -- feeding --------------------------------------------------------------------

    def feed(self, line: str) -> List[Event]:
        line = line.rstrip("\r\n")
        if not line.strip():
            return []
        self.tail.append(line)
        if len(self.notable) < 50 and any(rx.search(line) for rx in CAUSE_RE):
            self.notable.append(line)
        events: List[Event] = []

        if self.ready_probe and self.ready_probe in line:
            events += self._finish()
            self.ready = True
            events.append(("ready",))
            return events

        if any(rx.search(line) for rx in IGNORE_RE):
            return []

        # A later phase announcing itself finishes the current one, whatever it said.
        for i in range(self._index + 1, len(self.phases)):
            phase = self.phases[i]
            if any(rx.search(line) for rx in phase.start):
                events += self._finish()
                self._index, self._done = i, False
                events.append(("start", phase.key, phase.label))
                events += self._extract(phase, line)
                return events

        phase = self.current
        if phase is None:
            return events
        events += self._extract(phase, line)
        if any(rx.search(line) for rx in phase.done):
            events += self._finish()
        return events

    def _extract(self, phase: Phase, line: str) -> List[Event]:
        events: List[Event] = []
        m = TQDM_RE.search(line)
        if m:
            done, total = int(m.group(2)), int(m.group(3))
            if total > 0:
                self._progress[phase.key] = (done, total)
                events.append(("progress", done, total))
        if phase.detail is not None:
            text = phase.detail(line)
            if text:
                self._detail[phase.key] = text
                events.append(("detail", text))
        return events

    def _finish(self) -> List[Event]:
        if self._index < 0 or self._done:
            return []
        self._done = True
        phase = self.phases[self._index]
        return [("done", phase.key, phase.done_label, self.detail_for(phase.key))]


# ------------------------------------------------------------------------ diagnosis

_LOG_PREFIX_RE = re.compile(
    r"^(?:\(\w+(?:_\w+)* pid=\d+\)\s*)?"                            # (EngineCore pid=60)
    r"(?:(?:ERROR|INFO|WARNING|DEBUG)\s+\d\d-\d\d \d\d:\d\d:\d\d\s+\[?[\w./:]+\]?\s*)?"
    r"(?:\d{4}-\d\d-\d\d[ T][\d:.,]+\s*\|\s*\w+\s*\|\s*\w+\s*\|\s*)?"  # tt-metal logger
)

_EVIDENCE_RE = re.compile(r"RuntimeError|Error:|error:|FAILED|failed to|Traceback")


def _clean(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line).strip()


def _evidence(tail: Sequence[str], pattern: Optional[str] = None) -> str:
    """The one line worth quoting: the one matching ``pattern`` if given, else the last
    error-shaped line, else the last line."""
    cleaned = [_clean(ln) for ln in tail]
    cleaned = [ln for ln in cleaned if ln and not ln.startswith(("^", "File \"", "return ",
                                                                  "self.", "raise "))]
    if pattern:
        for ln in cleaned:
            if re.search(pattern, ln):
                return ln
    for ln in reversed(cleaned):
        if _EVIDENCE_RE.search(ln) and "Traceback" not in ln:
            return ln
    return cleaned[-1] if cleaned else ""


def diagnose_boot(tail: Sequence[str], *, exited: bool, target: str,
                  extra_args: Optional[Sequence[str]] = None,
                  timeout_s: int = 1800) -> dict:
    """Classify why a boot did not reach ready — text in, dict out.

    ``cause`` is the card title; ``detail`` one sentence of explanation; ``evidence`` one
    quoted log line; ``actions`` what to try. Never a dump of the last N lines: a container
    that CRASHED and one that is merely slow need different next steps, and the most common
    crash here is an argument tt-model forwarded to the engine.
    """
    joined = "\n".join(tail)
    follow = f"tt-model logs {target} -f"
    stop = f"tt-model stop {target}"

    if not exited:
        return {
            "cause": f"the server did not report ready within {timeout_s // 60} min",
            "detail": "The container is still running. A cold boot JIT-compiles kernels, "
                      "which can take ~10 min the first time; it may simply be slow.",
            "evidence": _evidence(tail),
            "actions": [f"keep watching:  {follow}", f"give up:        {stop}"],
        }

    bad = [a for a in (extra_args or []) if a.startswith("-")]
    flag_rx = r"unrecognized arguments|error: argument|no such option"
    if bad and re.search(flag_rx, joined):
        return {
            "cause": "the container exited — the engine rejected a flag",
            "detail": f"{', '.join(bad)} was passed through to the engine, which rejected it. "
                      f"tt-model's own flags must come BEFORE the target "
                      f"(`tt-model serve --flag {target}`); anything after it goes to the "
                      f"engine verbatim. If the flag is not a tt-model flag either, drop it.",
            "evidence": _evidence(tail, flag_rx),
            "actions": [f"tt-model serve {' '.join(bad)} {target}"],
        }

    held_rx = r"Sysmem mapped at unexpected NOC address|CHIP_IN_USE|stale process holding"
    if re.search(held_rx, joined):
        return {
            "cause": "the container exited — another process is holding the Tenstorrent device",
            "detail": "The card is already open in another process (usually another served "
                      "model, or a stale one that did not release it).",
            "evidence": _evidence(tail, held_rx),
            "actions": ["docker ps            (another model server running?)",
                        "tt-model list        then `tt-model stop <that model>`",
                        "tt-smi -r            (reset the card if nothing else is using it)"],
        }

    if "Address already in use" in joined:
        m = re.search(r"port (\d+)", joined)
        port = m.group(1) if m else "<port>"
        return {
            "cause": f"the container exited — port {port} is already taken",
            "detail": "Another process was listening on the published port when the "
                      "server tried to bind.",
            "evidence": _evidence(tail, r"Address already in use"),
            "actions": [f"lsof -i :{port}", f"tt-model serve --port <other> {target}"],
        }

    return {
        "cause": "the container exited before the server was ready",
        "detail": "The engine stopped during boot. The reason is in what it printed on the "
                  "way out.",
        "evidence": _evidence(tail),
        "actions": [f"full log:  tt-model logs {target}"],
    }


def summarize(diag: dict, tail: Sequence[str] = ()) -> str:
    """The diagnosis as plain text, for an exception message or a non-TTY run."""
    lines = [diag["cause"], f"  {diag['detail']}"]
    if diag.get("evidence"):
        lines.append(f"  log · {diag['evidence']}")
    if diag.get("actions"):
        lines.append("  try:")
        lines += [f"    {a}" for a in diag["actions"]]
    if tail:
        lines.append("  last output:")
        lines += [f"    {ln}" for ln in list(tail)[-8:]]
    return "\n".join(lines)
