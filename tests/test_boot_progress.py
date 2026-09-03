# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The boot-log parser behind `serve`'s progress view, against REAL boots.

tests/fixtures/boot_logs holds two captured container logs (ANSI stripped, tqdm `\\r`
fragments split onto their own lines, exactly as wait_ready's text-mode pipe yields them):

- vllm_fork_llama31_8b.log — a tt-inference-server (vllm-fork style) Llama-3.1-8B boot on
  two chips that reached ready. That runner silences uvicorn, so the stock ready line is
  appended at the end (marked with a comment line).
- vllm_plugin_qwen3_device_held.log — a stock `vllm serve` boot that died in the engine core
  because another process held the card.
"""

from pathlib import Path

import pytest

from tt_kernel.boot_progress import (BootTracker, TQDM_RE, TT_DIT_PHASES, VLLM_PHASES,
                                     diagnose_boot, summarize)

FIXTURES = Path(__file__).parent / "fixtures" / "boot_logs"
READY = "Application startup complete"


def _replay(name, phases=VLLM_PHASES, probe=READY):
    t = BootTracker(phases, probe)
    events = []
    for line in (FIXTURES / name).read_text().splitlines():
        events += t.feed(line)
    return t, events


class TestLlamaBoot:
    def test_walks_every_landmark_in_order_and_ends_ready(self):
        t, events = _replay("vllm_fork_llama31_8b.log")
        starts = [e[1] for e in events if e[0] == "start"]
        assert starts == ["engine", "device", "weights", "kv", "warmup", "server"]
        dones = [e[1] for e in events if e[0] == "done"]
        assert dones == starts, "every started phase must be finished"
        assert events[-1] == ("ready",) and t.ready
        assert sum(1 for e in events if e == ("ready",)) == 1

    def test_details_are_extracted_from_the_lines_that_state_them(self):
        t, events = _replay("vllm_fork_llama31_8b.log")
        done = {e[1]: e[3] for e in events if e[0] == "done"}
        assert done["device"] == "2 chips · mesh (1, 2)"
        assert done["kv"] == "133,120 tokens"
        assert done["warmup"] == "86s of warmup"
        assert done["engine"].startswith("vLLM 0.1.dev"), done["engine"]

    def test_the_firmware_version_is_not_mistaken_for_the_engine_version(self):
        """UMD logs `firmware bundle version: 19.13.1` in the same window."""
        _, events = _replay("vllm_fork_llama31_8b.log")
        assert not any(e[0] == "detail" and "19.13" in e[1] for e in events)

    def test_tqdm_progress_attaches_to_the_phase_that_owns_it(self):
        _, events = _replay("vllm_fork_llama31_8b.log")
        phase = None
        seen = {}
        for e in events:
            if e[0] == "start":
                phase = e[1]
            elif e[0] == "progress":
                seen.setdefault(phase, []).append((e[1], e[2]))
        assert seen["weights"][-1] == (32, 32)      # per-layer conversion to device
        assert seen["kv"][-1] == (32, 32)           # `Allocating TT kv caches ... 32/32`
        assert "warmup" not in seen and "server" not in seen

    def test_a_phase_with_no_extracted_fact_reports_its_final_count(self):
        _, events = _replay("vllm_fork_llama31_8b.log")
        done = {e[1]: e[3] for e in events if e[0] == "done"}
        assert done["weights"] == "32/32"
        assert done["server"] is None               # nothing to say; say nothing

    def test_side_channel_chatter_produces_no_events(self):
        t = BootTracker(VLLM_PHASES, READY)
        t.feed("INFO 09-02 [__init__.py] Platform plugin tt is activated")
        assert t.feed("2026-09-02 17:34:43,613 - utils.prompt_client - INFO - 🔄 Tensor cache "
                      "generation in progress. Waited 30.0s, next check in 10.0s") == []
        assert t.feed("utils.cache_monitor - INFO - 🔍 No cache content found") == []


class TestQwenCrash:
    def test_stops_in_the_device_phase_and_never_reports_ready(self):
        t, events = _replay("vllm_plugin_qwen3_device_held.log")
        assert [e[1] for e in events if e[0] == "start"] == ["engine", "device"]
        assert not t.ready
        assert t.current is not None and t.current.key == "device"
        assert {e[1]: e[3] for e in events if e[0] == "done"}["engine"] == "vLLM 0.24.0"

    def test_the_cause_survives_250_lines_of_traceback(self):
        """The line that says WHY scrolls out of any sane tail; the tracker keeps it aside."""
        t, _ = _replay("vllm_plugin_qwen3_device_held.log")
        assert len(t.tail) == 40 and not any("CHIP_IN_USE" in ln for ln in t.tail)
        diag = diagnose_boot(t.evidence(), exited=True, target="org/x")
        assert "another process is holding the Tenstorrent device" in diag["cause"]
        assert "CHIP_IN_USE" in diag["evidence"]
        assert any("tt-smi -r" in a for a in diag["actions"])


class TestDiagnosis:
    def test_a_passthrough_flag_is_blamed_by_name(self):
        diag = diagnose_boot(["vllm: error: unrecognized arguments: --refresh"], exited=True,
                             target="org/x", extra_args=["--refresh"])
        assert "rejected a flag" in diag["cause"]
        assert "--refresh was passed through to the engine" in diag["detail"]
        assert "must come BEFORE the target" in diag["detail"]

    def test_a_taken_port_names_the_port(self):
        diag = diagnose_boot(["ERROR:    [Errno 98] error while attempting to bind on address "
                              "('0.0.0.0', 8000): address already in use",
                              "OSError: [Errno 98] Address already in use"],
                             exited=True, target="org/x")
        assert "port" in diag["cause"]
        assert any("lsof" in a for a in diag["actions"])

    def test_still_running_is_not_reported_as_a_crash(self):
        diag = diagnose_boot(["still booting"], exited=False, target="org/x", timeout_s=1800)
        assert "did not report ready within 30 min" in diag["cause"]
        assert "still running" in diag["detail"]
        assert any("tt-model stop org/x" in a for a in diag["actions"])

    def test_an_unknown_crash_quotes_the_last_error_line_not_the_traceback(self):
        tail = ["(EngineCore pid=60) Traceback (most recent call last):",
                '(EngineCore pid=60)   File "/x.py", line 1, in f',
                "(EngineCore pid=60) RuntimeError: boom: could not open device"]
        diag = diagnose_boot(tail, exited=True, target="org/x")
        assert "container exited" in diag["cause"]
        assert diag["evidence"] == "RuntimeError: boom: could not open device"

    def test_summary_carries_cause_evidence_actions_and_tail(self):
        diag = diagnose_boot(["boom"], exited=True, target="org/x")
        text = summarize(diag, ["boom"])
        assert "container exited" in text and "boom" in text and "try:" in text


class TestShapes:
    def test_tqdm_regex_reads_counts_not_the_rounded_percent(self):
        m = TQDM_RE.search(" 97%|█████████▋| 31/32 [01:17<00:02,  2.49s/it]")
        assert (int(m.group(2)), int(m.group(3))) == (31, 32)

    def test_a_later_landmark_finishes_the_current_phase(self):
        """Log order is the truth: a missing or reworded completion line must not leave a
        phase spinning forever."""
        t = BootTracker(VLLM_PHASES, READY)
        t.feed("Opening user mode device driver")
        events = t.feed("Checkpoint directory: /weights")
        assert events[0][:3] == ("done", "device", "Tenstorrent device opened")
        assert events[1][:2] == ("start", "weights")

    def test_a_repeated_start_line_does_not_restart_the_phase(self):
        t = BootTracker(VLLM_PHASES, READY)
        t.feed("Warming up prefill for sequence length: 128")
        assert t.feed("Warming up prefill for sequence length: 1024") == []

    def test_unknown_boots_produce_no_rows_but_still_reach_ready(self):
        t = BootTracker(TT_DIT_PHASES, READY)
        for ln in ["something unrelated", "more of it"]:
            assert t.feed(ln) == []
        assert t.feed("INFO:     Application startup complete.") == [("ready",)]

    @pytest.mark.parametrize("phases", [VLLM_PHASES, TT_DIT_PHASES])
    def test_phase_tables_have_unique_keys(self, phases):
        keys = [p.key for p in phases]
        assert len(keys) == len(set(keys))
