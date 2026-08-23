# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""The interrupt guard, against a fake long-running child (sleep, not docker).

The contract: an accidental Ctrl-C on a TTY costs NOTHING (warn + continue); a
deliberate double-press cancels the whole process group; unattended contexts
(SIGTERM, or SIGINT with no TTY) cancel immediately because nobody is there to
press twice.
"""

import os
import signal
import subprocess
import time

import pytest

from tt_model.build import InterruptGuard


def _spawn_sleeper(guard):
    return guard.spawn(["sleep", "60"], stdout=subprocess.DEVNULL)


def test_child_gets_its_own_process_group():
    """The whole basis of the guard: a terminal Ctrl-C must reach tt-model, not the
    child — same-group children would be killed directly with nothing to intercept."""
    with InterruptGuard("the build", tty=True) as g:
        proc = _spawn_sleeper(g)
        assert os.getpgid(proc.pid) != os.getpgid(0)
        assert os.getpgid(proc.pid) == proc.pid
        g._cancel()
        proc.wait(timeout=5)


def test_first_sigint_on_tty_warns_and_the_child_survives(capsys):
    with InterruptGuard("the build", tty=True, window_s=30) as g:
        proc = _spawn_sleeper(g)
        g._handle(signal.SIGINT, None)
        assert not g.cancelled
        assert proc.poll() is None                     # STILL RUNNING
        err = capsys.readouterr().err
        assert "STILL RUNNING" in err
        assert "Press Ctrl-C again" in err
        g._cancel()
        proc.wait(timeout=5)


def test_second_sigint_within_window_cancels_the_group():
    with InterruptGuard("the build", tty=True, window_s=30) as g:
        proc = _spawn_sleeper(g)
        g._handle(signal.SIGINT, None)
        g._handle(signal.SIGINT, None)
        assert g.cancelled
        proc.wait(timeout=5)
        assert proc.poll() is not None


def test_window_expiry_disarms(capsys):
    """If the second press comes after the window, it is a fresh warning, not a
    cancel — an old accidental press must not arm a future one."""
    with InterruptGuard("the build", tty=True, window_s=0.05) as g:
        proc = _spawn_sleeper(g)
        g._handle(signal.SIGINT, None)
        time.sleep(0.1)
        g._handle(signal.SIGINT, None)
        assert not g.cancelled
        assert proc.poll() is None
        g._cancel()
        proc.wait(timeout=5)


def test_sigint_without_tty_cancels_immediately():
    with InterruptGuard("the build", tty=False) as g:
        proc = _spawn_sleeper(g)
        g._handle(signal.SIGINT, None)
        assert g.cancelled
        proc.wait(timeout=5)


def test_sigterm_cancels_immediately_even_on_tty():
    with InterruptGuard("the build", tty=True) as g:
        proc = _spawn_sleeper(g)
        g._handle(signal.SIGTERM, None)
        assert g.cancelled
        proc.wait(timeout=5)


def test_stage_tracking_feeds_the_warning_card(capsys):
    with InterruptGuard("the build", tty=True) as g:
        g.note_line("#12 [builder 4/9] RUN ./build_metal.sh --enable-ccache\n")
        g.note_line("some non-stage output\n")
        assert g.current_stage == "builder 4/9"
        proc = _spawn_sleeper(g)
        g._handle(signal.SIGINT, None)
        assert "builder 4/9" in capsys.readouterr().err
        g._cancel()
        proc.wait(timeout=5)


def test_handlers_restored_on_exit():
    before = (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM))
    with InterruptGuard("x", tty=True):
        assert signal.getsignal(signal.SIGINT) != before[0]
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == before
