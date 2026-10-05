# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model stop` for a v5/v6 bundle, which serves as a host process rather than a container.

serve records the server's PID in the install folder; stop signals exactly that process.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, localdb

_runner = CliRunner()
_ID = "org/thin"


@pytest.fixture
def install(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    inst = tmp_path / "models" / "org" / "thin"
    inst.mkdir(parents=True)
    run_sh = inst / "run.sh"
    run_sh.write_text('echo "$$" > "$(dirname "$0")/seen.pid"\n')
    localdb.record(_ID, {"repo_id": _ID, "self_contained": True, "install_dir": str(inst),
                         "bundle_path": str(inst), "run_script": str(run_sh)})
    return inst


def _spawn(argv0):
    """A stand-in server whose /proc cmdline starts with ``argv0``."""
    p = subprocess.Popen(["bash", "-c", f'exec -a "{argv0}" sleep 60'])
    time.sleep(0.2)
    return p


def test_serve_records_the_pid_that_runs_run_sh(install, monkeypatch):
    real_run = subprocess.run
    during = {}

    def _run(argv, **kw):
        r = real_run(argv, **kw)
        during["pid"] = (install / cli.SERVE_PID_FILE).read_text().strip()
        return r

    monkeypatch.setattr(cli.subprocess, "run", _run)
    res = _runner.invoke(cli.app, ["serve", _ID, "--local-only", "--port", "20000"])
    assert res.exit_code == 0, res.output
    assert during["pid"] == (install / "seen.pid").read_text().strip()
    assert not (install / cli.SERVE_PID_FILE).exists()  # cleaned up once the server exits


def test_stop_terminates_the_recorded_server(install):
    proc = _spawn(f"{install}/venv/bin/python")
    (install / cli.SERVE_PID_FILE).write_text(str(proc.pid))
    res = _runner.invoke(cli.app, ["stop", _ID])
    assert res.exit_code == 0, res.output
    assert proc.wait(timeout=5) is not None
    assert not (install / cli.SERVE_PID_FILE).exists()


def test_stop_ignores_a_recycled_pid(install):
    stranger = _spawn("/usr/bin/some-other-program")
    try:
        (install / cli.SERVE_PID_FILE).write_text(str(stranger.pid))
        res = _runner.invoke(cli.app, ["stop", _ID])
        assert res.exit_code == 0, res.output
        assert "nothing running" in res.output
        assert stranger.poll() is None  # not ours: left alone
    finally:
        stranger.kill()
        stranger.wait()


def test_stop_with_nothing_running(install):
    res = _runner.invoke(cli.app, ["stop", _ID])
    assert res.exit_code == 0, res.output
    assert "nothing running" in res.output


def test_stop_finds_a_server_whose_run_script_is_nested(tmp_path, monkeypatch):
    """serve and stop must agree on where the PID file lives even when run.sh is not at the
    install root (``bundled.run_script`` may name e.g. ``bin/run.sh``)."""
    import threading

    import typer

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    inst = tmp_path / "models" / "org" / "nested"
    (inst / "bin").mkdir(parents=True)
    run_sh = inst / "bin" / "run.sh"
    # run.sh execs the server, as a generated one does; argv0 marks it as this install's.
    run_sh.write_text(f'exec -a "{inst}/venv/bin/python" sleep 60\n')
    entry = {"repo_id": "org/nested", "self_contained": True, "install_dir": str(inst),
             "bundle_path": str(inst), "run_script": str(run_sh)}

    def _serve():
        try:
            cli._serve_self_contained(entry, print_only=False)
        except typer.Exit:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    pid_file = inst / cli.SERVE_PID_FILE
    for _ in range(50):  # wait for serve to record the PID
        if pid_file.exists() and pid_file.read_text().strip():
            break
        time.sleep(0.1)
    assert pid_file.exists(), "serve wrote its PID somewhere stop does not look"

    cli._stop_self_contained(entry)
    t.join(timeout=10)
    assert not t.is_alive(), "stop did not terminate the server serve started"
