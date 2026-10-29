# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model ps`: what is running on this host, on which chips, at which endpoint."""

import json
import subprocess
import time

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container, localdb

from test_container_cli import _wire

runner = CliRunner()


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _row(name, model, profile, instance, port, devices):
    return {"name": name, "status": "Up 3 minutes", "model": model, "profile": profile,
            "instance": instance, "port": port, "devices": devices}


def test_ps_says_when_nothing_is_running(home):
    res = runner.invoke(cli.app, ["ps"])
    assert res.exit_code == 0
    assert "nothing running" in res.output


def test_ps_lists_containers_by_repo_with_chips_and_endpoint(home, monkeypatch):
    manifest = _wire(home)
    name = json.loads(manifest.read_text())["name"]
    localdb.record("org/x", {"repo_id": "org/x", "container": True, "manifest": str(manifest)})
    monkeypatch.setattr(container, "labelled_containers", lambda **kw: [
        _row(f"tt-model-{name}-p150.b", name, "p150", "b", "20001", "1"),
        _row(f"tt-model-{name}-p150", name, "p150", "", "20000", "0"),
    ])
    res = runner.invoke(cli.app, ["ps"])
    assert res.exit_code == 0, res.output
    lines = res.output.splitlines()
    assert lines[0].split() == ["MODEL", "PROFILE", "NAME", "CHIPS", "ENDPOINT", "STATUS"]
    assert "org/x" in lines[1] and "http://127.0.0.1:20000" in lines[1]
    assert " b " in lines[2] and "http://127.0.0.1:20001" in lines[2]


def test_ps_json_is_machine_readable(home, monkeypatch):
    monkeypatch.setattr(container, "labelled_containers", lambda **kw: [
        _row("tt-model-m-p300.a", "m", "p300", "a", "20002", "2,3")])
    res = runner.invoke(cli.app, ["ps", "--json"])
    rows = json.loads(res.output)
    assert rows[0]["chips"] == "2,3" and rows[0]["url"] == "http://127.0.0.1:20002"
    assert rows[0]["repo"] == "m"  # unknown to localdb: falls back to the manifest name


def test_ps_finds_a_running_bundle_server(home):
    inst = home / "models" / "org" / "thin"
    inst.mkdir(parents=True)
    localdb.record("org/thin", {"repo_id": "org/thin", "self_contained": True,
                                "install_dir": str(inst), "run_script": str(inst / "run.sh")})
    proc = subprocess.Popen(["bash", "-c", f'exec -a "{inst}/venv/bin/python" python3 -c "import time; time.sleep(60)" --port 20007'],
                            env={"TT_VISIBLE_DEVICES": "3"})
    try:
        time.sleep(0.2)
        (inst / cli.SERVE_PID_FILE).write_text(str(proc.pid))
        rows = json.loads(runner.invoke(cli.app, ["ps", "--json"]).output)
        assert [(r["repo"], r["chips"], r["port"], r["kind"]) for r in rows] == \
            [("org/thin", "3", "20007", "bundle")]
    finally:
        proc.kill()
