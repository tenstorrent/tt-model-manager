# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""CLI wiring for the container (v5.1) path.

Two things are under test: that the container commands do the right thing, and — just as
important — that the v5 flow through the SAME commands is unchanged.
"""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container, container_cli
from tt_kernel.container_manifest import ContainerManifest
from tt_kernel.manifest import Manifest

from test_container_manifest import BASE

runner = CliRunner()


def _wire(tmp_path: Path, **over) -> Path:
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    wire = m.to_wire(image_tag="tt-model/my-model:abc123", tt_metal_version="0.72.1",
                     tt_kernel_version="0.1.0")
    p = tmp_path / "tt_kernel_manifest.json"
    p.write_text(wire.to_json())
    return p


# ------------------------------------------------------------------ v5 is unchanged


def test_package_without_from_metal_says_what_to_pass():
    """--from-metal became optional so --container could exist; omitting BOTH must still
    be a clear error, not a traceback."""
    res = runner.invoke(cli.app, ["package", "org/x", "--out", "/tmp/x"])
    assert res.exit_code != 0
    assert "--from-metal is required" in res.output
    assert "--container" in res.output  # and it points at the alternative


def test_the_v5_list_command_still_exists_and_is_not_shadowed():
    res = runner.invoke(cli.app, ["--help"])
    assert "List locally installed bundles" in res.output


# ------------------------------------------------------------------ package dispatch


def test_package_container_routes_to_the_container_path(tmp_path, monkeypatch):
    called = {}

    def fake(manifest_path, out_root=None):
        called["path"] = manifest_path
        called["out"] = out_root
        return tmp_path / "build" / "my-model"

    monkeypatch.setattr(container_cli, "package_container", fake)
    res = runner.invoke(cli.app, ["package", "--container", "m.yaml", "--out", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert called == {"path": "m.yaml", "out": str(tmp_path)}


def test_a_manifest_error_is_reported_not_raised(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise container_cli.ContainerCliError("serve profile 'p': block_size is required")

    monkeypatch.setattr(container_cli, "package_container", boom)
    res = runner.invoke(cli.app, ["package", "--container", "m.yaml"])
    assert res.exit_code != 0
    assert "block_size is required" in res.output


# ------------------------------------------------------------------ target resolution


def test_a_local_manifest_path_resolves(tmp_path):
    """How an author serves a build before pushing it."""
    assert container_cli.resolve_target(str(_wire(tmp_path))) is not None


def test_a_v5_manifest_path_does_not_resolve_as_a_container(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(Manifest(schema_version="5", name="n", tt_metal_version="v",
                          arch="blackhole",
                          producer={"tt_kernel_version": "0.1.0", "created_at": "now"}
                          ).to_json())
    assert container_cli.resolve_target(str(p)) is None


def test_garbage_resolves_to_none_rather_than_raising(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("not json")
    assert container_cli.resolve_target(str(p)) is None
    assert container_cli.resolve_target("org/never-pulled") is None


def test_stop_on_an_unpulled_target_says_to_pull_it():
    res = runner.invoke(cli.app, ["stop", "org/nope"])
    assert res.exit_code != 0
    assert "tt-model pull org/nope" in res.output


# ------------------------------------------------------------------ serve


def _manifest(tmp_path, **over) -> Manifest:
    return Manifest.from_json(_wire(tmp_path, **over).read_text())


def test_serve_print_emits_the_docker_run_without_running_it(tmp_path, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    container_cli.serve_container(_manifest(tmp_path), print_only=True)
    out = capsys.readouterr().out
    assert "docker run" in out and "--device /dev/tenstorrent" in out
    assert not ran


def test_serve_refuses_when_the_container_already_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running",
                        lambda name=None: [{"name": name, "status": "Up 3 minutes"}])
    with pytest.raises(container_cli.ContainerCliError, match="already exists"):
        container_cli.serve_container(_manifest(tmp_path))


def test_serve_starts_the_container(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    container_cli.serve_container(_manifest(tmp_path))
    assert ran and ran[0][:2] == ["docker", "run"]
    assert "tt-model-my-model-p150x4" in ran[0]


def test_serve_picks_the_named_profile(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: ran.append(argv))
    two = json.loads(json.dumps(BASE))["serve_profiles"] + [
        {"name": "p150x2", "hardware": "p150x2", "mesh_device": "P150x2", "max_num_seqs": 8}]
    m = _manifest(tmp_path, serve_profiles=two, default_profile="p150x4")
    container_cli.serve_container(m, profile_name="p150x2")
    assert "tt-model-my-model-p150x2" in ran[0]


def test_an_unknown_profile_is_refused_with_the_available_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    with pytest.raises(container_cli.ContainerCliError, match="p150x4"):
        container_cli.serve_container(_manifest(tmp_path), profile_name="nope")


def test_follow_reports_when_the_server_never_became_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv: None)
    monkeypatch.setattr(container, "wait_ready", lambda *a, **k: False)
    with pytest.raises(container_cli.ContainerCliError, match="did not report ready"):
        container_cli.serve_container(_manifest(tmp_path), follow=True)


# ------------------------------------------------------------------ stop


def test_stop_reports_a_clean_shutdown(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(container, "running", lambda name=None: [{"name": name}])
    monkeypatch.setattr(container, "stop", lambda name, image=None: True)
    container_cli.stop_container(_manifest(tmp_path))
    out = capsys.readouterr().out
    assert "stopped 1" in out
    assert "mesh" not in out.lower()


def test_stop_warns_loudly_when_a_kill_forced_a_mesh_reset(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(container, "running", lambda name=None: [{"name": name}])
    monkeypatch.setattr(container, "stop", lambda name, image=None: False)
    container_cli.stop_container(_manifest(tmp_path))
    assert "mesh was left dirty" in capsys.readouterr().out


def test_stopping_nothing_says_so_rather_than_failing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    container_cli.stop_container(_manifest(tmp_path))
    assert "nothing running" in capsys.readouterr().out


# ------------------------------------------------------------------ logs / profiles


def test_logs_without_a_running_container_says_how_to_start_one(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "running", lambda name=None: [])
    with pytest.raises(container_cli.ContainerCliError, match="tt-model serve"):
        container_cli.logs_container(_manifest(tmp_path))


def test_profiles_marks_the_default(tmp_path, monkeypatch, capsys):
    two = json.loads(json.dumps(BASE))["serve_profiles"] + [
        {"name": "p150x2", "hardware": "p150x2", "mesh_device": "P150x2", "max_num_seqs": 8}]
    m = _manifest(tmp_path, serve_profiles=two, default_profile="p150x4")
    container_cli.list_containers(m)
    out = capsys.readouterr().out
    assert "p150x4" in out and "default" in out and "p150x2" in out
