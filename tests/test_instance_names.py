# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``serve --name``: one package and profile running more than once, each addressable."""

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container, container_cli

from test_container_cli import _host_is_fine, _isolated_cache, _manifest  # noqa: F401 (fixtures)

runner = CliRunner()


class _R:
    def __init__(self, stdout="", rc=0):
        self.stdout, self.returncode = stdout, rc


def test_an_instance_gets_its_own_container_name(tmp_path):
    m = _manifest(tmp_path)
    profile = m.container.resolve_profile(None)
    base = container.container_name(m, profile)
    assert container.container_name(m, profile, "a") == f"{base}.a"
    assert container.container_name(m, profile, None) == base


@pytest.mark.parametrize("bad", ["", "A", "-a", "a_b", "a.b", "x" * 33])
def test_bad_instance_names_are_refused(bad):
    with pytest.raises(container.ContainerError, match="instance name"):
        container.check_instance_name(bad)


def test_compose_run_names_and_labels_the_instance(tmp_path):
    m = _manifest(tmp_path)
    argv = container.compose_run(m, m.container.resolve_profile(None), [], {}, device_ids=[2],
                                 hf_home_dir=tmp_path, cache_dir=tmp_path, weight_cache_dir=tmp_path,
                                 tensor_cache_dir=tmp_path, instance="b")
    assert argv[argv.index("--name") + 1].endswith(".b")
    assert f"{container.INSTANCE_LABEL}=b" in argv


def test_labelled_containers_reads_the_labels_and_filters(monkeypatch, real_labelled_containers):
    seen = {}

    def fake(argv, **kw):
        seen["argv"] = argv
        return _R("tt-model-x-p150.a\tUp 5 minutes\tx\tp150\ta\t20001\t2\n")
    monkeypatch.setattr(container, "_run", fake)
    rows = container.labelled_containers(model="x", instance="a")
    assert rows == [{"name": "tt-model-x-p150.a", "status": "Up 5 minutes", "model": "x",
                     "profile": "p150", "instance": "a", "port": "20001", "devices": "2"}]
    assert f"label={container.LABEL}=x" in seen["argv"]
    assert f"label={container.INSTANCE_LABEL}=a" in seen["argv"]
    assert "--all" not in seen["argv"]


def test_stop_without_a_name_stops_every_instance_and_with_one_only_that(tmp_path, monkeypatch):
    m = _manifest(tmp_path)
    profile = m.container.resolved_default()
    base = container.container_name(m, m.container.resolve_profile(None))
    up = {base, f"{base}.a", f"{base}.b"}
    monkeypatch.setattr(container, "is_running", lambda n: n in up)
    monkeypatch.setattr(container, "labelled_containers", lambda **kw: [
        {"name": f"{base}.{i}", "status": "Up", "model": m.name, "profile": profile,
         "instance": i, "port": "", "devices": ""} for i in ("b", "a")])
    stopped = []
    monkeypatch.setattr(container, "stop", lambda n, image=None: (
        stopped.append(n), container.StopOutcome(True, "not_needed"))[1])
    container_cli.stop_container(m)
    assert stopped == [base, f"{base}.a", f"{base}.b"]
    stopped.clear()
    container_cli.stop_container(m, instance="a")
    assert stopped == [f"{base}.a"]


def test_serve_with_a_name_launches_that_instance_and_hints_with_it(tmp_path, monkeypatch, capsys):
    ran = []
    monkeypatch.setattr(container, "running", lambda name=None: [])
    monkeypatch.setattr(container, "run_checked", lambda argv, **kw: ran.append(argv))
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    container_cli.serve_container(_manifest(tmp_path), target="org/x", detach=True, instance="a")
    argv = ran[0]
    assert argv[argv.index("--name") + 1].endswith(".a")
    assert "tt-model logs org/x --name a -f" in capsys.readouterr().out


def test_serve_refuses_a_bad_instance_name_before_any_work(tmp_path, monkeypatch):
    monkeypatch.setattr(container, "run_checked", lambda argv, **kw: pytest.fail("ran"))
    with pytest.raises(container.ContainerError, match="instance name"):
        container_cli.serve_container(_manifest(tmp_path), target="org/x", instance="Bad")


def test_curl_by_name_targets_that_instances_port(monkeypatch):
    monkeypatch.setattr(container, "labelled_containers", lambda **kw: [
        {"name": "tt-model-x-p150.a", "status": "Up", "model": "x", "profile": "p150",
         "instance": kw.get("instance"), "port": "20005", "devices": "0"}])
    res = runner.invoke(cli.app, ["curl", "hi", "--name", "a", "--model", "m", "--print"])
    assert res.exit_code == 0, res.output
    assert "http://127.0.0.1:20005/v1/chat/completions" in res.output


def test_curl_by_an_unknown_name_says_how_to_start_it(monkeypatch):
    res = runner.invoke(cli.app, ["curl", "hi", "--name", "nope", "--model", "m", "--print"])
    assert res.exit_code != 0
    assert "tt-model serve <id> --name nope" in res.output
