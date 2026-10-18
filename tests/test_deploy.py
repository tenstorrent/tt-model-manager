# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model deploy`: the file, and the planner that keeps instances from colliding."""

import pytest

from tt_kernel import deploy
from tt_kernel.manifest import Manifest

from test_container_cli import _wire

QB2 = {0: "A", 1: "A", 2: "B", 3: "B"}


def _manifest(tmp_path, profiles):
    profs = [{"name": n, "hardware": hw, "mesh_device": mesh, "max_num_seqs": 32} for n, hw, mesh in profiles]
    return Manifest.from_json(_wire(tmp_path, serve_profiles=profs, default_profile=profiles[0][0]).read_text())


@pytest.fixture
def pkg(tmp_path):
    return _manifest(tmp_path, [("p150", "p150", "P150"), ("p300", "p300", "P300"), ("p300x2", "p300x2", "P300x2")])


def _dep(tmp_path, body):
    p = tmp_path / "d.yaml"
    p.write_text(body)
    return deploy.load(str(p))


def _plan(dep, pkg, *, claimed=(), port_free=lambda p: True, hugepages=16):
    return deploy.plan(dep, {i.model: pkg for i in dep.instances}, inventory=[0, 1, 2, 3],
                       claimed=set(claimed), boards=QB2, port_free=port_free, free_hugepages=hugepages)


def test_two_2chip_instances_get_one_board_each_and_their_own_ports(tmp_path, pkg):
    dep = _dep(tmp_path, """
deployment: gptoss-64u
instances:
  - {name: a, model: org/m, profile: p300}
  - {name: b, model: org/m, profile: p300}
""")
    plan = _plan(dep, pkg)
    assert [(p.instance.name, p.chips, p.port) for p in plan] == [("a", [0, 1], 20000), ("b", [2, 3], 20001)]
    assert plan[0].container != plan[1].container


def test_explicit_pins_are_placed_first_and_auto_picks_route_around_them(tmp_path, pkg):
    dep = _dep(tmp_path, """
instances:
  - {name: tts, model: org/m, profile: p150}
  - {name: llm, model: org/m, profile: p300, chips: [0, 1], port: 20050}
""")
    plan = {p.instance.name: p for p in _plan(dep, pkg)}
    assert plan["llm"].chips == [0, 1] and plan["llm"].port == 20050
    assert plan["tts"].chips == [2]


def test_overlapping_chips_are_refused_and_point_at_cohosting(tmp_path, pkg):
    dep = _dep(tmp_path, """
instances:
  - {name: big, model: org/m, profile: p300x2, chips: [0, 1, 2, 3]}
  - {name: small, model: org/m, profile: p300, chips: [0, 1]}
""")
    with pytest.raises(deploy.DeployError, match="already taken by big.*co-host"):
        _plan(dep, pkg)


def test_chips_held_by_a_running_server_are_not_handed_out(tmp_path, pkg):
    dep = _dep(tmp_path, "instances:\n  - {name: a, model: org/m, profile: p300}\n")
    assert _plan(dep, pkg, claimed={0})[0].chips == [2, 3]
    with pytest.raises(deploy.DeployError, match="whole boards"):
        _plan(dep, pkg, claimed={1, 2})


def test_busy_and_duplicate_ports_are_refused(tmp_path, pkg):
    dup = _dep(tmp_path, """
instances:
  - {name: a, model: org/m, port: 20001}
  - {name: b, model: org/m, port: 20001}
""")
    with pytest.raises(deploy.DeployError, match="given to two"):
        _plan(dup, pkg)
    studio = _dep(tmp_path, "instances:\n  - {name: a, model: org/m, port: 8000}\n")
    with pytest.raises(deploy.DeployError, match="in use"):
        _plan(studio, pkg)


def test_auto_ports_skip_ports_that_are_taken(tmp_path, pkg):
    dep = _dep(tmp_path, "instances:\n  - {name: a, model: org/m}\n  - {name: b, model: org/m}\n")
    plan = _plan(dep, pkg, port_free=lambda p: p != 20000)
    assert [p.port for p in plan] == [20001, 20002]


def test_too_few_hugepages_is_refused_before_anything_starts(tmp_path, pkg):
    dep = _dep(tmp_path, "instances:\n  - {name: a, model: org/m, profile: p300x2}\n")
    with pytest.raises(deploy.DeployError, match="hugepage"):
        _plan(dep, pkg, hugepages=3)


@pytest.mark.parametrize("body, match", [
    ("instances: []\n", "non-empty"),
    ("instances:\n  - {model: org/m}\n", "needs `name`"),
    ("instances:\n  - {name: a, model: org/m, gpu: 1}\n", "unknown key"),
    ("instances:\n  - {name: a, model: org/m}\n  - {name: a, model: org/m}\n", "used twice"),
    ("instances:\n  - {name: a, model: org/m, args: --max-model-len 4096}\n", "`args` is a list"),
    ("instances:\n  - {name: a, model: org/m, chips: \"23\"}\n", "list of chip ids"),
    ("instances:\n  - {name: a, model: org/m, chips: 0}\n", "list of chip ids"),
    ("instances:\n  - {name: a, model: org/m, port: high}\n", "`port` is `auto` or a number"),
])
def test_bad_files_are_refused_with_the_reason(tmp_path, body, match):
    with pytest.raises(deploy.DeployError, match=match):
        _dep(tmp_path, body)


def test_a_wrong_chip_count_names_what_the_profile_needs(tmp_path, pkg):
    dep = _dep(tmp_path, "instances:\n  - {name: a, model: org/m, profile: p300, chips: [0]}\n")
    with pytest.raises(deploy.DeployError, match="needs 2 distinct"):
        _plan(dep, pkg)


def test_an_instance_that_serves_several_ports_gets_a_whole_free_range(tmp_path, pkg):
    dep = _dep(tmp_path, "instances:\n  - {name: group, model: org/m}\n  - {name: solo, model: org/m}\n")
    plan = deploy.plan(dep, {"org/m": pkg}, inventory=[0, 1, 2, 3], claimed=set(), boards=QB2,
                       port_free=lambda p: p != 20002, free_hugepages=16,
                       ports_needed=lambda m, prof: 3 if prof.name else 1)
    assert [p.port for p in plan] == [20003, 20006]


def test_deploy_up_rolls_back_and_explains_when_an_instance_does_not_come_up(tmp_path, pkg, monkeypatch):
    from typer.testing import CliRunner

    from tt_kernel import cli, container, container_cli

    dep = _dep(tmp_path, "instances:\n  - {name: a, model: org/m, profile: p300}\n  - {name: b, model: org/m, profile: p300}\n")
    plan = _plan(dep, pkg)
    monkeypatch.setattr(cli, "_deploy_plan", lambda f: (dep, plan))
    started, stopped, removed = [], [], []
    monkeypatch.setattr(container_cli, "serve_container", lambda m, **kw: started.append(kw["instance"]))
    monkeypatch.setattr(container_cli, "pull_dir", lambda repo: tmp_path)
    monkeypatch.setattr(container_cli, "stop_container", lambda m, **kw: stopped.append(kw["instance"]))
    monkeypatch.setattr(container, "container_exists", lambda n: True)
    monkeypatch.setattr(container, "is_running", lambda n: False)
    monkeypatch.setattr(container, "remove", lambda n, force=False: removed.append(n))
    monkeypatch.setattr(container, "wait_ready", lambda name, probe, timeout_s=0: container.ReadyResult(
        not name.endswith(".b"), name.endswith(".b"), ["RuntimeError: Invalid device ID in TT_VISIBLE_DEVICES: 2"], 1.0))
    res = CliRunner().invoke(cli.app, ["deploy", "up", "d.yaml"])
    assert res.exit_code != 0
    assert started == ["a", "b"] and stopped == ["b", "a"] and len(removed) == 2
    assert "✗" in res.output and "Invalid device ID" in res.output
    assert "stopped the deployment" in res.output


@pytest.fixture
def up(tmp_path, pkg, monkeypatch):
    """`deploy up` of instances a and b with docker stubbed out; returns (run, calls)."""
    from typer.testing import CliRunner

    from tt_kernel import cli, container, container_cli

    dep = _dep(tmp_path, "instances:\n  - {name: a, model: org/m, profile: p300}\n  - {name: b, model: org/m, profile: p300}\n")
    plan = _plan(dep, pkg)
    monkeypatch.setattr(cli, "_deploy_plan", lambda f: (dep, plan))
    calls = {"started": [], "stopped": [], "fail_start": set(), "ready": lambda name: True}

    def serve(m, **kw):
        if kw["instance"] in calls["fail_start"]:
            raise RuntimeError("image load failed")
        calls["started"].append(kw["instance"])
    monkeypatch.setattr(container_cli, "serve_container", serve)
    monkeypatch.setattr(container_cli, "pull_dir", lambda repo: tmp_path)
    monkeypatch.setattr(container_cli, "stop_container", lambda m, **kw: calls["stopped"].append(kw["instance"]))
    monkeypatch.setattr(container, "container_exists", lambda n: False)

    def wait(name, probe, timeout_s=0):
        ok = calls["ready"](name)
        return container.ReadyResult(ok, False, [] if ok else ["still loading"], 1.0)
    monkeypatch.setattr(container, "wait_ready", wait)
    return (lambda *flags: CliRunner().invoke(cli.app, ["deploy", "up", "d.yaml", *flags])), calls


def test_keep_going_stops_an_instance_that_timed_out_and_exits_1(up):
    run, calls = up
    calls["ready"] = lambda name: not name.endswith(".b")   # b is still booting at the deadline
    res = run("--keep-going")
    assert res.exit_code == 1
    assert calls["stopped"] == ["b"], "b would keep booting on its chips, outside the deployment"


def test_a_start_failure_of_any_kind_rolls_back_what_started(up):
    run, calls = up
    calls["fail_start"] = {"b"}
    res = run()
    assert res.exit_code != 0 and "could not start b" in res.output
    assert calls["started"] == ["a"] and calls["stopped"] == ["a"]


def test_keep_going_keeps_what_started_when_another_cannot_start(up):
    run, calls = up
    calls["fail_start"] = {"b"}
    res = run("--keep-going")
    assert res.exit_code == 1 and "1 of 2 instance(s) ready" in res.output
    assert calls["started"] == ["a"] and calls["stopped"] == []


def test_ctrl_c_while_waiting_rolls_back_every_started_instance(up):
    run, calls = up

    def interrupted(name):
        raise KeyboardInterrupt
    calls["ready"] = interrupted
    run()
    assert calls["stopped"] == ["b", "a"]
