# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model deploy` with a co-host group: the fit check decides shared or separate chips."""

import json

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container, container_cli, deploy
from tt_kernel.launchers import CohostLauncher, LauncherError
from tt_kernel.manifest import Manifest

from test_container_cli import _wire

FITS = {"verdict": "shared", "fits": True, "members": {"gptoss": {"status": "ok"}, "llama8b": {"status": "ok"}}}
# the shape of tt-cohost's fit report (python -m tt_cohost.fit)
CLASH = {"fit_report_version": 1, "verdict": "separate", "fits": False,
         "reason": "llama8b failed in compile: L1 clash (its programs need L1 another member holds): "
                   "Statically allocated circular buffers on core range clash with L1 buffers",
         "failure": {"member": "llama8b", "phase": "compile", "kind": "l1_clash",
                     "error": "Statically allocated circular buffers on core range clash with L1 buffers"},
         "members": {"gptoss": {"status": "aborted", "phase": "compile", "error": "the group stopped"},
                     "llama8b": {"status": "failed", "phase": "compile", "error_kind": "l1_clash",
                                 "error": "Statically allocated circular buffers on core range clash with L1 buffers"}}}


@pytest.fixture
def group(tmp_path):
    """A 4-chip co-host package of gptoss + llama8b; each member names its own package."""
    raw = json.loads(_wire(tmp_path).read_text())
    raw["name"] = "llm-pair"
    raw["container"]["kind"] = "cohost"
    raw["container"]["runtime"] = {"members": [
        {"name": "gptoss", "kind": "vllm", "model": "openai/gpt-oss-20b", "port_offset": 0,
         "standalone": {"model": "org/gptoss", "profile": "p300"}},
        {"name": "llama8b", "kind": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct", "port_offset": 1,
         "standalone": {"model": "org/llama", "profile": "p300"}}]}
    raw["container"]["serve_profiles"] = [{"name": "p150x4", "hardware": "p300x2", "mesh_device": "P150x4"}]
    raw["container"]["default_profile"] = "p150x4"
    return Manifest.from_json(json.dumps(raw))


@pytest.fixture
def single(tmp_path):
    profs = [{"name": "p300", "hardware": "p300", "mesh_device": "P300", "max_num_seqs": 32}]
    return Manifest.from_json(_wire(tmp_path, serve_profiles=profs, default_profile="p300").read_text())


def _file(tmp_path, body):
    p = tmp_path / "d.yaml"
    p.write_text(body)
    return str(p)


GROUP_ONLY = "instances:\n  - {name: llm, model: org/group}\n"


# ------------------------------------------------------------------------- the file


def test_a_group_lists_what_serves_its_members_on_separate_chips(tmp_path):
    dep = deploy.load(_file(tmp_path, """
instances:
  - name: llm
    model: org/group
    separate:
      - {name: gptoss, model: org/gptoss, profile: p300}
      - {name: llama, model: org/llama, chips: [2, 3]}
"""))
    assert [(i.name, i.model, i.profile, i.chips) for i in dep.instances[0].separate] == [
        ("gptoss", "org/gptoss", "p300", None), ("llama", "org/llama", None, [2, 3])]


@pytest.mark.parametrize("body, match", [
    ("instances:\n  - {name: llm, model: org/g, separate: [{name: a, model: m, separate: []}]}\n", "unknown key"),
    ("instances:\n  - {name: llm, model: org/g, separate: [{name: llm, model: m}]}\n", "used twice"),
    ("instances:\n  - {name: llm, model: org/g, separate: {name: a}}\n", "list of instances"),
])
def test_bad_separate_lists_are_refused(tmp_path, body, match):
    with pytest.raises(deploy.DeployError, match=match):
        deploy.load(_file(tmp_path, body))


def test_without_a_list_the_members_own_packages_serve_them(tmp_path, group):
    inst = deploy.load(_file(tmp_path, GROUP_ONLY)).instances[0]
    assert [(i.name, i.model, i.profile) for i in deploy.separate_instances(inst, group, "p150x4")] == [
        ("llm-gptoss", "org/gptoss", "p300"), ("llm-llama8b", "org/llama", "p300")]


def test_a_member_with_no_package_of_its_own_is_named(tmp_path, group):
    del group.container.runtime["members"][1]["standalone"]
    inst = deploy.load(_file(tmp_path, GROUP_ONLY)).instances[0]
    with pytest.raises(deploy.DeployError, match="member\\(s\\) llama8b.*`separate:`"):
        deploy.separate_instances(inst, group, "p150x4")


def test_the_reason_names_the_member_the_phase_and_the_error():
    assert deploy.fit_summary("llm", CLASH) == (
        "llm: the models do not fit on shared chips (llama8b failed in compile: L1 clash (its programs need L1 "
        "another member holds): Statically allocated circular buffers on core range clash with L1 buffers)")
    no_reason = {"verdict": "separate", "failure": CLASH["failure"]}
    assert "(llama8b failed in compile: Statically allocated" in deploy.fit_summary("llm", no_reason)


def test_standalone_must_name_a_package():
    runtime = {"members": [{"name": "a", "kind": "vllm", "model": "m", "standalone": "org/a"}]}
    with pytest.raises(LauncherError, match="standalone is"):
        CohostLauncher().members(runtime, None)


# ------------------------------------------------------------------------ deploy up


@pytest.fixture
def up(tmp_path, group, single, monkeypatch):
    """`deploy up` on a 4-chip box with docker stubbed out; returns (run, calls)."""
    calls = {"served": [], "fits": [], "verdict": FITS}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_require_container", lambda model: group if model == "org/group" else single)
    monkeypatch.setattr(container, "all_device_ids", lambda: [0, 1, 2, 3])
    monkeypatch.setattr(container, "_claimed_devices", lambda inventory: set())
    monkeypatch.setattr(container, "chip_boards", lambda: {0: "A", 1: "A", 2: "B", 3: "B"})
    monkeypatch.setattr(container, "port_is_free", lambda port: True)
    monkeypatch.setattr(container, "free_hugepages_1g", lambda: 16)

    def fit(m, **kw):
        calls["fits"].append((kw["instance"], kw["chips"]))
        return calls["verdict"]
    monkeypatch.setattr(container_cli, "fit_check", fit)
    monkeypatch.setattr(container_cli, "fit_files", lambda m, prof, target=None: (tmp_path / "fit", prof))
    monkeypatch.setattr(container_cli, "ensure_weights", lambda *a, **k: calls.setdefault("weights", []).append(a[1]))
    monkeypatch.setattr(container_cli, "serve_container",
                        lambda m, **kw: calls["served"].append((kw["instance"], kw["device_id"])))
    monkeypatch.setattr(container_cli, "pull_dir", lambda repo: tmp_path)
    monkeypatch.setattr(container, "wait_ready", lambda *a, **k: container.ReadyResult(True, False, [], 1.0))

    def run(body=GROUP_ONLY, *flags):
        _file(tmp_path, body)
        return CliRunner().invoke(cli.app, ["deploy", "up", "d.yaml", *flags])
    return run, calls


def test_a_group_that_fits_is_served_on_shared_chips(up):
    run, calls = up
    res = run()
    assert res.exit_code == 0, res.output
    assert calls["weights"] == ["org/group"], "the trial boot needs the weights before it starts"
    assert "tail -f" in res.output
    assert calls["fits"] == [("llm", [0, 1, 2, 3])]
    assert calls["served"] == [("llm", "0,1,2,3")]


def test_a_group_that_does_not_fit_falls_back_to_separate_chips_and_says_why(up):
    run, calls = up
    calls["verdict"] = CLASH
    res = run()
    assert res.exit_code == 0, res.output
    out = " ".join(res.output.split())
    assert "llama8b failed in compile: L1 clash" in out and "clash with L1 buffers" in out
    assert "serving llm-gptoss, llm-llama8b on separate chips" in out
    assert calls["served"] == [("llm-gptoss", "0,1"), ("llm-llama8b", "2,3")]


def test_placement_separate_skips_the_check_and_serves_the_file_list(up):
    run, calls = up
    res = run("instances:\n  - {name: llm, model: org/group, separate: [{name: a, model: org/a, profile: p300}]}\n",
              "--placement", "separate")
    assert res.exit_code == 0, res.output
    assert calls["fits"] == [] and calls["served"] == [("a", "0,1")]


def test_placement_shared_serves_the_group_without_the_check(up):
    run, calls = up
    calls["verdict"] = CLASH
    res = run(GROUP_ONLY, "--placement", "shared")
    assert res.exit_code == 0, res.output
    assert calls["fits"] == [] and calls["served"] == [("llm", "0,1,2,3")]


def test_no_fallback_for_a_group_that_does_not_fit_starts_nothing(up, group):
    run, calls = up
    calls["verdict"] = CLASH
    del group.container.runtime["members"][0]["standalone"]
    res = run()
    out = " ".join(res.output.split())
    assert res.exit_code != 0 and calls["served"] == []
    assert "llama8b failed in compile" in out and "gptoss" in out and "--placement shared" in out


def test_an_unknown_placement_is_refused(up):
    run, _ = up
    res = run(GROUP_ONLY, "--placement", "maybe")
    assert res.exit_code != 0 and "auto, shared, separate" in res.output


def test_down_also_stops_the_members_a_fallback_started(up, tmp_path, monkeypatch):
    _file(tmp_path, GROUP_ONLY)
    stopped = []
    monkeypatch.setattr(container_cli, "stop_container", lambda m, **kw: stopped.append(kw["instance"]))
    monkeypatch.setattr(container, "is_running", lambda name: name.endswith(".llm-llama8b"))
    res = CliRunner().invoke(cli.app, ["deploy", "down", "d.yaml"])
    assert res.exit_code == 0, res.output
    assert stopped == ["llm", "llm-llama8b"]


# ------------------------------------------------------------------------ fit check


@pytest.fixture
def fit_env(tmp_path, group, monkeypatch):
    """container_cli.fit_check with docker stubbed: the fake `docker run` writes ``verdict``."""
    state = {"runs": [], "verdict": FITS}
    monkeypatch.setattr(container, "model_cache_dir", lambda m: tmp_path / "cache")
    monkeypatch.setattr(container, "loaded_digest", lambda ref: "sha256:" + "c" * 64)
    monkeypatch.setattr(container, "ensure_mount_sources", lambda m: None)
    monkeypatch.setattr(container, "docker_is_rootless", lambda: False)
    monkeypatch.setattr(container, "container_exists", lambda name: False)

    class _Done:
        returncode = 0

    def fake_run(argv, **kw):
        state["runs"].append(argv)
        if state["verdict"] is not None:
            (tmp_path / "cache" / "fit" / f"p150x4.{'c' * 12}.verdict.json").write_text(json.dumps(state["verdict"]))
        kw["stdout"].write("INFO fit: llama8b compile\n")
        return _Done()
    monkeypatch.setattr(container_cli.subprocess, "run", fake_run)

    def check():
        return container_cli.fit_check(group, profile_name=None, port=21000, chips=[0, 1, 2, 3],
                                       instance="llm", target="org/group")
    return check, state, tmp_path / "cache" / "fit"


def test_the_fit_check_boots_the_group_on_its_planned_chips(fit_env):
    check, state, fit_dir = fit_env
    assert check()["verdict"] == "shared"
    argv = state["runs"][0]
    key = f"p150x4.{'c' * 12}"
    assert argv[-9:] == ["python", "-m", "tt_cohost.fit", "--config", f"/cache/fit/{key}.group.json",
                         "--json", f"/cache/fit/{key}.verdict.json", "--workdir", f"/cache/fit/{key}.work"]
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--device"] == [
        f"/dev/tenstorrent/{c}:/dev/tenstorrent/{c}" for c in range(4)]
    assert argv[argv.index("--name") + 1].endswith(".llm-fit")
    config = json.loads((fit_dir / f"{key}.group.json").read_text())
    assert [(m["name"], m["port"]) for m in config["members"]] == [("gptoss", 21000), ("llama8b", 21001)]


def test_a_shared_verdict_is_kept_for_the_same_image(fit_env):
    check, state, _ = fit_env
    check()
    again = check()
    assert again["verdict"] == "shared" and again["cached"] and len(state["runs"]) == 1


def test_a_separate_verdict_is_checked_again_next_time(fit_env):
    check, state, _ = fit_env
    state["verdict"] = CLASH
    assert check()["verdict"] == "separate"
    check()
    assert len(state["runs"]) == 2


def test_a_check_that_writes_no_verdict_counts_as_not_fitting(fit_env):
    check, state, _ = fit_env
    state["verdict"] = None
    verdict = check()
    assert verdict["verdict"] == "separate" and "llama8b compile" in verdict["reason"]
