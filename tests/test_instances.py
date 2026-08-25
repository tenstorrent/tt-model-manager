# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the tt-metal instance registry + selection (instances.py) and its wiring into
pull (pin) / serve (replay). No hardware, no network, no real subprocess probes.
"""

import json

import pytest
from typer.testing import CliRunner

from tt_kernel import bundles, cli, hub, instances, localdb, metal, runtime, toolchain
from tt_kernel.device import DeviceInfo
from tt_kernel.instances import Instance, InstanceVersions

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolate_config(monkeypatch, tmp_path):
    """Redirect the registry file to a tmp XDG_CONFIG_HOME for every test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


# --------------------------------------------------------------------- registry file
def test_registry_add_remove_roundtrip():
    instances.add_instance("m073", "/opt/tt/073/bin/python", tt_metal_home="/opt/tt/073",
                           env={"LD_LIBRARY_PATH": "/l"})
    instances.add_instance("m080", "/opt/tt/080/bin/python", tt_metal_home="/opt/tt/080")
    names = {i.name for i in instances.registry_instances()}
    assert names == {"m073", "m080"}
    got = next(i for i in instances.registry_instances() if i.name == "m073")
    assert got.tt_metal_home == "/opt/tt/073" and got.env == {"LD_LIBRARY_PATH": "/l"}
    assert got.activation_env()["TT_METAL_HOME"] == "/opt/tt/073"

    assert instances.remove_instance("m073") is True
    assert {i.name for i in instances.registry_instances()} == {"m080"}
    assert instances.remove_instance("m073") is False  # already gone


def test_add_replaces_same_name():
    instances.add_instance("x", "/a/python")
    instances.add_instance("x", "/b/python")
    regs = [i for i in instances.registry_instances() if i.name == "x"]
    assert len(regs) == 1 and regs[0].python == "/b/python"


# --------------------------------------------------------------------- scan
def _make_checkout(root, name, with_python=True):
    co = root / name
    (co / "tt_metal" / "hw" / "inc").mkdir(parents=True)
    if with_python:
        bindir = co / "build" / "python_env" / "bin"
        bindir.mkdir(parents=True)
        (bindir / "python").write_text("#!/bin/sh\n")
    return co


def test_scan_finds_checkouts_and_flags_unlaunchable(tmp_path):
    root = tmp_path / "src"
    root.mkdir()
    _make_checkout(root, "metal-good", with_python=True)
    _make_checkout(root, "metal-nobuild", with_python=False)
    pairs = dict((p[0].split("/")[-1], p[1]) for p in
                 ((str(h), py) for h, py in instances.scan_checkouts([str(root)])))
    assert pairs["metal-good"] is not None
    assert pairs["metal-nobuild"] is None
    # Only the launchable one becomes a selectable instance.
    insts = instances.scan_instances([str(root)])
    assert [i.name for i in insts] == ["scan:metal-good"]
    assert insts[0].tt_metal_home.endswith("metal-good")


# --------------------------------------------------------------------- dedup / precedence
def test_all_instances_dedup_and_precedence(monkeypatch):
    active = Instance(name="active", python="/venv/bin/python", source="active")
    monkeypatch.setattr(instances, "active_instance", lambda: active)
    # A registry entry and a scan entry that resolve to the SAME python -> registry wins.
    monkeypatch.setattr(instances, "registry_instances",
                        lambda: [Instance(name="reg", python="/shared/python", source="registry")])
    monkeypatch.setattr(instances, "scan_instances",
                        lambda roots=None: [Instance(name="scan:x", python="/shared/python", source="scan")])
    alls = instances.all_instances(roots=[])
    assert [i.name for i in alls] == ["active", "reg"]  # scan dupe dropped, active always kept


# --------------------------------------------------------------------- select
def _insts(monkeypatch, table):
    """table: {name: (python, ttnn, vllm, plugin)} -> wire all_instances + probe_versions."""
    objs = [Instance(name=n, python=p, source="registry") for n, (p, *_v) in table.items()]
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: objs)
    vers = {n: InstanceVersions(t, vl, pl) for n, (_p, t, vl, pl) in table.items()}
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: vers[inst.name])


def test_select_newest_satisfying(monkeypatch):
    _insts(monkeypatch, {
        "old": ("/a", "0.71.0", "0.24.1", "0.3.0"),   # ttnn too old
        "good": ("/b", "0.73.0", "0.24.1", "0.3.2"),  # satisfies
        "newer_good": ("/c", "0.75.0", "0.26.0", "0.3.9"),  # satisfies, newer
        "toonew": ("/d", "0.80.0", "0.26.0", "0.3.9"),  # ttnn out of upper bound
    })
    res = instances.select(ttnn=">=0.72,<0.76", vllm=">=0.24", plugin=">=0.3,<0.4")
    assert res.chosen.name == "newer_good"


def test_select_excludes_on_vllm_or_plugin(monkeypatch):
    _insts(monkeypatch, {
        "bad_vllm": ("/a", "0.73.0", "0.20.0", "0.3.2"),    # vllm too old
        "bad_plugin": ("/b", "0.73.0", "0.24.1", "0.5.0"),  # plugin out of range
    })
    res = instances.select(ttnn=">=0.72,<0.76", vllm=">=0.24", plugin=">=0.3,<0.4")
    assert res.chosen is None
    assert all(not c.satisfies for c in res.candidates)


def test_select_gitsha_assumed_ok(monkeypatch):
    _insts(monkeypatch, {"dev": ("/a", "deadbeef", None, None)})
    res = instances.select(ttnn=">=0.72,<0.76", vllm=">=0.24", plugin=">=0.3,<0.4")
    assert res.chosen.name == "dev"  # unparseable/None versions ⇒ assume OK


# --------------------------------------------------------------------- CLI: instances
def test_cli_instances_add_list_remove(monkeypatch):
    monkeypatch.setattr(instances, "scan_instances", lambda roots=None: [])
    monkeypatch.setattr(instances, "scan_checkouts", lambda roots=None: [])
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.73.0", "0.24.1", "0.3.2"))
    add = runner.invoke(cli.app, ["instances", "add", "--name", "m073",
                                  "--python", "/opt/tt/073/bin/python", "--env", "LD_LIBRARY_PATH=/l"])
    assert add.exit_code == 0, add.output
    lst = runner.invoke(cli.app, ["instances", "list"])
    assert lst.exit_code == 0 and "m073" in lst.output and "ttnn=0.73.0" in lst.output
    rm = runner.invoke(cli.app, ["instances", "remove", "m073"])
    assert rm.exit_code == 0 and "Removed" in rm.output


# --------------------------------------------------------------------- pull pins / serve replays
def _v4_manifest_file(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "platform": {"ttnn": ">=0.72,<0.76"},
        "runtime": {"kind": "vllm", "version": ">=0.24", "plugin_version": ">=0.3,<0.4"},
        "entrypoint": {"class": "ttl:LagunaForCausalLM", "arch_name": "LagunaForCausalLM"},
        "weights": {"repo": "p/L"}, "resources": {"max_num_seqs": 8}, "env": {"MESH_DEVICE": "P150"},
    }))
    return p


def _fake_hub(monkeypatch, tmp_path):
    import shutil
    remote = tmp_path / "remote"

    def push_folder(rid, staged, commit_message=""):
        if remote.exists():
            shutil.rmtree(remote)
        shutil.copytree(staged, remote)

    def download_bundle(rid, revision, dest):
        shutil.copytree(remote, dest, dirs_exist_ok=True)
        from pathlib import Path
        return Path(dest)

    # The repo does not exist yet, so push takes the create path and never touches
    # visibility on an existing repo (see cli._ensure_repo).
    monkeypatch.setattr(hub, "repo_exists", lambda *a, **k: False)
    monkeypatch.setattr(hub, "create_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "set_visibility", lambda *a, **k: None)
    monkeypatch.setattr(hub, "tag_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "push_folder", push_folder)
    monkeypatch.setattr(hub, "download_bundle", download_bundle)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4, source="test"))
    monkeypatch.setattr(toolchain, "check_toolchain",
                        lambda python=None: toolchain.ToolchainReport(components=[]))


def test_pull_pins_newest_satisfying_then_serve_replays(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    # Two candidates; m073 satisfies, m080 is too new. Give m073 an existing python so serve
    # uses the pin directly (no re-resolve).
    py073 = tmp_path / "py073"
    py073.write_text("#!/bin/sh\n")
    table = {
        "m073": Instance(name="m073", python=str(py073), tt_metal_home="/opt/073",
                         env={"LD_LIBRARY_PATH": "/l73"}, source="registry"),
        "m080": Instance(name="m080", python="/opt/080/python", source="registry"),
    }
    vers = {"m073": InstanceVersions("0.73.0", "0.24.1", "0.3.2"),
            "m080": InstanceVersions("0.80.0", "0.26.0", "0.3.9")}
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: list(table.values()))
    monkeypatch.setattr(instances, "probe_versions", lambda inst, use_cache=True: vers[inst.name])

    mp = _v4_manifest_file(tmp_path)
    assert runner.invoke(cli.app, ["push", "acme/l", "--private", "--backend", "vllm",
                                   "--manifest", str(mp)]).exit_code == 0
    pull = runner.invoke(cli.app, ["pull", "acme/l", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    assert "instance: m073" in pull.output

    entry = localdb.get("acme/l")
    assert entry["instance_name"] == "m073" and entry["instance_python"] == str(py073)
    assert entry["platform_ttnn"] == ">=0.72,<0.76"
    assert entry["runtime_plugin_version"] == ">=0.3,<0.4"

    serve = runner.invoke(cli.app, ["serve", "acme/l", "--print", "--local-only"])
    assert serve.exit_code == 0, serve.output
    assert serve.output.count(str(py073)) >= 1          # launched under the pinned interpreter
    assert "TT_METAL_HOME=/opt/073" in serve.output      # activation env threaded in
    assert "LD_LIBRARY_PATH=/l73" in serve.output
    assert "MESH_DEVICE=P150" in serve.output            # bundle launch env still applied


def test_pull_blocks_when_none_in_range_then_force(monkeypatch, tmp_path):
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    active = Instance(name="active", python="/venv/bin/python", source="active")
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: [active])
    monkeypatch.setattr(instances, "active_instance", lambda: active)
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.80.0", "0.20.0", None))

    mp = _v4_manifest_file(tmp_path)
    runner.invoke(cli.app, ["push", "acme/oor", "--private", "--backend", "vllm", "--manifest", str(mp)])
    blocked = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole"])
    assert blocked.exit_code == 1 and "--force" in blocked.output
    forced = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole", "--force"])
    assert forced.exit_code == 0, forced.output
    assert localdb.get("acme/oor")["instance_name"] == "active"  # forced fallback to active


def test_serve_reresolves_when_pin_missing(monkeypatch, tmp_path):
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    # Pin an instance whose python does not exist -> serve must re-resolve from stored ranges.
    good_py = tmp_path / "good"
    good_py.write_text("#!/bin/sh\n")
    replacement = Instance(name="m074", python=str(good_py), tt_metal_home="/opt/074", source="registry")
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: [replacement])
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.74.0", "0.24.1", "0.3.5"))

    # Seed a localdb entry with a dead pin + the ranges.
    dest = tmp_path / "bundles" / "acme__x"
    dest.mkdir(parents=True)
    bundles.write_vllm_metadata(dest, {
        "arch": "B", "main_class": "m:C", "hf_weights": "p/w",
        "launch": {"default": {"command": ["python3", "server_example_tt.py", "--model", "p/w"], "env": {}}},
    })
    localdb.record("acme/x", {
        "name": "x", "backend": "vllm", "build_key": None, "bundle_path": str(dest),
        "instance_name": "gone", "instance_python": "/does/not/exist/python",
        "platform_ttnn": ">=0.72,<0.76", "runtime_version": ">=0.24",
        "runtime_plugin_version": ">=0.3,<0.4", "installed_at": "now",
    })
    serve = runner.invoke(cli.app, ["serve", "acme/x", "--print", "--local-only"])
    assert serve.exit_code == 0, serve.output
    assert "re-resolving" in serve.output and "m074" in serve.output
    assert str(good_py) in serve.output and "TT_METAL_HOME=/opt/074" in serve.output


def test_v3_bundle_no_selection(monkeypatch, tmp_path):
    """A bundle with no platform/runtime ranges must not trigger instance selection."""
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.73.0")
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.24.1")
    monkeypatch.setattr(metal, "_vllm_plugin_version", lambda: None)
    # If selection were invoked, this would blow up (all_instances not patched to anything real).
    called = {"n": 0}
    monkeypatch.setattr(instances, "select",
                        lambda **k: called.__setitem__("n", called["n"] + 1) or instances.SelectionResult(None, [], "x"))

    # Author a legacy verbatim vLLM bundle (no v4 blocks) via --bundle-dir.
    folder = tmp_path / "b"
    folder.mkdir()
    bundles.write_vllm_metadata(folder, {
        "arch": "B", "main_class": "m:C", "hf_weights": "p/w",
        "launch": {"default": {"command": ["python3", "server_example_tt.py", "--model", "p/w"], "env": {}}},
    })
    runner.invoke(cli.app, ["push", "acme/legacy", "--private", "--backend", "vllm", "--bundle-dir", str(folder)])
    pull = runner.invoke(cli.app, ["pull", "acme/legacy", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    assert called["n"] == 0  # no instance selection for a range-less bundle
    assert localdb.get("acme/legacy")["instance_name"] is None


# =====================================================================================
# Review #11 (jzhengTT) regression tests
# =====================================================================================

# --- G1: version_satisfies must not crash on a None/absent range -----------------------
def test_version_satisfies_none_spec_is_none():
    # A real installed version with an undeclared (None) range must return None, not raise.
    assert toolchain.version_satisfies("0.72.0", None) is None
    assert toolchain.version_satisfies("0.72.0", "") is None


def test_select_with_partial_ranges_does_not_crash(monkeypatch):
    # A v4 manifest declaring only platform.ttnn (vllm/plugin None) must still select.
    _insts(monkeypatch, {"m": ("/p", "0.73.0", "0.24.0", "0.3.0")})
    res = instances.select(ttnn=">=0.72", vllm=None, plugin=None)
    assert res.chosen is not None and res.chosen.name == "m"


# --- G2: the pin must apply to the common launch-command forms -------------------------
@pytest.mark.parametrize("first,replaced", [
    ("python", True), ("python3", True), ("python3.10", True),
    ("/opt/tt/build/python_env/bin/python", True),
    ("vllm", False), ("bash", False),
])
def test_vllm_serve_argv_pins_python_forms(first, replaced):
    argv = runtime.vllm_serve_argv([first, "server_example_tt.py"], python="/pin/python")
    assert (argv[0] == "/pin/python") is replaced
    assert runtime.is_python_command(first) is replaced


# --- G10: activation env derives all three pin vars, prepended not clobbered -----------
def test_activation_env_derives_pythonpath_and_ld_library_path():
    inst = Instance(name="s", python="/p", tt_metal_home="/opt/tt-0.75", source="scan")
    env = inst.activation_env()
    assert env["TT_METAL_HOME"] == "/opt/tt-0.75"
    assert env["PYTHONPATH"] == "/opt/tt-0.75"
    assert env["LD_LIBRARY_PATH"] == "/opt/tt-0.75/build/lib"


def test_vllm_serve_env_prepends_path_vars(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
    monkeypatch.setenv("PYTHONPATH", "/existing")
    act = {"TT_METAL_HOME": "/opt/tt", "LD_LIBRARY_PATH": "/opt/tt/build/lib", "PYTHONPATH": "/opt/tt"}
    env = runtime.vllm_serve_env("/bundles", {}, activation_env=act)
    assert env["LD_LIBRARY_PATH"] == "/opt/tt/build/lib:/usr/lib"   # prepended, system kept
    assert env["PYTHONPATH"] == "/opt/tt:/existing"
    assert env["TT_METAL_HOME"] == "/opt/tt"                        # plain override


# --- G4: an unreadable scan root must be skipped, not crash the command ----------------
def test_scan_skips_unreadable_root(monkeypatch, tmp_path):
    from pathlib import Path
    bad = tmp_path / "denied"
    bad.mkdir()
    orig = Path.iterdir

    def fake_iterdir(self):
        if str(self) == str(bad):
            raise PermissionError("nope")
        return orig(self)

    monkeypatch.setattr(Path, "iterdir", fake_iterdir)
    # Must return cleanly (empty), not raise.
    assert instances.scan_checkouts([str(bad)]) == []


# --- G5 + G7: cache is keyed on (python, tt_metal_home)+mtime and skips failed probes ---
def _fake_probe_run(monkeypatch, by_home):
    """subprocess.run stub: emit versions chosen by the env's TT_METAL_HOME."""
    class _P:
        def __init__(self, out): self.stdout = out; self.returncode = 0
    def run(args, capture_output=True, text=True, timeout=None, env=None, check=True):
        home = (env or {}).get("TT_METAL_HOME")
        if home not in by_home:
            raise FileNotFoundError("boom")
        return _P(by_home[home])
    monkeypatch.setattr(instances.subprocess, "run", run)


def test_probe_cache_keyed_on_home_not_just_python(monkeypatch):
    # Two instances share one interpreter but differ in TT_METAL_HOME -> distinct versions.
    _fake_probe_run(monkeypatch, {
        "/opt/tt-0.72": "0.72.0|0.24.0|0.3.0",
        "/opt/tt-0.76": "0.76.0|0.25.0|0.3.9",
    })
    a = Instance(name="a", python="/usr/bin/python3", tt_metal_home="/opt/tt-0.72", source="registry")
    b = Instance(name="b", python="/usr/bin/python3", tt_metal_home="/opt/tt-0.76", source="registry")
    assert instances._cache_key(a) != instances._cache_key(b)
    assert instances.probe_versions(a).ttnn == "0.72.0"
    assert instances.probe_versions(b).ttnn == "0.76.0"   # not the cached 0.72 from `a`


def test_failed_probe_not_cached(monkeypatch):
    _fake_probe_run(monkeypatch, {})  # every probe raises
    inst = Instance(name="x", python="/usr/bin/python3", source="registry")
    assert instances.probe_versions(inst).ttnn is None
    # Nothing cached, so a later successful probe isn't masked by a stale None.
    assert instances._load().get("version_cache", {}) == {}


# --- G9: a corrupt registry is surfaced + moved aside, not silently erased -------------
def test_corrupt_registry_moved_aside(monkeypatch):
    instances.add_instance("keep", "/p/python")           # a real entry exists
    path = instances._registry_path()
    path.write_text("{ this is not json")                 # corrupt it
    assert instances._load() == {}                         # reads empty…
    assert path.with_suffix(".json.corrupt").is_file()     # …but the bad file is preserved
    # A subsequent add starts fresh (doesn't resurrect the corrupt content) and persists.
    instances.add_instance("new", "/p2/python")
    assert {i.name for i in instances.registry_instances()} == {"new"}


# --- G8: `instances list --for` drops the compat column on a fetch failure -------------
def test_instances_list_fetch_failure_skips_compat(monkeypatch):
    monkeypatch.setattr(instances, "all_instances",
                        lambda roots=None: [Instance(name="a", python="/p", source="registry")])
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.73.0", "0.24.0", "0.3.0"))
    monkeypatch.setattr(instances, "scan_checkouts", lambda roots=None: [])

    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(hub, "fetch_manifest", boom)
    res = runner.invoke(cli.app, ["instances", "list", "--for", "acme/x"])
    assert res.exit_code == 0, res.output
    assert "skipping compatibility" in res.output
    assert "✓" not in res.output and "✗" not in res.output   # no false compatibility marks


# --- G6: a None probe result must not blank out the real installed version -------------
def test_none_probe_does_not_nullify_gate(monkeypatch, tmp_path):
    """Selected instance reports ttnn (in range) but vLLM=None; the local vLLM (out of range)
    must still gate — a failed vLLM probe can't silently turn the check into a no-op."""
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tmp_path)
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.73.0")
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.20.0")      # ambient vLLM out of range
    monkeypatch.setattr(metal, "_vllm_plugin_version", lambda: None)
    sel = Instance(name="sel", python="/p", tt_metal_home="/opt/tt", source="registry")
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: [sel])
    monkeypatch.setattr(instances, "select",
                        lambda **k: instances.SelectionResult(
                            chosen=sel,
                            candidates=[instances.Candidate(sel, InstanceVersions("0.73.0", None, None), True)],
                            reason="sel"))
    monkeypatch.setattr(instances, "probe_versions",
                        lambda inst, use_cache=True: InstanceVersions("0.73.0", None, None))
    mp = _v4_manifest_file(tmp_path)  # runtime.version ">=0.24"
    assert runner.invoke(cli.app, ["push", "acme/g6", "--private", "--backend", "vllm",
                                   "--manifest", str(mp)]).exit_code == 0
    # vLLM 0.20 < 0.24 must block (probe None fell back to the real ambient 0.20), needs --force.
    blocked = runner.invoke(cli.app, ["pull", "acme/g6", "--arch", "blackhole"])
    assert blocked.exit_code == 1 and "--force" in blocked.output
