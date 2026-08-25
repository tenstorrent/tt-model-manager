# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the v4 unified "model + manifest" packaging model: schema back-compat,
range-aware resolution, launch rendering, and the push->pull->serve CLI round-trip that
renders vllm_metadata.json from the manifest. No hardware, no network.
"""

import json

import pytest
from typer.testing import CliRunner

from tt_kernel import bundles, cli, hub, instances, localdb, metal, runtime, toolchain
from tt_kernel.device import DeviceInfo
from tt_kernel.manifest import (
    Capabilities,
    Entrypoint,
    Manifest,
    Platform,
    Producer,
    Resources,
    Runtime,
    WeightsRef,
    compare,
)

runner = CliRunner()


# --------------------------------------------------------------------- fixtures
def _v4_manifest(**over):
    """A minimal valid v4 Manifest; override any field."""
    base = dict(
        schema_version="4",
        name="Laguna",
        tt_metal_version="0.73.0",
        arch="blackhole",
        device_count=4,
        producer=Producer(tt_kernel_version="0", created_at="t"),
        entrypoint=Entrypoint(**{"class": "ttl.gen:LagunaForCausalLM", "arch_name": "LagunaForCausalLM"}),
        weights=WeightsRef(repo="poolside/Laguna-XS-2.1"),
    )
    base.update(over)
    return Manifest(**base)


def _env(**over):
    base = dict(arch="blackhole", device_count=4, tt_metal_version="0.73.0", vllm_version="0.24.1")
    base.update(over)
    return metal.LocalEnv(**base)


# --------------------------------------------------------------------- schema
def test_v4_manifest_parses():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"))
    assert m.is_v4 is True
    j = m.to_json()
    m2 = Manifest.from_json(j)
    assert m2.is_v4 and m2.entrypoint.cls == "ttl.gen:LagunaForCausalLM"
    assert m2.weights.repo_id == "poolside/Laguna-XS-2.1"


def test_v4_json_aliases_accepted():
    """An authored manifest uses the natural `class` / `repo` aliases."""
    text = json.dumps({
        "schema_version": "4", "name": "L", "tt_metal_version": "0.73.0", "arch": "blackhole",
        "producer": {"tt_kernel_version": "0", "created_at": "t"},
        "entrypoint": {"class": "a:B", "arch_name": "B"},
        "weights": {"repo": "x/y"},
    })
    m = Manifest.from_json(text)
    assert m.entrypoint.cls == "a:B" and m.weights.repo_id == "x/y"


def test_v3_manifest_still_parses():
    text = json.dumps({
        "schema_version": "3", "name": "legacy", "tt_metal_version": "0.72.0", "arch": "blackhole",
        "build_key": 4242, "producer": {"tt_kernel_version": "0", "created_at": "t"},
    })
    m = Manifest.from_json(text)
    assert m.schema_version == "3" and m.is_v4 is False and m.build_key == 4242


def test_unknown_schema_rejected():
    text = json.dumps({
        "schema_version": "99", "name": "x", "tt_metal_version": "0", "arch": "blackhole",
        "producer": {"tt_kernel_version": "0", "created_at": "t"},
    })
    with pytest.raises(ValueError, match="Unsupported bundle schema_version"):
        Manifest.from_json(text)


# --------------------------------------------------------------------- version ranges
@pytest.mark.parametrize("installed,spec,expected", [
    ("0.73.0", ">=0.72,<0.76", True),
    ("0.80.0", ">=0.72,<0.76", False),
    ("0.72.0-5-gabc", ">=0.72,<0.76", True),   # git-describe decoration tolerated
    ("v0.74.1", ">=0.72,<0.76", True),         # leading v stripped
    ("deadbeef", ">=0.72,<0.76", None),        # bare sha -> assume OK
    (None, ">=0.72", None),                    # unresolved -> assume OK
    ("0.73.0", "not-a-spec!!", None),          # malformed spec -> None, never raises
    # Prereleases must keep their patch level (regression: was truncated to 0.72 before).
    ("0.72.3rc1", ">=0.72.2", True),           # rc of a patch that IS in range
    ("0.56.0rc1", ">=0.56.0", False),          # too-old prerelease correctly rejected
    ("1.1.3+light", ">=1.1.0", True),          # local/build metadata tolerated
])
def test_version_satisfies(installed, spec, expected):
    assert toolchain.version_satisfies(installed, spec) is expected


def test_compare_v4_in_range_compatible():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    r = compare(m, _env())
    assert r.compatible and not r.issues


def test_compare_v4_out_of_range_is_forceable_not_fatal():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    r = compare(m, _env(tt_metal_version="0.80.0", vllm_version="0.20.0"))
    assert not r.compatible and r.forceable and not r.has_fatal
    fields = {i.field for i in r.issues}
    assert "platform.ttnn" in fields and "runtime.vllm" in fields


def test_compare_v4_arch_still_fatal():
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"))
    r = compare(m, _env(arch="wormhole_b0"))
    assert r.has_fatal and any(i.field == "arch" and i.fatal for i in r.issues)


def test_compare_v4_dev_checkout_not_blocked():
    """A bare git sha for ttnn/vllm must not be reported out-of-range."""
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    r = compare(m, _env(tt_metal_version="deadbeef", vllm_version=None))
    assert r.compatible and not r.issues


# --------------------------------------------------------------------- render
def test_render_composes_launch_command():
    m = _v4_manifest(
        resources=Resources(max_model_len=131072, max_num_seqs=8, block_size=64,
                             trace_region_bytes=1500000000),
        capabilities=Capabilities(tool_parser="poolside_v1", reasoning_parser="poolside_v1"),
        env={"MESH_DEVICE": "P150"},
    )
    md = bundles.render_vllm_metadata(m)
    assert md["arch"] == "LagunaForCausalLM"
    assert md["main_class"] == "ttl.gen:LagunaForCausalLM"
    assert md["hf_weights"] == "poolside/Laguna-XS-2.1"
    cmd = md["launch"]["default"]["command"]
    assert cmd[:4] == ["python3", "server_example_tt.py", "--model", "poolside/Laguna-XS-2.1"]
    for flag, val in [("--max_model_len", "131072"), ("--max_num_seqs", "8"),
                      ("--block_size", "64"), ("--trace_region_size", "1500000000"),
                      ("--tool-call-parser", "poolside_v1"),
                      ("--reasoning_parser", "poolside_v1")]:
        assert cmd[cmd.index(flag) + 1] == val
    # vLLM rejects --tool-call-parser unless auto tool choice is enabled alongside it.
    assert "--enable-auto-tool-choice" in cmd
    env = md["launch"]["default"]["env"]
    assert env["VLLM_USE_V1"] == "1" and env["MESH_DEVICE"] == "P150"


def test_render_tool_parser_uses_vllm_flag_names():
    """``capabilities.tool_parser`` must render the flags vLLM actually accepts.

    vLLM's parser normalizes underscores to dashes, so the old ``--tool_parser`` became
    ``--tool-parser`` — not a vLLM flag — and the server refused to start. The real flag is
    ``--tool-call-parser``, and vLLM additionally requires ``--enable-auto-tool-choice``
    beside it. Verified against vllm 0.24 serving Qwen3-Coder with ``qwen3_coder``.
    """
    m = _v4_manifest(capabilities=Capabilities(tool_parser="qwen3_coder"))
    cmd = bundles.render_vllm_metadata(m)["launch"]["default"]["command"]
    assert "--tool_parser" not in cmd and "--tool-parser" not in cmd
    i = cmd.index("--tool-call-parser")
    assert cmd[i + 1] == "qwen3_coder"
    assert cmd[i - 1] == "--enable-auto-tool-choice"


def test_render_no_tool_flags_without_capability():
    """No tool_parser declared => neither flag appears (auto-tool-choice alone is an error)."""
    cmd = bundles.render_vllm_metadata(_v4_manifest())["launch"]["default"]["command"]
    assert "--enable-auto-tool-choice" not in cmd and "--tool-call-parser" not in cmd


def test_render_extra_args_appended_and_override_replaces():
    m = _v4_manifest(resources=Resources(
        max_num_seqs=8, extra_args=["--enable-prefix-caching"],
        command_override={"blackhole-4card": ["python3", "bh.py", "--fast"]},
    ))
    md = bundles.render_vllm_metadata(m)
    assert md["launch"]["default"]["command"][-1] == "--enable-prefix-caching"
    assert md["launch"]["blackhole-4card"]["command"] == ["python3", "bh.py", "--fast"]


def test_render_requires_entrypoint():
    m = _v4_manifest()
    m.entrypoint = None
    with pytest.raises(ValueError, match="entrypoint"):
        bundles.render_vllm_metadata(m)


# --------------------------------------------------------------------- CLI round-trip
def _fake_hub(monkeypatch, *, tt_metal="0.73.0", vllm="0.24.1"):
    """Redirect hub I/O to a local remote dir; deterministic device + versions."""
    import shutil
    remotes = {}

    def push_folder(repo_id, staged, commit_message=""):
        dst = remotes[repo_id] = staged.parent / f"remote__{bundles.model_key(repo_id)}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(staged, dst)

    def download_bundle(repo_id, revision, dest):
        shutil.copytree(remotes[repo_id], dest, dirs_exist_ok=True)
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
    monkeypatch.setattr(metal, "resolve_version", lambda: tt_metal)
    monkeypatch.setattr(metal, "_vllm_version", lambda: vllm)
    monkeypatch.setattr(metal, "_vllm_plugin_version", lambda: None)
    # Silence the toolchain warning path (no real tt-metal/vLLM in the test env).
    monkeypatch.setattr(toolchain, "check_toolchain",
                        lambda python=None: toolchain.ToolchainReport(components=[]))
    # Hermetic instance resolution: the active env is the only candidate (no real fs scan /
    # ~/.config), and it reports the mocked versions. Mirrors a single-build box.
    active = instances.Instance(name="active", python="/venv/bin/python", source="active")
    monkeypatch.setattr(instances, "all_instances", lambda roots=None: [active])
    monkeypatch.setattr(
        instances, "probe_versions",
        lambda inst, use_cache=True: instances.InstanceVersions(ttnn=tt_metal, vllm=vllm, plugin=None),
    )
    return remotes  # {repo_id: published-folder Path}


def _write_v4_manifest_file(tmp_path, **extra):
    doc = {
        "platform": {"ttnn": ">=0.72,<0.76"},
        "runtime": {"kind": "vllm", "version": ">=0.24"},
        "target": "p150x4",
        "mesh": {"devices": 4, "topology": "1x4", "fabric": "FABRIC_1D_RING"},
        "entrypoint": {"class": "ttl.gen:LagunaForCausalLM", "arch_name": "LagunaForCausalLM"},
        "weights": {"repo": "poolside/Laguna-XS-2.1"},
        "resources": {"max_num_seqs": 8},
        "capabilities": {"tool_parser": "poolside_v1"},
        "env": {"MESH_DEVICE": "P150"},
    }
    doc.update(extra)
    p = tmp_path / "laguna.json"
    p.write_text(json.dumps(doc))
    return p


def test_v4_push_pull_serve_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    remotes = _fake_hub(monkeypatch)

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "generator_vllm.py").write_text("# adapter code\n")
    mp = _write_v4_manifest_file(tmp_path)

    push = runner.invoke(cli.app, ["push", "acme/laguna", "--private", "--backend", "vllm",
                                   "--manifest", str(mp), "--bundle-dir", str(adapter)])
    assert push.exit_code == 0, push.output

    # The published bundle ships NO vllm_metadata.json — it's rendered on pull.
    import pathlib
    remote = remotes["acme/laguna"]
    assert not (remote / "vllm_bundle" / bundles.VLLM_METADATA_NAME).exists()
    assert (remote / "vllm_bundle" / "generator_vllm.py").exists()  # adapter code shipped
    assert json.loads((remote / "tt_kernel_manifest.json").read_text())["schema_version"] == "4"

    pull = runner.invoke(cli.app, ["pull", "acme/laguna", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    assert "rendered vllm_metadata.json" in pull.output

    rendered = pathlib.Path(localdb.get("acme/laguna")["bundle_path"]) / bundles.VLLM_METADATA_NAME
    assert rendered.exists()
    md = json.loads(rendered.read_text())
    assert md["main_class"] == "ttl.gen:LagunaForCausalLM"

    serve = runner.invoke(cli.app, ["serve", "acme/laguna", "--print", "--local-only"])
    assert serve.exit_code == 0, serve.output
    assert "server_example_tt.py" in serve.output
    assert "--max_num_seqs 8" in serve.output
    assert "--enable-auto-tool-choice --tool-call-parser poolside_v1" in serve.output
    assert "MESH_DEVICE=P150" in serve.output


def _published_manifest(remotes, repo_id):
    return json.loads((remotes[repo_id] / "tt_kernel_manifest.json").read_text())


def test_push_v4_prefers_mesh_devices_and_normalizes_arch(monkeypatch, tmp_path):
    """F3 + F8: authored mesh.devices beats the pusher's card count; arch is normalized."""
    remotes = _fake_hub(monkeypatch)
    # Pusher is a SINGLE-card box; the model targets 4 (mesh.devices) and is pushed with --arch bh.
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=1, source="test"))
    mp = _write_v4_manifest_file(tmp_path)  # mesh.devices == 4
    r = runner.invoke(cli.app, ["push", "acme/m", "--private", "--backend", "vllm",
                                "--manifest", str(mp), "--arch", "bh"])
    assert r.exit_code == 0, r.output
    man = _published_manifest(remotes, "acme/m")
    assert man["device_count"] == 4       # mesh wins over the 1-card pusher
    assert man["arch"] == "blackhole"     # 'bh' normalized, not stamped raw


def test_push_v4_weights_filters_apply_to_manifest_repo(monkeypatch, tmp_path):
    """F4: --weights-allow filters the manifest's own repo, even without --weights."""
    remotes = _fake_hub(monkeypatch)
    mp = _write_v4_manifest_file(tmp_path)  # weights.repo declared in the manifest
    r = runner.invoke(cli.app, ["push", "acme/w", "--private", "--backend", "vllm",
                                "--manifest", str(mp), "--weights-allow", "*.safetensors"])
    assert r.exit_code == 0, r.output
    w = _published_manifest(remotes, "acme/w")["weights"]
    assert w["repo_id"] == "poolside/Laguna-XS-2.1"       # authored repo preserved
    assert w["allow_patterns"] == ["*.safetensors"]        # filter applied to it


def test_push_v4_capability_tags(monkeypatch, tmp_path):
    """F5: --capability is honored on the v4 path (was silently dropped)."""
    _fake_hub(monkeypatch)
    tags = {}
    monkeypatch.setattr(hub, "tag_repo", lambda rid, t: tags.__setitem__(rid, t))
    mp = _write_v4_manifest_file(tmp_path)
    r = runner.invoke(cli.app, ["push", "acme/c", "--private", "--backend", "vllm",
                                "--manifest", str(mp), "--capability", "moe",
                                "--capability", "sliding-window-attention"])
    assert r.exit_code == 0, r.output
    assert "moe" in tags["acme/c"] and "sliding-window-attention" in tags["acme/c"]


def test_pull_v4_platform_only_manifest_clean_error(monkeypatch, tmp_path):
    """F6: a v4-ish manifest with no entrypoint errors cleanly instead of a raw ValueError."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    bdir = tmp_path / "bundles"
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(bdir))
    remotes = _fake_hub(monkeypatch)
    # Hand-craft a remote whose manifest sets platform (is_v4) + a vLLM runner but NO entrypoint.
    remote = tmp_path / "remote__acme__bad"
    remote.mkdir()
    (remote / "tt_kernel_manifest.json").write_text(json.dumps({
        "schema_version": "4", "name": "bad", "tt_metal_version": "0.73.0", "arch": "blackhole",
        "device_count": 4, "build_key": None, "producer": {"tt_kernel_version": "0", "created_at": "t"},
        "platform": {"ttnn": ">=0.72"},
        "runner": {"backend": "vllm", "bundle_dir": "vllm_bundle"},
    }))
    remotes["acme/bad"] = remote
    r = runner.invoke(cli.app, ["pull", "acme/bad", "--arch", "blackhole"])
    assert r.exit_code == 1
    assert "entrypoint" in r.output.lower() and "Traceback" not in r.output
    assert not (bdir / "acme__bad").exists()  # nothing installed


def test_pull_v4_fails_closed_on_unindexed_folder(monkeypatch, tmp_path):
    """F10: a shipped folder the manifest doesn't index is refused, not silently installed."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    bdir = tmp_path / "bundles"
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(bdir))
    remotes = _fake_hub(monkeypatch)
    remote = tmp_path / "remote__acme__tamper"
    (remote / "vllm_bundle").mkdir(parents=True)
    (remote / "vllm_bundle" / "evil.py").write_text("print('unverified')\n")  # present…
    (remote / "tt_kernel_manifest.json").write_text(json.dumps({          # …but NOT indexed
        "schema_version": "4", "name": "t", "tt_metal_version": "0.73.0", "arch": "blackhole",
        "device_count": 4, "build_key": None, "producer": {"tt_kernel_version": "0", "created_at": "t"},
        "entrypoint": {"class": "a:B", "arch_name": "B"}, "files": [],
        "runner": {"backend": "vllm", "bundle_dir": "vllm_bundle"},
    }))
    remotes["acme/tamper"] = remote
    r = runner.invoke(cli.app, ["pull", "acme/tamper", "--arch", "blackhole"])
    assert r.exit_code == 1 and "unverified" in r.output.lower()
    assert not (bdir / "acme__tamper").exists()


def test_serve_force_when_pull_out_of_range(monkeypatch, tmp_path):
    """F1: `serve` accepts --force so the documented one-liner is followable out of range."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tt_metal="0.80.0", vllm="0.24.1")  # ttnn out of the manifest's range
    mp = _write_v4_manifest_file(tmp_path)
    assert runner.invoke(cli.app, ["push", "acme/s", "--private", "--backend", "vllm",
                                   "--manifest", str(mp)]).exit_code == 0

    blocked = runner.invoke(cli.app, ["serve", "acme/s", "--print"])
    assert blocked.exit_code == 1 and "--force" in blocked.output

    forced = runner.invoke(cli.app, ["serve", "acme/s", "--print", "--force"])
    assert forced.exit_code == 0, forced.output
    assert "server_example_tt.py" in forced.output


def test_v4_push_builtin_entrypoint_no_bundle_dir(monkeypatch, tmp_path):
    """An entrypoint that references a tt-metal built-in needs no shipped code folder."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch)
    mp = _write_v4_manifest_file(tmp_path)

    push = runner.invoke(cli.app, ["push", "acme/builtin", "--private", "--backend", "vllm",
                                   "--manifest", str(mp)])
    assert push.exit_code == 0, push.output
    pull = runner.invoke(cli.app, ["pull", "acme/builtin", "--arch", "blackhole"])
    assert pull.exit_code == 0, pull.output
    import pathlib
    rendered = pathlib.Path(localdb.get("acme/builtin")["bundle_path"]) / bundles.VLLM_METADATA_NAME
    assert rendered.exists()


def test_v4_pull_out_of_range_blocks_then_forces(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch, tt_metal="0.80.0", vllm="0.20.0")  # both out of the declared range
    mp = _write_v4_manifest_file(tmp_path)
    runner.invoke(cli.app, ["push", "acme/oor", "--private", "--backend", "vllm", "--manifest", str(mp)])

    blocked = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole"])
    assert blocked.exit_code == 1
    assert "--force" in blocked.output

    forced = runner.invoke(cli.app, ["pull", "acme/oor", "--arch", "blackhole", "--force"])
    assert forced.exit_code == 0, forced.output


def test_search_target_and_arch_tags(monkeypatch):
    captured = {}

    class _M:
        def __init__(self, i):
            self.id, self.private, self.downloads, self.last_modified = i, False, 1, ""

    class _Api:
        @staticmethod
        def list_models(filter=None, search=None, limit=50):
            captured["filter"] = filter
            return [_M("acme/laguna")]

    monkeypatch.setattr(hub, "_api", lambda: _Api())
    res = runner.invoke(cli.app, ["search", "laguna", "--arch", "Blackhole", "--target", "P150x4"])
    assert res.exit_code == 0, res.output
    assert captured["filter"] == ["tt-model-cache", "blackhole", "p150x4"]  # ANDed, lowercased


def test_doctor_reports_bundle_ranges(monkeypatch):
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"),
                     target="p150x4")
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: m)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4, source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.80.0")  # out of range
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.24.1")    # in range
    monkeypatch.setattr(toolchain, "check_toolchain",
                        lambda python=None: toolchain.ToolchainReport(components=[]))

    res = runner.invoke(cli.app, ["doctor", "acme/laguna", "--arch", "blackhole"])
    # An out-of-range requirement must fail the gate (doctor's documented contract), not exit 0.
    assert res.exit_code == 1, res.output
    assert "Bundle requirements" in res.output
    assert "require >=0.72,<0.76, installed 0.80.0" in res.output  # flagged out of range
    assert "target: p150x4" in res.output


def test_doctor_names_the_instance_that_would_serve(monkeypatch):
    """`doctor <bundle>` must name the instance it resolves, not just the ranges.

    REGRESSION: this reporting block sat after `_report_bundle_requirements`'s `return`,
    so it never executed — doctor printed a bundle's required ranges and then went quiet
    about which of the host's registered builds actually satisfies them, which is the one
    thing a user runs `doctor <bundle>` to find out. Dead code prints nothing and fails
    nothing, so only an output assertion catches it.
    """
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: m)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4,
                                                              source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.73.0")
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.24.1")
    monkeypatch.setattr(toolchain, "check_toolchain",
                        lambda python=None: toolchain.ToolchainReport(components=[]))

    chosen = instances.Instance(name="sel", python="/venv/bin/python3", source="registry")
    monkeypatch.setattr(instances, "select",
                        lambda **kw: instances.SelectionResult(
                            chosen=chosen, candidates=[], reason="sel (ttnn=0.73.0)"))

    res = runner.invoke(cli.app, ["doctor", "acme/laguna", "--arch", "blackhole"])
    assert res.exit_code == 0, res.output
    assert "would link to instance" in res.output, res.output
    assert "sel (ttnn=0.73.0)" in res.output


def test_doctor_reports_when_no_instance_is_selectable(monkeypatch):
    """The other half of the same block: an unsatisfiable set says so, and says what to do."""
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: m)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4,
                                                              source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.73.0")
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.24.1")
    monkeypatch.setattr(toolchain, "check_toolchain",
                        lambda python=None: toolchain.ToolchainReport(components=[]))
    monkeypatch.setattr(instances, "select",
                        lambda **kw: instances.SelectionResult(
                            chosen=None, candidates=[], reason="nothing registered"))

    res = runner.invoke(cli.app, ["doctor", "acme/laguna", "--arch", "blackhole"])
    assert "no instance selectable" in res.output, res.output
    assert "tt-model instances add" in res.output


def test_doctor_in_range_exits_zero(monkeypatch):
    m = _v4_manifest(platform=Platform(ttnn=">=0.72,<0.76"), runtime=Runtime(version=">=0.24"))
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: m)
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=4, source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.73.0")   # in range
    monkeypatch.setattr(metal, "_vllm_version", lambda: "0.24.1")     # in range
    monkeypatch.setattr(toolchain, "check_toolchain",
                        lambda python=None: toolchain.ToolchainReport(components=[]))
    res = runner.invoke(cli.app, ["doctor", "acme/laguna", "--arch", "blackhole"])
    assert res.exit_code == 0, res.output
