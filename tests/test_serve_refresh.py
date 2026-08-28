# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Offline tests for `serve --refresh`: the opt-in path that re-pulls + re-installs an already
installed self-contained bundle when the Hub has a newer revision, so a republished source is not
served with stale launch params. The Hub download, the install.sh subprocess, and the launch are
all stubbed — no hardware, no network.
"""

from pathlib import Path

from typer.testing import CliRunner

from tt_kernel import cli, localdb, metal, packaging
from tt_kernel.manifest import WeightsRef

_runner = CliRunner()


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # localdb
    monkeypatch.setenv("TT_MODEL_MODELS_DIR", str(tmp_path / "models"))  # install dir
    monkeypatch.setattr(metal, "local_env", lambda **k: metal.LocalEnv(arch="blackhole", device_count=1))


def _staged_bundle(tmp_path):
    """Stage a v5 self-contained bundle (ttnn wheel tagged for THIS interpreter so it's compatible)."""
    win = tmp_path / "in"
    win.mkdir()
    ttnn = win / f"ttnn-0.75.0-{packaging.host_python_tag()}-{packaging.host_python_tag()}-linux_x86_64.whl"
    ttnn.write_bytes(b"whl")
    plugin = win / "vllm_tt_plugin-0.3.0-py3-none-any.whl"
    plugin.write_bytes(b"whl")
    metal_dir = tmp_path / "metal"
    metal_dir.mkdir()
    (metal_dir / "requirements.txt").write_text("torch==2.11.0\n")
    staged = tmp_path / "snap_src"
    packaging.stage_package(
        staged, name="llama-3.2-3b-tt", arch="blackhole",
        ttnn_wheel=ttnn, plugin_wheel=plugin, metal_dir=metal_dir,
        vllm_metadata={"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"},
        tt_kernel_version="0.0.0", weights=WeightsRef(repo="unsloth/Llama-3.2-3B-Instruct"),
        tt_metal_version="0.75.0",
    )
    return staged


def _record_installed(tmp_path, *, revision, pinned=False):
    inst = tmp_path / "models" / "myorg" / "llama-3.2-3b-tt"
    inst.mkdir(parents=True, exist_ok=True)
    run_sh = inst / "run.sh"
    run_sh.write_text("#!/usr/bin/env bash\necho serving\n")
    (inst / "venv").mkdir(exist_ok=True)  # marks a real install for the reuse/reinstall gate
    localdb.record("myorg/llama-3.2-3b-tt", {
        "repo_id": "myorg/llama-3.2-3b-tt", "self_contained": True,
        "install_dir": str(inst), "bundle_path": str(inst), "run_script": str(run_sh),
        "python": str(inst / "venv/bin/python"), "revision": revision, "pinned": pinned,
    })


def _install_fakes(staged, monkeypatch):
    """Wire download_bundle + install_self_contained fakes; return a dict capturing their calls."""
    seen = {"download_revs": [], "installs": 0}

    def _fake_download(repo_id, revision, dest):
        import shutil
        seen["download_revs"].append(revision)
        snap = Path(dest) / f"snap{seen['installs']}"
        shutil.copytree(staged, snap)
        return snap

    def _fake_install(bundle_dir, venv_dir):
        (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
        py = venv_dir / "bin" / "python"
        py.write_text("#!/bin/sh\n")
        seen["installs"] += 1
        return py

    monkeypatch.setattr(cli.hub, "download_bundle", _fake_download)
    monkeypatch.setattr(cli.runtime, "install_self_contained", _fake_install)
    return seen


def _stub_serve_run(monkeypatch):
    def _fake_run(argv, **kw):
        class _R:
            returncode = 0
        return _R()
    monkeypatch.setattr(cli.subprocess, "run", _fake_run)


def test_serve_refresh_reinstalls_when_newer_tip(monkeypatch, tmp_path):
    # A newer revision is published; `serve --refresh` must re-pull + re-install BEFORE serving.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", "myorg/llama-3.2-3b-tt", "--refresh", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 1, "a newer tip with --refresh should reinstall"
    assert seen["download_revs"] == ["newsha1111"]  # fetched exactly the resolved tip
    assert localdb.get("myorg/llama-3.2-3b-tt")["revision"] == "newsha1111"  # record updated


def test_serve_without_refresh_serves_installed_and_never_reinstalls(monkeypatch, tmp_path):
    # Default behavior is unchanged: a newer tip only WARNS; it must not download or reinstall.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", "myorg/llama-3.2-3b-tt", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0, "no --refresh: must serve the installed bundle, never reinstall"
    assert seen["download_revs"] == []
    assert "There is an update to myorg/llama-3.2-3b-tt" in res.output  # advisory still fires
    assert localdb.get("myorg/llama-3.2-3b-tt")["revision"] == "oldsha0000"  # unchanged


def test_serve_refresh_noop_when_up_to_date(monkeypatch, tmp_path):
    # Tip == installed: --refresh resolves the tip, sees no diff, and serves as-is (no download).
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="samesha000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "samesha000")

    res = _runner.invoke(cli.app, ["serve", "myorg/llama-3.2-3b-tt", "--refresh", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


def test_serve_refresh_noop_when_offline(monkeypatch, tmp_path):
    # Hub unreachable (latest_revision -> None): never block a serve; serve the installed bundle.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: None)

    res = _runner.invoke(cli.app, ["serve", "myorg/llama-3.2-3b-tt", "--refresh", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


def test_serve_refresh_ignored_with_local_only(monkeypatch, tmp_path):
    # --local-only must not touch the Hub even alongside --refresh.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("latest_revision must not be called with --local-only")
    monkeypatch.setattr(cli.hub, "latest_revision", _boom)

    res = _runner.invoke(cli.app, ["serve", "myorg/llama-3.2-3b-tt", "--refresh", "--local-only", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


def test_serve_refresh_skips_pinned_install(monkeypatch, tmp_path):
    # A pinned install (user chose @revision) must not be moved off it, even by --refresh.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000", pinned=True)
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", "myorg/llama-3.2-3b-tt", "--refresh", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []
    assert localdb.get("myorg/llama-3.2-3b-tt")["revision"] == "oldsha0000"
