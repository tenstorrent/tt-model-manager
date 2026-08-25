# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Offline tests for the v5 self-contained consumer path: pull installs the shipped platform into
the bundle's own venv (mocked pip), serve runs the bundle's run.sh. No hardware, no network — the
HF download and the install.sh subprocess are stubbed.
"""

from pathlib import Path

from typer.testing import CliRunner

from tt_kernel import cli, localdb, metal, packaging, runtime
from tt_kernel.manifest import BundledPlatform, Manifest, WeightsRef, WheelArtifact

_runner = CliRunner()


def _fake_wheel(dirpath, filename, content=b"whl"):
    p = dirpath / filename
    p.write_bytes(content)
    return p


def _staged_bundle(tmp_path):
    """Stage a v5 self-contained bundle (ttnn wheel tagged for THIS interpreter so it's compatible)."""
    win = tmp_path / "in"
    win.mkdir()
    ttnn = _fake_wheel(win, f"ttnn-0.75.0-{packaging.host_python_tag()}-{packaging.host_python_tag()}-linux_x86_64.whl")
    plugin = _fake_wheel(win, "vllm_tt_plugin-0.3.0-py3-none-any.whl")
    metal_dir = tmp_path / "metal"
    (metal_dir).mkdir()
    (metal_dir / "requirements.txt").write_text("torch==2.11.0\n")
    staged = tmp_path / "snap"
    packaging.stage_package(
        staged, name="llama-3.2-3b-tt", arch="blackhole",
        ttnn_wheel=ttnn, plugin_wheel=plugin, metal_dir=metal_dir,
        vllm_metadata={"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"},
        tt_kernel_version="0.0.0", weights=WeightsRef(repo="unsloth/Llama-3.2-3B-Instruct"),
        tt_metal_version="0.75.0",
    )
    return staged


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # localdb
    monkeypatch.setenv("TT_MODEL_MODELS_DIR", str(tmp_path / "models"))  # install dir
    monkeypatch.setattr(metal, "local_env", lambda **k: metal.LocalEnv(arch="blackhole", device_count=1))


def test_glibc_floor_of_tag():
    assert packaging.glibc_floor_of_tag("manylinux_2_39_x86_64") == (2, 39)
    assert packaging.glibc_floor_of_tag("manylinux_2_35_x86_64") == (2, 35)
    assert packaging.glibc_floor_of_tag("manylinux2014_x86_64") == (2, 17)
    assert packaging.glibc_floor_of_tag("linux_x86_64") is None  # unrepaired: no declared floor
    assert packaging.glibc_floor_of_tag("any") is None


def test_host_incompatible_wheels_flags_glibc_too_old(monkeypatch):
    # A manylinux_2_39 wheel on a glibc-2.35 host (Ubuntu 22.04) must be flagged clearly.
    monkeypatch.setattr(packaging, "host_glibc", lambda: (2, 35))
    monkeypatch.setattr(packaging, "host_python_tag", lambda: "cp312")
    bundled = BundledPlatform(
        ttnn_wheel=WheelArtifact(path="wheels/ttnn-0.75.0-cp312-cp312-manylinux_2_39_x86_64.whl",
                                 sha256="x", python_tag="cp312", abi_tag="cp312",
                                 platform_tag="manylinux_2_39_x86_64"),
    )
    problems = packaging.host_incompatible_wheels(bundled)
    assert any("needs glibc >= 2.39" in p and "2.35" in p for p in problems)


def test_host_incompatible_wheels_ok_when_glibc_new_enough(monkeypatch):
    monkeypatch.setattr(packaging, "host_glibc", lambda: (2, 39))
    monkeypatch.setattr(packaging, "host_python_tag", lambda: "cp312")
    bundled = BundledPlatform(
        ttnn_wheel=WheelArtifact(path="wheels/ttnn-0.75.0-cp312-cp312-manylinux_2_35_x86_64.whl",
                                 sha256="x", python_tag="cp312", abi_tag="cp312",
                                 platform_tag="manylinux_2_35_x86_64"),
    )
    assert packaging.host_incompatible_wheels(bundled) == []


def test_host_incompatible_wheels_flags_python_mismatch():
    bundled = BundledPlatform(
        ttnn_wheel=WheelArtifact(path="wheels/ttnn-0.75.0-cp999-cp999-linux_x86_64.whl",
                                 sha256="x", python_tag="cp999", abi_tag="cp999", platform_tag="linux_x86_64"),
        plugin_wheel=WheelArtifact(path="wheels/p-0.3-py3-none-any.whl", sha256="y",
                                   python_tag="py3", abi_tag="none", platform_tag="any"),
    )
    problems = packaging.host_incompatible_wheels(bundled)
    assert any("cp999" in p for p in problems)  # ttnn flagged
    assert not any("py3-none-any" in p for p in problems)  # universal plugin skipped


def test_pull_installs_self_contained(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    staged = _staged_bundle(tmp_path)

    def _fake_download(repo_id, revision, dest):
        import shutil
        snap = Path(dest) / "snap"
        shutil.copytree(staged, snap)
        return snap

    installed = {}

    def _fake_install(bundle_dir, venv_dir):
        (venv_dir / "bin").mkdir(parents=True)
        py = venv_dir / "bin" / "python"
        py.write_text("#!/bin/sh\n")
        installed["dir"] = bundle_dir
        return py

    monkeypatch.setattr(cli.hub, "download_bundle", _fake_download)
    monkeypatch.setattr(cli.runtime, "install_self_contained", _fake_install)

    res = _runner.invoke(cli.app, ["pull", "myorg/llama-3.2-3b-tt"])
    assert res.exit_code == 0, res.output

    entry = localdb.get("myorg/llama-3.2-3b-tt")
    assert entry and entry["self_contained"] is True
    assert Path(entry["run_script"]).is_file()
    assert Path(entry["python"]).exists()
    # the platform install was invoked against the materialized folder
    assert installed["dir"] == Path(entry["install_dir"])
    assert (Path(entry["install_dir"]) / "wheels").is_dir()


def test_serve_self_contained_print(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    # record an installed self-contained bundle with a real run.sh
    inst = tmp_path / "models" / "myorg" / "llama-3.2-3b-tt"
    inst.mkdir(parents=True)
    run_sh = inst / "run.sh"
    run_sh.write_text("#!/usr/bin/env bash\necho serving\n")
    localdb.record("myorg/llama-3.2-3b-tt", {
        "repo_id": "myorg/llama-3.2-3b-tt", "self_contained": True,
        "install_dir": str(inst), "run_script": str(run_sh), "python": str(inst / "venv/bin/python"),
    })

    calls = {}

    def _fake_run(argv, **kw):
        calls["argv"] = argv
        calls["env"] = kw.get("env", {})

        class _R:
            returncode = 0

        return _R()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    res = _runner.invoke(cli.app, ["serve", "myorg/llama-3.2-3b-tt", "--print"])
    assert res.exit_code == 0, res.output
    # --print asks run.sh to echo the resolved command via TT_MODEL_PRINT=1
    assert calls["argv"][0] == "bash" and str(run_sh) in calls["argv"]
    assert calls["env"].get("TT_MODEL_PRINT") == "1"
