# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A v6 bundle keeps its HF weights in the shared HF cache, not inside the install folder.

An update replaces the install folder, so weights kept inside it were downloaded again on every
update. v5 bundles keep their per-folder cache.
"""

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_kernel import cli, container_cli, localdb, packaging
from tt_kernel.manifest import Manifest, Mesh, Producer, WeightsRef

runner = CliRunner()


def _thin_run_sh(tmp_path):
    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    staged = tmp_path / "bundle"
    packaging.stage_thin_package(
        staged, name="m", arch="blackhole", model_py=model_py, tt_kernel_version="0",
        vllm_metadata={"arch": "A", "main_class": "model:C"},
        weights=WeightsRef(repo="org/weights", revision="abc123"),
    )
    return staged


def _v5_run_sh():
    return packaging.render_run_sh(Manifest(
        schema_version="5", name="m", arch="blackhole", tt_metal_version="0.75.0",
        producer=Producer(tt_kernel_version="0.0.0", created_at="2026-08-20T00:00:00Z"),
        weights=WeightsRef(repo="org/model"), mesh=Mesh(devices=1, topology="P150"),
    ))


def _hf_home(run_sh: str, here: Path, env: dict) -> Path:
    """HF_HOME after run.sh's cache block runs in order (its XDG_CACHE_HOME override included)."""
    lines = run_sh.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("export HF_HOME="))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("export TORCHINDUCTOR_CACHE_DIR="))
    script = f"HERE={here}\n" + "\n".join(lines[start:end + 1]) + '\necho "$HF_HOME"'
    out = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True,
                         check=True).stdout.strip()
    return Path(out)


@pytest.mark.parametrize("xdg", [False, True])
def test_v6_run_sh_defaults_hf_home_to_the_shared_cache(tmp_path, xdg):
    staged = _thin_run_sh(tmp_path)
    env = {"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"}
    if xdg:
        env["XDG_CACHE_HOME"] = str(tmp_path / "xdg")
    want = (tmp_path / "xdg" if xdg else tmp_path / "home" / ".cache") / "huggingface"
    assert _hf_home((staged / "run.sh").read_text(), staged, env) == want


def test_v6_run_sh_still_honours_an_explicit_hf_home(tmp_path):
    staged = _thin_run_sh(tmp_path)
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "HF_HOME": str(tmp_path / "mine")}
    assert _hf_home((staged / "run.sh").read_text(), staged, env) == tmp_path / "mine"


def test_v5_run_sh_keeps_its_per_folder_cache(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin"}
    assert _hf_home(_v5_run_sh(), tmp_path / "install", env) == tmp_path / "install" / ".hf"


def _installed_thin(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    install = _thin_run_sh(tmp_path)
    localdb.record("org/bundle", {"install_dir": str(install), "self_contained": True})
    purged = []
    monkeypatch.setattr(container_cli, "_purge_hf", lambda r, w: purged.append(r))
    return install, purged


def test_rm_of_a_v6_bundle_keeps_its_shared_weights_by_default(tmp_path, monkeypatch):
    install, purged = _installed_thin(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rm", "org/bundle"])
    assert res.exit_code == 0, res.output
    assert not install.exists() and purged == []
    assert "weights kept (org/weights)" in res.output


def test_rm_include_weights_deletes_a_v6_bundles_weights(tmp_path, monkeypatch):
    install, purged = _installed_thin(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rm", "org/bundle", "--include-weights"])
    assert res.exit_code == 0, res.output
    assert not install.exists() and purged == ["org/weights"]
