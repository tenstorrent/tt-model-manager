# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`pull --with-weights` must put the weights where the bundle's run.sh will look for them.

run.sh defaults HF_HOME to <install>/.hf and the server loads weights by repo id, so weights
downloaded anywhere else are never read and the first serve downloads them all again.
"""

import os
import shutil
import subprocess
from pathlib import Path

import huggingface_hub
import pytest
from typer.testing import CliRunner

from tt_kernel import cli, metal, packaging
from tt_kernel.manifest import WeightsRef

_runner = CliRunner()


def _thin_bundle(tmp_path):
    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    staged = tmp_path / "snap_src"
    packaging.stage_thin_package(
        staged, name="m", arch="blackhole", model_py=model_py, tt_kernel_version="0",
        vllm_metadata={"arch": "A", "main_class": "model:C"},
        weights=WeightsRef(repo="org/weights", revision="abc123"),
    )
    return staged


def _serve_time_hub_cache(run_sh: Path, env: dict) -> Path:
    """Evaluate run.sh's own HF_HOME line the way bash will at serve time."""
    line = next(ln for ln in run_sh.read_text().splitlines() if ln.startswith("export HF_HOME="))
    out = subprocess.run(["bash", "-c", f'HERE={run_sh.parent}\n{line}\necho "$HF_HOME"'],
                         env=env, capture_output=True, text=True, check=True).stdout.strip()
    return Path(env.get("HF_HUB_CACHE") or Path(out) / "hub")


@pytest.mark.parametrize("user_hf_home", [False, True])
def test_pull_with_weights_fills_the_cache_run_sh_reads(monkeypatch, tmp_path, user_hf_home):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("TT_MODEL_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    if user_hf_home:
        monkeypatch.setenv("HF_HOME", str(tmp_path / "shared-hf"))
    else:
        monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(metal, "local_env", lambda **k: metal.LocalEnv(arch="blackhole", device_count=1))
    staged = _thin_bundle(tmp_path)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "bundlesha")

    def _download(repo_id, revision, dest):
        shutil.copytree(staged, Path(dest) / "snap")
        return Path(dest) / "snap"

    monkeypatch.setattr(cli.hub, "download_bundle", _download)
    monkeypatch.setattr(cli.runtime, "install_self_contained",
                        lambda bundle, venv: venv / "bin" / "python")
    seen = {}

    def _snapshot(**kw):
        seen.update(kw)
        return str(Path(kw["cache_dir"]) / "models--org--weights" / "snapshots" / "abc123")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _snapshot)

    res = _runner.invoke(cli.app, ["pull", "org/bundle", "--with-weights"])
    assert res.exit_code == 0, res.output
    install = tmp_path / "models" / "org" / "bundle"
    assert "local_dir" not in seen
    assert seen["revision"] == "abc123"
    assert Path(seen["cache_dir"]) == _serve_time_hub_cache(install / "run.sh", dict(os.environ))
