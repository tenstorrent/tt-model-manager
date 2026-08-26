# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for v6 "thin" bundles (issue #29): the per-model venv is built from pip dependency pins
(ttnn / TTTv2 / models wheel) + optional generic_op wheels, not from embedded platform wheels.
No hardware, no network — staging + rendering + compat rules only.

DRAFT: this reflects the #29 plan; the bundle becomes fully installable once TTTv2 and the models
wheel are published so requirements can pin real versions.
"""

import json

from typer.testing import CliRunner

from tt_kernel import cli, metal, packaging
from tt_kernel.manifest import Manifest, Mesh, Resources, WeightsRef, compare

_runner = CliRunner()


def _stage_thin(tmp_path, requirements=None, wheels_dir=None):
    model_py = tmp_path / "model.py"
    model_py.write_text("class QwenForCausalLM:  # the runner\n    pass\n")
    staged = tmp_path / "thin"
    m = packaging.stage_thin_package(
        staged, name="qwen-thin", arch="blackhole", model_py=model_py,
        vllm_metadata={"arch": "QwenForCausalLM", "main_class": "model:QwenForCausalLM"},
        tt_kernel_version="0.0.0", requirements=requirements, wheels_dir=wheels_dir,
        weights=WeightsRef(repo="Qwen/Qwen3-4B"), mesh=Mesh(devices=1, topology="P150"),
        resources=Resources(max_num_seqs=32, block_size=64),
    )
    return staged, m


def test_thin_layout_and_manifest(tmp_path):
    staged, m = _stage_thin(tmp_path)
    # manifest
    assert m.schema_version == "6"
    assert m.is_thin is True and m.has_own_venv is True and m.is_self_contained is False
    assert m.deps.requirements == "requirements.txt" and m.deps.wheels_dir is None
    assert m.entrypoint.arch_name == "QwenForCausalLM"
    assert m.weights.repo_id == "Qwen/Qwen3-4B"
    # layout: model.py + requirements + metadata subfolder + scripts; NO wheels/ or metal/
    assert (staged / "model.py").is_file()
    assert (staged / "requirements.txt").is_file()
    assert (staged / "vllm_models" / "qwen-thin" / "vllm_metadata.json").is_file()
    assert not (staged / "wheels").exists() and not (staged / "metal").exists()
    m2 = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    assert m2.is_thin and m2.deps.python == "3.12"


def test_thin_default_requirements_template_has_todo_pins(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    req = (staged / "requirements.txt").read_text()
    assert "ttnn>=0.77" in req                       # engine resolves from PyPI today
    assert "tt-metal-models" in req                  # the models wheel (incl. tt_transformers)
    assert "tt-metal#54340" in req                   # tracks the upstream packaging PR (#29 M0)
    assert "SFPI" in req and "NOT listed" in req     # SFPI is an external box dep


def test_thin_install_sh_builds_venv_from_pins(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    inst = (staged / "install.sh").read_text()
    assert "uv venv --relocatable" in inst and 'UV_PYTHON_INSTALL_DIR="$HERE/.python"' in inst
    assert "-r \"$HERE/requirements.txt\"" in inst   # installs from the pins
    assert "--no-index" not in inst                  # thin pulls ttnn/TTTv2 from the index
    assert "wheels/" not in inst                     # no embedded platform wheels


def test_thin_install_sh_find_links_when_wheels_shipped(tmp_path):
    wd = tmp_path / "ops"
    wd.mkdir()
    (wd / "my_ops-0.1-py3-none-any.whl").write_bytes(b"PK\x03\x04")
    staged, m = _stage_thin(tmp_path, wheels_dir=wd)
    assert m.deps.wheels_dir == "custom_ops"
    assert (staged / "custom_ops" / "my_ops-0.1-py3-none-any.whl").is_file()
    inst = (staged / "install.sh").read_text()
    assert '--find-links "$HERE/custom_ops"' in inst  # bundled generic_op wheel is discoverable


def test_thin_scripts_are_owner_rw_only_not_executable(tmp_path):
    # Least privilege (Cycode SAST): the generated scripts are run via `bash <script>`, so they need
    # no execute bit and no group/other access — mode 0o600.
    staged, _ = _stage_thin(tmp_path)
    for s in ("install.sh", "run.sh"):
        mode = (staged / s).stat().st_mode & 0o777
        assert mode == 0o600, f"{s} is {oct(mode)}, expected 0o600"


def test_thin_run_sh_pythonpath_is_bundle_root_not_metal(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    run = (staged / "run.sh").read_text()
    assert 'export PYTHONPATH="$HERE:' in run          # model.py imports from the bundle root
    assert "$HERE/metal" not in run                    # v6 has no embedded metal tree
    assert "_ttnncpp" in run and 'EXTRA_MODELS_DIR="$HERE/vllm_models"' in run  # engine wiring unchanged
    assert "Qwen/Qwen3-4B" in run


def test_thin_compare_gates_only_on_arch_and_device_count(tmp_path):
    _, m = _stage_thin(tmp_path)
    # matching arch -> compatible (no host tt-metal/version gates for a thin bundle)
    ok = compare(m, metal.LocalEnv(arch="blackhole", device_count=1, tt_metal_version="0.99-different"))
    assert ok.compatible is True and ok.issues == []
    # wrong arch is still fatal
    bad = compare(m, metal.LocalEnv(arch="wormhole_b0", device_count=1))
    assert bad.has_fatal and any(i.field == "arch" for i in bad.issues)


def test_cli_package_thin_stage_only(tmp_path):
    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    out = tmp_path / "staged"
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--arch-name", "QwenForCausalLM", "--main-class", "model:C",
        "--weights", "Qwen/Qwen3-4B", "--mesh", "P150", "--out", str(out),
    ])
    assert res.exit_code == 0, res.output
    m = Manifest.from_json((out / "tt_kernel_manifest.json").read_text())
    assert m.is_thin and m.arch == "blackhole"
    assert (out / "model.py").is_file() and (out / "requirements.txt").is_file()
