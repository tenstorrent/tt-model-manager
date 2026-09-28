# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for v6 "thin" bundles (issue #29): the per-model venv is built from pip dependency pins
(ttnn / TTTv2 / models wheel) + optional generic_op wheels, not from embedded platform wheels.
No hardware, no network — staging + rendering + compat rules only.

DRAFT: this reflects the #29 plan; the bundle becomes fully installable once TTTv2 and the models
wheel are published so requirements can pin real versions.
"""

import json
import shlex
import subprocess
import sys

from typer.testing import CliRunner

from tt_kernel import cli, metal, packaging
from tt_kernel.manifest import Manifest, Mesh, Resources, WeightsRef, compare

_runner = CliRunner()


def _stage_thin(tmp_path, requirements=None, plugin_wheel=None, extra_wheels=None,
                models_wheels=None, vllm_wheel=None, with_vllm=True, constraints=None,
                vllm_common_requirements=None):
    model_py = tmp_path / "model.py"
    model_py.write_text("class QwenForCausalLM:  # the runner\n    pass\n")
    staged = tmp_path / "thin"
    m = packaging.stage_thin_package(
        staged, name="qwen-thin", arch="blackhole", model_py=model_py,
        vllm_metadata={"arch": "QwenForCausalLM", "main_class": "model:QwenForCausalLM"},
        tt_kernel_version="0.0.0", requirements=requirements,
        constraints=constraints,
        plugin_wheel=plugin_wheel, extra_wheels=extra_wheels,
        models_wheels=models_wheels,
        vllm_wheel=vllm_wheel, vllm_common_requirements=vllm_common_requirements,
        with_vllm=with_vllm,
        weights=WeightsRef(repo="Qwen/Qwen3-4B"), mesh=Mesh(devices=1, topology="P150"),
        resources=Resources(max_num_seqs=32, block_size=64),
    )
    return staged, m


def test_thin_layout_and_manifest(tmp_path):
    staged, m = _stage_thin(tmp_path)
    # manifest
    assert m.schema_version == "6"
    assert m.is_thin is True and m.has_own_venv is True and m.is_self_contained is False
    assert m.deps.requirements == "requirements.txt" and m.deps.wheels_dir is None and m.deps.wheels == []
    assert m.entrypoint.arch_name == "QwenForCausalLM"
    assert m.weights.repo_id == "Qwen/Qwen3-4B"
    # vLLM: empty-target build recorded (no wheel by default -> built from source), overrides shipped
    assert m.deps.vllm is not None and m.deps.vllm.target_device == "empty" and m.deps.vllm.wheel is None
    assert m.deps.vllm.overrides == "vllm-overrides.txt"
    # layout: model.py + requirements + vllm-overrides + metadata subfolder + scripts; NO wheels/ or metal/
    assert (staged / "model.py").is_file()
    assert (staged / "requirements.txt").is_file()
    assert (staged / "vllm-overrides.txt").is_file()
    assert (staged / "vllm_models" / "qwen-thin" / "vllm_metadata.json").is_file()
    assert not (staged / "wheels").exists() and not (staged / "metal").exists()
    m2 = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    assert m2.is_thin and m2.deps.python == "3.12"


def test_thin_default_requirements_template_has_todo_pins(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    req = (staged / "requirements.txt").read_text()
    assert "ttnn>=0.77" in req                       # engine resolves from PyPI today
    assert "tt-metal-models" in req                  # the models wheel (incl. tt_transformers)
    assert "tt-metal#54478" in req                   # tracks the upstream packaging PR (#29 M0)
    assert "SFPI" in req and "NOT listed" in req     # SFPI is an external box dep
    # vLLM must NOT be a pin here — it's the empty-target build done by install.sh; the plugin ships
    # as a bundled wheel. A resolvable `vllm` pin would clobber the empty build with the CUDA wheel.
    assert "VLLM_TARGET_DEVICE=empty" in req
    assert "vllm-tt-plugin" in req
    assert "\nvllm==" not in req and "\nvllm>=" not in req  # never a resolvable vllm pin


def test_thin_ships_vllm_overrides_matching_the_plugin(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    ov = (staged / "vllm-overrides.txt").read_text()
    # These are the exact pins the plugin's docs/vllm-overrides.txt uses so ttnn's numpy<2 survives.
    assert "opencv-python-headless==4.11.0.86" in ov
    assert "numpy>=1.24.4,<2" in ov


def test_thin_install_sh_builds_empty_target_vllm_from_source(tmp_path):
    staged, _ = _stage_thin(tmp_path)   # no --vllm-wheel -> from source
    inst = (staged / "install.sh").read_text()
    # ttnn/models install BEFORE vLLM (torch + numpy<2 first), then the empty-target vLLM steps.
    i_req = inst.index('-r "$HERE/requirements.txt"')
    i_vllm = inst.index("VLLM_TARGET_DEVICE=empty")
    assert i_req < i_vllm
    # common deps under the TT override set, then vLLM from source with --no-deps --no-binary.
    assert '--override "$HERE/vllm-overrides.txt"' in inst
    assert "requirements/common.txt" in inst                 # fetched from the pinned upstream tag
    assert "--no-deps --no-binary vllm vllm==0.25.1" in inst


def test_thin_prebuilt_vllm_wheel_installed_by_path_not_built(tmp_path):
    vw = tmp_path / "vllm-0.25.1-cp312-cp312-linux_x86_64.whl"; vw.write_bytes(b"PK\x03\x04")
    staged, m = _stage_thin(tmp_path, vllm_wheel=vw)
    assert m.deps.vllm.wheel == f"wheels/{vw.name}"
    assert (staged / "wheels" / vw.name).is_file()
    inst = (staged / "install.sh").read_text()
    assert "--no-binary vllm" not in inst                    # not built from source
    assert f'--no-deps "$HERE/wheels/{vw.name}"' in inst     # installed by path


def test_thin_stages_resolved_constraints_and_pinned_vllm_common(tmp_path):
    constraints = tmp_path / "qualification.lock"
    constraints.write_text("numpy==1.26.4\nrequests==2.34.2\n")
    common = tmp_path / "common.txt"
    common.write_text("requests>=2.26.0\n")

    staged, manifest = _stage_thin(
        tmp_path,
        constraints=constraints,
        vllm_common_requirements=common,
    )

    assert (staged / "constraints.txt").read_bytes() == constraints.read_bytes()
    assert (staged / "vllm-common.txt").read_bytes() == common.read_bytes()
    assert manifest.deps.vllm.common_requirements == "vllm-common.txt"
    install = (staged / "install.sh").read_text()
    assert "raw.githubusercontent.com" not in install
    assert install.count('"${TT_MODEL_CONSTRAINTS_ARGS[@]}"') == 3
    assert '--constraint "$HERE/constraints.txt"' in install
    assert '-r "$HERE/vllm-common.txt"' in install


def test_thin_no_vllm_skips_the_vllm_step(tmp_path):
    staged, m = _stage_thin(tmp_path, with_vllm=False)
    assert m.deps.vllm is None
    assert not (staged / "vllm-overrides.txt").exists()
    inst = (staged / "install.sh").read_text()
    assert "VLLM_TARGET_DEVICE" not in inst and "requirements/common.txt" not in inst


def test_thin_install_sh_builds_venv_from_pins(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    inst = (staged / "install.sh").read_text()
    assert "uv venv --relocatable" in inst and 'UV_PYTHON_INSTALL_DIR="$HERE/.python"' in inst
    assert 'UV_PYTHON_BIN_DIR="$HERE/.python/bin"' in inst
    assert 'UV_CACHE_DIR="$HERE/.uv-cache"' in inst
    assert "-r \"$HERE/requirements.txt\"" in inst   # installs from the pins
    assert "--no-index" not in inst                  # thin pulls ttnn/TTTv2 from the index
    assert "wheels/" not in inst                     # no embedded platform wheels


def test_thin_installer_can_pin_uv_without_new_schema_field(tmp_path):
    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    staged = tmp_path / "thin"
    packaging.stage_thin_package(
        staged,
        name="uv-pinned",
        arch="blackhole",
        model_py=model_py,
        vllm_metadata={"arch": "QwenForCausalLM", "main_class": "model:C"},
        tt_kernel_version="0.0.0",
        env={"TT_MODEL_UV_VERSION": "0.12.11"},
    )
    install = (staged / "install.sh").read_text()
    assert 'UVVER="0.12.11"' in install
    assert "https://astral.sh/uv/0.12.11/install.sh" in install
    assert "awk '{print $2}'" in install


def test_thin_validates_exact_external_sfpi_before_vllm(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    inst = (staged / "install.sh").read_text()
    assert inst.index('sfpi-version') > inst.index('-r "$HERE/requirements.txt"')
    assert inst.index('sfpi-version') < inst.index("VLLM_TARGET_DEVICE=empty")
    assert "SFPI mismatch" in inst and "sfpi_version" in inst and "sfpi_build" in inst
    assert 'local_sfpi="$TTNN_DIR/runtime/sfpi"' in inst
    assert "system_sfpi=/opt/tenstorrent/sfpi" in inst


def test_thin_reconciles_vllm_override_metadata_then_checks_environment(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    inst = (staged / "install.sh").read_text()
    reconcile = inst.index("reconciled vLLM dependency metadata")
    plugin_install = inst.index('"$HERE/wheels/vllm_tt_plugin') if "vllm_tt_plugin" in inst else -1
    check = inst.index('uv pip check --python "$PYBIN"')
    assert reconcile > plugin_install
    assert check > reconcile
    assert "opencv-python-headless==" in inst
    assert 'distribution("vllm")._path' in inst
    assert "urlsafe_b64encode" in inst and 'record_key = f"{metadata.parent.name}/METADATA"' in inst
    assert 'lineterminator="\\n"' in inst


def test_thin_run_pins_runtime_root_and_revalidates_sfpi(tmp_path):
    staged, _ = _stage_thin(tmp_path)
    run = (staged / "run.sh").read_text()
    assert 'export TT_METAL_RUNTIME_ROOT="$TTNN_DIR"' in run
    assert run.index('export TT_METAL_RUNTIME_ROOT="$TTNN_DIR"') < run.index("sfpi-version")
    assert run.index("sfpi-version") < run.index("LD_PRELOAD=")
    assert 'export TT_METAL_CACHE="${TT_METAL_CACHE:-$HERE/.cache}"' in run


def test_sfpi_gate_prefers_local_and_rejects_wrong_build(tmp_path):
    """The rendered gate checks the same local-first path that TT-Metal will select."""
    from tt_kernel.packaging import _render_sfpi_validation

    ttnn_dir = tmp_path / "ttnn"
    (ttnn_dir / "tt_metal").mkdir(parents=True)
    (ttnn_dir / "tt_metal" / "sfpi-version").write_text(
        "sfpi_version='7.78.0'\nsfpi_build='935'\n"
    )
    compiler = ttnn_dir / "runtime" / "sfpi" / "compiler" / "bin" / "riscv-tt-elf-g++"
    compiler.parent.mkdir(parents=True)
    (ttnn_dir / "runtime" / "sfpi" / "include").mkdir(parents=True)
    (ttnn_dir / "runtime" / "sfpi" / "include" / "sfpi_lib.h").write_text("// test\n")
    compiler.write_text("#!/bin/sh\necho 'riscv-tt-elf-g++ (tenstorrent/sfpi:7.60.0[704]) 15.1.0'\n")
    compiler.chmod(0o700)
    script = tmp_path / "gate.sh"
    script.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        f"PYBIN={shlex.quote(sys.executable)}\nTTNN_DIR={shlex.quote(str(ttnn_dir))}\n"
        + _render_sfpi_validation(discover_ttnn=False)
    )
    wrong = subprocess.run(["bash", str(script)], text=True, capture_output=True)
    assert wrong.returncode != 0
    assert "need 7.78.0[935]" in wrong.stderr and str(ttnn_dir / "runtime" / "sfpi") in wrong.stderr

    compiler.write_text("#!/bin/sh\necho 'riscv-tt-elf-g++ (tenstorrent/sfpi:7.78.0[935]) 15.1.0'\n")
    ok = subprocess.run(["bash", str(script)], text=True, capture_output=True)
    assert ok.returncode == 0, ok.stderr
    assert f"validated SFPI 7.78.0[935] at {ttnn_dir}/runtime/sfpi" in ok.stderr


def test_thin_ships_plugin_and_ops_as_wheels_by_path(tmp_path):
    # No custom vLLM fork — the vLLM integration is vllm-tt-plugin, shipped as a wheel.
    pw = tmp_path / "vllm_tt_plugin-0.1.0-py3-none-any.whl"; pw.write_bytes(b"PK\x03\x04")
    ow = tmp_path / "my_ops-0.1-py3-none-any.whl"; ow.write_bytes(b"PK\x03\x04")
    staged, m = _stage_thin(tmp_path, plugin_wheel=pw, extra_wheels=[ow])
    # recorded in deps.wheels in order — plugin, then ops — and shipped in wheels/ (no vllm fork)
    assert m.deps.wheels == [f"wheels/{pw.name}", f"wheels/{ow.name}"]
    assert m.deps.wheels_dir == "wheels"
    assert not any("vllm-" in w and "plugin" not in w for w in m.deps.wheels)  # no bare vllm fork wheel
    for w in (pw, ow):
        assert (staged / "wheels" / w.name).is_file()
    inst = (staged / "install.sh").read_text()
    assert f'"$HERE/wheels/{pw.name}"' in inst and f'"$HERE/wheels/{ow.name}"' in inst
    assert '--find-links "$HERE/wheels"' in inst
    assert '-r "$HERE/requirements.txt"' in inst


def test_thin_models_wheel_is_installed_by_exact_path(tmp_path):
    # The staged wheel must win even when an index carries the same package version.
    mw = tmp_path / "tt_metal_models-0.77.0-py3-none-any.whl"; mw.write_bytes(b"PK\x03\x04")
    staged, m = _stage_thin(tmp_path, models_wheels=[mw])
    assert m.deps.models_wheels == [f"wheels/{mw.name}"]
    assert m.deps.wheels == []
    assert m.deps.wheels_dir == "wheels"
    assert (staged / "wheels" / mw.name).is_file()
    inst = (staged / "install.sh").read_text()
    req_line = next(line for line in inst.splitlines() if '-r "$HERE/requirements.txt"' in line)
    assert '--find-links "$HERE/wheels"' in req_line
    assert f'"$HERE/wheels/{mw.name}"' in req_line
    assert inst.index(req_line) < inst.index("VLLM_TARGET_DEVICE=empty")


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
    ok = compare(m, metal.LocalEnv(arch="blackhole", device_count=1))
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


def _plain(text: str) -> str:
    """Collapse rich's box-drawing + wrapping so a phrase can be matched across lines."""
    return " ".join(text.replace("│", " ").replace("─", " ").split())


def test_cli_package_thin_is_marked_beta_and_unsupported(tmp_path):
    # v6 thin is a draft format: the CLI must say so both in `--help` (so it is visible before
    # anyone runs it) and on every run (so a staged bundle is never mistaken for supported output).
    top = _plain(_runner.invoke(cli.app, ["--help"]).output)
    assert "BETA" in top.split("package-thin", 1)[1][:120]

    cmd_help = _plain(_runner.invoke(cli.app, ["package-thin", "--help"]).output)
    assert "BETA" in cmd_help and "NOT SUPPORTED" in cmd_help.upper()

    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    res = _runner.invoke(cli.app, [
        "package-thin", "--model-py", str(model_py), "--arch", "blackhole",
        "--arch-name", "QwenForCausalLM", "--main-class", "model:C",
        "--out", str(tmp_path / "staged"),
    ])
    assert res.exit_code == 0, res.output
    assert "package-thin is BETA and not supported" in _plain(res.output)
