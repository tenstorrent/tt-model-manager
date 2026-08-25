# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Offline tests for v5 self-contained bundle staging (packaging.stage_package). No hardware,
no network: fake wheel files + a fake metal tree are staged and the running-folder layout +
manifest are asserted.
"""

import json

from typer.testing import CliRunner

from tt_kernel import cli, packaging
from tt_kernel.manifest import Capabilities, Manifest, Mesh, Producer, WeightsRef

_runner = CliRunner()


def test_parse_wheel_tags():
    t = packaging.parse_wheel_tags("ttnn-0.75.0-cp312-cp312-linux_x86_64.whl")
    assert t == {"python_tag": "cp312", "abi_tag": "cp312", "platform_tag": "linux_x86_64"}
    # build-tag variant + none-any plugin wheel
    assert packaging.parse_wheel_tags("vllm_tt_plugin-0.3.0-py3-none-any.whl")["platform_tag"] == "any"
    assert packaging.parse_wheel_tags("not-a-wheel.txt")["python_tag"] is None


def _fake_wheel(dirpath, filename, content=b"PK\x03\x04 fake wheel"):
    p = dirpath / filename
    p.write_bytes(content)
    return p


def _run_sh_manifest(**over):
    base = dict(
        schema_version="5",
        name="m",
        arch="blackhole",
        tt_metal_version="0.75.0",
        producer=Producer(tt_kernel_version="0.0.0", created_at="2026-08-20T00:00:00Z"),
        weights=WeightsRef(repo="org/model"),
        mesh=Mesh(devices=1, topology="P150"),
    )
    base.update(over)
    return Manifest(**base)


def test_render_run_sh_tool_parser_uses_vllm_flag_names():
    """The self-contained run.sh must emit the SAME vLLM flags the compose path does (PR #16).

    vLLM's FlexibleArgumentParser normalizes '_'->'-', so '--tool_parser' becomes the nonexistent
    '--tool-parser'; the real flag is '--tool-call-parser', and vLLM hard-errors on it without
    '--enable-auto-tool-choice'. This is the shipped launch path — before the fix it dropped the
    capability entirely, so a bundle declaring tool_parser silently got no tool calling.
    """
    run = packaging.render_run_sh(
        _run_sh_manifest(capabilities=Capabilities(tool_parser="qwen3_coder",
                                                   reasoning_parser="qwen3_coder"))
    )
    assert "--tool_parser" not in run and "--tool-parser" not in run
    assert "--enable-auto-tool-choice --tool-call-parser qwen3_coder" in run
    assert "--reasoning_parser qwen3_coder" in run


def test_render_run_sh_no_tool_flags_without_capability():
    """No tool_parser declared => neither flag appears (bare --enable-auto-tool-choice is an error)."""
    run = packaging.render_run_sh(_run_sh_manifest())
    assert "--enable-auto-tool-choice" not in run and "--tool-call-parser" not in run


def test_stage_package_layout(tmp_path):
    # fake author artifacts
    wheels = tmp_path / "in_wheels"
    wheels.mkdir()
    ttnn = _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn-bytes")
    plugin = _fake_wheel(wheels, "vllm_tt_plugin-0.3.0-py3-none-any.whl")
    metal = tmp_path / "metal_src"
    (metal / "models" / "tt_transformers" / "tt").mkdir(parents=True)
    (metal / "requirements.txt").write_text("torch==2.11.0\ntransformers==5.12.1\n")
    (metal / "__pycache__").mkdir()
    (metal / "__pycache__" / "junk.pyc").write_bytes(b"junk")  # must be ignored
    (metal / "models" / "tt_transformers" / "tt" / "attention.py").write_text("# blocks\n")

    vmeta = {"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"}
    staged = tmp_path / "staged"

    manifest = packaging.stage_package(
        staged,
        name="llama-3.2-3b-tt",
        arch="blackhole",
        ttnn_wheel=ttnn,
        plugin_wheel=plugin,
        metal_dir=metal,
        vllm_metadata=vmeta,
        tt_kernel_version="0.0.0",
        weights=WeightsRef(repo="unsloth/Llama-3.2-3B-Instruct"),
        mesh=Mesh(devices=1, topology="P150"),
        env={"HF_HUB_ENABLE_HF_TRANSFER": "1"},
        tt_metal_version="0.75.0",
    )

    # layout
    assert (staged / "wheels" / "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl").read_bytes() == b"ttnn-bytes"
    assert (staged / "wheels" / "vllm_tt_plugin-0.3.0-py3-none-any.whl").is_file()
    assert (staged / "metal" / "models" / "tt_transformers" / "tt" / "attention.py").is_file()
    assert not (staged / "metal" / "__pycache__").exists()  # ignored
    assert (staged / "requirements.txt").read_text().startswith("torch==2.11.0")
    # vllm_metadata.json lives in a per-model SUBFOLDER under vllm_models/ (EXTRA_MODELS_DIR contract)
    meta = staged / "vllm_models" / "llama-3.2-3b-tt" / "vllm_metadata.json"
    assert meta.is_file()
    assert json.loads(meta.read_text())["arch"] == "LlamaForCausalLM"
    assert not (staged / "vllm_metadata.json").exists()  # NOT at the root (plugin would miss it)
    for s in ("install.sh", "run.sh"):
        assert (staged / s).stat().st_mode & 0o111  # executable
    # run.sh wired the non-obvious env + serving args the TT backend requires
    run = (staged / "run.sh").read_text()
    assert "_ttnncpp" in run and "TT_VLLM_BUILTIN_MODELS=0" in run
    assert 'EXTRA_MODELS_DIR="$HERE/vllm_models"' in run
    assert "find_spec" in run                      # ttnn located without importing it
    assert 'export HF_MODEL=' in run               # adapter reads HF_MODEL from env
    assert "--max_num_seqs 32" in run and "--block_size 64" in run  # TT backend defaults
    assert "unsloth/Llama-3.2-3B-Instruct" in run and "P150" in run

    # manifest
    m2 = Manifest.from_json((staged / "tt_kernel_manifest.json").read_text())
    assert m2.schema_version == "5"
    assert m2.is_self_contained is True
    assert m2.bundled.ttnn_wheel.python_tag == "cp312"
    assert m2.bundled.ttnn_wheel.sha256 == packaging.sha256_file(ttnn)
    assert [w.path for w in m2.bundled.wheels] == [
        "wheels/ttnn-0.75.0-cp312-cp312-linux_x86_64.whl",
        "wheels/vllm_tt_plugin-0.3.0-py3-none-any.whl",
    ]
    assert m2.entrypoint.arch_name == "LlamaForCausalLM"
    assert m2.weights.repo_id == "unsloth/Llama-3.2-3B-Instruct"


def test_cli_package_stage_only(tmp_path):
    """`tt-model package ... --out <dir>` (no repo_id) stages the folder, no network."""
    wheels = tmp_path / "w"
    wheels.mkdir()
    _fake_wheel(wheels, "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl", b"ttnn")
    _fake_wheel(wheels, "vllm_tt_plugin-0.3.0-py3-none-any.whl")
    metal = tmp_path / "metal"
    metal.mkdir()
    (metal / "requirements.txt").write_text("torch==2.11.0\n")
    out = tmp_path / "staged"

    res = _runner.invoke(
        cli.app,
        [
            "package",
            "--from-metal", str(metal),
            "--wheels-dir", str(wheels),
            "--arch", "blackhole",
            "--arch-name", "LlamaForCausalLM",
            "--main-class", "generator_vllm:LlamaForCausalLM",
            "--weights", "unsloth/Llama-3.2-3B-Instruct",
            "--mesh", "P150",
            "--no-repair", "--no-vendor-deps",
            "--out", str(out),
        ],
    )
    assert res.exit_code == 0, res.output
    m = Manifest.from_json((out / "tt_kernel_manifest.json").read_text())
    assert m.is_self_contained and m.arch == "blackhole"
    # wheels_dir auto-classified ttnn + plugin
    assert m.bundled.ttnn_wheel is not None and m.bundled.plugin_wheel is not None
    assert (out / "wheels" / "ttnn-0.75.0-cp312-cp312-linux_x86_64.whl").is_file()


def test_cli_package_requires_ttnn_wheel(tmp_path):
    metal = tmp_path / "metal"
    metal.mkdir()
    res = _runner.invoke(
        cli.app,
        ["package", "--from-metal", str(metal), "--arch", "blackhole",
         "--arch-name", "X", "--main-class", "m:C", "--out", str(tmp_path / "s")],
    )
    assert res.exit_code != 0
