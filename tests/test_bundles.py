# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for vLLM bundle materialization (bundles.py), the serve helpers (runtime.py),
and the ``serve`` CLI flow. No hardware, no network.
"""

import json

from typer.testing import CliRunner

from tt_kernel import bundles, cli, localdb, runtime

runner = CliRunner()


def _make_bundle_folder(tmp_path, **meta):
    src = tmp_path / "src_bundle"
    src.mkdir()
    (src / bundles.VLLM_METADATA_NAME).write_text(json.dumps(meta))
    (src / "generator_vllm.py").write_text("# adapter code\n")
    return src


# --------------------------------------------------------------- resolve_bundles_dir
def test_resolve_bundles_dir_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv(bundles.ENV_BUNDLES_DIR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert bundles.resolve_bundles_dir() == tmp_path / ".cache" / "tt-model" / "bundles"
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "envdir"))
    assert bundles.resolve_bundles_dir() == tmp_path / "envdir"
    assert bundles.resolve_bundles_dir(str(tmp_path / "flagdir")) == tmp_path / "flagdir"


def test_model_key_flattens_namespace():
    assert bundles.model_key("org/Model-8B") == "org__Model-8B"
    assert bundles.model_key("bare") == "bare"


# --------------------------------------------------------------- install / remove
def test_install_and_remove_bundle(tmp_path):
    src = _make_bundle_folder(tmp_path, arch="LlamaForCausalLM", main_class="m:C")
    bdir = tmp_path / "bundles"
    dest = bundles.install_bundle(src, bdir, "org__m")
    assert (dest / bundles.VLLM_METADATA_NAME).is_file()
    assert (dest / "generator_vllm.py").is_file()
    # Re-install overwrites cleanly.
    dest2 = bundles.install_bundle(src, bdir, "org__m")
    assert dest2 == dest
    assert bundles.remove_bundle(bdir, "org__m") is True
    assert not dest.exists()
    assert bundles.remove_bundle(bdir, "org__m") is False


# --------------------------------------------------------------- read_vllm_metadata
def test_read_vllm_metadata(tmp_path):
    src = _make_bundle_folder(tmp_path, arch="LlamaForCausalLM", main_class="m:C",
                              hf_weights="org/w", launch={"default": {"command": ["x"]}})
    md = bundles.read_vllm_metadata(src)
    assert md.arch == "LlamaForCausalLM" and md.main_class == "m:C"
    assert md.hf_weights == "org/w" and "default" in md.launch


def test_read_vllm_metadata_missing(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        bundles.read_vllm_metadata(tmp_path)


# --------------------------------------------------------------- machine selection
def test_machine_candidates_order(monkeypatch):
    monkeypatch.setenv(bundles.ENV_MACHINE, "custom")
    monkeypatch.setattr(bundles.device, "detect",
                        lambda arch_override=None: bundles.device.DeviceInfo(arch="blackhole", device_count=1))
    cands = bundles.machine_candidates()
    assert cands[0] == "custom"
    assert "blackhole-1card" in cands and "blackhole" in cands and cands[-1] == "default"


def test_select_launch_match_and_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv(bundles.ENV_MACHINE, raising=False)
    monkeypatch.setattr(bundles.device, "detect",
                        lambda arch_override=None: bundles.device.DeviceInfo(arch="blackhole", device_count=1))
    md = bundles.VllmMetadata(raw={"launch": {
        "blackhole": {"command": ["a"], "env": {"K": "V"}},
        "default": {"command": ["b"]},
    }})
    key, spec = bundles.select_launch(md)
    assert key == "blackhole" and spec.command == ["a"] and spec.env == {"K": "V"}

    md2 = bundles.VllmMetadata(raw={"launch": {"default": {"command": ["b"]}}})
    key2, spec2 = bundles.select_launch(md2)
    assert key2 == "default" and spec2.command == ["b"]

    md3 = bundles.VllmMetadata(raw={"launch": {}})
    assert bundles.select_launch(md3) == (None, None)


# --------------------------------------------------------------- runtime helpers
def test_vllm_serve_env_overlays(monkeypatch):
    env = runtime.vllm_serve_env("/bundles", {"MESH_DEVICE": "P150", "VLLM_USE_V1": "1"})
    assert env[runtime.ENV_EXTRA_MODELS_DIR] == "/bundles"
    assert env["MESH_DEVICE"] == "P150" and env["VLLM_USE_V1"] == "1"


def test_vllm_serve_argv_python_override():
    argv = runtime.vllm_serve_argv(["python3", "server.py", "--model", "x"], python="/venv/bin/python")
    assert argv[0] == "/venv/bin/python"
    argv2 = runtime.vllm_serve_argv(["server", "--x"], python="/venv/bin/python")
    assert argv2[0] == "server"  # only swapped when first token is python/python3


# --------------------------------------------------------------- serve CLI flow
def _seed_vllm_installed(repo_id, bundle_path):
    localdb.record(repo_id, {
        "name": repo_id.split("/")[-1], "backend": "vllm", "build_key": None,
        "arch": "blackhole", "bundle_path": str(bundle_path), "installed_at": "now",
    })


def test_serve_print_local_only(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))  # isolate localdb
    # Lay a bundle folder into a bundles_dir with a 'default' launch entry.
    src = _make_bundle_folder(
        tmp_path, arch="LlamaForCausalLM",
        main_class="models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
        launch={"default": {"command": ["python3", "server_example_tt.py", "--model",
                                         "org/m", "--port", "8100"],
                            "env": {"MESH_DEVICE": "P150"}}},
    )
    dest = bundles.install_bundle(src, tmp_path / "bundles", "org__m")
    _seed_vllm_installed("org/m", dest)

    res = runner.invoke(cli.app, ["serve", "org/m", "--print", "--local-only"])
    assert res.exit_code == 0, res.output
    assert "EXTRA_MODELS_DIR=" in res.output
    assert "server_example_tt.py" in res.output
    assert "MESH_DEVICE=P150" in res.output
    assert "http://localhost:8100" in res.output  # endpoint parsed from --port


def test_serve_local_only_missing_bundle_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    res = runner.invoke(cli.app, ["serve", "org/absent", "--print", "--local-only"])
    assert res.exit_code == 1


# --------------------------------------------------- push -> pull -> serve round-trip
def _fake_hub(monkeypatch):
    """Redirect hub I/O to a local 'remote' dir so push/pull need no network."""
    import shutil
    from tt_kernel import hub, metal
    from tt_kernel.device import DeviceInfo
    remotes = {}

    def push_folder(repo_id, staged, commit_message=""):
        dst = remotes[repo_id] = staged.parent / f"remote__{bundles.model_key(repo_id)}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(staged, dst)

    def download_bundle(repo_id, revision, dest):
        shutil.copytree(remotes[repo_id], dest, dirs_exist_ok=True)
        return __import__("pathlib").Path(dest)

    # The repo does not exist yet, so push takes the create path and never touches
    # visibility on an existing repo (see cli._ensure_repo).
    monkeypatch.setattr(hub, "repo_exists", lambda *a, **k: False)
    monkeypatch.setattr(hub, "create_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "set_visibility", lambda *a, **k: None)
    monkeypatch.setattr(hub, "tag_repo", lambda *a, **k: None)
    monkeypatch.setattr(hub, "push_folder", push_folder)
    monkeypatch.setattr(hub, "download_bundle", download_bundle)
    # Deterministic env: pretend a single blackhole card + known tt-metal version.
    monkeypatch.setattr(metal, "detect_device",
                        lambda arch_override=None: DeviceInfo(arch="blackhole", device_count=1, source="test"))
    monkeypatch.setattr(metal, "resolve_version", lambda: "0.72.0")
    monkeypatch.setattr(metal, "local_env",
                        lambda **k: metal.LocalEnv(tt_metal_version="0.72.0", arch="blackhole",
                                                   device_count=1, harvesting_mask=0, build_key=None))


def test_push_pull_serve_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv(bundles.ENV_BUNDLES_DIR, str(tmp_path / "bundles"))
    _fake_hub(monkeypatch)

    src = _make_bundle_folder(
        tmp_path, arch="LlamaForCausalLM",
        main_class="models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
        hf_weights="meta-llama/Llama-3.1-8B-Instruct",
        launch={"default": {"command": ["python3", "server_example_tt.py", "--model",
                                        "meta-llama/Llama-3.1-8B-Instruct"], "env": {}}},
    )

    push = runner.invoke(cli.app, ["push", "acme/llama", "--private", "--backend", "vllm",
                                   "--bundle-dir", str(src)])
    assert push.exit_code == 0, push.output
    assert "Pushed vLLM bundle" in push.output

    pull = runner.invoke(cli.app, ["pull", "acme/llama"])
    assert pull.exit_code == 0, pull.output
    assert "vLLM bundle ->" in pull.output

    # Installed and recorded with the vllm backend.
    entry = localdb.get("acme/llama")
    assert entry and entry["backend"] == "vllm" and entry["bundle_path"]

    serve = runner.invoke(cli.app, ["serve", "acme/llama", "--print", "--local-only"])
    assert serve.exit_code == 0, serve.output
    assert "server_example_tt.py" in serve.output and "EXTRA_MODELS_DIR=" in serve.output

    # rm removes the folder + index entry.
    rm = runner.invoke(cli.app, ["rm", "acme/llama"])
    assert rm.exit_code == 0, rm.output
    assert localdb.get("acme/llama") is None


# ------------------------------------------------ passthrough args reach vLLM (F15)
def _seed_print_bundle(tmp_path, port="8100"):
    src = _make_bundle_folder(
        tmp_path, arch="LlamaForCausalLM",
        main_class="models.tt_transformers.tt.generator_vllm:LlamaForCausalLM",
        launch={"default": {"command": ["python3", "server_example_tt.py", "--model",
                                        "org/m", "--port", port],
                            "env": {"MESH_DEVICE": "P150"}}},
    )
    dest = bundles.install_bundle(src, tmp_path / "bundles", "org__m")
    _seed_vllm_installed("org/m", dest)
    return dest


def test_serve_passes_extra_args_through_to_vllm(monkeypatch, tmp_path):
    """`serve` declares allow_extra_args and documents passthrough, but only the
    self-contained path ever received it — `_serve_vllm` had no extra_args parameter, so on
    the host-provisioned path every argument after the bundle id was silently dropped. A
    silently ignored flag is worse than a rejected one."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_print_bundle(tmp_path)
    res = runner.invoke(cli.app, ["serve", "org/m", "--print", "--local-only",
                                  "--port", "7009", "--max-num-seqs", "8"])
    assert res.exit_code == 0, res.output
    assert "--port 7009" in res.output
    assert "--max-num-seqs 8" in res.output


def test_user_port_wins_over_the_bundle_default(monkeypatch, tmp_path):
    """Passthrough is appended after the bundle's own command so argparse last-wins gives
    the user's --port priority; the bundle default must still be present but overridden."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_print_bundle(tmp_path, port="8100")
    res = runner.invoke(cli.app, ["serve", "org/m", "--print", "--local-only",
                                  "--port", "7009"])
    assert res.exit_code == 0, res.output
    cmd = res.output.strip().splitlines()[-1]
    assert cmd.index("--port 8100") < cmd.index("--port 7009")


def test_announced_endpoint_matches_the_overridden_port(monkeypatch, tmp_path):
    """The endpoint was parsed from the bundle's launch command, so `serve --port 7009`
    announced :8100 while vLLM bound 7009. Announcing a port we are not binding also means
    a port preflight would check the wrong one."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_print_bundle(tmp_path, port="8100")
    res = runner.invoke(cli.app, ["serve", "org/m", "--print", "--local-only",
                                  "--port", "7009"])
    assert res.exit_code == 0, res.output
    assert "http://localhost:7009" in res.output
    assert "http://localhost:8100" not in res.output


def test_no_passthrough_leaves_the_bundle_command_alone(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _seed_print_bundle(tmp_path, port="8100")
    res = runner.invoke(cli.app, ["serve", "org/m", "--print", "--local-only"])
    assert res.exit_code == 0, res.output
    assert "http://localhost:8100" in res.output


# ----------------------------------------- the serving adapter must exist (preflight)
class TestAdapterRoot:
    def test_extracts_the_top_level_package(self):
        assert runtime.adapter_root(
            "models.autoports.qwen_qwen3_32b.tt.generator_vllm:Qwen3ForCausalLM") == "models"

    def test_handles_a_bare_module(self):
        assert runtime.adapter_root("mypkg:Cls") == "mypkg"

    def test_empty_is_none(self):
        assert runtime.adapter_root("") is None
        assert runtime.adapter_root(None) is None


class TestModuleImportable:
    def test_true_for_a_stdlib_module(self):
        assert runtime.module_importable("json") is True

    def test_false_for_a_missing_module(self):
        assert runtime.module_importable("definitely_not_a_module_xyz") is False

    def test_fails_open_when_the_interpreter_cannot_be_probed(self):
        """An unanswerable check must not become a blocker — that would refuse to serve a
        model that would have worked."""
        assert runtime.module_importable("json", python="/nonexistent/python") is True


def test_serve_blocks_when_the_adapter_tree_is_absent(monkeypatch, tmp_path):
    """The ttnn PyPI wheel ships no models/ tree, so a fully green toolchain still cannot
    serve a bundle whose main_class lives under models.*. Without this the failure arrives
    as an ImportError from inside vLLM, after startup has already burned ~18 seconds."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = _make_bundle_folder(
        tmp_path, arch="Qwen3ForCausalLM",
        main_class="models.autoports.qwen_qwen3_32b.tt.generator_vllm:Qwen3ForCausalLM",
        launch={"default": {"command": ["python3", "server.py", "--port", "8123"], "env": {}}},
    )
    dest = bundles.install_bundle(src, tmp_path / "bundles", "org__m")
    _seed_vllm_installed("org/m", dest)
    monkeypatch.setattr(runtime, "vllm_available", lambda python=None: True)
    res = runner.invoke(cli.app, ["serve", "org/m", "--local-only"])
    assert res.exit_code == 1
    assert "serving adapter is missing" in res.output
    assert "models" in res.output
    assert "no models/ tree" in res.output


def test_serve_does_not_block_when_the_adapter_is_present(monkeypatch, tmp_path):
    """Guard against the check refusing a bundle that would have served."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    src = _make_bundle_folder(
        tmp_path, arch="X", main_class="json:Whatever",
        launch={"default": {"command": ["python3", "server.py", "--port", "8124"], "env": {}}},
    )
    dest = bundles.install_bundle(src, tmp_path / "bundles", "org__ok")
    _seed_vllm_installed("org/ok", dest)
    monkeypatch.setattr(runtime, "vllm_available", lambda python=None: True)
    res = runner.invoke(cli.app, ["serve", "org/ok", "--local-only", "--print"])
    assert "serving adapter is missing" not in res.output
