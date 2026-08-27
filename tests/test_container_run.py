# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`docker run` composition and launcher argv — the whole serve path, offline.

These are golden-string tests on purpose. Every flag here was expensive to learn (the
hugepages mount regex, the tool-parser pairing, the legacy runner's joined args), so the
tests assert the exact bytes rather than a shape: a silent change to any of them is a
boot failure ten minutes later on hardware.
"""

import json
from pathlib import Path

import pytest

from tt_kernel import container
from tt_kernel.container_manifest import ContainerManifest, ContainerManifestError
from tt_kernel.launchers import LauncherError, launcher_for
from tt_kernel.manifest import Manifest

from test_container_manifest import BASE


def _wire(**over) -> Manifest:
    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    m.validate_semantics()
    return m.to_wire(
        image_tag="tt-model/my-model:abc123",
        tt_metal_version="0.72.1",
        tt_kernel_version="0.1.0",
        hostname="h",
        created_at="2026-01-01T00:00:00+00:00",
    )


# ------------------------------------------------------------------ docker run


def _run_argv(m, **kw):
    profile = m.container.resolve_profile()
    launcher = launcher_for(m.container.kind)
    return container.compose_run(
        m, profile,
        launcher.serve_argv(m, profile),
        launcher.serve_env(m, profile),
        hf_home_dir=Path("/home/u/.cache/huggingface"),
        cache_dir=Path("/home/u/.cache/tt-model/my-model/cache"),
        include_hf_token=False,
        **kw,
    )


def test_the_hugepages_mount_is_verbatim():
    """umd regex-matches /proc/mounts for exactly `/dev/hugepages-1G`; a subdirectory or
    a different dst silently fails that match and device-open fails minutes later."""
    argv = _run_argv(_wire())
    i = argv.index("--mount")
    assert argv[i + 1] == "type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G"


def test_the_device_and_ipc_flags_are_present():
    argv = _run_argv(_wire())
    assert argv[argv.index("--device") + 1] == "/dev/tenstorrent"
    assert argv[argv.index("--ipc") + 1] == "host"


def test_the_hf_cache_is_mounted_read_write_and_pointed_at_by_HF_HOME():
    """Model classes snapshot_download at load time and --trust-remote-code writes to
    HF_MODULES_CACHE, so a read-only mount breaks serving, not just downloading."""
    argv = _run_argv(_wire())
    assert "/home/u/.cache/huggingface:/hf" in argv       # no :ro suffix
    assert "HF_HOME=/hf" in argv


def test_the_kernel_cache_is_persisted_on_the_host():
    argv = _run_argv(_wire())
    assert "/home/u/.cache/tt-model/my-model/cache:/cache" in argv
    assert "TT_METAL_CACHE=/cache" in argv


def test_the_port_is_published_from_the_profile():
    argv = _run_argv(_wire())
    assert argv[argv.index("--publish") + 1] == "8000:8000"


def test_the_container_is_named_per_model_and_profile():
    argv = _run_argv(_wire())
    assert argv[argv.index("--name") + 1] == "tt-model-my-model-p150x4"


def test_the_image_is_the_last_thing_before_the_command():
    argv = _run_argv(_wire())
    assert "tt-model/my-model:abc123" in argv
    assert argv.index("tt-model/my-model:abc123") < argv.index("python")


def test_hf_token_is_passed_by_NAME_never_by_value(monkeypatch):
    """`--print` displays this argv and `ps` shows it; the value must not be in it."""
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    m = _wire()
    argv = container.compose_run(
        m, m.container.resolve_profile(), ["x"], {},
        hf_home_dir=Path("/h"), cache_dir=Path("/c"),
    )
    assert "--env" in argv and "HF_TOKEN" in argv
    assert "hf_secret" not in " ".join(argv)


def test_no_hf_token_flag_when_the_env_has_none(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    m = _wire()
    argv = container.compose_run(
        m, m.container.resolve_profile(), ["x"], {},
        hf_home_dir=Path("/h"), cache_dir=Path("/c"),
    )
    assert "HF_TOKEN" not in argv


def test_compose_run_is_deterministic():
    """Env ordering must not wobble, or --print output churns between runs."""
    assert _run_argv(_wire()) == _run_argv(_wire())


def test_a_non_container_manifest_is_refused():
    m = Manifest(schema_version="5", name="n", tt_metal_version="v", arch="blackhole",
                 producer={"tt_kernel_version": "0.1.0", "created_at": "now"})
    with pytest.raises(container.ContainerError, match="not a container package"):
        container.image_ref(m)


# ------------------------------------------------------------------ registry pluggability


def test_an_hf_hosted_image_runs_its_local_tag_and_needs_no_docker_pull():
    m = _wire()
    assert container.image_ref(m) == "tt-model/my-model:abc123"
    assert container.compose_pull(m) is None


def test_a_registry_hosted_image_runs_and_pulls_by_reference():
    m = _wire(image={"registry": "ghcr.io/tenstorrent"})
    assert container.image_ref(m) == "ghcr.io/tenstorrent/my-model:abc123"
    assert container.compose_pull(m) == [
        "docker", "pull", "ghcr.io/tenstorrent/my-model:abc123"]


# ------------------------------------------------------------------ launcher argv


def test_vllm_serve_argv_golden():
    m = _wire()
    p = m.container.resolve_profile()
    assert launcher_for("vllm").serve_argv(m, p) == [
        "python", "-m", "models.common.readiness_check.run_vllm_server",
        "--stages", "serve",
        "--model-dir", "models/common",
        "--hf-model", "org/Weights-7B",
        "--mesh-device", "(1, 4)",
        "--max-num-seqs", "32",
        "--max-model-len", "131072",
        "--block-size", "64",
        "--port", "8000",
    ]


def test_the_mesh_is_forwarded_as_a_grid_not_as_MESH_DEVICE():
    """The readiness runner takes --mesh-device; it does not read a MESH_DEVICE env var."""
    m = _wire()
    p = m.container.resolve_profile()
    assert "MESH_DEVICE" not in launcher_for("vllm").serve_env(m, p)
    assert "(1, 4)" in launcher_for("vllm").serve_argv(m, p)


def test_serve_env_carries_the_model_id():
    """tt_transformers-style adapters read HF_MODEL from env, not from vLLM's --model."""
    m = _wire()
    env = launcher_for("vllm").serve_env(m, m.container.resolve_profile())
    assert env["HF_MODEL"] == "org/Weights-7B"


def test_extra_server_args_are_joined_into_one_string():
    """This runner takes --additional-server-args as ONE string, not loose argv."""
    m = _wire(serve={"port": 8000, "block_size": 64,
                     "args": ["--trust-remote-code", ["--seed", "0"]]})
    argv = launcher_for("vllm").serve_argv(m, m.container.resolve_profile())
    assert argv[argv.index("--additional-server-args") + 1] == "--trust-remote-code --seed 0"


def test_tool_parser_is_emitted_with_enable_auto_tool_choice():
    """vLLM hard-errors on --tool-call-parser without --enable-auto-tool-choice."""
    m = _wire(serve={"port": 8000, "block_size": 64,
                     "capabilities": {"tool_parser": "hermes"}})
    argv = launcher_for("vllm").serve_argv(m, m.container.resolve_profile())
    joined = argv[argv.index("--additional-server-args") + 1]
    assert joined == "--enable-auto-tool-choice --tool-call-parser hermes"


def test_reasoning_parser_keeps_its_underscore():
    """typer normalises '_'->'-'; '--reasoning_parser' is the spelling that survives."""
    m = _wire(serve={"port": 8000, "block_size": 64,
                     "capabilities": {"reasoning_parser": "deepseek_r1"}})
    argv = launcher_for("vllm").serve_argv(m, m.container.resolve_profile())
    assert "--reasoning_parser deepseek_r1" in argv[argv.index("--additional-server-args") + 1]


def test_tt_additional_config_becomes_tt_config():
    m = _wire(serve={"port": 8000, "block_size": 64,
                     "additional_config": {"tt": {"k": 1}}})
    argv = launcher_for("vllm").serve_argv(m, m.container.resolve_profile())
    assert json.loads(argv[argv.index("--tt-config") + 1]) == {"k": 1}


def test_server_timeout_is_passed_when_set():
    m = _wire(serve={"port": 8000, "block_size": 64, "server_timeout": 900})
    argv = launcher_for("vllm").serve_argv(m, m.container.resolve_profile())
    assert argv[argv.index("--server-timeout") + 1] == "900"


# ------------------------------------------------------------------ kind validation


def test_an_unknown_kind_is_refused():
    with pytest.raises(LauncherError, match="unsupported kind"):
        launcher_for("tensorrt")


def test_vllm_is_the_only_kind_and_it_means_the_fork():
    """`kind: vllm` must mean what runtime.kind="vllm" has always meant in this repo:
    the tenstorrent/vllm fork. One word, one meaning."""
    from tt_kernel.launchers import KINDS

    assert sorted(KINDS) == ["vllm"]


def test_the_fork_requires_repo_and_ref():
    raw = json.loads(json.dumps(BASE))
    raw["runtime"] = {"vllm": {"repo": "https://github.com/tenstorrent/vllm"},
                      "model_dir": "models/common"}
    with pytest.raises(ContainerManifestError, match=r"requires runtime.vllm"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_a_runtime_plugin_block_is_refused():
    raw = json.loads(json.dumps(BASE))
    raw["runtime"]["plugin"] = {"repo": "x", "ref": "y"}
    with pytest.raises(ContainerManifestError, match="takes no runtime.plugin"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_an_unknown_runtime_key_is_refused():
    raw = json.loads(json.dumps(BASE))
    raw["runtime"]["nonsense"] = 1
    with pytest.raises(ContainerManifestError, match="does not understand runtime.nonsense"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_model_dir_must_be_covered_by_the_code_allowlist():
    """Otherwise the launcher looks for a directory that never entered the image."""
    raw = json.loads(json.dumps(BASE))
    raw["runtime"]["model_dir"] = "models/not_shipped"
    with pytest.raises(ContainerManifestError, match="not covered by source.code"):
        ContainerManifest.model_validate(raw).validate_semantics()


# ------------------------------------------------------------------ stop semantics


def test_reset_mesh_runs_tt_smi_from_the_image_not_the_host():
    """A container consumer has no host tt-smi; the image already carries one."""
    argv = container.compose_reset_mesh("tt-model/x:1")
    assert argv[:4] == ["docker", "run", "--rm", "--device"]
    assert argv[argv.index("--entrypoint") + 1] == "tt-smi"
    assert argv[-3:] == ["tt-model/x:1", "-r", "all"]


def _fake_docker(monkeypatch, *, running_state, exit_code):
    calls = []

    class R:
        def __init__(self, stdout="", rc=0):
            self.stdout, self.returncode = stdout, rc

    def fake(argv, **kw):
        calls.append(argv)
        if "{{.State.Running}}" in argv:
            return R(running_state)
        if "{{.State.ExitCode}}" in argv:
            return R(exit_code)
        return R()

    monkeypatch.setattr(container, "_run", fake)
    return calls


def test_a_clean_sigterm_stop_does_not_reset_the_mesh(monkeypatch):
    calls = _fake_docker(monkeypatch, running_state="true", exit_code="0")
    assert container.stop("c", image="img") is True
    assert not any("--entrypoint" in c for c in calls)
    stop_cmd = next(c for c in calls if c[:2] == ["docker", "stop"])
    assert stop_cmd[stop_cmd.index("--timeout") + 1] == str(container.STOP_TIMEOUT_S)


def test_a_sigkilled_container_triggers_a_mesh_reset(monkeypatch):
    """137 means the grace period expired: the mesh was never closed, eth cores are
    dirty, and the NEXT boot fails unless it is reset now."""
    calls = _fake_docker(monkeypatch, running_state="true", exit_code="137")
    assert container.stop("c", image="img") is False
    assert any("--entrypoint" in c for c in calls)


def test_an_already_stopped_container_is_just_removed(monkeypatch):
    calls = _fake_docker(monkeypatch, running_state="false", exit_code="")
    assert container.stop("c", image="img") is True
    assert any(c[:3] == ["docker", "rm", "c"] for c in calls)
    assert not any("--entrypoint" in c for c in calls)
