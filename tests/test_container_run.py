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

from test_container_manifest import BASE, FORK


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
        weight_cache_dir=Path("/home/u/.cache/tt-model/my-model/weights"),
        include_hf_token=False,
        **kw,
    )


def test_the_container_runs_as_the_host_user():
    """Everything it writes lands in bind mounts the host user owns (HF cache, kernel
    cache), so it must write AS that user. A baked-in uid cannot work — 1000 and 1001 are
    both common — and a mismatch fails as Permission denied from the JIT, minutes in."""
    import os

    argv = _run_argv(_wire())
    assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"


def test_mount_sources_are_created_as_the_host_user(tmp_path):
    """Otherwise the docker daemon creates them as ROOT and the container cannot write."""
    m = _wire()
    hf, cache = tmp_path / "hf", tmp_path / "c" / "cache"
    container.ensure_mount_sources(m, hf_home_dir=hf, cache_dir=cache)
    assert hf.is_dir() and cache.is_dir()


def test_compose_run_creates_nothing(tmp_path):
    """`serve --print` must not touch the filesystem — composition stays pure."""
    m = _wire()
    hf, cache = tmp_path / "hf", tmp_path / "c" / "cache"
    container.compose_run(m, m.container.resolve_profile(), ["x"], {},
                          hf_home_dir=hf, cache_dir=cache, include_hf_token=False)
    assert not hf.exists() and not cache.exists()


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


def test_the_converted_weight_cache_is_persisted_on_the_host():
    """Without this the container reconverts tens of GB on every start: FLUX.2 measured
    8-9 minutes cold against 40 seconds once warm. tt_dit reads TT_DIT_CACHE_DIR and
    silently skips caching when it is unset, so the miss produced no error at all."""
    argv = _run_argv(_wire())
    assert "/home/u/.cache/tt-model/my-model/weights:/weight-cache" in argv
    assert "TT_DIT_CACHE_DIR=/weight-cache" in argv


def test_every_cache_mount_has_a_variable_pointing_at_it():
    """The failure mode this guards is a mount nothing reads. A cache dir with no
    variable aimed at it is not a slow cache, it is no cache, and it looks fine in
    `docker inspect`."""
    argv = _run_argv(_wire())
    dests = {v.split(":", 1)[1] for i, v in enumerate(argv)
             if i and argv[i - 1] == "--volume"}
    envs = " ".join(argv)
    for dest in dests:
        assert f"={dest}" in envs, f"{dest} is mounted but no env var points at it"


def test_the_weight_cache_source_is_created_before_run(tmp_path):
    """docker creates a missing bind source as ROOT, and the container runs as the host
    user, so an uncreated dir fails as Permission denied minutes into a boot."""
    m = _wire()
    container.ensure_mount_sources(
        m,
        hf_home_dir=tmp_path / "hf",
        cache_dir=tmp_path / "cache",
        weight_cache_dir=tmp_path / "weights",
    )
    assert (tmp_path / "weights").is_dir()


def test_the_port_is_published_from_the_profile():
    argv = _run_argv(_wire())
    assert argv[argv.index("--publish") + 1] == "8000:8000"


def test_the_container_is_named_per_model_and_profile():
    argv = _run_argv(_wire())
    assert argv[argv.index("--name") + 1] == "tt-model-my-model-p150x4"


def test_the_image_is_the_last_thing_before_the_command():
    argv = _run_argv(_wire())
    assert "tt-model/my-model:abc123" in argv
    assert argv.index("tt-model/my-model:abc123") < argv.index("vllm")


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


def test_plugin_serve_argv_golden():
    m = _wire()
    p = m.container.resolve_profile()
    assert launcher_for("vllm-plugin").serve_argv(m, p) == [
        "vllm", "serve", "org/Weights-7B",
        "--max-model-len", "131072",
        "--max-num-seqs", "32",
        "--block-size", "64",
        "--port", "8000",
    ]


def test_the_plugin_kind_hands_the_mesh_over_in_the_environment():
    m = _wire()
    env = launcher_for("vllm-plugin").serve_env(m, m.container.resolve_profile())
    assert env["MESH_DEVICE"] == "P150x4"
    assert env["HF_MODEL"] == "org/Weights-7B"


def test_fork_serve_argv_golden():
    m = _wire(**FORK)
    p = m.container.resolve_profile()
    assert launcher_for("vllm-fork").serve_argv(m, p) == [
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


def test_the_fork_forwards_the_mesh_as_a_grid_not_as_MESH_DEVICE():
    """The readiness runner takes --mesh-device; it does not read a MESH_DEVICE env var."""
    m = _wire(**FORK)
    p = m.container.resolve_profile()
    assert "MESH_DEVICE" not in launcher_for("vllm-fork").serve_env(m, p)
    assert "(1, 4)" in launcher_for("vllm-fork").serve_argv(m, p)


def test_the_fork_joins_extra_server_args_into_one_string():
    """That runner takes --additional-server-args as ONE string, not loose argv."""
    m = _wire(**FORK, serve={"port": 8000, "block_size": 64,
                             "args": ["--trust-remote-code", ["--seed", "0"]]})
    argv = launcher_for("vllm-fork").serve_argv(m, m.container.resolve_profile())
    assert argv[argv.index("--additional-server-args") + 1] == "--trust-remote-code --seed 0"


def test_tool_parser_is_emitted_with_enable_auto_tool_choice():
    """vLLM hard-errors on --tool-call-parser without --enable-auto-tool-choice."""
    m = _wire(serve={"port": 8000, "block_size": 64,
                     "capabilities": {"tool_parser": "hermes"}})
    argv = launcher_for("vllm-plugin").serve_argv(m, m.container.resolve_profile())
    i = argv.index("--enable-auto-tool-choice")
    assert argv[i + 1:i + 3] == ["--tool-call-parser", "hermes"]


def test_reasoning_parser_keeps_its_underscore():
    """typer normalises '_'->'-'; '--reasoning_parser' is the spelling that survives."""
    m = _wire(serve={"port": 8000, "block_size": 64,
                     "capabilities": {"reasoning_parser": "deepseek_r1"}})
    argv = launcher_for("vllm-plugin").serve_argv(m, m.container.resolve_profile())
    assert argv[argv.index("--reasoning_parser") + 1] == "deepseek_r1"


def test_the_plugin_kind_passes_additional_config_as_json():
    m = _wire(serve={"port": 8000, "block_size": 64,
                     "additional_config": {"tt": {"k": 1}}})
    argv = launcher_for("vllm-plugin").serve_argv(m, m.container.resolve_profile())
    assert json.loads(argv[argv.index("--additional-config") + 1]) == {"tt": {"k": 1}}


def test_the_fork_lifts_the_tt_block_into_tt_config():
    m = _wire(**FORK, serve={"port": 8000, "block_size": 64,
                             "additional_config": {"tt": {"k": 1}}})
    argv = launcher_for("vllm-fork").serve_argv(m, m.container.resolve_profile())
    assert json.loads(argv[argv.index("--tt-config") + 1]) == {"k": 1}


def test_server_timeout_is_passed_when_set():
    m = _wire(**FORK, serve={"port": 8000, "block_size": 64, "server_timeout": 900})
    argv = launcher_for("vllm-fork").serve_argv(m, m.container.resolve_profile())
    assert argv[argv.index("--server-timeout") + 1] == "900"


# ------------------------------------------------------------------ kind validation


def test_an_unknown_kind_is_refused():
    with pytest.raises(LauncherError, match="unsupported kind"):
        launcher_for("tensorrt")


def test_neither_kind_is_called_plain_vllm():
    """runtime.kind="vllm" already means the fork in a v4 manifest; reusing the bare word
    here would give one field two meanings depending on which schema you read."""
    from tt_kernel.launchers import KINDS

    assert "vllm" not in KINDS
    # the two vLLM arrangements are both named for what they ARE, not for their age
    assert {"vllm-fork", "vllm-plugin"} <= set(KINDS)


def test_the_plugin_kind_requires_a_plugin_source():
    raw = json.loads(json.dumps(BASE))
    del raw["runtime"]["plugin"]
    with pytest.raises(ContainerManifestError, match="requires runtime.plugin"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_the_fork_kind_refuses_a_runtime_plugin_block():
    raw = json.loads(json.dumps(BASE))
    raw.update(json.loads(json.dumps(FORK)))
    raw["runtime"]["plugin"] = {"repo": "x", "ref": "y"}
    with pytest.raises(ContainerManifestError, match="takes no runtime.plugin"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_extra_models_dir_must_be_covered_by_the_code_allowlist():
    raw = json.loads(json.dumps(BASE))
    raw["runtime"]["extra_models_dir"] = "models/not_shipped"
    with pytest.raises(ContainerManifestError, match="not covered by source.code"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_an_unknown_runtime_key_is_refused():
    raw = json.loads(json.dumps(BASE))
    raw["runtime"]["nonsense"] = 1
    with pytest.raises(ContainerManifestError, match="does not understand runtime.nonsense"):
        ContainerManifest.model_validate(raw).validate_semantics()


def test_model_dir_must_be_covered_by_the_code_allowlist():
    """Otherwise the launcher looks for a directory that never entered the image."""
    raw = json.loads(json.dumps(BASE))
    raw.update(json.loads(json.dumps(FORK)))
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


# ------------------------------------------------------------------ host preflight
#
# Every check here exists because the UNCHECKED failure is late and unrecognisable. The
# hugepages one is the worst: umd regex-matches /proc/mounts, so a wrong mount point
# fails silently and the container dies on device open, minutes into a boot.

GOOD_MOUNTS = (
    "proc /proc proc rw,relatime 0 0\n"
    "hugetlbfs /dev/hugepages-1G hugetlbfs rw,mode=777,pagesize=1024M 0 0\n"
)


def _pf(tmp_path, monkeypatch, *, version="29.5.3", mounts=GOOD_MOUNTS, devices=True,
        need_devices=True):
    monkeypatch.setattr(container, "_docker_version", lambda: version)
    m = tmp_path / "mounts"
    m.write_text(mounts)
    dev = tmp_path / "tenstorrent"
    if devices:
        dev.mkdir()
    return container.preflight(need_devices=need_devices, proc_mounts=m, dev_root=dev)


def _fail_names(reqs):
    return [r.name for r in container.preflight_failures(reqs)]


def test_a_healthy_host_passes_everything(tmp_path, monkeypatch):
    assert _fail_names(_pf(tmp_path, monkeypatch)) == []


def test_missing_docker_is_reported_with_the_group_hint(tmp_path, monkeypatch):
    reqs = _pf(tmp_path, monkeypatch, version=None)
    bad = container.preflight_failures(reqs)
    assert [r.name for r in bad] == ["docker"]
    assert "docker` group" in bad[0].fix


def test_docker_too_old_is_refused_with_the_reason(tmp_path, monkeypatch):
    """< 25 does not emit an OCI layout from `docker save`, which only shows up after a
    multi-hour build otherwise."""
    bad = container.preflight_failures(_pf(tmp_path, monkeypatch, version="24.0.7"))
    assert [r.name for r in bad] == ["docker"]
    assert "OCI layout" in bad[0].fix and "skopeo" in bad[0].fix


def test_docker_25_is_accepted(tmp_path, monkeypatch):
    assert _fail_names(_pf(tmp_path, monkeypatch, version="25.0.0")) == []


def test_absent_tt_devices_are_reported(tmp_path, monkeypatch):
    bad = container.preflight_failures(_pf(tmp_path, monkeypatch, devices=False))
    assert "tt devices" in [r.name for r in bad]
    assert "tt-kmd" in next(r for r in bad if r.name == "tt devices").fix


def test_hugepages_mounted_at_the_wrong_path_is_caught(tmp_path, monkeypatch):
    """This is the failure this whole check exists for."""
    wrong = "hugetlbfs /dev/hugepages hugetlbfs rw,pagesize=2M 0 0\n"
    bad = container.preflight_failures(_pf(tmp_path, monkeypatch, mounts=wrong))
    assert "hugepages" in [r.name for r in bad]
    assert "/proc/mounts" in next(r for r in bad if r.name == "hugepages").fix


def test_a_hugepages_subdirectory_does_not_count(tmp_path, monkeypatch):
    sub = "hugetlbfs /dev/hugepages-1G/sub hugetlbfs rw 0 0\n"
    assert "hugepages" in _fail_names(_pf(tmp_path, monkeypatch, mounts=sub))


def test_a_non_hugetlbfs_mount_at_the_path_does_not_count(tmp_path, monkeypatch):
    tmpfs = "tmpfs /dev/hugepages-1G tmpfs rw 0 0\n"
    assert "hugepages" in _fail_names(_pf(tmp_path, monkeypatch, mounts=tmpfs))


def test_byte_moving_operations_do_not_require_a_card(tmp_path, monkeypatch):
    """pull and push work fine on a machine with no hardware attached."""
    reqs = _pf(tmp_path, monkeypatch, devices=False, mounts="", need_devices=False)
    assert [r.name for r in reqs] == ["docker"]
    assert _fail_names(reqs) == []


def test_serve_preflights_but_print_does_not(tmp_path, monkeypatch):
    """--print composes a command without running it, so it must work anywhere — on a
    laptop with no card, for instance."""
    from tt_kernel import container_cli

    called = []
    monkeypatch.setattr(container_cli, "require_host",
                        lambda **k: called.append(k) or (_ for _ in ()).throw(
                            container_cli.ContainerCliError("no host")))
    m = _wire()
    container_cli.serve_container(m, print_only=True)   # must not raise
    assert called == []

# ------------------------------------------------------ HF_HUB_CACHE outside HF_HOME
#
# snapshot_download resolves through huggingface_hub's HF_HUB_CACHE, which honours both
# HF_HOME and HF_HUB_CACHE; the mount was derived from HF_HOME alone. A user who points
# HF_HUB_CACHE at a scratch disk therefore had the host download to one place and the
# container look in another, and the model silently re-downloaded the weights.

HF = Path("/home/u/.cache/huggingface")


def test_the_hub_cache_is_not_mounted_twice_when_it_sits_inside_hf_home():
    """Default and HF_HOME-only setups: /hf already contains hub/, so nothing is added."""
    argv = _run_argv(_wire(), hub_cache_dir=HF / "hub")
    assert "HF_HUB_CACHE=/hf-hub" not in argv
    assert "/hf-hub" not in " ".join(argv)


def test_an_unpinned_hub_cache_follows_hf_home_rather_than_the_environment():
    """Purity: pinning hf_home_dir pins the whole story, so composition must not read the
    developer's real HF_HUB_CACHE."""
    assert "HF_HUB_CACHE=/hf-hub" not in _run_argv(_wire())


def test_a_hub_cache_outside_hf_home_is_mounted_and_pointed_at():
    argv = _run_argv(_wire(), hub_cache_dir=Path("/mnt/big/hub"))
    assert "/mnt/big/hub:/hf-hub" in argv
    assert "HF_HUB_CACHE=/hf-hub" in argv
    assert "HF_HOME=/hf" in argv  # still carries the token store


def test_hub_cache_reads_the_huggingface_hub_constant(monkeypatch, tmp_path):
    """Resolved by asking the library, not by re-deriving HF_HOME/HF_HUB_CACHE precedence."""
    import huggingface_hub.constants as c
    monkeypatch.setattr(c, "HF_HUB_CACHE", str(tmp_path / "elsewhere"), raising=False)
    assert container.hub_cache() == tmp_path / "elsewhere"


def test_ensure_mount_sources_creates_the_hub_dir_too(tmp_path):
    """Every bind-mount source must exist first, or the daemon creates it as ROOT and the
    container -- running as the host user -- cannot write it."""
    hub = tmp_path / "scratch" / "hub"
    container.ensure_mount_sources(_wire(), hf_home_dir=tmp_path / "hf",
                                   cache_dir=tmp_path / "c",
                                   weight_cache_dir=tmp_path / "w", hub_cache_dir=hub)
    assert hub.is_dir()


def test_a_symlinked_hub_cache_is_judged_by_where_it_lands(tmp_path):
    """Spelling must not decide it: a symlink INTO HF_HOME needs no second mount."""
    real = tmp_path / "hf" / "hub"
    real.mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    m = _wire()
    argv = container.compose_run(
        m, m.container.resolve_profile(), ["srv"], {},
        hf_home_dir=tmp_path / "hf", cache_dir=tmp_path / "c",
        weight_cache_dir=tmp_path / "w", hub_cache_dir=link, include_hf_token=False)
    assert "HF_HUB_CACHE=/hf-hub" not in argv


# ------------------------------------------------------------------ free-port walk


def test_pick_free_port_skips_a_real_listener():
    """20000 busy -> 20001 (or the next free one): the increment is real, not cosmetic."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        got = container.pick_free_port(busy, attempts=10)
        assert got > busy
        assert container.port_is_free(got)


def test_pick_free_port_returns_the_preferred_port_when_free():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    # the socket is closed, so the port is free again
    assert container.pick_free_port(free, attempts=10) == free


def test_pick_free_port_gives_up_with_the_fix_in_the_message(monkeypatch):
    monkeypatch.setattr(container, "port_is_free", lambda p: False)
    with pytest.raises(container.ContainerError, match="--port"):
        container.pick_free_port(20000, attempts=3)


def test_compose_run_defaults_to_20000_when_the_profile_names_no_port():
    m = _wire()
    profile = m.container.resolve_profile().model_copy(update={"port": None})
    argv = container.compose_run(m, profile, ["srv"], {},
                                 hf_home_dir=Path("/hf"), cache_dir=Path("/c"),
                                 weight_cache_dir=Path("/w"), include_hf_token=False)
    assert argv[argv.index("--publish") + 1] == "20000:20000"
