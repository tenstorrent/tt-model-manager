# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""docker run composition and CLI serve/stop semantics — no docker daemon needed.

The launch is asserted through `serve --print` (the composed argv IS the contract),
the same pattern the old suite used to test launches without spawning anything.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tt_model import cli, container, hardware
from tt_model.manifest import load_manifest

from conftest import EXAMPLES

runner = CliRunner()
LAGUNA = str(EXAMPLES / "laguna-xs-2.1.yaml")


def _print_argv(*args):
    res = runner.invoke(cli.app, ["serve", *args, "--print"])
    assert res.exit_code == 0, res.output
    return res.output


# ------------------------------------------------------------- the docker run line
def test_compose_run_has_every_load_bearing_flag(laguna):
    from tt_model.types import TYPES

    prof = laguna.resolve_profile()
    t = TYPES[laguna.type]
    argv = container.compose_run(laguna, prof, t.serve_argv(laguna, prof),
                                 t.serve_env(laguna, prof))
    joined = " ".join(argv)
    assert "--device /dev/tenstorrent" in joined
    assert "--ipc host" in joined
    # VERBATIM: umd regex-matches /proc/mounts for exactly this mountpoint pair
    assert "--mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G" in joined
    # HF cache read-write at HF_HOME (weights are the only thing on the host)
    assert ":/hf" in joined and "--env HF_HOME=/hf" in joined
    assert "readonly" not in joined
    # per-model persistent JIT cache
    assert ":/cache" in joined and "--env TT_METAL_CACHE=/cache" in joined
    assert "--publish 8000:8000" in joined
    assert "--env MESH_DEVICE=P150x4" in joined
    assert "--env HF_MODEL=poolside/Laguna-XS-2.1" in joined
    # the allow-list trap: VLLM_PLUGINS must never be set
    assert "VLLM_PLUGINS" not in joined
    assert argv[-1] == "8000" and "vllm" in argv  # the serve command rides at the end


def test_hf_token_passed_by_name_only(monkeypatch, laguna):
    """The token must never be embedded in the argv (visible in `ps`)."""
    from tt_model.types import TYPES

    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    prof = laguna.resolve_profile()
    t = TYPES[laguna.type]
    argv = container.compose_run(laguna, prof, t.serve_argv(laguna, prof), {})
    assert "HF_TOKEN" in argv
    assert not any("hf_secret" in a for a in argv)


# --------------------------------------------------------------- profile selection
def test_serve_print_uses_and_names_the_default():
    out = _print_argv(LAGUNA)
    assert "profile: p150x4 (default)" in out
    assert "tt-model-laguna-xs-2.1-p150x4" in out


def test_serve_print_profile_override():
    out = _print_argv(LAGUNA, "--profile", "p150x2")
    assert "tt-model-laguna-xs-2.1-p150x2" in out
    assert "65536" in out


def test_unknown_profile_errors_with_the_available_names():
    res = runner.invoke(cli.app, ["serve", LAGUNA, "--profile", "nope", "--print"])
    assert res.exit_code == 1
    assert "p150x4" in res.output and "p150x2" in res.output


def test_hardware_mismatch_warns_and_suggests_never_substitutes(monkeypatch, tmp_path):
    """2 chips detected, default profile wants 4: refuse with a suggestion; --force
    proceeds WITH THE ASKED-FOR PROFILE (silent substitution is how someone benchmarks
    the wrong deployment without knowing it)."""
    byid = tmp_path / "by-id"
    byid.mkdir()
    for serial in ("blackhole-AAA", "blackhole-BBB"):
        (byid / serial).touch()
    monkeypatch.setattr(hardware, "BY_ID", byid)

    res = runner.invoke(cli.app, ["serve", LAGUNA, "--print"])
    assert res.exit_code == 1
    assert "p150x2" in res.output          # the suggestion
    assert "--force" in res.output

    res = runner.invoke(cli.app, ["serve", LAGUNA, "--print", "--force"])
    assert res.exit_code == 0
    assert "tt-model-laguna-xs-2.1-p150x4" in res.output   # NOT substituted


def test_hardware_match_passes_silently(monkeypatch, tmp_path):
    byid = tmp_path / "by-id"
    byid.mkdir()
    for i in range(4):
        (byid / f"blackhole-{i:03d}").touch()
    monkeypatch.setattr(hardware, "BY_ID", byid)
    out = _print_argv(LAGUNA)
    assert "does NOT fit" not in out


# ---------------------------------------------------------------------- hardware.py
def test_detect_reads_arch_and_count(tmp_path):
    byid = tmp_path / "by-id"
    byid.mkdir()
    for i in range(4):
        (byid / f"blackhole-{i:03d}").touch()
    host = hardware.detect(byid)
    assert host.arch == "blackhole" and host.chips == 4


def test_detect_no_devices(tmp_path):
    assert hardware.detect(tmp_path / "missing") is None


def test_smaller_profile_fits_bigger_host(laguna):
    host = hardware.HostDevices(arch="blackhole", chips=4)
    assert hardware.fitting_profiles(laguna, host) == ["p150x4", "p150x2"]
    host2 = hardware.HostDevices(arch="blackhole", chips=2)
    assert hardware.fitting_profiles(laguna, host2) == ["p150x2"]


def test_wrong_arch_fits_nothing(laguna):
    host = hardware.HostDevices(arch="wormhole", chips=4)
    assert hardware.fitting_profiles(laguna, host) == []


# ----------------------------------------------------------------------------- stop
def test_stop_is_sigterm_first_never_rm_f(monkeypatch, laguna):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)

        class R:
            returncode = 0
            stdout = "true" if "inspect" in argv and "Running" in argv[-2] else "0"
            stderr = ""
        return R()

    monkeypatch.setattr(container, "_run", fake_run)
    container.stop("tt-model-x-p")
    joined = [" ".join(c) for c in calls]
    assert any(f"docker stop --timeout {container.STOP_TIMEOUT_S}" in j for j in joined)
    assert not any("rm -f" in j for j in joined)


def test_stop_resets_mesh_only_after_a_kill(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)

        class R:
            returncode = 0
            stderr = ""
            stdout = "true"
        if "{{.State.ExitCode}}" in argv:
            R.stdout = "137"      # 128+SIGKILL: the grace period expired
        return R()

    monkeypatch.setattr(container, "_run", fake_run)
    clean = container.stop("tt-model-x-p", image="img")
    assert clean is False
    reset = [c for c in calls if "tt-smi" in c]
    assert reset and "-r" in reset[0] and "--rm" in reset[0]


def test_stop_clean_shutdown_skips_the_reset(monkeypatch):
    calls = []

    def fake_run(argv, **kw):
        calls.append(argv)

        class R:
            returncode = 0
            stderr = ""
            stdout = "true"
        if "{{.State.ExitCode}}" in argv:
            R.stdout = "0"
        return R()

    monkeypatch.setattr(container, "_run", fake_run)
    assert container.stop("tt-model-x-p", image="img") is True
    assert not any("tt-smi" in c for c in calls)


# ------------------------------------------------------------------------ exclusivity
def test_serve_refuses_while_another_model_runs(monkeypatch):
    monkeypatch.setattr(container, "running",
                        lambda name_filter=None: [{"name": "tt-model-other-p",
                                                   "image": "x", "status": "Up 2 hours",
                                                   "ports": ""}])
    res = runner.invoke(cli.app, ["serve", LAGUNA])
    assert res.exit_code == 1
    assert "already running" in res.output
    assert "tt-model-other-p" in res.output


def test_image_tag_prefers_built_block(laguna):
    assert container.image_tag(laguna) == "tt-model/laguna-xs-2.1:dev"
    laguna.built = {"image": "tt-model/laguna-xs-2.1:9b415f820"}
    assert container.image_tag(laguna) == "tt-model/laguna-xs-2.1:9b415f820"
