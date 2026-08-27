# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Packaging a container: provenance, staging, build argv, and the interrupt guard.

A cold build is 2.5-4 hours, so everything up to and after `docker build` is tested
without one: a fake git repo on disk, a fake `docker` on PATH, and no daemon.
"""

import json
import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

from tt_kernel import build
from tt_kernel.build import BuildError, InterruptGuard

from test_container_manifest import BASE


# ------------------------------------------------------------------ fixtures


def _fake_metal(root: Path, *, commit: bool = True) -> Path:
    """A directory that passes the tt-metal shape check, optionally a git repo."""
    metal = root / "tt-metal"
    (metal / "tt_metal" / "python_env").mkdir(parents=True)
    (metal / "tt_metal" / "python_env" / "requirements-dev.txt").write_text(
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "torch==2.11.0 ; platform_machine == 'x86_64'\n"
    )
    (metal / "models" / "common").mkdir(parents=True)
    (metal / "models" / "common" / "mod.py").write_text("x = 1\n")
    (metal / "build_metal.sh").write_text("#!/bin/bash\n")
    # things that must NOT reach the build context
    (metal / ".cpmcache").mkdir()
    (metal / ".cpmcache" / "huge").write_text("x" * 100)
    (metal / "build_Release").mkdir()
    (metal / "build_Release" / "obj").write_text("x")
    (metal / "docs").mkdir()
    (metal / "docs" / "d.md").write_text("d")
    if commit:
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "init", "-q", str(metal)], check=True)
        subprocess.run(["git", "-C", str(metal), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(metal), "commit", "-qm", "i"], check=True, env=env)
    return metal


def _manifest_file(tmp_path: Path, metal: Path, **over) -> Path:
    raw = json.loads(json.dumps(BASE))
    raw["source"] = dict(raw["source"], tt_metal=str(metal), code=["models/common"])
    raw.update(over)
    p = tmp_path / "tt-model.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p


def _no_network(monkeypatch):
    """resolve_git_ref must not reach the network in these tests."""
    monkeypatch.setattr(build, "resolve_git_ref", lambda repo, ref: "s" * 40)


# ------------------------------------------------------------------ provenance


def test_a_directory_without_tt_metal_is_refused(tmp_path):
    (tmp_path / "nope").mkdir()
    m_path = _manifest_file(tmp_path, tmp_path / "nope")
    with pytest.raises(BuildError, match="does not look like a tt-metal checkout"):
        build.stage(m_path, out_root=tmp_path / "out")


def test_the_head_sha_and_clean_state_are_recorded(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert len(staged.built["tt_metal"]["sha"]) == 40
    assert staged.built["tt_metal"]["dirty"] is False


def test_an_uncommitted_tree_is_packaged_but_recorded_as_dirty(tmp_path, monkeypatch):
    """Packaging a dirty tree is the point of the hermetic default — but it is RECORDED."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    (metal / "models" / "common" / "mod.py").write_text("x = 2  # uncommitted\n")
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.built["tt_metal"]["dirty"] is True


def test_runtime_refs_are_pinned_to_shas(tmp_path, monkeypatch):
    """A branch name in the manifest becomes a sha in the published one — the whole
    point: a plugin that moves under a validated model is the bug this prevents."""
    monkeypatch.setattr(build, "resolve_git_ref", lambda repo, ref: "a" * 40)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.manifest.runtime["plugin"]["sha"] == "a" * 40
    assert staged.built["plugin"] == {
        "repo": "https://github.com/tenstorrent/vllm-tt-plugin", "sha": "a" * 40}


def test_a_full_sha_needs_no_remote_lookup():
    assert build.resolve_git_ref("https://x/y", "b" * 40) == "b" * 40


def test_an_unresolvable_ref_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(build.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, "", ""))
    with pytest.raises(BuildError, match="could not resolve ref"):
        build.resolve_git_ref("https://x/y", "no-such-branch")


@pytest.mark.parametrize("desc,expected", [
    ("v0.72.0-0-gabc1234567", "0.72.0"),
    ("v0.72.0-14-gabc1234567", "0.72.1.dev14"),
    ("v0.72.0-14-gabc1234567-dirty", "0.72.1.dev14+gabc1234567"),
])
def test_scm_version_mirrors_setuptools_scm(tmp_path, monkeypatch, desc, expected):
    monkeypatch.setattr(build, "_git", lambda *a, **k: desc)
    assert build.scm_version(tmp_path) == expected


def test_scm_version_falls_back_when_git_says_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "_git", lambda *a, **k: None)
    assert build.scm_version(tmp_path) == "0.0.0.dev0"


# ------------------------------------------------------------------ staging


def test_the_build_context_excludes_the_expensive_trees(tmp_path, monkeypatch):
    """Handing BuildKit the raw checkout would transfer .git (~5.7 GB) before the
    Dockerfile's excludes ever run."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    ctx_metal = staged.metal.context
    for gone in (".git", ".cpmcache", "build_Release", "docs", "models"):
        assert not (ctx_metal / gone).exists(), gone
    assert (ctx_metal / "tt_metal").is_dir()   # the tree itself survives


def test_models_is_excluded_from_the_context_so_the_allowlist_is_the_only_source(
        tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert not (staged.metal.context / "models").exists()
    assert (staged.ctx / "code" / "models" / "common" / "mod.py").exists()


def test_a_missing_code_entry_is_an_error_not_a_skip(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    src = dict(BASE["source"], tt_metal=str(metal), code=["models/common", "models/gone"])
    with pytest.raises(BuildError, match="models/gone"):
        build.stage(_manifest_file(tmp_path, metal, source=src), out_root=tmp_path / "out")


def test_weights_and_junk_never_ride_along_in_code(tmp_path, monkeypatch):
    """Weights are a POINTER in this design; a stray .safetensors would silently make
    the repo enormous."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    (metal / "models" / "common" / "w.safetensors").write_text("W")
    (metal / "models" / "common" / "__pycache__").mkdir()
    (metal / "models" / "common" / "__pycache__" / "x.pyc").write_text("c")
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    code = staged.ctx / "code" / "models" / "common"
    assert (code / "mod.py").exists()
    assert not (code / "w.safetensors").exists()
    assert not (code / "__pycache__").exists()


def test_the_generated_scripts_and_docker_assets_land_in_the_context(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    for f in ("install_engine.sh", "verify.sh", "Dockerfile", "entrypoint.sh"):
        assert (staged.ctx / f).is_file(), f
    assert (staged.ctx / "install_engine.sh").read_text().startswith("#!/bin/bash\nset -euxo")


def test_a_missing_lock_file_is_an_error(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    rt = dict(json.loads(json.dumps(BASE))["runtime"], lock="nope.lock")
    with pytest.raises(BuildError, match="does not exist"):
        build.stage(_manifest_file(tmp_path, metal, runtime=rt), out_root=tmp_path / "out")


def test_the_code_digest_changes_when_the_code_changes(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    a = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    (metal / "models" / "common" / "mod.py").write_text("x = 99\n")
    b = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out2")
    assert a.built["code_sha256"] != b.built["code_sha256"]


# ------------------------------------------------------------------ build argv


def test_build_argv_carries_the_named_metal_context_and_the_args(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    argv = build.build_argv(staged)
    assert argv[:2] == ["docker", "build"]
    assert f"metalsrc={staged.metal.context}" in argv
    assert f"--build-arg" in argv
    joined = " ".join(argv)
    assert "METAL_MODE=local" in joined
    assert "TT_MODEL_KIND=vllm" in joined
    assert "MODEL_PROFILES=p150x4" in joined
    assert argv[-1] == str(staged.ctx)


def test_builtin_models_is_suppressed_only_when_an_extension_ships(tmp_path, monkeypatch):
    """"0" for a builtin-registry model would register ZERO architectures."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    plain = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "o1")
    assert plain.build_args["TT_VLLM_BUILTIN_MODELS"] == ""
    assert plain.build_args["EXTRA_MODELS_DIR"] == ""

    rt = dict(json.loads(json.dumps(BASE))["runtime"], extension="models/common/ext")
    ext = build.stage(_manifest_file(tmp_path, metal, runtime=rt), out_root=tmp_path / "o2")
    assert ext.build_args["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert ext.build_args["EXTRA_MODELS_DIR"].endswith("/extra_models")


def test_the_image_tag_encodes_the_build_not_the_hardware(tmp_path, monkeypatch):
    """One image serves every profile, so the tag names the metal commit."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.image.startswith("tt-model/my-model:")
    assert staged.image.split(":")[1] == staged.built["tt_metal"]["sha"][:9]


# ------------------------------------------------------------------ model card


def _built(**over):
    b = {"tt_metal": {"sha": "a" * 40, "dirty": False, "scm_version": "0.72.1"},
         "vllm": {"repo": "r", "sha": "b" * 40},
         "code_sha256": "c" * 64, "created_at": "2026-01-01T00:00:00+00:00",
         "tt_model_version": "0.1.0"}
    b.update(over)
    return b


def _card(**over):
    from tt_kernel.container_manifest import ContainerManifest

    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    return build.render_model_card(m, _built(), ["models/common/"])


def test_the_card_lists_every_profile_and_marks_the_default():
    card = _card()
    assert "`p150x4`" in card and "*(default)*" in card


def test_the_card_pins_provenance():
    card = _card()
    assert "a" * 40 in card and "b" * 40 in card


def test_the_card_flags_a_dirty_build():
    from tt_kernel.container_manifest import ContainerManifest

    m = ContainerManifest.model_validate(json.loads(json.dumps(BASE)))
    card = build.render_model_card(
        m, _built(tt_metal={"sha": "a" * 40, "dirty": True}), [])
    assert "dirty tree" in card


def test_the_card_says_weights_are_not_baked_in():
    assert "never baked into the image" in _card()


def test_the_card_includes_the_authors_quickstart():
    assert "point it here" in _card(card={"quickstart": "point it here"}).lower()


# ------------------------------------------------------------------ interrupt guard


def test_the_first_ctrl_c_on_a_tty_only_warns():
    warnings = []
    g = InterruptGuard("the build", tty=True, warn=warnings.append,
                       on_cancel_note="costs the stage")
    g._handle(signal.SIGINT, None)
    assert not g.cancelled
    assert "STILL RUNNING" in warnings[0]
    assert "costs the stage" in warnings[0]


def test_a_second_ctrl_c_within_the_window_cancels():
    g = InterruptGuard("the build", tty=True, warn=lambda s: None)
    g._handle(signal.SIGINT, None)
    g._handle(signal.SIGINT, None)
    assert g.cancelled


def test_a_second_ctrl_c_after_the_window_only_warns_again():
    """The guard re-arms; a Ctrl-C an hour later must not cancel by surprise."""
    warnings = []
    g = InterruptGuard("the build", window_s=0.01, tty=True, warn=warnings.append)
    g._handle(signal.SIGINT, None)
    time.sleep(0.02)
    g._handle(signal.SIGINT, None)
    assert not g.cancelled and len(warnings) == 2


def test_without_a_tty_the_first_interrupt_cancels():
    """Nobody is there to press twice."""
    g = InterruptGuard("the build", tty=False, warn=lambda s: None)
    g._handle(signal.SIGINT, None)
    assert g.cancelled


def test_sigterm_always_cancels_immediately():
    g = InterruptGuard("the build", tty=True, warn=lambda s: None)
    g._handle(signal.SIGTERM, None)
    assert g.cancelled


def test_the_warning_names_the_stage_being_lost():
    warnings = []
    g = InterruptGuard("the build", tty=True, warn=warnings.append)
    g.note_line("#24 [builder 5/9] RUN ./build_metal.sh --enable-ccache\n")
    g._handle(signal.SIGINT, None)
    assert "builder 5/9" in warnings[0]


def test_signal_handlers_are_restored_on_exit():
    before = signal.getsignal(signal.SIGINT)
    with InterruptGuard("x", warn=lambda s: None):
        assert signal.getsignal(signal.SIGINT) is not before
    assert signal.getsignal(signal.SIGINT) is before


def test_the_child_runs_in_its_own_process_group():
    """A terminal Ctrl-C otherwise reaches the child directly and there is nothing
    left to intercept."""
    with InterruptGuard("x", warn=lambda s: None) as g:
        p = g.spawn([sys.executable, "-c", "import os; print(os.getpgrp())"],
                    stdout=subprocess.PIPE, text=True)
        out, _ = p.communicate()
    assert int(out.strip()) != os.getpgrp()


def test_elapsed_is_human_readable():
    g = InterruptGuard("x", warn=lambda s: None)
    g.started = time.monotonic() - 3700
    assert g.elapsed().startswith("1h")


# ------------------------------------------------------------------ run_build


def _fake_docker(tmp_path, monkeypatch, script: str):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    exe = bindir / "docker"
    exe.write_text("#!/bin/bash\n" + script)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")


def test_a_failing_build_reports_the_log_path(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, 'echo "#1 [prep 1/2] boom"\nexit 3\n')
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    with pytest.raises(BuildError, match="exit 3"):
        build.run_build(staged)
    assert build.build_log_path("my-model").exists()


def test_output_is_teed_to_the_log_and_echoed(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, 'echo "#7 [builder 1/9] hello"\nexit 0\n')
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    seen = []
    build.run_build(staged, echo=seen.append)
    assert any("hello" in ln for ln in seen)
    assert "hello" in build.build_log_path("my-model").read_text()
