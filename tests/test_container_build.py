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
    # the submodule sentinel tt-metal's CMakeLists.txt checks for
    sentinel = metal / "tt_metal" / "third_party" / "umd" / "CMakeLists.txt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_text("# umd\n")
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


def test_the_build_tag_is_provisional(tmp_path, monkeypatch):
    """`stage` cannot know the image's digest — it does not exist until the build runs — so
    it names the output provisionally and `finalize` retags by digest."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.image == f"tt-model/my-model:build-{staged.built['tt_metal']['sha'][:9]}"
    assert staged.digest is None


def test_the_published_tag_is_the_image_digest(tmp_path, monkeypatch):
    """The old tag came from tt-metal's HEAD, which is not the image's identity: a
    republish that changed the plugin pin or the allowlist without moving that sha reused
    the tag, so a consumer's pull skipped the load and served the previous image."""
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, FAKE_DOCKER)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    build.run_build(staged)
    final = build.retag_to_digest(staged)
    assert final == "tt-model/my-model:ee5226aec965"
    assert staged.digest.startswith("sha256:")
    assert staged.built["image"] == final
    assert staged.built["image_digest"] == staged.digest


def test_two_builds_with_the_same_content_get_the_same_tag(tmp_path, monkeypatch):
    """And the corollary: an unrelated commit in the metal tree no longer mints a new
    10 GB tag for byte-identical content."""
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, FAKE_DOCKER)
    metal = _fake_metal(tmp_path)
    tags = []
    for i in range(2):
        staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / f"out{i}")
        build.run_build(staged)
        tags.append(build.retag_to_digest(staged))
    assert tags[0] == tags[1]


def test_a_missing_digest_is_a_clear_error(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, 'exit 1\n')
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    with pytest.raises(BuildError, match="could not read the digest"):
        build.retag_to_digest(staged)


# ------------------------------------------------------------------ model card


def _built(**over):
    b = {"tt_metal": {"sha": "a" * 40, "dirty": False, "scm_version": "0.72.1",
                      "pushed": True,
                      "remote": "https://github.com/tenstorrent/tt-metal"},
         "vllm": {"repo": "https://github.com/tenstorrent/vllm", "sha": "b" * 40},
         "code_sha256": "c" * 64, "created_at": "2026-01-01T00:00:00+00:00",
         "tt_model_version": "0.1.0"}
    b.update(over)
    return b


def _card(**over):
    from tt_kernel.container_manifest import ContainerManifest

    raw = json.loads(json.dumps(BASE))
    raw.update(over)
    m = ContainerManifest.model_validate(raw)
    return build.render_model_card(m, _built())


def test_a_single_profile_card_states_the_hardware_up_front_with_no_table():
    card = _card()
    assert "Runs on **p150x4**" in card
    assert "--profile" not in card
    assert "## Serve profiles" not in card


def test_a_multi_profile_card_lists_every_profile_and_marks_the_default():
    profiles = [
        {"name": "p150x2", "hardware": "p150x2", "mesh_device": "P150x2",
         "max_num_seqs": 8, "max_model_len": 65536},
        {"name": "p150x4", "hardware": "p150x4", "mesh_device": "P150x4",
         "max_num_seqs": 32, "max_model_len": 131072},
    ]
    card = _card(serve_profiles=profiles, default_profile="p150x4")
    assert "`p150x2`" in card and "`p150x4` *(default)*" in card
    assert "--profile" in card
    assert "Runs on **p150x2** or **p150x4**" in card


def test_the_card_names_the_tool_and_schema_and_links_the_repo():
    card = _card()
    assert "https://github.com/tenstorrent/tt-model-manager" in card
    assert "manifest schema 5.1" in card
    assert "0.1.0" in card  # the tt-model version that built it


def test_the_card_leads_with_the_authors_description():
    card = _card(card={"description": "Intended for agentic coding."})
    assert card.index("Intended for agentic coding.") < card.index("## Quickstart")


def test_the_quickstart_sets_expectations():
    card = _card()
    assert "tt-model pull  you/my-model --with-weights" in card
    assert "`pull --with-weights` downloads the Docker image" in card
    assert "tt-model serve" in card
    assert "several minutes" in card
    assert "Application startup complete" in card


def test_the_card_pins_provenance():
    card = _card()
    assert "a" * 40 in card and "b" * 40 in card


def test_the_card_flags_a_dirty_build():
    from tt_kernel.container_manifest import ContainerManifest

    m = ContainerManifest.model_validate(json.loads(json.dumps(BASE)))
    card = build.render_model_card(
        m, _built(tt_metal={"sha": "a" * 40, "dirty": True}))
    assert "dirty tree" in card


def test_the_card_says_weights_are_not_baked_in():
    assert "not in the image" in _card()


def test_the_card_includes_the_authors_quickstart():
    assert "point it here" in _card(card={"quickstart": "point it here"}).lower()


def test_the_card_never_shows_the_internal_kind_slug_as_prose():
    card = _card()
    assert "serving stack" not in card
    assert "**serving stack**" not in card


def test_the_card_deep_links_every_pushed_pin():
    from tt_kernel.container_manifest import ContainerManifest

    m = ContainerManifest.model_validate(json.loads(json.dumps(BASE)))
    built = _built(
        tt_metal={"sha": "a" * 40, "dirty": False, "pushed": True,
                  "remote": "git@github.com:tenstorrent/tt-metal.git"},
        plugin={"sha": "d" * 40,
                "repo": "https://github.com/tenstorrent/vllm-tt-plugin"},
        vllm=None,  # no built entry -> the row falls back to runtime.vllm.version
    )
    card = build.render_model_card(m, built)
    # ssh remotes are normalized to browsable https commit URLs
    assert f"https://github.com/tenstorrent/tt-metal/commit/{'a' * 40}" in card
    assert f"https://github.com/tenstorrent/vllm-tt-plugin/commit/{'d' * 40}" in card
    # a released vLLM links to the actual release, and the plugin is named officially
    assert "https://github.com/vllm-project/vllm/releases/tag/v0.24.0" in card
    assert "vllm-tt-plugin" in card
    assert "| plugin |" not in card


def test_a_commit_that_is_not_public_is_not_shown_at_all():
    from tt_kernel.container_manifest import ContainerManifest

    m = ContainerManifest.model_validate(json.loads(json.dumps(BASE)))
    built = _built(
        tt_metal={"sha": "a" * 40, "dirty": False, "pushed": False,
                  "remote": "https://github.com/tenstorrent/tt-metal"},
        plugin={"sha": "d" * 40, "path": "/home/me/vllm-tt-plugin", "dirty": True},
    )
    card = build.render_model_card(m, built)
    # not the sha, not a link to it — the cell says the commit is unpublished instead
    assert "a" * 40 not in card
    assert "d" * 40 not in card
    assert "commit not published" in card
    assert "dirty tree" in card


def test_the_card_has_no_shipped_code_listing_and_no_what_is_inside():
    card = _card()
    assert "Shipped code" not in card
    assert "What is inside" not in card
    assert "byte-identical" in card  # the fact moved into Provenance


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


# ------------------------------------------------------------------ end to end
#
# stage -> build -> finalize, against a fake `docker` that behaves like the real one for
# the three things we ask of it: build, save (an OCI-layout tar), and run (the env freeze).
# This is the only test that exercises finalize(), which is what actually produces the
# directory `push` uploads.

FAKE_DOCKER = r'''
case "$1" in
  build) echo "#7 [builder 1/9] RUN ./build_metal.sh"; exit 0 ;;
  save)
    d=$(mktemp -d)
    mkdir -p "$d/blobs/sha256"
    printf '{"imageLayoutVersion":"1.0.0"}' > "$d/oci-layout"
    printf '{"manifests":[]}' > "$d/index.json"
    printf 'layer-bytes' > "$d/blobs/sha256/aaaaaaaaaaaaaaaa"
    tar -C "$d" -cf - .
    exit 0 ;;
  run) echo "torch==2.11.0+cpu"; echo "vllm==0.24.0"; exit 0 ;;
  image)
    # `image inspect --format {{.Id}}` — the digest the final tag is derived from — and
    # `image rm` for dropping the provisional tag.
    case "$2" in
      inspect) echo "sha256:ee5226aec96561844c9371059d44338202eff46d1a2572815b716215205cbb0d"; exit 0 ;;
      rm) exit 0 ;;
    esac
    exit 1 ;;
  tag) exit 0 ;;
esac
exit 1
'''


def test_package_end_to_end_produces_a_publishable_directory(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, FAKE_DOCKER)
    metal = _fake_metal(tmp_path)

    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    build.run_build(staged)
    out = build.finalize(staged)

    # the four things a consumer's pull depends on
    assert (out / "tt_kernel_manifest.json").is_file()
    assert (out / "README.md").is_file()
    assert (out / "code" / "models" / "common" / "mod.py").is_file()
    assert (out / "image" / "oci-layout").is_file()
    assert (out / "image" / "blobs" / "sha256" / "aaaaaaaaaaaaaaaa").is_file()

    # the build context is gone; only the publishable tree remains
    assert not staged.ctx.exists()


def test_the_published_manifest_is_a_readable_v5_1_document(tmp_path, monkeypatch):
    from tt_kernel.manifest import Manifest

    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, FAKE_DOCKER)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    build.run_build(staged)
    out = build.finalize(staged)

    m = Manifest.from_json((out / "tt_kernel_manifest.json").read_text())
    assert m.schema_version == "5.1" and m.is_container
    assert m.container.image.tag == staged.image
    assert m.container.resolve_profile().name == "p150x4"
    assert m.weights.repo_id == "org/Weights-7B"
    # provenance survived into the published document
    assert m.container.built["tt_metal"]["sha"] == staged.built["tt_metal"]["sha"]
    assert m.container.built["repo"] == "you/my-model"


def test_the_env_is_frozen_from_the_image_when_no_lock_was_supplied(tmp_path, monkeypatch):
    """The first build resolves live; freezing is what makes every later build reproducible."""
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, FAKE_DOCKER)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    build.run_build(staged)
    out = build.finalize(staged)

    lock = (out / "requirements.lock").read_text()
    assert "torch==2.11.0+cpu" in lock and "vllm==0.24.0" in lock
    assert staged.manifest.runtime["lock"] == "requirements.lock"


def test_an_existing_lock_is_passed_through_untouched(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    _fake_docker(tmp_path, monkeypatch, FAKE_DOCKER)
    metal = _fake_metal(tmp_path)
    (tmp_path / "requirements.lock").write_text("pinned==1.0\n")
    rt = dict(json.loads(json.dumps(BASE))["runtime"], lock="requirements.lock")
    staged = build.stage(_manifest_file(tmp_path, metal, runtime=rt), out_root=tmp_path / "out")
    build.run_build(staged)
    out = build.finalize(staged)
    assert (out / "requirements.lock").read_text() == "pinned==1.0\n"


def test_uninitialised_submodules_are_caught_before_any_docker_work(tmp_path, monkeypatch):
    """tt-metal's CMakeLists refuses to configure without them. Catching it at stage time
    turns a failure that costs a base-image pull + a CMake configure into an instant one,
    and names the command that fixes it."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    sentinel = metal / build.SUBMODULE_SENTINEL
    assert sentinel.is_file()          # _fake_metal creates it
    sentinel.unlink()
    with pytest.raises(BuildError, match="submodule update --init --recursive"):
        build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")


# ------------------------------------------------------------------ the ignore list


def test_a_package_named_like_a_bring_up_artifact_is_NOT_dropped(tmp_path, monkeypatch):
    """`readiness_*` used to be in CODE_IGNORE and silently ate
    models/common/readiness_check — a real package models import. The failure surfaced
    only as a ModuleNotFoundError inside the finished image, after the full build."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    rc = metal / "models" / "common" / "readiness_check"
    rc.mkdir(parents=True)
    (rc / "contract.py").write_text("class Generator: pass\n")
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert (staged.ctx / "code" / "models" / "common" / "readiness_check" / "contract.py").is_file()


def test_the_ignore_list_holds_no_pattern_that_could_match_a_package_name():
    """Every entry must be build detritus or model weights — never a bare name that a
    real python package could plausibly have."""
    for pat in build.CODE_IGNORE:
        assert pat.startswith("*") or pat.startswith(".") or pat in (
            "__pycache__", "venv", "logs"
        ), f"{pat!r} could match a package directory"


def test_weights_are_still_dropped_from_inside_an_allowlisted_path(tmp_path, monkeypatch):
    """The narrowing must not reopen the 60 GB image hole."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    (metal / "models" / "common" / "w.safetensors").write_text("W")
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert not (staged.ctx / "code" / "models" / "common" / "w.safetensors").exists()


def test_whatever_the_ignore_list_drops_is_reported(tmp_path, monkeypatch):
    """A skip must never be silent — that is the whole failure mode."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    (metal / "models" / "common" / "w.safetensors").write_text("W")
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert any("w.safetensors" in p for p in staged.code_skipped)


def test_nothing_skipped_means_an_empty_report(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.code_skipped == []


def test_the_default_serve_script_lands_in_the_build_context(tmp_path, monkeypatch):
    """It becomes the image's CMD, so a bare `docker run` is configured correctly rather
    than silently misconfigured."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    script = (staged.ctx / "serve-default.sh").read_text()
    assert script.startswith("#!/bin/bash")
    assert "exec vllm serve" in script
    assert "/dev/tenstorrent" in script


# ------------------------------------------------------------------ local plugin checkout
#
# v5 shipped the author's own plugin WHEEL, so the plugin got the same hermetic treatment
# as tt-metal. v5.1 could only clone a pushed ref, which meant an author iterating on the
# plugin could not package what they were actually running — and a local-only commit
# failed the build outright, an hour in.


def _fake_plugin(root: Path, *, commit: bool = True) -> Path:
    d = root / "vllm-tt-plugin"
    (d / "src" / "vllm_tt_plugin").mkdir(parents=True)
    (d / "src" / "vllm_tt_plugin" / "__init__.py").write_text("x = 1\n")
    (d / "pyproject.toml").write_text("[project]\nname='vllm-tt-plugin'\nversion='0.1.0'\n")
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "x.pyc").write_text("c")
    if commit:
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
        subprocess.run(["git", "init", "-q", str(d)], check=True)
        subprocess.run(["git", "-C", str(d), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(d), "commit", "-qm", "i"], check=True, env=env)
    return d


def _with_local_plugin(tmp_path, metal, plugin, **over):
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt["plugin"] = {"path": str(plugin)}
    return _manifest_file(tmp_path, metal, runtime=rt, **over)


def test_a_local_plugin_checkout_is_staged_into_the_build_context(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal, plugin = _fake_metal(tmp_path), _fake_plugin(tmp_path)
    staged = build.stage(_with_local_plugin(tmp_path, metal, plugin),
                         out_root=tmp_path / "out")
    ctx = staged.ctx / "plugin-src"
    assert (ctx / "pyproject.toml").is_file()
    assert (ctx / "src" / "vllm_tt_plugin" / "__init__.py").is_file()
    assert not (ctx / "__pycache__").exists()   # build detritus dropped
    assert not (ctx / ".git").exists()


def test_the_install_line_uses_the_staged_tree_not_a_clone(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal, plugin = _fake_metal(tmp_path), _fake_plugin(tmp_path)
    build.stage(_with_local_plugin(tmp_path, metal, plugin), out_root=tmp_path / "out")
    from tt_kernel.container_manifest import load_container_manifest
    from tt_kernel.launchers import launcher_for

    m = load_container_manifest(_with_local_plugin(tmp_path, metal, plugin))
    lines = "\n".join(launcher_for(m.kind).install_lines(m))
    assert "/ctx/plugin-src" in lines
    assert "git clone" not in lines


def test_the_local_plugin_sha_and_dirty_flag_are_recorded(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal, plugin = _fake_metal(tmp_path), _fake_plugin(tmp_path)
    staged = build.stage(_with_local_plugin(tmp_path, metal, plugin),
                         out_root=tmp_path / "out")
    rec = staged.built["plugin"]
    assert len(rec["sha"]) == 40 and rec["dirty"] is False
    assert rec["path"] == str(plugin)


def test_an_uncommitted_plugin_is_packaged_but_flagged_dirty(tmp_path, monkeypatch):
    """The point of the hermetic default — but never claimed as pinned."""
    _no_network(monkeypatch)
    metal, plugin = _fake_metal(tmp_path), _fake_plugin(tmp_path)
    (plugin / "src" / "vllm_tt_plugin" / "__init__.py").write_text("x = 2  # local\n")
    staged = build.stage(_with_local_plugin(tmp_path, metal, plugin),
                         out_root=tmp_path / "out")
    assert staged.built["plugin"]["dirty"] is True


def test_a_path_that_is_not_a_python_package_is_refused(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    nope = tmp_path / "nope"
    nope.mkdir()
    with pytest.raises(BuildError, match="does not look like a python package"):
        build.stage(_with_local_plugin(tmp_path, metal, nope), out_root=tmp_path / "out")


def test_the_context_dir_exists_even_without_a_local_plugin(tmp_path, monkeypatch):
    """The Dockerfile COPYs it unconditionally; a missing dir would break every other
    model's build."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert (staged.ctx / "plugin-src").is_dir()
    assert not any((staged.ctx / "plugin-src").iterdir())


def test_a_local_plugin_ships_what_git_considers_the_project(tmp_path, monkeypatch):
    """A hardcoded exclude list is not enough: a real plugin checkout was carrying a 30 GB
    gitignored model_cache/ beside 928 KB of src/. The project already declares what is
    not part of it."""
    _no_network(monkeypatch)
    metal, plugin = _fake_metal(tmp_path), _fake_plugin(tmp_path)
    (plugin / ".gitignore").write_text("model_cache/\n")
    junk = plugin / "model_cache"
    junk.mkdir()
    (junk / "huge.bin").write_text("x" * 1000)
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "-C", str(plugin), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(plugin), "commit", "-qm", "ignore"], check=True, env=env)

    staged = build.stage(_with_local_plugin(tmp_path, metal, plugin),
                         out_root=tmp_path / "out")
    ctx = staged.ctx / "plugin-src"
    assert (ctx / "pyproject.toml").is_file()
    assert not (ctx / "model_cache").exists(), "gitignored artifacts must not be staged"


def test_uncommitted_edits_to_tracked_files_still_ship(tmp_path, monkeypatch):
    """The hermetic default: package what you validated, not what you remembered to commit."""
    _no_network(monkeypatch)
    metal, plugin = _fake_metal(tmp_path), _fake_plugin(tmp_path)
    (plugin / "src" / "vllm_tt_plugin" / "__init__.py").write_text("x = 99  # uncommitted\n")
    staged = build.stage(_with_local_plugin(tmp_path, metal, plugin),
                         out_root=tmp_path / "out")
    got = (staged.ctx / "plugin-src" / "src" / "vllm_tt_plugin" / "__init__.py").read_text()
    assert "uncommitted" in got
    assert staged.built["plugin"]["dirty"] is True


def test_a_non_git_plugin_directory_still_works(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    plugin = _fake_plugin(tmp_path, commit=False)
    staged = build.stage(_with_local_plugin(tmp_path, metal, plugin),
                         out_root=tmp_path / "out")
    assert (staged.ctx / "plugin-src" / "pyproject.toml").is_file()


# ------------------------------------------------------------------ fork provenance
#
# Local mode records the LOCAL checkout's HEAD, which is exactly right for a fork — but a
# 40-character sha with no remote is unfindable, and on a personal fork guessing the
# upstream repo does not help. The plugin always recorded its repo; tt-metal did not.


def test_the_remote_and_branch_are_recorded_for_a_local_checkout(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    subprocess.run(["git", "-C", str(metal), "remote", "add", "origin",
                    "https://github.com/someone/tt-metal.git"], check=True)
    subprocess.run(["git", "-C", str(metal), "checkout", "-qb", "my/fork-branch"], check=True)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    rec = staged.built["tt_metal"]
    assert rec["remote"] == "https://github.com/someone/tt-metal.git"
    assert rec["branch"] == "my/fork-branch"


def test_the_tracking_fork_is_recorded_when_it_contains_head(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    fork = tmp_path / "fork.git"
    subprocess.run(["git", "init", "-q", "--bare", str(fork)], check=True)
    subprocess.run(
        ["git", "-C", str(metal), "remote", "add", "origin", "https://github.com/upstream/tt-metal.git"],
        check=True,
    )
    subprocess.run(["git", "-C", str(metal), "remote", "add", "fork", str(fork)], check=True)
    subprocess.run(["git", "-C", str(metal), "checkout", "-qb", "my/fork-branch"], check=True)
    subprocess.run(["git", "-C", str(metal), "push", "-qu", "fork", "HEAD"], check=True)

    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    rec = staged.built["tt_metal"]
    assert rec["pushed"] is True
    assert rec["remote"] == str(fork)


def test_an_unpushed_commit_is_recorded_as_not_pushed(tmp_path, monkeypatch):
    """Packaging local work stays allowed — it is the hermetic default — but a manifest
    saying "built from <sha>" must not imply anyone can obtain that commit."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.built["tt_metal"]["pushed"] is False


def test_a_commit_reachable_from_a_remote_branch_is_recorded_as_pushed(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    # a bare repo to push into, so `git branch -r --contains HEAD` finds it
    bare = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    subprocess.run(["git", "-C", str(metal), "remote", "add", "origin", str(bare)], check=True)
    subprocess.run(["git", "-C", str(metal), "push", "-q", "origin", "HEAD:main"], check=True)
    subprocess.run(["git", "-C", str(metal), "fetch", "-q", "origin"], check=True)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.built["tt_metal"]["pushed"] is True


def test_the_fork_sha_is_the_local_head_not_an_upstream_one(tmp_path, monkeypatch):
    """Nothing assumes upstream tt-metal: the recorded sha is whatever the author's
    checkout is on, which is the whole point of the hermetic default."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    head = subprocess.run(["git", "-C", str(metal), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.built["tt_metal"]["sha"] == head


def test_an_unpushed_commit_is_reported_as_normal_not_as_a_defect(tmp_path, monkeypatch,
                                                                  capsys):
    """The target user is a community developer on a local branch or fork. The image ships
    the tree, so nothing ever resolves this sha — presenting it as something to fix made a
    normal situation read like a problem."""
    from tt_kernel import container_cli

    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    monkeypatch.setattr(container_cli, "run_build", lambda *a, **k: None)
    monkeypatch.setattr(container_cli, "finalize", lambda *a, **k: tmp_path / "out")
    try:
        container_cli.package_container(str(_manifest_file(tmp_path, metal)),
                                        out_root=str(tmp_path / "out"))
    except Exception:
        pass
    out = " ".join(capsys.readouterr().out.split())
    assert "that is fine" in out
    assert "push the branch" not in out          # no imperative to fix a non-problem
    assert "⚠" not in out                        # informational, not a warning

def _rt(**over):
    rt = json.loads(json.dumps(BASE))["runtime"]
    rt.update(over)
    return rt


def test_a_local_vllm_wheel_is_staged_and_installed(tmp_path, monkeypatch):
    """v5's --vllm-wheel: the binary the author actually ran, not a rebuild that may
    resolve differently."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    wheel = tmp_path / "vllm-0.24.0+empty-py3-none-any.whl"
    wheel.write_text("wheel")
    staged = build.stage(
        _manifest_file(tmp_path, metal, runtime=_rt(vllm={"wheel": str(wheel)})),
        out_root=tmp_path / "out")
    assert (staged.ctx / "wheels" / wheel.name).is_file()
    assert staged.built["vllm"] == {"wheel": wheel.name}

    from tt_kernel.container_manifest import load_container_manifest
    from tt_kernel.launchers import launcher_for
    m = load_container_manifest(
        _manifest_file(tmp_path, metal, runtime=_rt(vllm={"wheel": str(wheel)})))
    lines = "\n".join(launcher_for(m.kind).install_lines(m))
    assert "/ctx/wheels/vllm-*.whl" in lines
    assert "--no-binary vllm" not in lines      # no sdist rebuild


def test_a_local_vllm_source_tree_is_staged(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    vllm = tmp_path / "vllm"
    vllm.mkdir()
    (vllm / "pyproject.toml").write_text("[project]\nname='vllm'\nversion='0.24.0'\n")
    staged = build.stage(
        _manifest_file(tmp_path, metal, runtime=_rt(vllm={"path": str(vllm)})),
        out_root=tmp_path / "out")
    assert (staged.ctx / "vllm-src" / "pyproject.toml").is_file()


def test_extra_wheels_are_staged(tmp_path, monkeypatch):
    """v5's --extra-wheel."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    extra = tmp_path / "mylib-1.0-py3-none-any.whl"
    extra.write_text("w")
    staged = build.stage(
        _manifest_file(tmp_path, metal, runtime=_rt(wheels=[str(extra)])),
        out_root=tmp_path / "out")
    assert (staged.ctx / "wheels" / extra.name).is_file()


def test_an_extra_wheel_named_like_vllm_is_refused(tmp_path, monkeypatch):
    """The Dockerfile installs /ctx/wheels/vllm-*.whl, so such a name would be silently
    picked up as the engine."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    bad = tmp_path / "vllm-plugin-extras-1.0.whl"
    bad.write_text("w")
    with pytest.raises(BuildError, match="runtime.vllm.wheel for that"):
        build.stage(_manifest_file(tmp_path, metal, runtime=_rt(wheels=[str(bad)])),
                    out_root=tmp_path / "out")


def test_a_missing_extra_wheel_is_an_error(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    with pytest.raises(BuildError, match="does not exist"):
        build.stage(_manifest_file(tmp_path, metal, runtime=_rt(wheels=["/nope.whl"])),
                    out_root=tmp_path / "out")


def test_the_wheel_and_vllm_context_dirs_always_exist(tmp_path, monkeypatch):
    """The Dockerfile COPYs them unconditionally."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert (staged.ctx / "wheels").is_dir() and (staged.ctx / "vllm-src").is_dir()


def test_a_pinned_weights_revision_reaches_snapshot_download(tmp_path, monkeypatch):
    """Recording a revision achieves nothing unless pull actually requests it — the gap
    that let an author's weights and a consumer's differ silently."""
    from tt_kernel import container_cli

    seen = {}
    monkeypatch.setattr(container_cli, "snapshot_download", None, raising=False)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda **k: seen.update(k) or "/w")
    from tt_kernel.manifest import WeightsRef
    container_cli._download_weights(
        WeightsRef(repo="org/M", revision="abc123", ignore_patterns=["*.pt"]))
    assert seen["revision"] == "abc123"
    assert seen["ignore_patterns"] == ["*.pt"]


def test_a_globbed_vllm_wheel_path_resolves(tmp_path, monkeypatch):
    """Wheel names carry version and platform tags, so authors write dist/vllm-*.whl."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "vllm-0.24.0+empty-cp312-cp312-linux_x86_64.whl").write_text("w")
    staged = build.stage(
        _manifest_file(tmp_path, metal, runtime=_rt(vllm={"wheel": str(dist / "vllm-*.whl")})),
        out_root=tmp_path / "out")
    assert (staged.ctx / "wheels" / "vllm-0.24.0+empty-cp312-cp312-linux_x86_64.whl").is_file()


def test_a_wheel_pattern_matching_nothing_is_an_error(tmp_path, monkeypatch):
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    with pytest.raises(BuildError, match="no wheel matches"):
        build.stage(_manifest_file(tmp_path, metal,
                                   runtime=_rt(vllm={"wheel": str(tmp_path / "vllm-*.whl")})),
                    out_root=tmp_path / "out")


def test_the_source_commits_are_passed_as_build_args(tmp_path, monkeypatch):
    """So `docker inspect <digest-tagged image>` can still answer which commits built it."""
    _no_network(monkeypatch)
    metal = _fake_metal(tmp_path)
    staged = build.stage(_manifest_file(tmp_path, metal), out_root=tmp_path / "out")
    assert staged.build_args["MODEL_TT_METAL_SHA"] == staged.built["tt_metal"]["sha"]
    assert staged.build_args["MODEL_PLUGIN_SHA"] == staged.built["plugin"]["sha"]
    # describe() is empty when the tree carries no release tag to describe from — a real
    # checkout has one, this fixture does not. Passed through either way, never None.
    assert staged.build_args["MODEL_TT_METAL_DESCRIBE"] == (
        staged.built["tt_metal"]["describe"] or "")
