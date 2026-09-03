# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Offline tests for `serve --refresh`: the opt-in path that re-pulls + re-installs an already
installed self-contained bundle when the Hub has a newer revision, so a republished source is not
served with stale launch params. The Hub download, the install.sh subprocess, and the launch are
all stubbed — no hardware, no network.

The redesign's contract (a refresh must be SAFE and NON-FATAL) is what most of these pin:

- a refresh that fails for any reason falls back to serving the still-intact old install (exit 0);
- no baseline revision recorded  => no refresh at all;
- the tip resolution is bounded (never `timeout=None`);
- pre-staged weights survive a refresh;
- a custom `--models-dir` install is refreshed IN PLACE, not orphaned to the default dir;
- an explicit `@sha` is honored (installed + re-pinned);
- `--refresh` overrides `--no-update-check` (the explicit opt-in still hits the Hub);
- `--print` performs NO re-pull/rebuild and its stdout stays a bare command.
"""

from pathlib import Path

from typer.testing import CliRunner

from tt_kernel import cli, localdb, metal, packaging
from tt_kernel.manifest import WeightsRef

_runner = CliRunner()

_ID = "myorg/llama-3.2-3b-tt"


def _isolate(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # localdb
    monkeypatch.setenv("TT_MODEL_MODELS_DIR", str(tmp_path / "models"))  # install dir
    monkeypatch.setattr(metal, "local_env", lambda **k: metal.LocalEnv(arch="blackhole", device_count=1))


def _staged_bundle(tmp_path):
    """Stage a v5 self-contained bundle (ttnn wheel tagged for THIS interpreter so it's compatible)."""
    win = tmp_path / "in"
    win.mkdir()
    ttnn = win / f"ttnn-0.75.0-{packaging.host_python_tag()}-{packaging.host_python_tag()}-linux_x86_64.whl"
    ttnn.write_bytes(b"whl")
    plugin = win / "vllm_tt_plugin-0.3.0-py3-none-any.whl"
    plugin.write_bytes(b"whl")
    metal_dir = tmp_path / "metal"
    metal_dir.mkdir()
    (metal_dir / "requirements.txt").write_text("torch==2.11.0\n")
    staged = tmp_path / "snap_src"
    packaging.stage_package(
        staged, name="llama-3.2-3b-tt", arch="blackhole",
        ttnn_wheel=ttnn, plugin_wheel=plugin, metal_dir=metal_dir,
        vllm_metadata={"arch": "LlamaForCausalLM", "main_class": "generator_vllm:LlamaForCausalLM"},
        tt_kernel_version="0.0.0", weights=WeightsRef(repo="unsloth/Llama-3.2-3B-Instruct"),
        tt_metal_version="0.75.0",
    )
    return staged


def _record_installed(tmp_path, *, revision, pinned=False, install_dir=None, weights_path=None):
    inst = Path(install_dir) if install_dir else tmp_path / "models" / "myorg" / "llama-3.2-3b-tt"
    inst.mkdir(parents=True, exist_ok=True)
    run_sh = inst / "run.sh"
    run_sh.write_text("#!/usr/bin/env bash\necho ORIGINAL\n")  # a local edit we must not lose
    (inst / "venv" / "bin").mkdir(parents=True, exist_ok=True)  # marks a real install for the reuse gate
    (inst / "venv" / "bin" / "python").write_text("#!/bin/sh\n# ORIGINAL interpreter\n")
    localdb.record(_ID, {
        "repo_id": _ID, "self_contained": True,
        "install_dir": str(inst), "bundle_path": str(inst), "run_script": str(run_sh),
        "python": str(inst / "venv/bin/python"), "revision": revision, "pinned": pinned,
        "weights": "unsloth/Llama-3.2-3B-Instruct" if weights_path else None,
        "weights_path": weights_path,
    })
    return inst


def _install_fakes(staged, monkeypatch, *, install_raises=False, download_raises=False):
    """Wire download_bundle + install_self_contained fakes; return a dict capturing their calls."""
    seen = {"download_revs": [], "installs": 0, "weights_calls": 0}

    def _fake_download(repo_id, revision, dest):
        import shutil
        if download_raises:
            raise ConnectionError("simulated network blip during refresh download")
        seen["download_revs"].append(revision)
        snap = Path(dest) / f"snap{len(seen['download_revs'])}"
        shutil.copytree(staged, snap)
        return snap

    def _fake_install(bundle_dir, venv_dir):
        if install_raises:
            import subprocess
            raise subprocess.CalledProcessError(1, ["bash", "install.sh"])
        (venv_dir / "bin").mkdir(parents=True, exist_ok=True)
        py = venv_dir / "bin" / "python"
        py.write_text("#!/bin/sh\n# REFRESHED interpreter\n")
        seen["installs"] += 1
        return py

    def _fake_weights(weights, dest):
        seen["weights_calls"] += 1
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "model.safetensors").write_bytes(b"w")
        return dest

    monkeypatch.setattr(cli.hub, "download_bundle", _fake_download)
    monkeypatch.setattr(cli.runtime, "install_self_contained", _fake_install)
    monkeypatch.setattr(cli.runtime, "download_weights", _fake_weights)
    return seen


def _stub_serve_run(monkeypatch):
    """Stub the launch so a real (non --print) serve returns exit 0 without execing anything."""
    def _fake_run(argv, **kw):
        class _R:
            returncode = 0
        return _R()
    monkeypatch.setattr(cli.subprocess, "run", _fake_run)


# --------------------------------------------------------------------------- happy path


def test_serve_refresh_reinstalls_when_newer_tip(monkeypatch, tmp_path):
    # A newer revision is published; `serve --refresh` re-pulls + re-installs BEFORE serving.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 1, "a newer tip with --refresh should reinstall"
    assert seen["download_revs"] == ["newsha1111"]  # fetched exactly the resolved tip
    rec = localdb.get(_ID)
    assert rec["revision"] == "newsha1111"  # record updated
    assert rec["pinned"] is False  # no @rev given
    # the refreshed interpreter is now in place
    assert "REFRESHED" in Path(rec["python"]).read_text()


# --------------------------------------------------------------------------- inert / no-op


def test_serve_without_refresh_serves_installed_and_never_reinstalls(monkeypatch, tmp_path):
    # Default behavior is unchanged: a newer tip only WARNS; it must not download or reinstall.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0, "no --refresh: must serve the installed bundle, never reinstall"
    assert seen["download_revs"] == []
    assert "There is an update to myorg/llama-3.2-3b-tt" in res.output  # advisory still fires
    assert localdb.get(_ID)["revision"] == "oldsha0000"  # unchanged


def test_serve_refresh_noop_when_up_to_date(monkeypatch, tmp_path):
    # Tip == installed: --refresh resolves the tip, sees no diff, and serves as-is (no download).
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="samesha000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "samesha000")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


def test_serve_refresh_noop_when_offline(monkeypatch, tmp_path):
    # Hub unreachable (latest_revision -> None): never block a serve; serve the installed bundle.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: None)

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


def test_serve_refresh_ignored_with_local_only(monkeypatch, tmp_path):
    # --local-only must not touch the Hub even alongside --refresh.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("latest_revision must not be called with --local-only")
    monkeypatch.setattr(cli.hub, "latest_revision", _boom)

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh", "--local-only", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


def test_serve_refresh_skips_pinned_install(monkeypatch, tmp_path):
    # A pinned install (user chose @revision) must not be moved off it, even by --refresh.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000", pinned=True)
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []
    assert localdb.get(_ID)["revision"] == "oldsha0000"


def test_serve_refresh_noop_when_no_recorded_revision(monkeypatch, tmp_path):
    # Finding #4: an install predating the `revision` field has no honest baseline; --refresh must
    # NOT wipe/rebuild it, and must not even hit the Hub. Guard runs BEFORE latest_revision.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision=None)
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("latest_revision must not be called with no recorded baseline")
    monkeypatch.setattr(cli.hub, "latest_revision", _boom)

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


# --------------------------------------------------------------------------- safe / non-fatal


def test_serve_refresh_download_failure_still_serves_old(monkeypatch, tmp_path):
    # Finding #2: a network blip during the re-pull must NOT be fatal. Warn + serve the old bundle.
    _isolate(monkeypatch, tmp_path)
    inst = _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch, download_raises=True)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output  # served, not a fatal exit
    assert seen["installs"] == 0
    assert localdb.get(_ID)["revision"] == "oldsha0000"  # record untouched
    assert "ORIGINAL" in (inst / "run.sh").read_text()  # old tree fully intact
    assert "failed" in res.output  # a warning was shown


def test_serve_refresh_install_failure_keeps_old_bundle_intact(monkeypatch, tmp_path):
    # Finding #1: an install.sh failure AFTER the download must not destroy the working install.
    # The old tree (run.sh, venv) must survive and be served; the record must be unchanged.
    _isolate(monkeypatch, tmp_path)
    inst = _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch, install_raises=True)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output  # still serves the old bundle
    assert seen["installs"] == 0
    rec = localdb.get(_ID)
    assert rec["revision"] == "oldsha0000"  # record NOT moved to the half-built install
    # the ORIGINAL tree is restored: its run.sh edit and its interpreter are still there
    assert "ORIGINAL" in (inst / "run.sh").read_text()
    assert "ORIGINAL" in Path(rec["python"]).read_text()
    assert Path(rec["python"]).is_file()


# --------------------------------------------------------------------------- bounded timeout


def test_serve_refresh_uses_bounded_timeout(monkeypatch, tmp_path):
    # Finding #3: the tip resolution must be bounded, never timeout=None (which hangs a serve on a
    # half-open network). Assert the actual kwarg, not a swallowed *a,**k.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="samesha000")
    _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)

    captured = {}

    def _capture(repo_id, revision=None, timeout="MISSING"):
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        captured["timeout"] = timeout
        return "samesha000"  # up to date -> no reinstall, we only care about the call args
    monkeypatch.setattr(cli.hub, "latest_revision", _capture)

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert captured["timeout"] is not None, "must not resolve with timeout=None"
    assert captured["timeout"] == 3.0, f"expected the bounded 3s timeout, got {captured['timeout']!r}"


# --------------------------------------------------------------------------- weights preserved


def test_serve_refresh_preserves_prestaged_weights(monkeypatch, tmp_path):
    # Finding #5: a user who pre-downloaded weights keeps them across a refresh (with_weights is
    # driven by the recorded weights_path), instead of silently deleting + nulling them.
    _isolate(monkeypatch, tmp_path)
    inst = _record_installed(tmp_path, revision="oldsha0000",
                             weights_path=str(tmp_path / "models" / "myorg" / "llama-3.2-3b-tt" / "weights"))
    (inst / "weights").mkdir(exist_ok=True)
    (inst / "weights" / "model.safetensors").write_bytes(b"old")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["weights_calls"] == 1, "pre-staged weights must be re-fetched, not dropped"
    rec = localdb.get(_ID)
    assert rec["weights_path"], "weights_path must survive the refresh (not be nulled)"
    assert Path(rec["weights_path"]).is_dir()


def test_serve_refresh_no_weights_when_none_prestaged(monkeypatch, tmp_path):
    # The converse: an install with no pre-staged weights does not suddenly fetch them on refresh.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")  # weights_path is None
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["weights_calls"] == 0


# --------------------------------------------------------------------------- custom models-dir


def test_serve_refresh_in_place_for_custom_models_dir(monkeypatch, tmp_path):
    # Finding #6: an install placed on a custom --models-dir must be refreshed IN PLACE (its
    # recorded install_dir), not orphaned into the default resolve_models_dir(None, repo_id).
    _isolate(monkeypatch, tmp_path)
    custom = tmp_path / "big_disk" / "llama"
    _record_installed(tmp_path, revision="oldsha0000", install_dir=str(custom))
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    # sanity: the default dir is a DIFFERENT path, so an in-place refresh is observable
    from tt_kernel import runtime
    default_dir = runtime.resolve_models_dir(None, _ID)
    assert Path(default_dir) != custom

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 1
    rec = localdb.get(_ID)
    assert Path(rec["install_dir"]) == custom, "refresh must stay on the custom models-dir"
    assert not Path(default_dir).exists(), "must not orphan a copy into the default models-dir"
    assert "REFRESHED" in (custom / "venv" / "bin" / "python").read_text()


# --------------------------------------------------------------------------- explicit @sha pin


def test_serve_refresh_honors_explicit_sha(monkeypatch, tmp_path):
    # Finding #7: `serve <id>@<sha> --refresh` must install exactly that rev and re-pin it, not move
    # to the default-branch tip.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)

    captured = {}

    def _resolve(repo_id, revision=None, timeout=None):
        captured["revision_arg"] = revision
        return "pinnedsha22" if revision == "v2-tag" else "tipsha9999"
    monkeypatch.setattr(cli.hub, "latest_revision", _resolve)

    res = _runner.invoke(cli.app, ["serve", f"{_ID}@v2-tag", "--refresh"])
    assert res.exit_code == 0, res.output
    assert captured["revision_arg"] == "v2-tag", "the user's @rev must be threaded into the resolve"
    assert seen["download_revs"] == ["pinnedsha22"]  # installed that exact rev
    rec = localdb.get(_ID)
    assert rec["revision"] == "pinnedsha22"
    assert rec["pinned"] is True, "an explicit @rev must be recorded as pinned"


# --------------------------------------------------------------------------- precedence


def test_refresh_overrides_no_update_check(monkeypatch, tmp_path):
    # Finding #8: --refresh is the explicit opt-in and still hits the Hub even with --no-update-check.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)
    monkeypatch.setattr(cli.hub, "latest_revision", lambda *a, **k: "newsha1111")

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh", "--no-update-check"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 1, "--refresh must win over --no-update-check and still refresh"
    assert seen["download_revs"] == ["newsha1111"]


def test_no_update_check_alone_suppresses_advisory(monkeypatch, tmp_path):
    # The other half of the precedence: --no-update-check WITHOUT --refresh skips the Hub entirely.
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("--no-update-check must skip the advisory Hub request")
    monkeypatch.setattr(cli.hub, "latest_revision", _boom)

    res = _runner.invoke(cli.app, ["serve", _ID, "--no-update-check", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []


# --------------------------------------------------------------------------- --print purity


def test_print_refresh_does_no_rebuild(monkeypatch, tmp_path):
    # Finding #9: `--refresh --print` must NOT hit the Hub, rmtree, or rebuild. stdout stays a bare
    # command (no refresh/install chatter); the "skipped" note goes to stderr.
    _isolate(monkeypatch, tmp_path)
    inst = _record_installed(tmp_path, revision="oldsha0000")
    seen = _install_fakes(_staged_bundle(tmp_path), monkeypatch)
    _stub_serve_run(monkeypatch)

    def _boom(*a, **k):
        raise AssertionError("--print must not resolve the tip or rebuild")
    monkeypatch.setattr(cli.hub, "latest_revision", _boom)

    res = _runner.invoke(cli.app, ["serve", _ID, "--refresh", "--print"])
    assert res.exit_code == 0, res.output
    assert seen["installs"] == 0 and seen["download_revs"] == []  # no rebuild, no re-pull
    assert "ORIGINAL" in (inst / "run.sh").read_text()  # install untouched
    # no refresh/install chatter anywhere in stdout
    stdout = res.output.replace(res.stderr, "")
    assert "↻ refreshing" not in stdout
    assert "Installing shipped wheels" not in stdout
    assert "refreshed self-contained bundle" not in stdout
    # the skip note went to stderr
    assert "--refresh skipped under --print" in res.stderr


# --------------------------------------------------------------------------- default port


def test_bundle_serve_defaults_to_port_20000_or_walks_upward(monkeypatch, tmp_path):
    """No --port anywhere: the launch gets `--port 20000` (or the next free port when
    this box has it taken) instead of inheriting vLLM's 8000."""
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    argvs = []

    def _fake_run(argv, **kw):
        argvs.append(argv)

        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    res = _runner.invoke(cli.app, ["serve", _ID, "--no-update-check"])
    assert res.exit_code == 0, res.output
    launch = argvs[-1]
    assert int(launch[launch.index("--port") + 1]) >= 20000


def test_bundle_serve_explicit_port_suppresses_the_default(monkeypatch, tmp_path):
    """--port is exact, appended last so argparse last-wins keeps the user's value."""
    _isolate(monkeypatch, tmp_path)
    _record_installed(tmp_path, revision="oldsha0000")
    argvs = []

    def _fake_run(argv, **kw):
        argvs.append(argv)

        class _R:
            returncode = 0
        return _R()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)
    res = _runner.invoke(cli.app, ["serve", "--port", "7009", _ID, "--no-update-check"])
    assert res.exit_code == 0, res.output
    launch = argvs[-1]
    ports = [launch[i + 1] for i, a in enumerate(launch) if a == "--port"]
    assert ports == ["7009"]
