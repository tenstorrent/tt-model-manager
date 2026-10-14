# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tt-model rm --all` — the tt-studio "purge all" equivalent.

Each bundle goes through the same removal `rm <id>` does; one failure must not shield
the rest; and the destructive sweep asks first unless --yes.
"""

from typer.testing import CliRunner

from tt_kernel import cli, container_cli, localdb

runner = CliRunner()


def _install_two_venv_bundles(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    dirs = {}
    for rid in ("org/alpha", "org/beta"):
        d = tmp_path / rid.replace("/", "__")
        d.mkdir()
        (d / "run.sh").write_text("#!/bin/sh\n")
        localdb.record(rid, {"install_dir": str(d), "self_contained": True})
        dirs[rid] = d
    return dirs


def test_rm_all_removes_every_installed_bundle_with_yes(tmp_path, monkeypatch):
    dirs = _install_two_venv_bundles(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rm", "--all", "--yes"])
    assert res.exit_code == 0, res.output
    assert localdb.all_entries() == []
    assert not any(d.exists() for d in dirs.values())
    assert "removed all 2" in res.output


def test_rm_all_asks_first_and_a_no_removes_nothing(tmp_path, monkeypatch):
    dirs = _install_two_venv_bundles(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rm", "--all"], input="n\n")
    assert res.exit_code == 1, res.output
    assert "Aborted" in res.output
    assert len(localdb.all_entries()) == 2
    assert all(d.exists() for d in dirs.values())


def test_rm_all_routes_a_container_entry_through_remove_container(tmp_path, monkeypatch):
    """A container entry has no install_dir; `--all` must take the container path for
    it (containers, image, caches), not silently drop the index row."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    localdb.record("org/boxed", {"container": True})
    seen = {}
    monkeypatch.setattr(container_cli, "load_pulled", lambda rid: object())

    def fake_remove(rid, mani, *, keep_cache, include_weights):
        seen[rid] = (keep_cache, include_weights)
        localdb.remove(rid)

    monkeypatch.setattr(container_cli, "remove_container", fake_remove)
    res = runner.invoke(cli.app, ["rm", "--all", "-y", "--include-weights"])
    assert res.exit_code == 0, res.output
    assert seen == {"org/boxed": (False, True)}


def test_rm_all_keeps_going_past_one_failure_and_reports_it(tmp_path, monkeypatch):
    dirs = _install_two_venv_bundles(tmp_path, monkeypatch)
    localdb.record("org/broken", {"container": True})
    monkeypatch.setattr(container_cli, "load_pulled", lambda rid: object())

    def boom(rid, mani, **kw):
        raise container_cli.ContainerCliError("docker daemon is down")

    monkeypatch.setattr(container_cli, "remove_container", boom)
    res = runner.invoke(cli.app, ["rm", "--all", "-y"])
    assert res.exit_code == 1, res.output
    assert not any(d.exists() for d in dirs.values()), "the healthy bundles still went"
    assert [e["repo_id"] for e in localdb.all_entries()] == ["org/broken"]
    assert "Removed 2 of 3" in res.output and "org/broken" in res.output


def test_rm_all_with_nothing_installed_is_a_quiet_no_op(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    res = runner.invoke(cli.app, ["rm", "--all", "-y"])
    assert res.exit_code == 0, res.output
    assert "nothing is installed" in res.output


def test_rm_refuses_both_an_id_and_all(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    res = runner.invoke(cli.app, ["rm", "org/alpha", "--all"])
    assert res.exit_code == 1
    assert "not both" in res.output


def test_rm_without_an_id_or_all_says_what_to_do(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    res = runner.invoke(cli.app, ["rm"])
    assert res.exit_code == 1
    assert "--all" in res.output


def test_rm_single_id_still_works(tmp_path, monkeypatch):
    dirs = _install_two_venv_bundles(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rm", "org/alpha"])
    assert res.exit_code == 0, res.output
    assert not dirs["org/alpha"].exists() and dirs["org/beta"].exists()
    assert [e["repo_id"] for e in localdb.all_entries()] == ["org/beta"]
