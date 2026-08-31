# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Repo-visibility semantics shared by push / package / package-thin.

Three orthogonal axes the CLI keeps separate:
  * push        — upload the bundle's files to an HF repo
  * public/private — the repo's visibility (--public / --private)
  * publish     — opt into the community-catalog listing (--publish), which requires --public

The rules under test:
  * a NEW repo is created PRIVATE by default (tt-model never makes something public by omission);
  * an EXISTING repo's visibility is never flipped unless the user passes the flag (and a flip is
    announced); and
  * --publish requires an explicit --public, refused before anything is uploaded.
All three commands route creation through `_ensure_repo`, so the unit tests below pin the core.
"""

from pathlib import Path

from typer.testing import CliRunner

from tt_kernel import cli, hub

runner = CliRunner()


# --------------------------------------------------------------- _ensure_repo (the shared core)
def test_new_repo_is_private_by_default(monkeypatch):
    created = {}
    monkeypatch.setattr(hub, "repo_exists", lambda r: False)
    monkeypatch.setattr(hub, "create_repo", lambda r, private: created.update(repo=r, private=private))
    cli._ensure_repo("me/new", None)  # user said nothing
    assert created == {"repo": "me/new", "private": True}


def test_new_repo_public_only_when_asked(monkeypatch):
    created = {}
    monkeypatch.setattr(hub, "repo_exists", lambda r: False)
    monkeypatch.setattr(hub, "create_repo", lambda r, private: created.update(private=private))
    cli._ensure_repo("me/new", False)  # --public
    assert created["private"] is False


def test_existing_repo_visibility_untouched_by_omission(monkeypatch):
    flips = []
    monkeypatch.setattr(hub, "repo_exists", lambda r: True)
    monkeypatch.setattr(hub, "set_visibility", lambda r, private: flips.append(private))
    cli._ensure_repo("me/exists", None)  # said nothing -> change nothing
    assert flips == []


def test_existing_repo_flipped_only_when_explicit(monkeypatch):
    flips = []
    monkeypatch.setattr(hub, "repo_exists", lambda r: True)
    monkeypatch.setattr(hub, "is_private_safe", lambda r: False)  # currently public
    monkeypatch.setattr(hub, "set_visibility", lambda r, private: flips.append(private))
    res_lines = []
    monkeypatch.setattr(cli.typer, "secho", lambda *a, **k: res_lines.append(a[0] if a else ""))
    cli._ensure_repo("me/exists", True)  # --private, and it means it
    assert flips == [True]  # flipped, explicitly
    assert any("Changed visibility" in line for line in res_lines)  # and announced


# --------------------------------------------------------------- package-thin routing / defaults
def _model_py(tmp_path: Path) -> Path:
    p = tmp_path / "model.py"
    p.write_text("class C:  # runner\n    pass\n")
    return p


def _thin_argv(tmp_path, *extra, repo="me/thin"):
    return [
        "package-thin", repo, "--model-py", str(_model_py(tmp_path)),
        "--arch", "blackhole", "--arch-name", "QwenForCausalLM", "--main-class", "model:C",
        "--weights", "Qwen/Qwen3-4B", "--mesh", "P150", *extra,
    ]


def test_package_thin_routes_creation_through_ensure_repo_private_by_default(monkeypatch, tmp_path):
    # The v6 push must go through _ensure_repo (private by default, no silent flip) rather than the
    # old unconditional create_repo + set_visibility that flipped an existing repo on every push.
    seen = {}
    monkeypatch.setattr(cli, "_ensure_repo", lambda repo_id, private: seen.update(repo=repo_id, private=private))
    monkeypatch.setattr(hub, "push_folder", lambda *a, **k: None)
    monkeypatch.setattr(hub, "tag_repo", lambda *a, **k: None)
    # If the old path were still live it would call these directly — make that a hard failure.
    monkeypatch.setattr(hub, "create_repo", lambda *a, **k: pytest_fail("create_repo called directly"))
    monkeypatch.setattr(hub, "set_visibility", lambda *a, **k: pytest_fail("set_visibility called directly"))

    res = runner.invoke(cli.app, _thin_argv(tmp_path))
    assert res.exit_code == 0, res.output
    assert seen == {"repo": "me/thin", "private": None}  # tri-state None => _ensure_repo makes it private


def test_package_thin_publish_implies_public(monkeypatch, tmp_path):
    # --publish needs no separate --public: it forces public (private=False into _ensure_repo) and
    # lists (package/package-thin list by writing the catalog tag via tag_repo). One flag, not two.
    from tt_kernel import TT_MODEL_CATALOG_TAG

    seen = {"ensure_private": "unset", "tags": None}
    monkeypatch.setattr(cli, "_ensure_repo",
                        lambda repo_id, private: seen.update(ensure_private=private))
    monkeypatch.setattr(hub, "push_folder", lambda *a, **k: None)
    monkeypatch.setattr(hub, "tag_repo", lambda repo_id, tags: seen.update(tags=list(tags)))

    res = runner.invoke(cli.app, _thin_argv(tmp_path, "--publish"))
    assert res.exit_code == 0, res.output
    assert seen["ensure_private"] is False  # --publish implied --public
    assert TT_MODEL_CATALOG_TAG in seen["tags"]  # and listed it (catalog tag written)


def test_package_thin_publish_with_private_conflicts(monkeypatch, tmp_path):
    # The one refusal: --publish (public by definition) can't be combined with --private.
    calls = []
    monkeypatch.setattr(cli, "_ensure_repo", lambda *a, **k: calls.append("ensure"))
    monkeypatch.setattr(hub, "push_folder", lambda *a, **k: calls.append("push"))
    res = runner.invoke(cli.app, _thin_argv(tmp_path, "--publish", "--private"))
    assert res.exit_code == 1
    assert "--public" in res.output  # message points at the resolution
    assert calls == []  # refused before anything happens


def pytest_fail(msg):  # tiny helper so the lambdas above read cleanly
    raise AssertionError(msg)
