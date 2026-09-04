# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Hub-failure classification: the wording matrix, with no network.

``classify_hub_error`` is pure (exception in, dict out) precisely so this can be a plain
unit test. The fixtures below are the *real* messages huggingface_hub produced on this
box — a classifier tested against invented wording only proves it handles the wording you
imagined.
"""

import pytest
from typer.testing import CliRunner

from tt_kernel import MANIFEST_NAME
from tt_kernel.hub import classify_hub_error, classify_repo_id

runner = CliRunner()


class _Resp:
    def __init__(self, code):
        self.status_code = code


def _exc(name, message, status=None):
    """Build a stand-in that looks like the huggingface_hub exception of that name."""
    cls = type(name, (Exception,), {})
    e = cls(message)
    if status is not None:
        e.response = _Resp(status)
    return e


# Captured verbatim from `tt-model pull mando2222/llama-3.2-3b-e2e`.
REPO_NOT_FOUND = (
    "404 Client Error. (Request ID: Root=1-6a877753-050422286266670657ac89bb;a1af2465)\n"
    "\n"
    "Repository Not Found for url: https://huggingface.co/api/models/x/y/revision/main.\n"
    "Please make sure you specified the correct `repo_id` and `repo_type`."
)
UNAUTHORIZED = "401 Client Error. (Request ID: Root=1-6a878f2d-60019c1d;350dfb25)"

# Captured verbatim from `snapshot_download("jashan/kabadoo")` with huggingface_hub 1.27.0.
# Note the LAST line: every not-found message now mentions a "private or gated repo", which
# is what used to route a plain typo to "the Hub recognised X but refused it".
REPO_NOT_FOUND_HF127 = (
    "404 Client Error. (Request ID: Root=1-6a99c88d-70f4e46d6a289145589a22ea;097dfff0)\n"
    "\n"
    "Repository Not Found for url: https://huggingface.co/api/models/jashan/kabadoo/revision/main.\n"
    "Please make sure you specified the correct `repo_id` and `repo_type`.\n"
    "If you are trying to access a private or gated repo, make sure you are authenticated "
    "and your token has the required permissions.\n"
    "For more details, see https://huggingface.co/docs/huggingface_hub/authentication"
)


def test_404_names_both_possibilities():
    """The Hub answers 404 for "no such repo" AND "private repo you can't see"; it will
    not distinguish them for an unauthorised caller. Asserting either one alone sends half
    of users down the wrong path."""
    d = classify_hub_error(_exc("RepositoryNotFoundError", REPO_NOT_FOUND, 404), "x/y")
    assert d["cause"] == "no such bundle, or no access"
    assert "id is wrong" in d["detail"]
    assert "private" in d["detail"]


def test_not_found_with_gated_boilerplate_is_still_not_found():
    """huggingface_hub 1.27 appends "...a private or gated repo..." to EVERY not-found
    message. A repo that does not exist must not be reported as one the Hub recognised."""
    d = classify_hub_error(
        _exc("RepositoryNotFoundError", REPO_NOT_FOUND_HF127, 404), "jashan/kabadoo")
    assert d["cause"] == "no such bundle, or no access"
    assert "recognised" not in d["detail"]
    assert not any("accept the terms" in a for a in d["actions"])


def test_not_found_type_wins_over_a_401_status():
    """An unauthenticated caller gets 401 from the Hub for a repo that simply does not
    exist; huggingface_hub still raises RepositoryNotFoundError. The type is the signal."""
    d = classify_hub_error(
        _exc("RepositoryNotFoundError", REPO_NOT_FOUND_HF127, 401), "jashan/kabadoo")
    assert d["cause"] == "no such bundle, or no access"


def test_a_bare_401_is_a_credentials_problem_not_a_gate():
    """A bare 401 (not a RepositoryNotFoundError, not a GatedRepoError) means the token
    itself was rejected. It used to share the gated card, so an expired token told the
    user to go "accept the terms" — a button that was never the problem here."""
    d = classify_hub_error(_exc("HfHubHTTPError", UNAUTHORIZED, 401), "x/y")
    assert d["cause"] == "your token was rejected"
    assert any("login" in a for a in d["actions"])
    assert not any("accept the terms" in a for a in d["actions"])


def test_403_is_authorisation():
    d = classify_hub_error(_exc("HfHubHTTPError", "403 Forbidden", 403), "x/y")
    assert d["cause"] == "not authorised"


def test_gated_repo_routes_to_access():
    d = classify_hub_error(_exc("GatedRepoError", "Access to model x/y is restricted"), "x/y")
    assert d["cause"] == "you do not have access"
    assert any("huggingface.co/x/y" in a for a in d["actions"])


def test_offline_beats_not_found():
    """An offline machine reports both a connection failure and an unresolvable repo.
    "You are offline" is the actionable half, so it must win."""
    d = classify_hub_error(
        _exc("LocalEntryNotFoundError", "Max retries exceeded ... not found"), "x/y")
    assert d["cause"] == "cannot reach the Hub"


def test_missing_manifest_is_not_a_bundle():
    d = classify_hub_error(
        _exc("EntryNotFoundError", "tt_kernel_manifest.json does not exist"), "x/y")
    assert d["cause"] == "not a tt-model bundle"


def test_a_real_missing_manifest_404_is_not_reported_as_a_missing_repo():
    """The Hub answers 404 for a missing FILE too, and a real EntryNotFoundError carries
    that exact status code — the earlier fixture above omits both the "404 Client Error."
    prefix and a status, so it never exercised the ``status == 404`` branch at all. With a
    realistic exception, status alone can't tell "no such repo" from "repo exists, no
    manifest" apart; only the class can. Checking the status-404 branch before the class
    check turned "this repo ships no manifest" into "no such bundle, or no access"."""
    real = ("404 Client Error. (Request ID: Root=1-6a9843cb-aaaa;bbbb)\n\n"
            "Entry Not Found for url: https://huggingface.co/x/y/resolve/main/"
            f"{MANIFEST_NAME}.")
    d = classify_hub_error(_exc("EntryNotFoundError", real, 404), "x/y")
    assert d["cause"] == "not a tt-model bundle"


def test_unknown_error_still_gives_a_card_and_an_escape_hatch():
    d = classify_hub_error(_exc("WeirdError", "something novel"), "x/y")
    assert d["cause"] == "the Hub request failed"
    assert any("--verbose" in a for a in d["actions"])


def test_every_branch_returns_the_full_card_contract():
    """failure_card reads all four keys; a branch missing one would raise at render time,
    i.e. while already reporting another error."""
    cases = [
        _exc("RepositoryNotFoundError", REPO_NOT_FOUND, 404),
        _exc("RepositoryNotFoundError", REPO_NOT_FOUND_HF127, 404),
        _exc("GatedRepoError", "Access to model x/y is restricted", 401),
        _exc("HfHubHTTPError", UNAUTHORIZED, 401),
        _exc("HfHubHTTPError", "403", 403),
        _exc("LocalEntryNotFoundError", "Max retries exceeded"),
        _exc("EntryNotFoundError", "tt_kernel_manifest.json missing"),
        _exc("WeirdError", "novel"),
    ]
    for exc in cases:
        d = classify_hub_error(exc, "x/y")
        assert set(d) == {"cause", "detail", "evidence", "actions"}, type(exc).__name__
        assert d["cause"] and d["detail"] and d["actions"]
    d = classify_repo_id("kabadoo")
    assert set(d) == {"cause", "detail", "evidence", "actions"}
    assert d["cause"] and d["detail"] and d["actions"]


def test_evidence_drops_the_request_id():
    """The Request ID is for a support ticket, not for the person reading the terminal."""
    d = classify_hub_error(_exc("RepositoryNotFoundError", REPO_NOT_FOUND, 404), "x/y")
    assert d["evidence"] == "404 Client Error."
    assert "Request ID" not in d["evidence"]


def test_evidence_survives_an_empty_message():
    d = classify_hub_error(_exc("WeirdError", ""), "x/y")
    assert d["evidence"] == ""


@pytest.mark.parametrize("marker", ["no such host", "dial tcp", "i/o timeout",
                                    "TLS handshake", "Connection refused"])
def test_network_markers_all_classify_as_offline(marker):
    d = classify_hub_error(_exc("OSError", f"failed: {marker}"), "x/y")
    assert d["cause"] == "cannot reach the Hub"


def test_cli_renders_a_card_not_a_traceback(monkeypatch):
    """The end-to-end contract: no stack frames reach the user by default."""
    from tt_kernel import cli, hub

    def boom(*a, **k):
        raise _exc("RepositoryNotFoundError", REPO_NOT_FOUND, 404)

    monkeypatch.setattr(hub, "fetch_manifest", boom)
    res = runner.invoke(cli.app, ["info", "x/y"])
    assert res.exit_code == 1
    assert "no such bundle, or no access" in res.output
    assert "Traceback" not in res.output
    assert "hf_raise_for_status" not in res.output


def test_push_replaces_code_and_image_rather_than_merging(monkeypatch, tmp_path):
    """A push must not leave behind what the bundle stopped shipping.

    `upload_folder` only adds unless told otherwise, so narrowing a `source.code`
    allowlist previously left every dropped file published: the repo advertised `code/`
    as byte-identical to the image while carrying files the image did not have. Image
    blobs are content-addressed and accumulate the same way across rebuilds.
    """
    from tt_kernel import hub

    seen = {}

    class _Api:
        def upload_folder(self, **kw):
            seen.update(kw)

    monkeypatch.setattr(hub, "_api", lambda: _Api())
    hub.push_folder("you/model", tmp_path, "msg")

    assert seen["delete_patterns"] == ["code/**", "image/**"]
    # .gitattributes is HF's own LFS config and the top-level files are rewritten every
    # push, so neither should be swept.
    assert not any(p == "*" or ".gitattributes" in p for p in seen["delete_patterns"])


def test_large_push_prunes_what_the_bundle_stopped_shipping(monkeypatch, tmp_path):
    """The CONTAINER path is push_large_folder, and upload_large_folder has no
    delete_patterns, so it only ever adds. Narrowing an allowlist previously left every
    dropped file published. Prune must run against the same folder that was uploaded."""
    from tt_kernel import hub

    (tmp_path / "code" / "models").mkdir(parents=True)
    (tmp_path / "code" / "models" / "keep.py").write_text("x")
    (tmp_path / "image" / "blobs").mkdir(parents=True)
    (tmp_path / "image" / "blobs" / "sha-keep").write_text("y")
    (tmp_path / "README.md").write_text("z")

    calls = {}

    class _Api:
        def upload_large_folder(self, **kw):
            calls["uploaded"] = True

        def list_repo_files(self, **kw):
            return [
                "code/models/keep.py",
                "code/models/tests/gone.py",      # dropped by a narrowed allowlist
                "image/blobs/sha-keep",
                "image/blobs/sha-stale",          # superseded blob from an older build
                "README.md",                      # rewritten every push
                ".gitattributes",                 # HF's own LFS config
            ]

        def delete_files(self, **kw):
            calls["deleted"] = sorted(kw["delete_patterns"])

    monkeypatch.setattr(hub, "_api", lambda: _Api())
    hub.push_large_folder("you/model", tmp_path)

    assert calls["uploaded"] is True
    assert calls["deleted"] == ["code/models/tests/gone.py", "image/blobs/sha-stale"]


def test_large_push_makes_no_delete_commit_when_nothing_is_stale(monkeypatch, tmp_path):
    from tt_kernel import hub

    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "a.py").write_text("x")

    calls = {}

    class _Api:
        def upload_large_folder(self, **kw):
            pass

        def list_repo_files(self, **kw):
            return ["code/a.py", ".gitattributes"]

        def delete_files(self, **kw):
            calls["deleted"] = True

    monkeypatch.setattr(hub, "_api", lambda: _Api())
    hub.push_large_folder("you/model", tmp_path)
    assert "deleted" not in calls


# ------------------------------------------------------------------ bare ids never reach the Hub
# `tt-model serve kabadoo` used to ask the Hub for a canonical (namespace-less) model, get a
# 404, and — worse — misreport it as "gated". A bundle is always namespace/name, so a bare
# name is refused with a definite card before any request is made.


def test_bare_name_is_not_a_repo_id():
    d = classify_repo_id("kabadoo")
    assert d["cause"] == "not a repo id"
    assert "namespace/name" in d["detail"]
    assert any(a.startswith("tt-model search kabadoo") for a in d["actions"])


@pytest.mark.parametrize("repo_id", ["ns/name", "tenstorrent/llama-3.1-8b"])
def test_namespaced_id_passes(repo_id):
    assert classify_repo_id(repo_id) is None


def _hub_must_not_be_called(monkeypatch):
    from tt_kernel import hub

    def boom(*a, **k):
        raise AssertionError("the Hub was asked about a bare id")

    for fn in ("latest_revision", "download_bundle", "fetch_manifest"):
        monkeypatch.setattr(hub, fn, boom)


@pytest.mark.parametrize("argv", [["pull", "kabadoo"], ["info", "kabadoo"]])
def test_pull_and_info_refuse_a_bare_name_before_the_hub(monkeypatch, argv):
    from tt_kernel import cli

    _hub_must_not_be_called(monkeypatch)
    res = runner.invoke(cli.app, argv)
    assert res.exit_code == 1, res.output
    assert "not a repo id" in res.output
    assert "recognised" not in res.output
    assert "Traceback" not in res.output


def test_serve_refuses_a_bare_name_that_is_not_installed(monkeypatch, tmp_path):
    from tt_kernel import cli, container_cli

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))  # empty localdb
    monkeypatch.setattr(container_cli, "resolve_target", lambda *a, **k: None)
    _hub_must_not_be_called(monkeypatch)
    res = runner.invoke(cli.app, ["serve", "kabadoo"])
    assert res.exit_code == 1, res.output
    assert "not a repo id" in res.output
    assert "recognised" not in res.output


def test_serve_still_serves_an_installed_bare_name(monkeypatch):
    """The shape check sits AFTER local resolution: whatever localdb already holds under a
    bare name keeps serving; only what would be sent to the Hub is judged."""
    from tt_kernel import cli, container_cli

    served = []
    monkeypatch.setattr(container_cli, "resolve_target", lambda *a, **k: None)
    monkeypatch.setattr(cli.localdb, "get", lambda rid: {"self_contained": True, "repo_id": rid})
    monkeypatch.setattr(cli, "_serve_self_contained", lambda entry, **k: served.append(entry))
    _hub_must_not_be_called(monkeypatch)
    res = runner.invoke(cli.app, ["serve", "kabadoo", "--no-update-check"])
    assert res.exit_code == 0, res.output
    assert served and served[0]["repo_id"] == "kabadoo"
