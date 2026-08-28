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

from tt_kernel.hub import classify_hub_error

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


def test_404_names_both_possibilities():
    """The Hub answers 404 for "no such repo" AND "private repo you can't see"; it will
    not distinguish them for an unauthorised caller. Asserting either one alone sends half
    of users down the wrong path."""
    d = classify_hub_error(_exc("RepositoryNotFoundError", REPO_NOT_FOUND, 404), "x/y")
    assert d["cause"] == "no such bundle, or no access"
    assert "id is wrong" in d["detail"]
    assert "private" in d["detail"]


def test_401_is_access_not_missing():
    d = classify_hub_error(_exc("HfHubHTTPError", UNAUTHORIZED, 401), "x/y")
    assert d["cause"] == "you do not have access"
    assert any("login" in a for a in d["actions"])


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


def test_unknown_error_still_gives_a_card_and_an_escape_hatch():
    d = classify_hub_error(_exc("WeirdError", "something novel"), "x/y")
    assert d["cause"] == "the Hub request failed"
    assert any("--verbose" in a for a in d["actions"])


def test_every_branch_returns_the_full_card_contract():
    """failure_card reads all four keys; a branch missing one would raise at render time,
    i.e. while already reporting another error."""
    cases = [
        _exc("RepositoryNotFoundError", REPO_NOT_FOUND, 404),
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
