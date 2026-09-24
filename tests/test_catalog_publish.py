# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the community-catalog opt-in: `--publish`, `publish`, `unpublish`.

The catalog is a pure index — the only effect of (un)publishing is flipping the
`tt-model-catalog` tag on a public repo. The Hub calls are monkeypatched; we assert the
tag transitions and the public-only guard.
"""

import json

import pytest
from typer.testing import CliRunner

from tt_kernel import TT_MODEL_CATALOG_TAG, cli, hub

runner = CliRunner()


def _entry_not_found(message="no manifest"):
    """The error huggingface_hub raises for a file missing from a repo that exists.

    Prefers the 1.x subclass, because that is what the library actually raises and the
    code under test catches the base class on purpose. `huggingface_hub.utils` is the
    import path production uses: the package floor is >=0.23, and `.errors` only exists
    from 0.25, so a test importing from `.errors` would quietly contradict the fix it
    guards the moment anyone resolved the floor.
    """
    from huggingface_hub.utils import EntryNotFoundError

    try:
        from huggingface_hub.errors import RemoteEntryNotFoundError as cls
    except ImportError:
        cls = EntryNotFoundError
    exc = cls.__new__(cls)          # 1.x demands a live response object
    Exception.__init__(exc, message)
    return exc


def _no_manifest(repo_id, rev):
    """A repo with no tt_kernel_manifest.json: not a container bundle, so `publish`'s card
    gate has nothing to judge and steps aside. The stub the pre-gate tests need, since
    `publish` now reads the published manifest before touching visibility."""
    raise _entry_not_found()


def test_publish_makes_a_private_repo_public_then_lists(monkeypatch):
    """`publish` implies public: a private repo is made public (announced) and then listed —
    publishing is the public+register step in one, not a refusal."""
    listings = []
    flipped = []
    monkeypatch.setattr(hub, "fetch_manifest", _no_manifest)
    monkeypatch.setattr(hub, "is_private", lambda repo_id: True)
    monkeypatch.setattr(hub, "set_visibility",
                        lambda repo_id, private: flipped.append((repo_id, private)))
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: listings.append((repo_id, listed)))

    res = runner.invoke(cli.app, ["publish", "me/private-bundle"])
    assert res.exit_code == 0, res.output
    assert flipped == [("me/private-bundle", False)]  # made public first
    assert listings == [("me/private-bundle", True)]  # then listed
    assert "public" in res.output.lower()             # and the flip was announced


def test_publish_reports_through_the_phase_body_not_raw_echo(monkeypatch):
    """Both lines go through console.py, and this test can tell the difference.

    `typer.secho` writes flush-left, unmarked, and wraps wherever the terminal happens
    to break — so a reader gets no gutter to scan and a wrapped line lands under the
    phase label instead of under its own marker. The assertions below are on exactly
    what console.py adds and secho cannot: a `!`/`✓` marker in a two-space gutter, and
    padding that every wrapped continuation line keeps.
    """
    # `publish` reads the published manifest before touching visibility (the card gate);
    # this test predates that, so give it a repo with no manifest so the gate steps aside.
    monkeypatch.setattr(hub, "fetch_manifest", _no_manifest)
    monkeypatch.setattr(hub, "is_private", lambda repo_id: True)
    monkeypatch.setattr(hub, "set_visibility", lambda repo_id, private: None)
    monkeypatch.setattr(hub, "set_catalog_listing", lambda repo_id, listed: None)

    res = runner.invoke(cli.app, ["publish", "me/private-bundle"])
    assert res.exit_code == 0, res.output
    lines = [ln for ln in res.output.splitlines() if ln.strip()]
    assert lines, res.output
    assert any(ln.startswith("  ! ") for ln in lines), res.output   # the warning note
    assert any(ln.startswith("  ✓ ") for ln in lines), res.output   # the milestone
    # Padding, not a string indent: continuation lines stay in the body column.
    assert all(ln.startswith("  ") for ln in lines), res.output


def test_publish_lists_public_repo(monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "fetch_manifest", _no_manifest)
    monkeypatch.setattr(hub, "is_private", lambda repo_id: False)
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: calls.append((repo_id, listed)))

    res = runner.invoke(cli.app, ["publish", "me/public-bundle"])
    assert res.exit_code == 0
    assert calls == [("me/public-bundle", True)]
    assert "catalog" in res.output.lower()


def test_unpublish_delists(monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: calls.append((repo_id, listed)))

    res = runner.invoke(cli.app, ["unpublish", "me/public-bundle"])
    assert res.exit_code == 0
    assert calls == [("me/public-bundle", False)]


def test_set_catalog_listing_adds_and_removes_tag(monkeypatch):
    """`set_catalog_listing` unions/removes exactly the catalog tag, preserving others."""
    pushed = {}

    class FakeCardData:
        def __init__(self, tags=None):
            self.tags = tags or []

    class FakeCard:
        def __init__(self, tags):
            self.data = FakeCardData(list(tags))

        def push_to_hub(self, repo_id, repo_type=None):
            pushed["tags"] = list(self.data.tags)

    def fake_load(repo_id):
        return FakeCard(["tt-model-cache", "blackhole"])

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "ModelCard",
                        type("MC", (), {"load": staticmethod(fake_load)}))
    monkeypatch.setattr(huggingface_hub, "ModelCardData", FakeCardData)

    hub.set_catalog_listing("me/x", listed=True)
    assert TT_MODEL_CATALOG_TAG in pushed["tags"]
    assert "blackhole" in pushed["tags"]  # existing tags preserved

    # Now removal: card already carries the catalog tag.
    def fake_load_listed(repo_id):
        return FakeCard(["tt-model-cache", "blackhole", TT_MODEL_CATALOG_TAG])

    monkeypatch.setattr(huggingface_hub, "ModelCard",
                        type("MC", (), {"load": staticmethod(fake_load_listed)}))
    hub.set_catalog_listing("me/x", listed=False)
    assert TT_MODEL_CATALOG_TAG not in pushed["tags"]
    assert "tt-model-cache" in pushed["tags"]


# -- the card gate -------------------------------------------------------------------
# A listing is what a stranger picks a model from, so it must say how the model performs
# and where it falls short. `publish` checks the PUBLISHED manifest: it runs against a
# repo, and the author's tt-model.yaml never left their machine. The check runs before
# the visibility flip, and fails CLOSED on anything but "there is no manifest to read".


def _wire_with_card(**card):
    """A published container manifest carrying an author's card block."""
    import json

    from tt_kernel.container_manifest import ContainerManifest

    BASE = {
        "schema": "5.1", "repo": "me/x", "name": "x", "weights": "org/W",
        "kind": "vllm-plugin", "arch": "blackhole",
        "source": {"tt_metal": "/tmp/tt-metal", "code": ["models/common"],
                   "ubuntu": "22.04", "python": "3.12"},
        "runtime": {"vllm": {"version": "0.24.0"},
                    "plugin": {"repo": "https://x/y", "ref": "abc"}},
        "serve": {"port": 8000, "block_size": 64},
        "serve_profiles": [{"name": "p150x4", "hardware": "p150x4",
                            "mesh_device": "P150x4", "max_num_seqs": 32,
                            "max_model_len": 131072}],
        "card": card or None,
    }
    m = ContainerManifest.model_validate(json.loads(json.dumps(BASE)))
    return m.to_wire(image_tag="t", tt_metal_version="0.1", tt_kernel_version="0.1")


def _wire_hub(monkeypatch, wire_or_exc, *, private=False):
    """Stub every Hub call `publish` makes; returns the list of side effects it caused."""
    effects = []

    def fetch(repo_id, rev):
        if isinstance(wire_or_exc, BaseException):
            raise wire_or_exc
        return wire_or_exc

    monkeypatch.setattr(hub, "fetch_manifest", fetch)
    monkeypatch.setattr(hub, "is_private", lambda repo_id: private)
    monkeypatch.setattr(hub, "set_visibility",
                        lambda repo_id, private: effects.append(("public", repo_id)))
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: effects.append(("list", repo_id, listed)))
    return effects


def test_publish_refuses_a_card_missing_the_required_sections(monkeypatch):
    effects = _wire_hub(monkeypatch, _wire_with_card(description="just a blurb"), private=True)
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code != 0
    assert effects == []  # not listed — and a PRIVATE repo was not flipped public on the way
    assert "performance" in res.output and "limitations" in res.output
    # The remedy, not just the diagnosis: the `push` and `repush` contexts both had
    # their wording asserted, so replacing this one with a stub string kept CI green.
    # It is also the one route where the fix means re-packaging AND re-pushing.
    assert "re-package, push, and run `tt-model publish` again" in res.output


def test_publish_lists_a_bundle_whose_card_is_complete(monkeypatch):
    effects = _wire_hub(
        monkeypatch, _wire_with_card(performance="41 ms/token", limitations="p150x4 only"))
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code == 0, res.output
    assert effects == [("list", "me/x", True)]


def test_publish_still_works_for_a_bundle_published_before_cards_rode_the_wire(monkeypatch):
    """`container.card` is None there. That is "cannot tell", not "sections missing" —
    it must not make an already-published bundle retroactively unlistable."""
    wire = _wire_with_card()          # no card block at all
    assert wire.container.card is None
    effects = _wire_hub(monkeypatch, wire)
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code == 0, res.output
    assert effects == [("list", "me/x", True)]


def test_publish_skips_the_gate_for_a_repo_with_no_manifest_at_all(monkeypatch):
    """No tt_kernel_manifest.json means this is not a container bundle; the gate has
    nothing to say about it and `publish` behaves as it always did.

    Raises the subclass hf_hub actually raises for a missing file, not the base class:
    the code catches the base on purpose, and a test that only ever sees the base would
    not notice if that stopped covering the real thing."""
    effects = _wire_hub(monkeypatch, _entry_not_found())
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code == 0, res.output
    assert effects == [("list", "me/x", True)]


def test_publish_skips_the_gate_for_a_v5_bundle_that_has_no_container_block(monkeypatch):
    """A v5/v6 bundle has a manifest but no `container:` at all, so there is no card to
    judge — a different exemption from "no manifest file" and from "container but no
    card", and the only one with no coverage before."""
    wire = _wire_with_card(performance="fast", limitations="none")
    wire.container = None          # what a v5/v6 manifest looks like on the wire
    assert not wire.is_container
    effects = _wire_hub(monkeypatch, wire)
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code == 0, res.output
    assert effects == [("list", "me/x", True)]


def test_publish_refuses_a_manifest_it_cannot_parse_without_blaming_the_hub(monkeypatch):
    """A fetched-but-unparseable manifest is a LOCAL problem — most likely a bundle
    pushed by a newer tt-model. Routed through classify_hub_error it read as "the Hub
    request failed", blaming the network for something a retry can never fix. Still a
    refusal (a card we cannot read is a promise we cannot check), but the remedy is an
    upgrade, not a rebuild."""
    effects = _wire_hub(monkeypatch, ValueError("Unsupported bundle schema_version '7'"),
                        private=True)
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code != 0
    assert effects == []
    assert "could not read the published manifest" in res.output
    assert "newer version" in res.output
    assert "the Hub request failed" not in res.output


def test_publish_fails_closed_when_the_manifest_cannot_be_read(monkeypatch):
    """A flaky Hub used to skip the gate AND still flip the repo public — the same command
    listing or refusing depending on the network. Listing is the hard-to-undo step, so an
    unreadable manifest is a refusal, and nothing else happens."""
    effects = _wire_hub(monkeypatch, ConnectionError("hub 503"), private=True)
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code != 0
    assert effects == []
    assert "was not made public" in res.output


def test_publish_fails_closed_when_offline(monkeypatch):
    """Offline raises LocalEntryNotFoundError, which INHERITS the not-found class the gate
    exempts. It must read as "cannot reach the Hub", never as "no manifest"."""
    from huggingface_hub.errors import LocalEntryNotFoundError

    effects = _wire_hub(monkeypatch, LocalEntryNotFoundError("offline"), private=True)
    res = runner.invoke(cli.app, ["publish", "me/x"])
    assert res.exit_code != 0
    assert effects == []
# -- the whitelist ---------------------------------------------------------------------
# Whitelisting copies a reviewed bundle into the Tenstorrent org. The copy IS the signal:
# only the DX team can write that namespace, so an author cannot grant themselves a
# review, which is what a tag on their own repo could never prevent. The review record
# lives on the copy's card, so it cannot drift from the artifact it describes.

from tt_kernel import TT_ORG, auth  # noqa: E402


def _manifest(weights="Qwen/Qwen3-32B"):
    """Just enough of a wire manifest for the target name to be derivable."""
    class _W:
        repo_id = weights

    return type("M", (), {"weights": _W() if weights else None})()


def _stub_whitelist(monkeypatch, *, tags=(TT_MODEL_CATALOG_TAG,), private=False,
                    sha="c" * 40, who={"name": "reviewer"}, repo_id=None,
                    target_exists=False, target_review=None, weights="Qwen/Qwen3-32B"):
    """Stub every Hub call `whitelist` makes; returns the ordered list of effects.

    Order matters and is asserted: a half-finished run (copied but not annotated) is a
    real state the command has to handle, so the tests need to see the sequence.
    """
    effects = []
    monkeypatch.setattr(hub, "repo_state",
                        lambda rid: hub.RepoState(list(tags), private, sha, repo_id or rid))
    monkeypatch.setattr(hub, "fetch_manifest", lambda rid, rev: _manifest(weights))
    monkeypatch.setattr(hub, "repo_exists", lambda rid: target_exists)
    monkeypatch.setattr(hub, "read_review", lambda rid: target_review)
    monkeypatch.setattr(auth, "whoami", lambda: who)
    monkeypatch.setattr(hub, "duplicate_into_org",
                        lambda src, dst: effects.append(("copy", src, dst)))
    monkeypatch.setattr(
        hub, "annotate_review",
        lambda rid, **kw: effects.append(("annotate", rid, kw["source"], kw["revision"],
                                          kw["reviewer"])))
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda rid, listed: effects.append(("list", rid, listed)))
    return effects


def test_whitelist_copies_the_bundle_into_the_org_and_records_the_review(monkeypatch):
    effects = _stub_whitelist(monkeypatch)
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code == 0, res.output
    assert effects == [
        ("copy", "me/listed", f"{TT_ORG}/Qwen3-32B"),
        ("annotate", f"{TT_ORG}/Qwen3-32B", "me/listed", "c" * 40, "reviewer"),
        ("list", f"{TT_ORG}/Qwen3-32B", True),
    ]
    assert f"{TT_ORG}/Qwen3-32B" in res.output


def test_whitelist_names_the_copy_after_the_weights_repo_not_the_bundle(monkeypatch):
    """The team's convention, and it is the better name: the canonical model id rather
    than an author's packaging slug (`someone/qwen3-32b-blackhole-v51`)."""
    effects = _stub_whitelist(monkeypatch, weights="openai/gpt-oss-120b")
    res = runner.invoke(cli.app, ["whitelist", "tt-hous/gpt-oss-120b-p150x4"])
    assert res.exit_code == 0, res.output
    assert effects[0] == ("copy", "tt-hous/gpt-oss-120b-p150x4", f"{TT_ORG}/gpt-oss-120b")


def test_whitelist_records_the_hubs_own_spelling_of_the_source(monkeypatch):
    effects = _stub_whitelist(monkeypatch, repo_id="Me/Listed")
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code == 0, res.output
    assert effects[1][2] == "Me/Listed"


def test_whitelist_lists_the_copy_explicitly(monkeypatch):
    """Not left to tag inheritance: a public-but-unlisted source would otherwise produce
    a copy nobody can find."""
    effects = _stub_whitelist(monkeypatch)
    runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert ("list", f"{TT_ORG}/Qwen3-32B", True) in effects


def test_whitelist_refuses_a_bundle_that_is_not_listed(monkeypatch):
    effects = _stub_whitelist(monkeypatch, tags=[])
    res = runner.invoke(cli.app, ["whitelist", "me/unlisted"])
    assert res.exit_code != 0
    assert effects == []
    assert "not in the community catalog" in res.output


def test_whitelist_refuses_a_private_bundle(monkeypatch):
    effects = _stub_whitelist(monkeypatch, private=True)
    res = runner.invoke(cli.app, ["whitelist", "me/private"])
    assert res.exit_code != 0
    assert effects == []
    assert "private" in res.output


def test_whitelist_needs_a_logged_in_reviewer(monkeypatch):
    effects = _stub_whitelist(monkeypatch, who=None)
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code != 0
    assert effects == []
    assert "login" in res.output


def test_whitelist_refuses_when_the_manifest_names_no_weights_repo(monkeypatch):
    """There is no name to copy to — the convention derives it from the weights repo."""
    effects = _stub_whitelist(monkeypatch, weights=None)
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code != 0
    assert effects == []
    assert "no weights repo" in res.output


def test_whitelist_refuses_a_name_already_taken_by_another_bundle(monkeypatch):
    """Two bundles of one model collide on the derived name. Never overwrite: the other
    copy is a reviewed artifact someone may be relying on."""
    effects = _stub_whitelist(
        monkeypatch, target_exists=True,
        target_review={hub.REVIEW_SOURCE_KEY: "someone-else/qwen3-32b-p300x2"})
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code != 0
    assert effects == []                     # nothing copied, nothing annotated
    assert "already exists" in res.output
    assert "someone-else/qwen3-32b-p300x2" in res.output
    assert "unwhitelist" in res.output       # and it names the way out


def test_whitelist_resumes_a_half_finished_run_without_copying_again(monkeypatch):
    """If annotate failed after the copy landed, re-running must finish the job rather
    than be permanently blocked by its own half-written state."""
    effects = _stub_whitelist(
        monkeypatch, target_exists=True,
        target_review={hub.REVIEW_SOURCE_KEY: "me/listed"})
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code == 0, res.output
    assert [e[0] for e in effects] == ["annotate", "list"]   # no second copy
    assert "re-recording" in res.output


def test_whitelist_matches_an_existing_copys_source_case_insensitively(monkeypatch):
    effects = _stub_whitelist(
        monkeypatch, target_exists=True,
        target_review={hub.REVIEW_SOURCE_KEY: "Me/Listed"})
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code == 0, res.output
    assert [e[0] for e in effects] == ["annotate", "list"]


def test_whitelist_does_not_confuse_an_unreadable_repo_with_an_unlisted_one(monkeypatch):
    """`repo_state` raises rather than failing to False, so a network blip cannot produce
    a confident, false "not in the catalog" refusal."""
    _stub_whitelist(monkeypatch)
    monkeypatch.setattr(hub, "repo_state",
                        lambda rid: (_ for _ in ()).throw(ConnectionError("hub 503")))
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code != 0
    assert "not in the community catalog" not in res.output


def test_whitelist_says_so_when_it_cannot_record_a_source_revision(monkeypatch):
    effects = _stub_whitelist(monkeypatch, sha=None)
    res = runner.invoke(cli.app, ["whitelist", "me/listed"])
    assert res.exit_code == 0, res.output
    assert effects[1][3] is None            # revision recorded as null
    assert "no source revision" in res.output


def test_unwhitelist_delists_the_copy_and_keeps_it(monkeypatch):
    calls = []
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda rid, listed: calls.append((rid, listed)))
    res = runner.invoke(cli.app, ["unwhitelist", f"{TT_ORG}/Qwen3-32B"])
    assert res.exit_code == 0, res.output
    assert calls == [(f"{TT_ORG}/Qwen3-32B", False)]
    assert "unchanged" in res.output        # the reviewed snapshot is kept


def test_unwhitelist_refuses_a_community_repo(monkeypatch):
    """Passing the original rather than the copy is the easy slip, and delisting someone
    else's repo is not ours to do."""
    calls = []
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda rid, listed: calls.append((rid, listed)))
    res = runner.invoke(cli.app, ["unwhitelist", "me/listed"])
    assert res.exit_code != 0
    assert calls == []
    assert f"not a {TT_ORG} copy" in res.output


# -- hub.duplicate_into_org / read_review / annotate_review ---------------------------


def test_duplicate_into_org_asks_for_a_server_side_model_copy(monkeypatch):
    """`duplicate_repo` copies git history and LFS on the Hub itself — a multi-GB bundle
    is one request that moves no data. `exist_ok=False` so an occupied name is an error
    the CLI diagnoses, never an overwrite of another bundle's reviewed copy."""
    seen = {}

    class _Api:
        def duplicate_repo(self, **kw):
            seen.update(kw)
            return "https://huggingface.co/Tenstorrent/Qwen3-32B"

    monkeypatch.setattr(hub, "_api", lambda: _Api())
    url = hub.duplicate_into_org("me/listed", f"{TT_ORG}/Qwen3-32B")
    assert url.endswith(f"{TT_ORG}/Qwen3-32B")
    assert seen["from_id"] == "me/listed"
    assert seen["to_id"] == f"{TT_ORG}/Qwen3-32B"
    assert seen["repo_type"] == "model"
    assert seen["exist_ok"] is False


def _fake_card(monkeypatch, *, data=None, text="# t\n\nbody", load_error=None):
    """Stand in for ModelCard.load, capturing what would be pushed."""
    pushed = {}

    class _Data:
        def __init__(self):
            for k, v in (data or {}).items():
                setattr(self, k, v)

    class _Card:
        def __init__(self):
            self.data = _Data()
            self.text = text

        def push_to_hub(self, repo_id, repo_type=None):
            pushed["repo_id"] = repo_id
            pushed["text"] = self.text
            pushed["data"] = {k: v for k, v in vars(self.data).items()}

    def load(repo_id):
        if load_error:
            raise load_error
        return _Card()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "ModelCard",
                        type("MC", (), {"load": staticmethod(load)}))
    return pushed


def test_annotate_review_writes_the_keys_tt_cli_reads(monkeypatch):
    pushed = _fake_card(monkeypatch, data={"license": "apache-2.0"})
    hub.annotate_review(f"{TT_ORG}/Qwen3-32B", source="me/listed", revision="d" * 40,
                        reviewer="sam", reviewed_at="2026-09-22T12:00:00Z")
    assert pushed["data"][hub.REVIEW_SOURCE_KEY] == "me/listed"
    assert pushed["data"][hub.REVIEW_REVISION_KEY] == "d" * 40
    assert pushed["data"][hub.REVIEW_REVIEWER_KEY] == "sam"
    assert pushed["data"][hub.REVIEW_DATE_KEY] == "2026-09-22T12:00:00Z"
    # the author's own frontmatter survives — card.data is mutated, not rebuilt
    assert pushed["data"]["license"] == "apache-2.0"
    # and the body gains the attribution, keeping what was there
    assert "body" in pushed["text"]
    assert "me/listed" in pushed["text"]
    assert "snapshot" in pushed["text"]          # later commits are not covered


def test_annotate_review_credits_the_author_in_the_cards_top_line(monkeypatch):
    """The team's bargain for whitelisting: the copy earns the Tenstorrent org's traffic,
    so the community author is credited where a reader lands, not under the fold."""
    pushed = _fake_card(monkeypatch, text="# Qwen3-32B\n\nthe author's own words")
    hub.annotate_review(f"{TT_ORG}/Qwen3-32B", source="someauthor/foo", revision="d" * 40,
                        reviewer="sam", reviewed_at="2026-09-22T12:00:00Z")
    body = pushed["text"]
    first = body.split("\n", 1)[0]
    assert first == f"<!-- {hub.ATTRIBUTION_MARKER} -->"
    # the author, and their profile, come before the model's own heading
    credit, heading = body.index("someauthor"), body.index("# Qwen3-32B")
    assert credit < heading
    assert "https://huggingface.co/someauthor" in body
    assert "the author's own words" in body      # nothing of theirs is displaced


def test_annotate_review_replaces_its_own_block_rather_than_stacking(monkeypatch):
    """`whitelist` resumes by re-annotating an already-copied source, so this runs again
    on a card it wrote. A second credit under the first would be the visible bug."""
    pushed = _fake_card(monkeypatch, text="# Qwen3-32B\n\nbody")
    hub.annotate_review(f"{TT_ORG}/Qwen3-32B", source="someauthor/foo", revision="a" * 40,
                        reviewer="sam", reviewed_at="2026-09-22T12:00:00Z")

    # feed the already-annotated card back in, as the Hub would on the resume run
    second = _fake_card(monkeypatch, text=pushed["text"])
    hub.annotate_review(f"{TT_ORG}/Qwen3-32B", source="someauthor/foo", revision="b" * 40,
                        reviewer="nate", reviewed_at="2026-09-23T12:00:00Z")
    body = second["text"]
    assert body.count(hub.ATTRIBUTION_MARKER) == 2       # one open, one close. Not four.
    assert body.count("# Qwen3-32B") == 1
    assert "bbbbbbbbb" in body and "aaaaaaaaa" not in body   # the newer review, only
    assert "nate" in body and "sam" not in body


def test_annotate_review_omits_the_revision_when_the_hub_gave_none(monkeypatch):
    """A review with no recorded sha is still a valid statement about the bundle; it just
    must not print an empty backtick pair where the commit should be."""
    pushed = _fake_card(monkeypatch)
    hub.annotate_review(f"{TT_ORG}/Qwen3-32B", source="someauthor/foo", revision=None,
                        reviewer="sam", reviewed_at="2026-09-22T12:00:00Z")
    block = hub._ATTRIBUTION_RE.search(pushed["text"]).group(0)
    assert "at `" not in block                   # no empty backticks where a sha would be
    assert "someauthor/foo" in block             # the credit still stands without one

    # and the sha IS shown, shortened, when the Hub reported one
    with_sha = _fake_card(monkeypatch)
    hub.annotate_review(f"{TT_ORG}/Qwen3-32B", source="someauthor/foo", revision="d" * 40,
                        reviewer="sam", reviewed_at="2026-09-22T12:00:00Z")
    assert "at `" + "d" * 9 + "`" in hub._ATTRIBUTION_RE.search(with_sha["text"]).group(0)


def test_annotate_review_refuses_rather_than_publishing_an_empty_readme(monkeypatch):
    """`tag_repo` may fall back to an empty card because it only writes tags. This writes
    the BODY, so a transient read failure must not replace a real README with nothing."""
    pushed = _fake_card(monkeypatch, load_error=ConnectionError("hub 503"))
    with pytest.raises(ConnectionError):
        hub.annotate_review(f"{TT_ORG}/x", source="me/listed", revision=None,
                            reviewer="sam", reviewed_at="2026-09-22T12:00:00Z")
    assert pushed == {}                          # nothing was pushed


def test_read_review_returns_none_for_a_repo_with_no_card(monkeypatch):
    from huggingface_hub.utils import EntryNotFoundError

    _fake_card(monkeypatch, load_error=EntryNotFoundError("no card"))
    assert hub.read_review(f"{TT_ORG}/x") is None


def test_read_review_propagates_a_real_failure(monkeypatch):
    """"Could not read" must never be mistaken for "no review recorded" — that is the
    difference between resuming and silently overwriting another bundle's copy."""
    _fake_card(monkeypatch, load_error=ConnectionError("hub 503"))
    with pytest.raises(ConnectionError):
        hub.read_review(f"{TT_ORG}/x")


def test_read_review_reports_the_recorded_source(monkeypatch):
    _fake_card(monkeypatch, data={hub.REVIEW_SOURCE_KEY: "me/listed"})
    assert hub.read_review(f"{TT_ORG}/x")[hub.REVIEW_SOURCE_KEY] == "me/listed"
