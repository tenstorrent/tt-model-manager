# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Tests for the community-catalog opt-in: `--publish`, `publish`, `unpublish`.

The catalog is a pure index — the only effect of (un)publishing is flipping the
`tt-model-catalog` tag on a public repo. The Hub calls are monkeypatched; we assert the
tag transitions and the public-only guard.
"""

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
