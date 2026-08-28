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


def test_publish_requires_public(monkeypatch):
    """`publish` refuses a private repo (the catalog is public by definition)."""
    calls = []
    monkeypatch.setattr(hub, "is_private", lambda repo_id: True)
    monkeypatch.setattr(hub, "set_catalog_listing",
                        lambda repo_id, listed: calls.append((repo_id, listed)))

    res = runner.invoke(cli.app, ["publish", "me/private-bundle"])
    assert res.exit_code == 1
    assert "private" in res.output.lower()
    assert calls == []  # nothing was listed


def test_publish_lists_public_repo(monkeypatch):
    calls = []
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
