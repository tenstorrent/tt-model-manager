# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""`tag_repo` must not clobber a repo's existing card metadata.

`tag_repo` runs after every `package`/`package-thin` push to add the discovery tags
(arch/board/kind/`tt-model-cache`). `set_catalog_listing` (tests in test_catalog_publish.py)
already proves the correct pattern — mutate `card.data.tags` in place — specifically to avoid
rebuilding `card.data` from scratch, which would drop every other frontmatter field. `tag_repo`
did not follow that pattern: it replaced `card.data` wholesale with a fresh `ModelCardData(tags=
...)`, silently dropping `license`, `pipeline_tag`, `library_name`, and `datasets` on every
push. A repo whose card sets those fields (as `episod/tt-tnt` and `episod/tt-tnt-1024` do) lost
them on the next `tt-model package`/`package-thin`.
"""

from tt_kernel import hub


def test_tag_repo_preserves_other_frontmatter_fields(monkeypatch):
    """Adding a discovery tag must not disturb license/pipeline_tag/datasets already on the card."""
    pushed = {}

    class FakeCardData:
        def __init__(self, tags=None, **extra):
            self.tags = list(tags or [])
            for k, v in extra.items():
                setattr(self, k, v)

    class FakeCard:
        def __init__(self, data):
            self.data = data

        def push_to_hub(self, repo_id, repo_type=None):
            pushed["data"] = self.data

    existing = FakeCardData(
        tags=["blackhole"],
        license="apache-2.0",
        pipeline_tag="text-generation",
        library_name="transformers",
        datasets=["episod/tt-tnt-corpus"],
    )

    def fake_load(repo_id):
        return FakeCard(existing)

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "ModelCard",
                        type("MC", (), {"load": staticmethod(fake_load)}))
    monkeypatch.setattr(huggingface_hub, "ModelCardData", FakeCardData)

    hub.tag_repo("episod/tt-tnt", ["tt-model-cache", "llama"])

    result = pushed["data"]
    assert set(result.tags) == {"blackhole", "tt-model-cache", "llama"}
    # The fields tag_repo has no business touching must survive untouched.
    assert result.license == "apache-2.0"
    assert result.pipeline_tag == "text-generation"
    assert result.library_name == "transformers"
    assert result.datasets == ["episod/tt-tnt-corpus"]


def test_tag_repo_still_works_on_a_card_with_no_frontmatter(monkeypatch):
    """A brand-new repo's card has no `data.tags` at all yet; tag_repo must still tag it."""
    pushed = {}

    class FakeCardData:
        def __init__(self, tags=None):
            self.tags = list(tags or [])

    class FakeCard:
        def __init__(self):
            self.data = object()  # no `tags` attribute at all, like ModelCard("")

        def push_to_hub(self, repo_id, repo_type=None):
            pushed["data"] = self.data

    def fake_load(repo_id):
        return FakeCard()

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "ModelCard",
                        type("MC", (), {"load": staticmethod(fake_load)}))
    monkeypatch.setattr(huggingface_hub, "ModelCardData", FakeCardData)

    hub.tag_repo("me/new-bundle", ["tt-model-cache", "blackhole"])

    assert set(pushed["data"].tags) == {"tt-model-cache", "blackhole"}
