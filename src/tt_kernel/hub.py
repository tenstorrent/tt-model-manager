# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Hugging Face Hub I/O. All storage, auth, visibility, LFS, and search live here.

Each bundle is one HF model repo (``repo_type="model"``) named ``namespace/name`` and
tagged ``tt-model-cache`` so ``search`` can filter for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from . import MANIFEST_NAME, TT_MODEL_CATALOG_TAG, TT_MODEL_TAG
from .manifest import Manifest

_REPO_TYPE = "model"


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def create_repo(repo_id: str, private: bool) -> str:
    """Create (or reuse) the model repo. Returns the repo URL."""
    from huggingface_hub import create_repo as hf_create_repo

    url = hf_create_repo(
        repo_id=repo_id, repo_type=_REPO_TYPE, private=private, exist_ok=True
    )
    return str(url)


def repo_exists(repo_id: str) -> bool:
    """Report whether the repo already exists on the Hub.

    ``push`` needs this to tell "I am creating this repo" from "I am adding a commit to
    someone's existing repo": visibility may be set from the flag in the first case, but
    must never be touched implicitly in the second (see ``cli._ensure_repo``).
    """
    from huggingface_hub import repo_exists as hf_repo_exists

    return bool(hf_repo_exists(repo_id=repo_id, repo_type=_REPO_TYPE))


def set_visibility(repo_id: str, private: bool) -> None:
    """Update repo visibility (newer hub uses update_repo_settings).

    Callers must only reach this when the user *asked* for a visibility change — flipping a
    repo public is not undoable in the sense that matters (whatever was in it was public for
    as long as it took to notice).
    """
    api = _api()
    if hasattr(api, "update_repo_settings"):
        api.update_repo_settings(repo_id=repo_id, repo_type=_REPO_TYPE, private=private)
    else:  # older huggingface_hub
        api.update_repo_visibility(repo_id=repo_id, repo_type=_REPO_TYPE, private=private)


def push_folder(repo_id: str, folder: Path, commit_message: str) -> None:
    """Upload an entire staged bundle folder. Large binaries go to LFS automatically."""
    _api().upload_folder(
        repo_id=repo_id,
        repo_type=_REPO_TYPE,
        folder_path=str(folder),
        commit_message=commit_message,
    )


def tag_repo(repo_id: str, tags: List[str]) -> None:
    """Best-effort: write a model card with metadata tags so search can filter."""
    from huggingface_hub import ModelCard, ModelCardData

    try:
        card = ModelCard.load(repo_id)
    except Exception:
        card = ModelCard("")
    existing = list(getattr(card.data, "tags", None) or [])
    card.data = ModelCardData(tags=sorted(set(existing) | set(tags)))
    card.push_to_hub(repo_id, repo_type=_REPO_TYPE)


def set_catalog_listing(repo_id: str, listed: bool) -> None:
    """Add or remove the community-catalog tag on a repo's model card.

    Listing is a deliberate opt-in, separate from ``push``: the web catalog shows only
    repos carrying ``TT_MODEL_CATALOG_TAG``. Removing it delists the repo from the
    catalog on the next crawl (the repo and its content are untouched — tt-model only
    ever flips a pointer tag; the repo stays under its owner's governance).
    """
    from huggingface_hub import ModelCard, ModelCardData

    try:
        card = ModelCard.load(repo_id)
    except Exception:
        card = ModelCard("")
    tags = set(getattr(card.data, "tags", None) or [])
    if listed:
        tags.add(TT_MODEL_CATALOG_TAG)
    else:
        tags.discard(TT_MODEL_CATALOG_TAG)
    card.data = ModelCardData(tags=sorted(tags))
    card.push_to_hub(repo_id, repo_type=_REPO_TYPE)


def is_private(repo_id: str) -> bool:
    """Report whether a repo is private (a listed catalog entry must be public)."""
    info = _api().model_info(repo_id)
    return bool(getattr(info, "private", False))


def is_private_safe(repo_id: str) -> Optional[bool]:
    """``is_private`` that answers ``None`` instead of raising when the Hub won't say.

    Visibility reporting must never be the thing that fails a push, so callers that only
    want to *describe* the current state (``push``) use this; callers that need a real
    answer to make a decision (``publish``) use :func:`is_private` and let it raise.
    """
    try:
        return is_private(repo_id)
    except Exception:  # noqa: BLE001 — network/permission/404: "unknown", not fatal
        return None


def latest_revision(repo_id: str, revision: Optional[str] = None,
                    timeout: Optional[float] = 3.0) -> Optional[str]:
    """Best-effort commit sha for the repo at ``revision`` (default: the default-branch tip).

    ``serve`` uses this to tell an installed bundle apart from a newer published one, and
    ``pull``/install resolves it BEFORE the download so the recorded sha is exactly what was
    fetched. Returns None when the Hub can't be reached or the repo/revision isn't found — an
    update check (or recording one) must never be the thing that fails a serve or an install.

    ``timeout`` bounds the request (default 3s): a half-open network must not hang the serve
    on a check that is only advisory. ``None`` waits indefinitely (the resolve-before-download
    caller passes a longer bound, since there it is load-bearing, not advisory).
    """
    try:
        return getattr(_api().model_info(repo_id, revision=revision, timeout=timeout), "sha", None)
    except Exception:  # noqa: BLE001 — offline / 404 / timeout / permission: "unknown", never fatal
        return None


def download_bundle(repo_id: str, revision: Optional[str], dest: Optional[str] = None) -> Path:
    """Snapshot-download a bundle and return the local snapshot path."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id=repo_id, repo_type=_REPO_TYPE, revision=revision, local_dir=dest
    )
    return Path(path)


def fetch_manifest(repo_id: str, revision: Optional[str]) -> Manifest:
    """Download just the manifest file and parse it (used by ``info``)."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id, repo_type=_REPO_TYPE, filename=MANIFEST_NAME, revision=revision
    )
    return Manifest.from_json(Path(path).read_text())


def search(
    query: str,
    limit: int = 50,
    catalog_only: bool = False,
    tags: Optional[List[str]] = None,
) -> List[dict]:
    """List tt-model cache repos matching a query, newest first.

    ``catalog_only`` restricts to repos opted into the community catalog (the same set the
    web frontend shows) rather than every pushed cache bundle. ``tags`` are additional repo
    tags ANDed with the base tag — e.g. an ``arch`` (``blackhole``) or a v4 ``target``
    (``p150x4``) written by ``push`` — so a consumer can ask "what runs on my box".
    """
    api = _api()
    base = TT_MODEL_CATALOG_TAG if catalog_only else TT_MODEL_TAG
    # list_models ANDs a list of filter tags; a lone string works too.
    filter_tags = [base] + [t for t in (tags or []) if t]
    results = api.list_models(
        filter=filter_tags if len(filter_tags) > 1 else base,
        search=query or None,
        limit=limit,
    )
    out: List[dict] = []
    for m in results:
        out.append(
            {
                "id": getattr(m, "id", None) or getattr(m, "modelId", None),
                "private": getattr(m, "private", None),
                "downloads": getattr(m, "downloads", None),
                "last_modified": str(getattr(m, "last_modified", "") or ""),
            }
        )
    return out
