# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Hugging Face Hub I/O. All storage, auth, visibility, LFS, and search live here.

Each bundle is one HF model repo (``repo_type="model"``) named ``namespace/name`` and
tagged ``tt-model-cache`` so ``search`` can filter for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from tqdm import tqdm as _tqdm

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
    """Upload an entire staged bundle folder. Large binaries go to LFS automatically.

    ``upload_folder`` takes no ``tqdm_class``, so the bridge here silences HF's writers and
    shows an indeterminate activity line rather than a byte count. An honest spinner beats
    a bar fighting our own output for the row.
    """
    with progress_bridge(f"Uploading to {repo_id}"):
        _api().upload_folder(
            repo_id=repo_id,
            repo_type=_REPO_TYPE,
            folder_path=str(folder),
            commit_message=commit_message,
        )


def push_large_folder(repo_id: str, folder: Path, *, num_workers: int = 4) -> None:
    """Upload a container package directory, whose ``image/`` is multi-GB.

    ``upload_large_folder`` rather than ``upload_folder``: it commits in batches, resumes
    per file after an interruption, and skips blobs the Hub already has. Since the OCI
    layout is content-addressed, two models built on the same tt-metal commit share most
    of their layers and the second push uploads only what actually differs.

    It manages its own progress reporting and takes no ``tqdm_class``, so the bridge here
    just silences HF's writers and shows one activity line.
    """
    with progress_bridge(f"Uploading to {repo_id}"):
        _api().upload_large_folder(
            repo_id=repo_id,
            repo_type=_REPO_TYPE,
            folder_path=str(folder),
            num_workers=num_workers,
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
    """Snapshot-download a bundle and return the local snapshot path.

    HF's own tqdm/xet bars are diverted into our activity row (see ``progress_bridge``) so
    only one writer ever owns the terminal.
    """
    from huggingface_hub import snapshot_download

    with progress_bridge(f"Downloading {repo_id}") as tqdm_class:
        path = snapshot_download(
            repo_id=repo_id, repo_type=_REPO_TYPE, revision=revision, local_dir=dest,
            tqdm_class=tqdm_class,
        )
    return Path(path)


def fetch_manifest(repo_id: str, revision: Optional[str]) -> Manifest:
    """Download just the manifest file and parse it (used by ``info``)."""
    from huggingface_hub import hf_hub_download

    with progress_bridge(f"Fetching manifest for {repo_id}"):
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


# --------------------------------------------------------------------- diagnosis
# Hub failures used to escape as a ~60-line Rich traceback (`tt-model pull <404>` and
# `tt-model info <404>` both did). Classification lives here, as a PURE function, so the
# 404/401/403/gated/offline matrix is unit-testable with no network and the CLI can render
# one card instead of a stack.
def _evidence(text: str) -> str:
    """The first line of an exception, minus tracking noise.

    huggingface_hub appends "(Request ID: Root=1-...;...)" to its HTTP errors. That is for
    a support ticket, not for the person reading their terminal, and it crowds out the
    part that matters.
    """
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line.split(" (Request ID:")[0].strip()


def classify_hub_error(exc: BaseException, repo_id: str) -> dict:
    """Map a huggingface_hub exception to ``{cause, detail, evidence, actions}``.

    Pure: exception in, dict out. Keys match what ``console.failure_card`` renders.

    The 404 wording is deliberately two-sided. The Hub answers 404 for *both* "no such
    repo" and "private repo you cannot see" — it will not distinguish them for an
    unauthorised caller — so asserting "it does not exist" would be a guess that sends
    the user down the wrong path half the time.
    """
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)

    # Offline / DNS / TLS beats everything: an unreachable Hub also can't confirm a repo,
    # and "you are offline" is the more useful half of that pair.
    if (name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled")
            or any(k in low for k in ("no such host", "dial tcp", "i/o timeout",
                                      "tls handshake", "connection refused",
                                      "connection error", "max retries exceeded",
                                      "temporary failure in name resolution"))):
        return {
            "cause": "cannot reach the Hub",
            "detail": "The request never got a response — this looks like a network or DNS problem, "
                      "not a missing bundle.",
            "evidence": _evidence(text),
            "actions": ["check your connection, then re-run",
                        "HF_HUB_OFFLINE=1 tt-model list   # what is already installed"],
        }

    if status == 401 or name == "GatedRepoError" or "gated" in low:
        return {
            "cause": "you do not have access",
            "detail": f"The Hub recognised {repo_id} but refused it for this token — it is gated, "
                      "or the token is missing, expired, or lacks read scope.",
            "evidence": _evidence(text),
            "actions": ["tt-model login", f"accept the terms at https://huggingface.co/{repo_id}"],
        }

    if status == 403:
        return {
            "cause": "not authorised",
            "detail": f"The Hub rejected this token for {repo_id}. It may lack read scope, or the "
                      "repo's terms may not be accepted for your account.",
            "evidence": _evidence(text),
            "actions": ["tt-model login", f"open https://huggingface.co/{repo_id}"],
        }

    if status == 404 or name == "RepositoryNotFoundError" or "not found" in low:
        return {
            "cause": "no such bundle, or no access",
            "detail": f"The Hub returned 404 for {repo_id}. Either the id is wrong, or the repo is "
                      "private and your token cannot see it — the Hub reports both the same way.",
            "evidence": _evidence(text),
            "actions": [f"tt-model search {repo_id.split('/')[-1]}   # find the right id",
                        "tt-model login                     # if it is private"],
        }

    if name == "EntryNotFoundError" or MANIFEST_NAME in text:
        return {
            "cause": "not a tt-model bundle",
            "detail": f"The repo exists but has no {MANIFEST_NAME}, so there is nothing for tt-model "
                      "to install.",
            "evidence": _evidence(text),
            "actions": ["tt-model search --catalog     # bundles published for tt-model"],
        }

    return {
        "cause": "the Hub request failed",
        "detail": f"{name} while talking to the Hugging Face Hub about {repo_id}.",
        "evidence": _evidence(text),
        "actions": ["re-run with --verbose for the full traceback"],
    }


# ----------------------------------------------------------------- progress bridge
# huggingface_hub writes its own tqdm bars, and hf_xet writes more ("Download complete",
# "Reconstruction complete"). With our spinner or a `secho` also writing, two processes
# fought for one terminal row: bars and status text interleaved mid-line, and unterminated
# bar output survived the process and painted over the next shell prompt — at which point
# leftover bytes were handed to bash as commands. So the CLI takes sole ownership of the
# row: HF's writers are silenced and their byte counts are re-reported through our own
# activity line.
class _Sink:
    """Where the real tqdm writes its bars. Nobody reads it."""

    def write(self, *a, **k):
        return 0

    def flush(self, *a, **k):
        pass

    def isatty(self):
        return False


class _ActivityTqdm(_tqdm):
    """A REAL tqdm whose output goes to ``console.activity`` instead of the terminal.

    This subclasses tqdm rather than imitating it, and that is the whole point. The
    previous stand-in reimplemented the slice of tqdm that huggingface_hub appeared to
    use, and hub kept reaching for more of it — first ``get_lock``/``set_lock`` (via
    ``tqdm.contrib.concurrent.thread_map``), then the wrapped-iterable protocol, then
    ``format_dict`` (via the xet progress reporter). Each was an AttributeError in a
    consumer's download, and the next hub release would have found another: the reporter
    also touches ``format_desc``, ``bytes_completed``, ``total_transfer_bytes`` and more.

    Subclassing makes that class of bug structurally impossible — every attribute exists
    and stays correct, because it IS tqdm. The only thing overridden is where the display
    goes: tqdm renders into a sink, and ``display()`` updates our one activity line.
    """

    label = "Working"
    _live = {}  # id -> (n, total), so concurrent file bars can be summed

    def __init__(self, *args, **kwargs):
        # tqdm still does all its bookkeeping; it just renders somewhere nobody looks.
        kwargs["file"] = _Sink()
        kwargs["disable"] = False
        kwargs.setdefault("leave", False)
        # Set BEFORE super().__init__: tqdm calls display() from its constructor, so an
        # attribute the override reads must already exist.
        #
        # Only byte-denominated bars are worth aggregating; a "Fetching 5 files" bar has a
        # different unit and would corrupt the byte total.
        self._bytes = kwargs.get("unit") in ("B", "iB")
        super().__init__(*args, **kwargs)
        if self._bytes:
            self._publish()

    # -- the only behaviour we change -------------------------------------------------
    def display(self, *args, **kwargs):
        if getattr(self, "_bytes", False):
            self._publish()
        return True

    def update(self, n=1):
        # Publish on every update rather than leaving it to display(), which tqdm
        # throttles by mininterval — the activity line would then sit still through a
        # burst of fast chunks, which is exactly when a download looks hung.
        r = super().update(n)
        if self._bytes:
            self._publish()
        return r

    def close(self):
        # Stop publishing BEFORE closing: tqdm.close() calls display() one last time,
        # which would re-add this bar to the aggregate immediately after it was removed.
        self._bytes = False
        super().close()
        _ActivityTqdm._live.pop(id(self), None)

    # -- our own aggregation ----------------------------------------------------------
    def _publish(self):
        _ActivityTqdm._live[id(self)] = (self.n, self.total or 0)
        self._render()

    @classmethod
    def _render(cls):
        done = sum(n for n, _ in cls._live.values())
        if not done:
            return
        from . import console

        console.activity.set(f"{cls.label}  {console.fmt_bytes(done)}")


def progress_bridge(label: str):
    """Silence HF/xet progress writers and route their byte counts to the activity row.

    Returns a ``tqdm_class`` to hand to ``snapshot_download``. Use as a context manager so
    the bars are restored on every exit path — leaving them disabled would silently strip
    progress from anything else in the process.
    """
    import contextlib

    from huggingface_hub.utils import disable_progress_bars, enable_progress_bars

    from . import console

    @contextlib.contextmanager
    def _ctx():
        _ActivityTqdm.label = label
        _ActivityTqdm._live.clear()
        disable_progress_bars()
        console.activity.start(label)
        try:
            yield _ActivityTqdm
        finally:
            console.activity.stop()
            enable_progress_bars()
            _ActivityTqdm._live.clear()

    return _ctx()
