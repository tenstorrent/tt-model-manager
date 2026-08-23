# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Hugging Face Hub I/O: upload, download, visibility, the model card, and error
diagnosis. One model == one HF model repo, tagged ``tt-model``.

Auth is huggingface_hub's own token store (``hf auth login`` / ``HF_TOKEN``); tt-model
keeps no token of its own.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from . import MANIFEST_NAME, TT_MODEL_TAG
from .manifest import Manifest

_REPO_TYPE = "model"


def _api():
    from huggingface_hub import HfApi

    return HfApi()


def repo_exists(repo_id: str) -> bool:
    from huggingface_hub import repo_exists as hf_repo_exists

    return bool(hf_repo_exists(repo_id=repo_id, repo_type=_REPO_TYPE))


def ensure_repo(repo_id: str, private: Optional[bool]) -> str:
    """Create the repo if needed; honour visibility ONLY when the user stated one.

    ``private`` is tri-state on purpose. ``None`` means the user said nothing — a new
    repo is then created private (the safe default), and an existing repo's visibility
    is never touched: silently flipping someone's public repo private (or the reverse)
    is the class of bug this signature exists to prevent.
    """
    from huggingface_hub import create_repo as hf_create_repo

    existed = repo_exists(repo_id)
    url = str(hf_create_repo(
        repo_id=repo_id, repo_type=_REPO_TYPE,
        private=True if private is None else private, exist_ok=True,
    ))
    if existed and private is not None:
        api = _api()
        if hasattr(api, "update_repo_settings"):
            api.update_repo_settings(repo_id=repo_id, repo_type=_REPO_TYPE, private=private)
        else:  # older huggingface_hub
            api.update_repo_visibility(repo_id=repo_id, repo_type=_REPO_TYPE, private=private)
    return url


def push_package(repo_id: str, staged: Path) -> None:
    """Upload a staged package directory (manifest, code/, README, image/).

    ``upload_large_folder`` is used because ``image/blobs`` is multi-GB: it uploads
    blob-by-blob with resume, so an interrupted push continues instead of restarting.
    It owns its own progress output.
    """
    _api().upload_large_folder(
        repo_id=repo_id,
        repo_type=_REPO_TYPE,
        folder_path=str(staged),
    )


def tag_repo(repo_id: str, tags: List[str]) -> None:
    """Best-effort: metadata tags on the model card so the Hub can filter."""
    try:
        from huggingface_hub import ModelCard, ModelCardData

        try:
            card = ModelCard.load(repo_id)
        except Exception:
            card = ModelCard.from_template(ModelCardData())
        existing = set(card.data.tags or [])
        card.data.tags = sorted(existing | set(tags))
        card.push_to_hub(repo_id, repo_type=_REPO_TYPE)
    except Exception:
        pass  # tags are a convenience; never fail a push over them


def download_package(repo_id: str, revision: Optional[str] = None) -> Path:
    """Snapshot the whole package repo into the HF cache; returns the snapshot path.

    The HF cache (not a local_dir) is deliberate: blobs shared between two models built
    on the same tt-metal commit are stored once, and a re-pull is a no-op.
    """
    from huggingface_hub import snapshot_download

    with progress_bridge(f"Pulling {repo_id}") as tqdm_class:
        path = snapshot_download(
            repo_id=repo_id, repo_type=_REPO_TYPE, revision=revision,
            tqdm_class=tqdm_class,
        )
    return Path(path)


def fetch_manifest(repo_id: str, revision: Optional[str] = None) -> Manifest:
    """Download just the manifest — enough for list/serve decisions before a full pull."""
    from huggingface_hub import hf_hub_download

    from .manifest import load_manifest

    path = hf_hub_download(
        repo_id=repo_id, repo_type=_REPO_TYPE, revision=revision, filename=MANIFEST_NAME
    )
    return load_manifest(path)


def download_weights(weights_repo: str, revision: Optional[str] = None) -> Path:
    """Fetch the model weights into the HOST HF cache — the only thing that ever
    lands on the host. The container bind-mounts this cache and never downloads."""
    from huggingface_hub import snapshot_download

    with progress_bridge(f"Downloading weights {weights_repo}") as tqdm_class:
        path = snapshot_download(repo_id=weights_repo, revision=revision, tqdm_class=tqdm_class)
    return Path(path)


# --------------------------------------------------------------------- model card
def render_model_card(m: Manifest, code_tree: List[str]) -> str:
    """The generated README.md for the HF repo: what this is, what it targets, and how
    to run it. The serve-profile table travels with the model instead of living in a
    quickstart doc someone has to find."""
    built = m.built or {}
    lines = [
        "---",
        "tags:",
        *[f"- {t}" for t in sorted({TT_MODEL_TAG, m.arch, m.type,
                                    *(p.hardware or "" for p in
                                      (m.resolve_profile(n) for n in m.profile_names())
                                      if p.hardware)})],
        "---",
        "",
        f"# {m.name}",
        "",
        f"A [tt-model]({'https://github.com/tenstorrent/tt-model-manager'}) package: a"
        f" self-contained Docker image serving **{m.weights}** on Tenstorrent"
        f" **{m.arch}** hardware.",
        "",
        "```bash",
        f"tt-model pull  {m.repo}     # image -> docker, weights -> host HF cache",
        f"tt-model serve {m.repo}     # docker run with the right devices and mounts",
        "```",
        "",
        "## Serve profiles",
        "",
        "| profile | hardware | mesh | max seqs | context | description |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    default = m.resolved_default()
    for name in m.profile_names():
        p = m.resolve_profile(name)
        star = " (default)" if name == default else ""
        lines.append(
            f"| `{name}`{star} | {p.hardware} | `{p.mesh_device}` | {p.max_num_seqs} "
            f"| {p.max_model_len or '—'} | {p.description or ''} |"
        )
    lines += [
        "",
        f"Select one with `tt-model serve {m.repo} --profile <name>`; the default is"
        f" `{default}`.",
        "",
        "## Provenance",
        "",
        f"- type: `{m.type}`",
        f"- weights: [`{m.weights}`](https://huggingface.co/{m.weights})"
        " — downloaded to the host HF cache at pull; never inside the image",
    ]
    for key in ("tt_metal", "vllm", "plugin", "image", "tt_model_version", "created_at"):
        if key in built:
            lines.append(f"- {key.replace('_', ' ')}: `{built[key]}`")
    lines += [
        "",
        "## What ships",
        "",
        "`code/` holds **exactly** the model's own code — the allowlist from the"
        " manifest, byte-identical to what is inside the image (both are staged from"
        " the same directory):",
        "",
        "```",
        *code_tree,
        "```",
        "",
        f"`image/` is the Docker image as an exploded OCI layout; `{MANIFEST_NAME}` is"
        " the full pinned manifest.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------- diagnosis
# Hub failures used to escape as a ~60-line traceback. Classification is a PURE
# function — exception in, dict out — so the 404/401/403/gated/offline matrix is
# unit-testable with no network and the CLI renders one card instead of a stack.
def _evidence(text: str) -> str:
    """The first line of an exception, minus tracking noise.

    huggingface_hub appends "(Request ID: Root=1-...;...)" to its HTTP errors. That is
    for a support ticket, not for the person reading their terminal.
    """
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return line.split(" (Request ID:")[0].strip()


def classify_hub_error(exc: BaseException, repo_id: str) -> dict:
    """Map a huggingface_hub exception to ``{cause, detail, evidence, actions}``.

    The 404 wording is deliberately two-sided. The Hub answers 404 for *both* "no such
    repo" and "private repo you cannot see" — it will not distinguish them for an
    unauthorised caller — so asserting "it does not exist" would be a guess that sends
    the user down the wrong path half the time.
    """
    name = type(exc).__name__
    text = str(exc)
    low = text.lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)

    # Offline / DNS / TLS beats everything: an unreachable Hub also can't confirm a
    # repo, and "you are offline" is the more useful half of that pair.
    if (name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled")
            or any(k in low for k in ("no such host", "dial tcp", "i/o timeout",
                                      "tls handshake", "connection refused",
                                      "connection error", "max retries exceeded",
                                      "temporary failure in name resolution"))):
        return {
            "cause": "cannot reach the Hub",
            "detail": "The request never got a response — this looks like a network or "
                      "DNS problem, not a missing package.",
            "evidence": _evidence(text),
            "actions": ["check your connection, then re-run",
                        "tt-model list   # what is already loaded locally"],
        }

    if status == 401 or name == "GatedRepoError" or "gated" in low:
        return {
            "cause": "you do not have access",
            "detail": f"The Hub recognised {repo_id} but refused it for this token — it "
                      "is gated, or the token is missing, expired, or lacks read scope.",
            "evidence": _evidence(text),
            "actions": ["hf auth login",
                        f"accept the terms at https://huggingface.co/{repo_id}"],
        }

    if status == 403:
        return {
            "cause": "not authorised",
            "detail": f"The Hub rejected this token for {repo_id}. It may lack read "
                      "scope, or the repo's terms may not be accepted for your account.",
            "evidence": _evidence(text),
            "actions": ["hf auth login", f"open https://huggingface.co/{repo_id}"],
        }

    if status == 404 or name == "RepositoryNotFoundError" or "not found" in low:
        return {
            "cause": "no such package, or no access",
            "detail": f"The Hub returned 404 for {repo_id}. Either the id is wrong, or "
                      "the repo is private and your token cannot see it — the Hub "
                      "reports both the same way.",
            "evidence": _evidence(text),
            "actions": [f"check the id: https://huggingface.co/{repo_id}",
                        "hf auth login   # if it is private"],
        }

    if name == "EntryNotFoundError" or MANIFEST_NAME in text:
        return {
            "cause": "not a tt-model package",
            "detail": f"The repo exists but has no {MANIFEST_NAME}, so there is nothing "
                      "for tt-model to run.",
            "evidence": _evidence(text),
            "actions": [f"https://huggingface.co/models?other={TT_MODEL_TAG}"],
        }

    return {
        "cause": "the Hub request failed",
        "detail": f"{name} while talking to the Hugging Face Hub about {repo_id}.",
        "evidence": _evidence(text),
        "actions": ["re-run with --verbose for the full traceback"],
    }


# ----------------------------------------------------------------- progress bridge
# huggingface_hub writes its own tqdm bars, and hf_xet writes more ("Download complete",
# "Reconstruction complete"). With our spinner also writing, two writers fought for one
# terminal row: bars interleaved mid-line, and unterminated bar output survived the
# process and painted over the next shell prompt — at which point leftover bytes were
# handed to bash as commands. So the CLI takes sole ownership of the row: HF's writers
# are silenced and their byte counts re-reported through our own activity line.
class _ActivityTqdm:
    """A tqdm stand-in that reports into ``console.activity`` instead of the terminal.

    huggingface_hub instantiates whatever ``tqdm_class`` it is handed, so implementing
    the slice it actually uses (init / update / close / context manager, plus ``total``
    and ``n``) is enough to divert the whole download's progress into one line we own.
    """

    label = "Working"
    _live = {}  # id -> (n, total), so concurrent file bars can be summed

    def __init__(self, *args, **kwargs):
        from . import console

        self._console = console
        self.total = kwargs.get("total")
        self.n = kwargs.get("initial", 0) or 0
        self.desc = kwargs.get("desc") or ""
        self.unit = kwargs.get("unit", "")
        self._key = id(self)
        # Only byte-denominated bars are worth aggregating; a "Fetching 5 files" bar
        # has a different unit and would corrupt the byte total.
        self._bytes = self.unit in ("B", "iB")
        if self._bytes:
            _ActivityTqdm._live[self._key] = (self.n, self.total or 0)
            self._render()

    # -- the tqdm surface huggingface_hub touches ---------------------------------
    def update(self, n=1):
        self.n += n or 0
        if self._bytes:
            _ActivityTqdm._live[self._key] = (self.n, self.total or 0)
            self._render()

    def close(self):
        _ActivityTqdm._live.pop(self._key, None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or ""

    def set_description_str(self, desc=None, refresh=True):
        self.desc = desc or ""

    def set_postfix(self, *a, **k):
        pass

    def set_postfix_str(self, *a, **k):
        pass

    def refresh(self, *a, **k):
        pass

    def reset(self, total=None):
        self.n = 0
        self.total = total

    def write(self, s, **k):
        pass  # tqdm.write() is a terminal escape hatch; we own the terminal here

    def __iter__(self):
        return iter(())

    @classmethod
    def _render(cls):
        done = sum(n for n, _ in cls._live.values())
        if not done:
            return
        from . import console

        console.activity.set(f"{cls.label}  {console.fmt_bytes(done)}")


def progress_bridge(label: str):
    """Silence HF/xet progress writers and route their byte counts to the activity row.

    Returns a ``tqdm_class`` to hand to ``snapshot_download``. Use as a context manager
    so the bars are restored on every exit path — leaving them disabled would silently
    strip progress from anything else in the process.
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
