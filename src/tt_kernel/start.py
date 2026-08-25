# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""``tt-model start`` — one guided path from nothing to a served model.

This orchestrates existing commands rather than reimplementing them: ``auth`` for the
token, ``toolchain``/``metal`` for the environment, ``hub`` to resolve the bundle, then the
same pull and serve code paths everything else uses. Behaviour is unchanged; what is new is
that the four steps are named up front, reported as they happen, and stop at the first one
that cannot succeed.

Interactive, but never *required*: every prompt has a non-interactive path (``--token``,
``$HF_TOKEN``, an existing HF token store, ``--yes``), and a non-TTY stdin skips prompting
entirely so this stays usable in CI.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from . import auth, console, hub, localdb, metal, runtime, toolchain

PHASES = ["Account", "Validate", "Hardware", "Model", "Serve"]
PHASE_DETAIL = {
    "Account": "Hugging Face token",
    "Validate": "tt-metal, vLLM, port",
    # Its own step, not a row in Validate: it answers a different question (is there a card,
    # and how many chips) from a different source (tt-smi, not the Python env), and its
    # answer is what the bundle's mesh has to match. Folded into the toolchain table it read
    # as one more version check.
    "Hardware": "tt-smi: arch + device count",
    "Model": "resolve + pull the bundle",
    "Serve": "launch the OpenAI server",
}


def stdin_is_interactive() -> bool:
    """Whether we may prompt at all.

    A prompt on a closed or piped stdin does not fail — it hangs, or reads EOF and takes a
    default the user never saw. Checking up front lets the caller degrade to "explain what
    is missing and exit" instead.
    """
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


# ----------------------------------------------------------------------------- account
@dataclass
class Account:
    name: Optional[str]
    source: str            # how we got the token
    logged_in: bool = False


def resolve_account(token: Optional[str] = None, *, allow_prompt: bool = True) -> Account:
    """Establish an HF identity, prompting only if there is no other way.

    Order: an explicit --token, then whatever huggingface_hub already has (its own token
    store or $HF_TOKEN), then a prompt. The token is read with getpass and handed straight
    to huggingface_hub — it is never echoed, logged, or held anywhere we render from.
    """
    if token:
        auth.login(token=token)
        me = auth.whoami()
        return Account(name=(me or {}).get("name"), source="--token", logged_in=bool(me))

    me = auth.whoami()
    if me:
        source = "$HF_TOKEN" if os.environ.get("HF_TOKEN") else "the HF token store"
        return Account(name=me.get("name"), source=source, logged_in=True)

    if not allow_prompt:
        return Account(name=None, source="none", logged_in=False)

    # A prompt must never run inside a capturing step() — it would be hidden and the CLI
    # would appear to hang. Callers arrange that; this only reads.
    secret = console.secret("Hugging Face token (input hidden): ")
    if not secret.strip():
        return Account(name=None, source="none", logged_in=False)
    auth.login(token=secret.strip())
    me = auth.whoami()
    return Account(name=(me or {}).get("name"), source="prompt", logged_in=bool(me))


# ---------------------------------------------------------------------------- validate
@dataclass
class Environment:
    report: object                       # toolchain.ToolchainReport
    arch: Optional[str]
    device_count: int
    device_source: Optional[str]
    port: int
    port_free: bool
    conflicts: list

    @property
    def blockers(self) -> List[str]:
        """What would stop a serve, in the order the user should fix it."""
        out = []
        for c in self.report.components:
            if not c.adequate:
                out.append(f"{c.name} is not available in this environment")
        if not self.port_free:
            out.append(f"port {self.port} is already in use")
        return out


def validate(port: int = 8000, *, arch_override: Optional[str] = None) -> Environment:
    """Check everything a serve needs, in one pass, before touching the network."""
    dev = metal.detect_device(arch_override=arch_override)
    return Environment(
        report=toolchain.check_toolchain(),
        arch=dev.arch,
        device_count=dev.device_count,
        device_source=dev.source,
        port=port,
        port_free=not runtime.port_in_use(port),
        conflicts=toolchain.check_environment(),
    )


# ------------------------------------------------------------------------------- model
def resolve_bundle(model: str) -> Tuple[str, str]:
    """Map what the user typed to an installed-or-installable bundle id.

    Returns ``(repo_id, how)``. Accepts a bundle id directly; for a bare HF model id it
    looks for an already-installed bundle that serves it, so `tt-model start Qwen/Qwen3-32B`
    works once `mando2222/Qwen3-32B-blackhole` is installed.
    """
    entry = localdb.get(model)
    if entry:
        return model, "installed"

    tail = model.split("/")[-1].lower()
    for e in localdb.all_entries():
        rid = e.get("repo_id") or ""
        if rid.split("/")[-1].lower().startswith(tail):
            return rid, f"installed bundle matching {model}"
    return model, "to pull"


def is_installed(repo_id: str) -> bool:
    """Whether the bundle is on disk — not merely recorded in the index.

    An index entry whose folder has been deleted made `start` skip the pull and fail three
    steps later at serve, while `_ensure_vllm_pulled` (which does check) would simply have
    re-pulled it. Trust the filesystem, not the bookkeeping.
    """
    entry = localdb.get(repo_id)
    path = entry.get("bundle_path") if entry else None
    return bool(path and Path(path).is_dir())


# ------------------------------------------------------------------- model discovery
@dataclass
class Choice:
    repo_id: str
    label: str
    servable: bool = True
    blocked_by: Optional[str] = None
    # What the bundle IS (backend · arch · …), without the reason it cannot run. The menu
    # renders the two in separate columns; `label` keeps them joined for callers that want
    # one string.
    meta: str = ""


def _servability(bundle_path: str) -> Tuple[bool, Optional[str]]:
    """Can this bundle's serving adapter actually be imported here?

    Checked at pick time, not just before launch. Auto-selecting "the only installed
    bundle" and then discovering at phase 4 that its adapter was never installed wastes the
    user's time and makes the guided path feel like it is guessing.
    """
    from pathlib import Path

    from . import bundles

    try:
        md = bundles.read_vllm_metadata(Path(bundle_path))
    except Exception:  # noqa: BLE001 — unreadable metadata is a different problem
        return True, None
    # The bundle folder itself counts: the TT plugin resolves adapters relative to each
    # EXTRA_MODELS_DIR entry, and some bundles ship their own models/ subtree.
    missing = runtime.missing_adapter_segment(getattr(md, "main_class", "") or "",
                                              search_paths=[str(bundle_path)])
    if missing:
        return False, f"{missing} is not importable"
    return True, None


def unregistered_bundles() -> List[Tuple[str, str]]:
    """``(repo_id, path)`` for bundle folders present on disk but absent from the index.

    A folder can be materialised without an index entry — pulled under a different
    XDG_CACHE_HOME, restored from a backup, or copied in by hand. It is servable in every
    practical sense (the plugin only needs it inside EXTRA_MODELS_DIR), so leaving it out of
    the menu hides a working model from its owner. Derived from the folder name, which
    `bundles.install_bundle` writes as ``namespace__name``.
    """
    from . import bundles as bundles_mod

    known = {e.get("repo_id") for e in localdb.all_entries()}
    out: List[Tuple[str, str]] = []
    try:
        root = bundles_mod.resolve_bundles_dir(None)
    except Exception:  # noqa: BLE001
        return out
    if not root.is_dir():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not (child / bundles_mod.VLLM_METADATA_NAME).is_file():
            continue
        repo_id = child.name.replace("__", "/", 1)
        if repo_id in known:
            continue
        out.append((repo_id, str(child)))
    return out


def installed_choices(*, check_servable: bool = True) -> List[Choice]:
    """Installed bundles as menu entries, each marked with whether it can serve here.

    `tt-model start` with no argument used to be a bare "Missing argument 'model'." — which
    is the one thing a guided command should not do. If there is something to serve, offer
    it; the user should not have to run `list` to find out what they already have.
    """
    out: List[Choice] = []
    for e in localdb.all_entries():
        repo_id = e.get("repo_id")
        bundle_path = e.get("bundle_path")
        if not repo_id or not bundle_path:
            continue
        bits = [b for b in (e.get("backend"), e.get("arch")) if b]
        if e.get("self_contained"):
            bits.append("self-contained")
        meta = " · ".join(bits)
        servable, blocked = (True, None)
        if check_servable:
            servable, blocked = _servability(bundle_path)
        if not servable and blocked:
            bits.append(blocked)
        suffix = f"  ({' · '.join(bits)})" if bits else ""
        out.append(Choice(repo_id=repo_id, label=f"{repo_id}{suffix}",
                          servable=servable, blocked_by=blocked, meta=meta))

    for repo_id, path in unregistered_bundles():
        servable, blocked = (True, None)
        if check_servable:
            servable, blocked = _servability(path)
        bits = ["on disk, not indexed"] + ([blocked] if blocked else [])
        out.append(Choice(repo_id=repo_id, label=f"{repo_id}  ({' · '.join(bits)})",
                          servable=servable, blocked_by=blocked,
                          meta="on disk, not indexed"))
    # Servable first, so a menu default is never a bundle we know cannot run.
    return sorted(out, key=lambda c: (not c.servable, c.repo_id))
