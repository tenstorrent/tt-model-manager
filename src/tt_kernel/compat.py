# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Backward-compat shims for the ``tt-kernel`` -> ``tt-model`` rename.

Old installs invoked the ``tt-kernel`` command, set ``TT_KERNEL_*`` env vars, and stored data
under ``~/.cache|.config/tt-kernel``. All of that keeps working:
- the ``tt-kernel`` console script still maps to the same app (see pyproject ``[project.scripts]``);
- ``TT_KERNEL_*`` env vars are honored as a fallback for their ``TT_MODEL_*`` replacements;
- an existing legacy data dir is used when the new ``tt-model`` one doesn't exist yet, so a
  pre-rename install keeps finding its bundles/instances/index.

Remove these shims once the rename has been out long enough that no one is on the old paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# Old/new brand tokens used to translate env-var names and on-disk dir names.
_LEGACY = "tt-kernel"
_CURRENT = "tt-model"
_ENV_LEGACY_PREFIX = "TT_KERNEL_"
_ENV_CURRENT_PREFIX = "TT_MODEL_"


def env(name: str) -> Optional[str]:
    """``os.environ.get(name)`` for a ``TT_MODEL_*`` var, falling back to the legacy
    ``TT_KERNEL_*`` name so old scripts/CI keep working. Returns None if neither is set."""
    val = os.environ.get(name)
    if val is not None:
        return val
    if name.startswith(_ENV_CURRENT_PREFIX):
        return os.environ.get(_ENV_LEGACY_PREFIX + name[len(_ENV_CURRENT_PREFIX):])
    return val


def data_dir(base: Path) -> Path:
    """Return ``base/"tt-model"``, but prefer an existing legacy ``base/"tt-kernel"`` when the
    new dir doesn't exist yet — so a pre-rename install keeps using its already-populated dir."""
    current = base / _CURRENT
    legacy = base / _LEGACY
    if not current.exists() and legacy.exists():
        return legacy
    return current


def cache_dir() -> Path:
    """The per-user cache root, honoring the ``tt-kernel`` -> ``tt-model`` rename shim.

    Use this instead of hardcoding ``~/.cache/tt-model``. On a pre-rename box (only
    ``~/.cache/tt-kernel`` exists), hardcoding the new name and writing under it CREATES
    ``~/.cache/tt-model`` — which flips ``data_dir`` to the new dir for every OTHER consumer
    too (``localdb``, ``runtime``), orphaning the legacy ``installed.json`` and making the
    already-installed bundles vanish from ``list``/``rm``. Routing through here keeps all
    state agreeing on one dir. See issue #62."""
    return data_dir(Path.home() / ".cache")


def invoked_as_legacy() -> bool:
    """True when the process was started via the old ``tt-kernel`` command name."""
    import sys

    return Path(sys.argv[0]).name == _LEGACY
