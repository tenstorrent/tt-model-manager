# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Detect the local device and tt-metal version without opening a device.

A self-contained bundle runs the engine from its own venv, so the only local facts that gate an
install are the device ``arch`` and ``device_count`` (see ``manifest.compare``). ``resolve_version``
is best-effort provenance, stamped into the manifest at package time.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from .device import detect as detect_device


@dataclass
class LocalEnv:
    """Everything ``manifest.compare`` needs about the local environment."""

    arch: Optional[str] = None
    device_count: int = 0


def _tt_metal_home() -> Optional[str]:
    return os.environ.get("TT_METAL_HOME") or os.environ.get("TT_METAL_RUNTIME_ROOT")


def _version_from_metadata() -> Optional[str]:
    """First version string among the tt-metal distributions, or None."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        for dist in ("ttnn", "tt-metal", "tt_metal", "metal-libs"):
            try:
                return version(dist)
            except PackageNotFoundError:
                continue
    except Exception:
        pass
    return None


def _version_from_git(home: str) -> Optional[str]:
    """``git describe`` in *home*, or None if it is not a git work tree."""
    if not shutil.which("git"):
        return None
    try:
        inside = subprocess.run(
            ["git", "-C", home, "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=10,
        )
        if inside.returncode != 0 or inside.stdout.strip() != "true":
            return None
    except (subprocess.SubprocessError, OSError):
        return None

    for argv in (
        ["git", "-C", home, "describe", "--tags", "--always", "--dirty"],
        ["git", "-C", home, "rev-parse", "HEAD"],
    ):
        try:
            out = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True)
            val = out.stdout.strip()
            if val:
                return val
        except (subprocess.SubprocessError, OSError):
            continue
    return None


def resolve_version() -> Optional[str]:
    """Resolve a tt-metal version string.

    Order: ``git describe`` in ``TT_METAL_HOME`` when that is a git work tree ->
    installed package metadata -> None.

    The git tree comes FIRST, and that ordering is the whole point of this
    function. tt-metal is very often installed editable from a source checkout,
    and an editable install writes its metadata exactly once -- at ``pip install
    -e`` time -- and never revisits it. Upgrading the checkout in place therefore
    leaves ``importlib.metadata`` reporting whatever the tree happened to be
    months ago, with no indication that it is stale.

    That is not hypothetical. Upgrading tt-metal to v0.77.0 in an editable tree
    left the metadata reading 0.65.1rc17.dev6200, and this function believed it.
    A probe that is wrong in the *stale* direction is worse than one that returns
    None: None is visibly missing, whereas stale looks authoritative.

    Metadata remains correct for a wheel install, which has no git tree, so it
    stays as the fallback rather than being removed.
    """
    home = _tt_metal_home()
    if home:
        from_git = _version_from_git(home)
        if from_git:
            return from_git
    return _version_from_metadata()


def local_env(arch_override: Optional[str] = None) -> LocalEnv:
    """Gather the local environment for compatibility comparison (arch + device_count)."""
    dev = detect_device(arch_override=arch_override)
    return LocalEnv(arch=dev.arch, device_count=dev.device_count)
