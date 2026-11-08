# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A tiny local index of installed bundles, for ``list`` and ``rm``.

Stored at ``~/.cache/tt-model/installed.json``. This is a convenience record only —
the source of truth for what's usable is always the tt-metal cache on disk.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from . import compat, console

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]


def _index_path() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return compat.data_dir(Path(base)) / "installed.json"


@contextlib.contextmanager
def _index_lock(path: Path) -> Iterator[None]:
    """Serialize read-modify-write cycles across concurrent tt-model processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    if fcntl is None:
        yield
        return
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _recover_from_disk(cache_root: Path) -> Dict[str, dict]:
    """Rebuild index entries from pulled container manifests and legacy venv dirs on disk."""
    recovered: Dict[str, dict] = {}
    pulled_root = cache_root / "pulled"
    if pulled_root.is_dir():
        for d in sorted(pulled_root.iterdir()):
            mpath = d / "tt_kernel_manifest.json"
            if not mpath.is_file():
                continue
            try:
                raw = json.loads(mpath.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            c = raw.get("container") if isinstance(raw, dict) else None
            repo_id = (
                (c.get("repo") if isinstance(c, dict) else None)
                or (raw.get("repo") if isinstance(raw, dict) else None)
                or d.name.replace("__", "/", 1)
            )
            if isinstance(repo_id, str) and "/" in repo_id:
                recovered[repo_id] = {
                    "repo_id": repo_id,
                    "container": True,
                    "manifest_dir": str(d),
                }
    return recovered


def _load(*, for_write: bool = False) -> Dict[str, dict]:
    path = _index_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        if not for_write:
            return {}
        corrupt = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
        with contextlib.suppress(OSError):
            os.replace(path, corrupt)
        console.note(
            f"install index {path} was unreadable and moved to {corrupt.name}; "
            "recovering entries from disk",
            marker="!",
            style="warning",
        )
        return _recover_from_disk(path.parent)
    except OSError:
        if for_write:
            raise
        return {}


def _save(data: Dict[str, dict]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".installed.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise


def record(repo_id: str, entry: dict) -> None:
    """Record (or replace) an installed bundle keyed by ``namespace/name``."""
    path = _index_path()
    with _index_lock(path):
        data = _load(for_write=True)
        data[repo_id] = entry
        _save(data)


def get(repo_id: str) -> Optional[dict]:
    return _load().get(repo_id)


def remove(repo_id: str) -> bool:
    path = _index_path()
    with _index_lock(path):
        data = _load(for_write=True)
        if repo_id in data:
            del data[repo_id]
            _save(data)
            return True
        return False


def all_entries() -> List[dict]:
    """Return installed entries, each augmented with its ``repo_id``."""
    return [{"repo_id": k, **v} for k, v in sorted(_load().items())]
