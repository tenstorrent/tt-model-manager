# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Docker image <-> exploded OCI layout on disk.

A container package carries its image as an *exploded* OCI layout
(``image/blobs/sha256/…``) rather than one giant tarball: layers are content-addressed
files, so HF/xet dedupes the blobs shared between models built on the same tt-metal
commit, a re-push uploads only what changed, and an interrupted multi-GB transfer
resumes per blob.

``skopeo`` does this conversion without a tar round-trip and is used when present; the
required path is plain ``docker save`` / ``docker load`` (Docker >= 25 emits and accepts
the OCI layout in its save tarballs).

This module is deliberately free of any tt-model concepts — it moves image bytes and
nothing else, so it is testable against a fake ``docker`` on PATH with no daemon.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Optional


class OciError(RuntimeError):
    """An image export/import that must not proceed. The message is user-facing."""


def _skopeo() -> Optional[str]:
    return shutil.which("skopeo")


def _extract_safely(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract a docker-save stream, refusing members that escape ``dest``.

    ``extractall(filter="data")`` is the one-liner for this, but it only exists from
    Python 3.11.4 — and this package declares ``requires-python = ">=3.9"``, where the
    keyword is a TypeError. ``tarfile.data_filter`` is the documented way to detect the
    feature, so use it when present and hand-check otherwise. The stream comes from the
    local docker daemon, but "trusted input" is not a reason to extract `../` paths.
    """
    if hasattr(tarfile, "data_filter"):
        tar.extractall(dest, filter="data")
        return
    for member in tar:
        name = member.name
        if name.startswith("/") or ".." in Path(name).parts:
            raise OciError(f"refusing to extract unsafe tar member {name!r}")
        tar.extract(member, dest)


def save(image: str, dest: Path) -> None:
    """Export a local docker image to an exploded OCI layout at ``dest``."""
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise OciError(f"refusing to export into non-empty {dest}")
    dest.mkdir(parents=True, exist_ok=True)

    sk = _skopeo()
    if sk:
        subprocess.run(
            [sk, "copy", f"docker-daemon:{image}", f"oci:{dest}:latest"], check=True
        )
        return

    # docker save emits an OCI-layout tar; stream-extract it straight into dest so the
    # image is never on disk twice (these are multi-GB).
    proc = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
            _extract_safely(tar, dest)
    finally:
        proc.stdout.close()
        if proc.wait() != 0:
            shutil.rmtree(dest, ignore_errors=True)
            raise OciError(f"docker save {image} failed")
    if not (dest / "oci-layout").exists():
        shutil.rmtree(dest, ignore_errors=True)
        raise OciError(
            "docker save did not produce an OCI layout — Docker >= 25 is required"
        )


def load(src: Path, expect_tag: Optional[str] = None) -> None:
    """Import an exploded OCI layout into the local docker daemon."""
    src = Path(src)
    if not (src / "oci-layout").exists():
        raise OciError(f"{src} is not an OCI layout (no oci-layout file)")

    sk = _skopeo()
    if sk and expect_tag:
        # ``:latest`` mirrors what save() writes. The reference is optional in the OCI
        # transport only when the index is unambiguous; naming it is symmetric and cannot
        # be wrong. (This branch runs only where skopeo is installed.)
        subprocess.run(
            [sk, "copy", f"oci:{src}:latest", f"docker-daemon:{expect_tag}"], check=True
        )
        return

    # Re-tar the directory and pipe it to docker load; nothing lands on disk twice.
    proc = subprocess.Popen(["docker", "load"], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        # dereference: an HF-cache snapshot is a symlink farm into blobs/, and docker
        # load needs the bytes, not the links.
        with tarfile.open(fileobj=proc.stdin, mode="w|", dereference=True) as tar:
            for path in sorted(src.rglob("*")):
                tar.add(path, arcname=str(path.relative_to(src)), recursive=False)
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise OciError("docker load failed")
