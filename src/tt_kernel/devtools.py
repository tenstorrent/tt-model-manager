# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Developer fixtures — the synthetic-cache generator behind ``tt-model dev``.

Ported from ``scripts/make_test_cache.sh`` so it lives with the code it fabricates for: the
layout below has to track ``cache.resolve_out_root``, and a shell script in another
directory drifts from it silently. Nothing here is part of the user-facing flow.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# The real on-disk shape (jit_compile_server.cpp): with --cache-dir/TT_METAL_CACHE set,
# tt-model glues the build_key onto the "tt-metal-cache" prefix, so build dirs are siblings
# directly under the root. See cache.resolve_out_root / _parent_and_prefix.
CACHE_PREFIX = "tt-metal-cache"
KERNELS = ("reader", "writer", "compute")
KERNEL_TARGETS = ("trisc0", "trisc1", "brisc")
FIRMWARE_TARGETS = ("trisc0", "trisc1", "brisc", "ncrisc", "erisc")


@dataclass
class TestCache:
    root: Path
    base: Path
    build_key: int
    kernel_count: int
    file_count: int
    wheel: Path = None


def _rand_bytes(low: int, high: int) -> bytes:
    """Varied sizes and contents, so per-file hashing and size reporting get exercised."""
    return os.urandom(low + int.from_bytes(os.urandom(2), "big") % (high - low))


def make_test_cache(root: str = "/tmp/ttk-test-cache", build_key: int = 4242,
                    *, with_runner: bool = False) -> TestCache:
    """Lay out a synthetic kernel cache (and optionally a real fake-runner wheel)."""
    root_path = Path(root).expanduser()
    base = root_path / f"{CACHE_PREFIX}{build_key}"
    if base.exists():
        shutil.rmtree(base)

    for kernel in KERNELS:
        for target in KERNEL_TARGETS:
            d = base / "kernels" / kernel / target
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{kernel}.elf").write_bytes(_rand_bytes(512, 4608))
            (d / f"{kernel}.hex").write_text(f"fake hex for {kernel}/{target}\n")
    for target in FIRMWARE_TARGETS:
        d = base / "firmware" / target
        d.mkdir(parents=True, exist_ok=True)
        (d / "fw.elf").write_bytes(_rand_bytes(256, 2304))

    result = TestCache(
        root=root_path, base=base, build_key=build_key,
        kernel_count=len(KERNELS),
        file_count=sum(1 for _ in base.rglob("*") if _.is_file()),
    )
    if with_runner:
        result.wheel = make_fake_runner_wheel(root_path)
    return result


def make_fake_runner_wheel(dest_dir: Path) -> Path:
    """Build a minimal but *real* wheel exposing ``fake_runner:Runner``.

    It needs a proper ``.dist-info/`` (METADATA + WHEEL + RECORD), not a bare METADATA
    file, or pip rejects it with "invalid wheel, .dist-info directory not found" — and pip
    is the only genuinely external step in the round-trip this fixture feeds.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    whl = dest_dir / "fake_runner-0.1-py3-none-any.whl"
    files = {
        "fake_runner/__init__.py":
            b'class Runner:\n    """Trivial fake runner for round-trip testing."""\n    pass\n',
        "fake_runner-0.1.dist-info/METADATA":
            b"Metadata-Version: 2.1\nName: fake-runner\nVersion: 0.1\n"
            b"Summary: round-trip test runner\n",
        "fake_runner-0.1.dist-info/WHEEL":
            b"Wheel-Version: 1.0\nGenerator: tt-model-dev\nRoot-Is-Purelib: true\n"
            b"Tag: py3-none-any\n",
    }
    record: List[str] = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record.append(f"{name},sha256={digest},{len(data)}")
    record.append("fake_runner-0.1.dist-info/RECORD,,")
    with zipfile.ZipFile(whl, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("fake_runner-0.1.dist-info/RECORD", ("\n".join(record) + "\n").encode())
    return whl


def push_recipe(cache: TestCache) -> List[Tuple[str, str]]:
    """``(description, command)`` pairs for what to do with the fixture just built."""
    if cache.wheel:
        return [("Push a v2 bundle (kernels + runner + weights)",
                 f'tt-model push <ns>/<name> --private --cache-dir "{cache.root}" '
                 f'--arch blackhole --tt-metal-version v0.99-test '
                 f'--python-package "{cache.wheel}" --runner-spec fake_runner:Runner '
                 f'--entry-point demo --weights org/model')]
    return [("Push it",
             f'tt-model push <ns>/<name> --private --cache-dir "{cache.root}" '
             f'--arch blackhole --tt-metal-version v0.99-test --model google/gemma-test')]
