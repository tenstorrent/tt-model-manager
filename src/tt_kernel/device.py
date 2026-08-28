# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Detect arch, harvesting mask, and device count from the local hardware.

Detection order (most authoritative first):
  1. ``tt-smi`` JSON snapshot — the canonical source for arch + harvesting + count.
  2. ``ARCH_NAME`` environment variable — set in most tt-metal dev environments.
  3. An explicit ``--arch`` flag passed through by the caller.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional

# tt-smi reports board/arch names that we normalise to tt-metal's ARCH_NAME values.
_ARCH_ALIASES = {
    "blackhole": "blackhole",
    "bh": "blackhole",
    "wormhole": "wormhole_b0",
    "wormhole_b0": "wormhole_b0",
    "wh": "wormhole_b0",
    "grayskull": "grayskull",
    "gs": "grayskull",
}

# tt-smi's snapshot reports the marketing board_type (e.g. "p150a"), not the arch.
# Map board prefixes to ARCH_NAME. Blackhole = p100/p150/p300 boards; Wormhole =
# n150/n300 boards; Grayskull = e75/e150/e300 boards.
_BOARD_ARCH_PREFIXES = (
    ("p100", "blackhole"),
    ("p150", "blackhole"),
    ("p300", "blackhole"),
    ("n150", "wormhole_b0"),
    ("n300", "wormhole_b0"),
    ("e75", "grayskull"),
    ("e150", "grayskull"),
    ("e300", "grayskull"),
)


def normalize_arch(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    v = value.strip().lower()
    if v in _ARCH_ALIASES:
        return _ARCH_ALIASES[v]
    for prefix, arch in _BOARD_ARCH_PREFIXES:
        if v.startswith(prefix):
            return arch
    return v


@dataclass
class DeviceInfo:
    arch: Optional[str] = None
    device_count: int = 0
    harvesting_mask: Optional[int] = None
    source: str = "none"  # "tt-smi" | "env" | "flag" | "none"


def _run_tt_smi_snapshot() -> Optional[dict]:
    """Run ``tt-smi`` and return its JSON snapshot, or None if unavailable.

    ``tt-smi -s`` dumps the snapshot to stdout (and overrides ``-f``), so we capture
    stdout directly. ``--snapshot_no_tty`` forces clean JSON when not on a terminal.
    """
    exe = shutil.which("tt-smi")
    if not exe:
        return None
    for argv in ([exe, "-s", "--snapshot_no_tty"], [exe, "-s"]):
        try:
            proc = subprocess.run(argv, capture_output=True, timeout=30, text=True)
        except (subprocess.SubprocessError, OSError):
            continue
        out = (proc.stdout or "").strip()
        if not out:
            continue
        # Be tolerant of any leading non-JSON banner lines.
        start = out.find("{")
        if start == -1:
            continue
        try:
            return json.loads(out[start:])
        except json.JSONDecodeError:
            continue
    return None


def _parse_snapshot(snap: dict) -> DeviceInfo:
    """Extract arch / count / harvesting from a tt-smi snapshot.

    tt-smi schemas vary across versions, so we probe the common shapes:
    ``device_info`` is a list with per-board ``board_info``/``board_type`` and a
    harvesting field. We take arch from the first board and treat all boards as the
    same arch (mixed-arch hosts are out of scope).
    """
    boards = snap.get("device_info") or snap.get("devices") or []
    if not isinstance(boards, list) or not boards:
        return DeviceInfo()

    first = boards[0]
    raw_arch = (
        first.get("board_info", {}).get("board_type")
        if isinstance(first.get("board_info"), dict)
        else None
    )
    raw_arch = raw_arch or first.get("board_type") or first.get("arch")
    arch = normalize_arch(raw_arch)

    harvest = _extract_harvesting(first)
    if harvest is None:
        # tt-smi 5.x reports it under smbus_telem as a hex string.
        telem = first.get("smbus_telem")
        if isinstance(telem, dict):
            hv = telem.get("HARVESTING_STATE")
            if isinstance(hv, str):
                try:
                    harvest = int(hv, 0)
                except ValueError:
                    harvest = None
            elif isinstance(hv, int):
                harvest = hv

    return DeviceInfo(
        arch=arch,
        device_count=len(boards),
        harvesting_mask=harvest,
        source="tt-smi",
    )


def _extract_harvesting(board: dict) -> Optional[int]:
    """Pull a harvesting mask out of a board entry, tolerating schema drift."""
    for key in ("harvesting", "harvesting_mask", "tensix_harvesting"):
        val = board.get(key)
        if val is None and isinstance(board.get("board_info"), dict):
            val = board["board_info"].get(key)
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            try:
                return int(val, 0)
            except ValueError:
                continue
        if isinstance(val, dict):
            # Some snapshots nest like {"mask": "0x0"}.
            inner = val.get("mask") or val.get("value")
            if isinstance(inner, int):
                return inner
            if isinstance(inner, str):
                try:
                    return int(inner, 0)
                except ValueError:
                    pass
    return None


def detect(arch_override: Optional[str] = None) -> DeviceInfo:
    """Detect device info, honouring an explicit ``--arch`` override last-resort.

    The override only fills in arch when hardware detection can't; tt-smi remains the
    source of harvesting/count which an override cannot provide.
    """
    snap = _run_tt_smi_snapshot()
    if snap is not None:
        info = _parse_snapshot(snap)
        if info.arch:
            return info

    env_arch = normalize_arch(os.environ.get("ARCH_NAME"))
    if env_arch:
        return DeviceInfo(arch=env_arch, device_count=0, harvesting_mask=None, source="env")

    if arch_override:
        return DeviceInfo(
            arch=normalize_arch(arch_override), device_count=0, harvesting_mask=None, source="flag"
        )

    return DeviceInfo()


def known_arches() -> List[str]:
    return sorted(set(_ARCH_ALIASES.values()))
