# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""Host hardware detection from /dev/tenstorrent/by-id — and nothing heavier.

The kernel driver names each device link ``<arch>-<board serial>``::

    blackhole-0174A1EE6E739850 -> ../0
    blackhole-71B46ADDE6B74928 -> ../1

so arch and chip count come for free: no tt-smi, no tt-metal, no python deps. The
detection is deliberately crude — it cannot tell a p150 from a p300 — so it is used to
*warn and suggest*, never to hard-block a launch the author knows is right
(``--force`` proceeds).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .manifest import Manifest, ServeProfile, hardware_chip_count

BY_ID = Path("/dev/tenstorrent/by-id")


@dataclass(frozen=True)
class HostDevices:
    arch: Optional[str]  # "blackhole" / "wormhole" / ... or None if mixed/unknown
    chips: int


def detect(by_id: Optional[Path] = None) -> Optional[HostDevices]:
    """What this host has, or None when no TT devices are visible."""
    by_id = by_id or BY_ID   # module attr read at call time, so tests can repoint it
    try:
        entries = [e.name for e in by_id.iterdir() if not e.name.startswith(".")]
    except OSError:
        return None
    if not entries:
        return None
    arches = {e.split("-", 1)[0].lower() for e in entries if "-" in e}
    return HostDevices(arch=arches.pop() if len(arches) == 1 else None, chips=len(entries))


def profile_fits(profile: ServeProfile, host: HostDevices, manifest_arch: str) -> bool:
    """Does this profile's hardware target fit what the host reports?"""
    if host.arch is not None:
        # the driver says "blackhole"; the manifest says "blackhole"/"wormhole_b0"
        if not manifest_arch.lower().startswith(host.arch):
            return False
    chips = hardware_chip_count(profile.hardware or "")
    return chips is None or chips <= host.chips


def fitting_profiles(m: Manifest, host: HostDevices) -> List[str]:
    return [p.name for p in m.serve_profiles
            if profile_fits(m.resolve_profile(p.name), host, m.arch)]
